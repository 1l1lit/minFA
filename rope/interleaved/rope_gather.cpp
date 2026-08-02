#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "kernel_operator.h"

using namespace AscendC;

#define AsI32() template ReinterpretCast<int32_t>()

constexpr int N = 48;
constexpr int FLTB = 4;
constexpr int SIZE = N * 4;

class Kernel {
public:
    __aicore__ inline Kernel(GM_ADDR K, GM_ADDR cos, GM_ADDR sin)
    {
        pipe.InitBuffer(ub, 192 * 1024);
        buf = ub.Get<uint8_t>();

        g_K.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(K), 2 * N);
        g_cos.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cos), N);
        g_sin.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sin), N);
    }

    __aicore__ inline void Main()
    {
        Init();
        Load();
        Playground();
        Store();
    }

private:
    __aicore__ inline void Init()
    {
        // Allocate buffer space
        cos = GetBuf<float>(0);
        sin = GetBuf<float>(SIZE);
        K = GetBuf<float>(2 * SIZE);

        cosG = GetBuf<float>(4 * SIZE);
        sinG = GetBuf<float>(6 * SIZE);
        KG = GetBuf<float>(8 * SIZE);

        csO = GetBuf<uint32_t>(8 * SIZE);
        KO = GetBuf<uint32_t>(0);
    }

    __aicore__ inline void Load()
    {
        DataCopy(K, g_K, 2 * N);
        DataCopy(cos, g_cos, N);
        DataCopy(sin, g_sin, N);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }

    __aicore__ inline void Playground()
    {
        // *** Scatter cos/sin ***
        int n = 8;
        for (int i = 0; i < n; i++) {
            csO(i) = (i / 2) * FLTB;
        }
        while (n < 2 * N) {
            if (n < N) {
                Adds(csO[n].AsI32(), csO.AsI32(), (n / 2) * FLTB, n);
                PipeBarrier<PIPE_V>();
                n *= 2;
            } else {
                Adds(csO[n].AsI32(), csO.AsI32(), (n / 2) * FLTB, 2 * N - n);
                PipeBarrier<PIPE_V>();
                break;
            }
        }
        Gather(cosG, cos, csO, 0, 2 * N);
        Gather(sinG, sin, csO, 0, 2 * N);
        PipeBarrier<PIPE_V>();

        // *** Negate sin (interleave) ***
        int repeatTimes = (2 * N) / 64;
        int remainder = (2 * N) % 64;
        uint64_t mask[2] = {0x5555555555555555, 0};
        Muls(sinG, sinG, -1.0f, mask, repeatTimes, {1, 1, 8, 8});
        if (remainder) {
            int offset = repeatTimes * 64;
            mask[0] = 0x55;
            Muls(sinG[offset], sinG[offset], -1.0f, mask, remainder / 8, {1, 1, 1, 1});
        }

        // *** Interleave K ***
        n = 8;
        for (int i = 0; i < n; i++) {
            KO(i) = (i ^ 1) * FLTB;
        }
        while (n < 2 * N) {
            if (n < N) {
                Adds(KO[n].AsI32(), KO.AsI32(), n * FLTB, n);
                PipeBarrier<PIPE_V>();
                n *= 2;
            } else {
                Adds(KO[n].AsI32(), KO.AsI32(), n * FLTB, 2 * N - n);
                PipeBarrier<PIPE_V>();
                break;
            }
        }
        Gather(KG, K, KO, 0, 2 * N);

        // *** Calculate ***
        Mul(K, K, cosG, 2 * N);
        PipeBarrier<PIPE_V>();
        MulAddDst(K, KG, sinG, 2 * N);
    }

    __aicore__ inline void Store()
    {
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(g_K, K, 2 * N);
    }

private:
    template <typename U>
    __aicore__ inline LocalTensor<U> GetBuf(uint32_t offset)
    {
        return buf[offset].template ReinterpretCast<U>();
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> ub;
    LocalTensor<uint8_t> buf;

    LocalTensor<float> K, cos, sin;
    LocalTensor<float> KG, cosG, sinG;
    LocalTensor<uint32_t> csO, KO;
    GlobalTensor<float> g_K, g_cos, g_sin;
};

__global__ __aicore__ void kernel_func(GM_ADDR K, GM_ADDR cos, GM_ADDR sin)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Kernel op(K, cos, sin);
    op.Main();
}

void rope_npu(std::vector<float>& K, std::vector<float>& cos, std::vector<float>& sin)
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    uint8_t* h_K = reinterpret_cast<uint8_t*>(K.data());
    uint8_t* h_cos = reinterpret_cast<uint8_t*>(cos.data());
    uint8_t* h_sin = reinterpret_cast<uint8_t*>(sin.data());
    uint8_t* d_K = nullptr;
    uint8_t* d_cos = nullptr;
    uint8_t* d_sin = nullptr;

    aclrtMalloc(reinterpret_cast<void**>(&d_K), 2 * SIZE, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&d_cos), SIZE, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&d_sin), SIZE, ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(d_K, 2 * SIZE, h_K, 2 * SIZE, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(d_cos, SIZE, h_cos, SIZE, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(d_sin, SIZE, h_sin, SIZE, ACL_MEMCPY_HOST_TO_DEVICE);

    kernel_func<<<1, nullptr, stream>>>(d_K, d_cos, d_sin);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(h_K, 2 * SIZE, d_K, 2 * SIZE, ACL_MEMCPY_DEVICE_TO_HOST);

    aclrtFree(d_K);
    aclrtFree(d_cos);
    aclrtFree(d_sin);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}

void rope_cpu(std::vector<float>& K, std::vector<float>& cos, std::vector<float>& sin)
{
    for (int i = 0; i < N; i++) {
        float real = K[2 * i] * cos[i] - K[2 * i + 1] * sin[i];
        float imag = K[2 * i] * sin[i] + K[2 * i + 1] * cos[i];
        K[2 * i] = real;
        K[2 * i + 1] = imag;
    }
}

int main()
{
    std::vector<float> cos(N);
    std::vector<float> sin(N);
    std::vector<float> K1(2 * N);
    std::vector<float> K2(2 * N);

    for (int i = 0; i < N; i++) {
        cos[i] = static_cast<float>(i + 1);
        sin[i] = static_cast<float>(i + 1);
    }

    std::cout << "Input K: " << std::endl;
    for (int i = 0; i < 2 * N; i++) {
        K1[i] = K2[i] = static_cast<float>(i + 1);
        std::cout << K1[i] << " ";
    }
    std::cout << std::endl;

    rope_npu(K1, cos, sin);
    rope_cpu(K2, cos, sin);

    if (K1 == K2) {
        std::cout << "Results match!" << std::endl;
    } else {
        std::cout << "Results do not match!" << std::endl;
        std::cout << "NPU output: " << std::endl;
        for (const auto& val : K1) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
        std::cout << "CPU output: " << std::endl;
        for (const auto& val : K2) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
