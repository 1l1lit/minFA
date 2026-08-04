#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "kernel_operator.h"

using namespace AscendC;

#define AsI32() template ReinterpretCast<int32_t>()

constexpr int N = 7;      // num_heads
constexpr int HD = 144;   // rotary_dim / 2，需为 8 的倍数
constexpr int D = 2 * HD; // rotary_dim

constexpr int FS = sizeof(float);
constexpr int HS = HD * FS;
constexpr int S = 2 * HS;

class Kernel {
public:
    __aicore__ inline Kernel(GM_ADDR K, GM_ADDR cos, GM_ADDR sin)
    {
        pipe.InitBuffer(ub, 192 * 1024);
        buf = ub.Get<uint8_t>();

        g_K.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(K), N * D);
        g_cos.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cos), HD);
        g_sin.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sin), HD);
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
        K = GetBuf<float>(0);
        cos = GetBuf<float>(N * S);
        sin = GetBuf<float>(N * S + HS);

        KG = GetBuf<float>(N * S);
        cosG = GetBuf<float>(2 * N * S);
        sinG = GetBuf<float>(2 * N * S + S);

        KO = GetBuf<uint32_t>(N * S);
        csO = GetBuf<uint32_t>(2 * N * S + S);
    }

    __aicore__ inline void Load()
    {
        DataCopy(K, g_K, N * D);
        DataCopy(cos, g_cos, HD);
        DataCopy(sin, g_sin, HD);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    }

    __aicore__ inline void Playground()
    {
        // *** Scatter cos/sin ***
        int n = 8;
        for (int i = 0; i < n; i++) {
            csO(i) = (i / 2) * FS;
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        while (n < D) {
            if (n < HD) {
                Adds(csO[n].AsI32(), csO.AsI32(), (n / 2) * FS, n);
                PipeBarrier<PIPE_V>();
                n *= 2;
            } else {
                Adds(csO[n].AsI32(), csO.AsI32(), (n / 2) * FS, D - n);
                PipeBarrier<PIPE_V>();
                break;
            }
        }
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Gather(cosG, cos, csO, 0, D);
        Gather(sinG, sin, csO, 0, D);
        PipeBarrier<PIPE_V>();

        // *** Negate sin (interleave) ***
        int repeat = D / 64;
        int offset = repeat * 64;
        uint64_t mask[1] = {0x5555555555555555};
        Muls(sinG, sinG, -1.0f, mask, repeat, {1, 1, 8, 8});
        if (offset < D) {
            mask[0] >>= (D - offset) / 8;
            Muls(sinG[offset], sinG[offset], -1.0f, mask, 1, {1, 1, 0, 0});
        }

        // *** Interleave K ***
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        n = 8;
        for (int i = 0; i < n; i++) {
            KO(i) = (i ^ 1) * FS;
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        while (n < N * D) {
            if (n < N * HD) {
                Adds(KO[n].AsI32(), KO.AsI32(), n * FS, n);
                PipeBarrier<PIPE_V>();
                n *= 2;
            } else {
                Adds(KO[n].AsI32(), KO.AsI32(), n * FS, N * D - n);
                PipeBarrier<PIPE_V>();
                break;
            }
        }
        Gather(KG, K, KO, 0, N * D);

        // *** Calculate ***
        n = 0;
        while (n < D) {
            int stride = Std::min(D - n, 64);
            Mul(K[n], K[n], cosG[n], stride, N, {1, 1, 1, D / 8, D / 8, 0});
            PipeBarrier<PIPE_V>();
            MulAddDst(K[n], KG[n], sinG[n], stride, N, {1, 1, 1, D / 8, D / 8, 0});
            n += stride;
        }
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    }

    __aicore__ inline void Store()
    {
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(g_K, K, N * D);
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
    LocalTensor<uint32_t> KO, csO;
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

    aclrtMalloc(reinterpret_cast<void**>(&d_K), N * S, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&d_cos), HS, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&d_sin), HS, ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(d_K, N * S, h_K, N * S, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(d_cos, HS, h_cos, HS, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(d_sin, HS, h_sin, HS, ACL_MEMCPY_HOST_TO_DEVICE);

    kernel_func<<<1, nullptr, stream>>>(d_K, d_cos, d_sin);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(h_K, N * S, d_K, N * S, ACL_MEMCPY_DEVICE_TO_HOST);

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
        for (int j = 0; j < HD; j++) {
            float real = K[i * D + 2 * j] * cos[j] - K[i * D + 2 * j + 1] * sin[j];
            float imag = K[i * D + 2 * j] * sin[j] + K[i * D + 2 * j + 1] * cos[j];
            K[i * D + 2 * j] = real;
            K[i * D + 2 * j + 1] = imag;
        }
    }
}

int main()
{
    std::vector<float> cos(HD);
    std::vector<float> sin(HD);
    std::vector<float> K1(N * D);
    std::vector<float> K2(N * D);

    for (int i = 0; i < HD; i++) {
        cos[i] = static_cast<float>(i + 1);
        sin[i] = static_cast<float>(i + 1);
    }
    for (int i = 0; i < N * D; i++) {
        K1[i] = static_cast<float>(i + 1);
        K2[i] = static_cast<float>(i + 1);
    }

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
