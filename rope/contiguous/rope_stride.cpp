#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "kernel_operator.h"

using namespace AscendC;

constexpr int N = 7;   // num_heads
constexpr int D = 144; // rotary_dim / 2，需为 8 的倍数
constexpr int S = D * sizeof(float);

class Kernel {
public:
    __aicore__ inline Kernel(GM_ADDR K, GM_ADDR cos, GM_ADDR sin)
    {
        pipe.InitBuffer(ub, 192 * 1024);
        buf = ub.Get<uint8_t>();

        g_K.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(K), 2 * N * D);
        g_cos.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cos), D);
        g_sin.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sin), D);
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
        Kl = GetBuf<float>(0);
        Kr = GetBuf<float>(N * S);
        cos = GetBuf<float>(2 * N * S);
        sin = GetBuf<float>(2 * N * S + S);
        tmp = GetBuf<float>(2 * N * S + 2 * S);
    }

    __aicore__ inline void Load()
    {
        DataCopy(Kl, g_K, {N, D / 8, D / 8, 0});
        DataCopy(Kr, g_K[D], {N, D / 8, D / 8, 0});
        DataCopy(cos, g_cos, D);
        DataCopy(sin, g_sin, D);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }

    __aicore__ inline void Playground()
    {
        int n = 0;
        while (n < D) {
            int stride = Std::min(D - n, 64);
            Mul(tmp[n], Kr[n], sin[n], stride, N, {1, 1, 1, D / 8, D / 8, 0});
            Mul(Kr[n], Kr[n], cos[n], stride, N, {1, 1, 1, D / 8, D / 8, 0});
            PipeBarrier<PIPE_V>();
            MulAddDst(Kr[n], Kl[n], sin[n], stride, N, {1, 1, 1, D / 8, D / 8, 0});
            Mul(Kl[n], Kl[n], cos[n], stride, N, {1, 1, 1, D / 8, D / 8, 0});
            n += stride;
        }
        PipeBarrier<PIPE_V>();
        Sub(Kl, Kl, tmp, N * D);
    }

    __aicore__ inline void Store()
    {
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(g_K, Kl, {N, D / 8, 0, D / 8});
        DataCopy(g_K[D], Kr, {N, D / 8, 0, D / 8});
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

    LocalTensor<float> Kl, Kr, cos, sin;
    LocalTensor<float> tmp;
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

    aclrtMalloc(reinterpret_cast<void**>(&d_K), 2 * N * S, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&d_cos), S, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&d_sin), S, ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(d_K, 2 * N * S, h_K, 2 * N * S, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(d_cos, S, h_cos, S, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(d_sin, S, h_sin, S, ACL_MEMCPY_HOST_TO_DEVICE);

    kernel_func<<<1, nullptr, stream>>>(d_K, d_cos, d_sin);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(h_K, 2 * N * S, d_K, 2 * N * S, ACL_MEMCPY_DEVICE_TO_HOST);

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
        for (int j = 0; j < D; j++) {
            float real = K[(2 * D) * i + j] * cos[j] - K[(2 * D) * i + j + D] * sin[j];
            float imag = K[(2 * D) * i + j] * sin[j] + K[(2 * D) * i + j + D] * cos[j];
            K[(2 * D) * i + j] = real;
            K[(2 * D) * i + j + D] = imag;
        }
    }
}

int main()
{
    std::vector<float> cos(D);
    std::vector<float> sin(D);
    std::vector<float> K1(2 * N * D);
    std::vector<float> K2(2 * N * D);

    for (int i = 0; i < D; i++) {
        cos[i] = static_cast<float>(i + 1);
        sin[i] = static_cast<float>(i + 1);
    }

    for (int i = 0; i < 2 * N * D; i++) {
        K1[i] = K2[i] = static_cast<float>(i + 1);
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
