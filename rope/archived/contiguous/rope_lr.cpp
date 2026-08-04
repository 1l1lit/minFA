#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "kernel_operator.h"

using namespace AscendC;

constexpr int N = 2;
constexpr int D = 8;
constexpr int S = D * sizeof(float);

class Kernel {
public:
    __aicore__ inline Kernel(GM_ADDR Kl, GM_ADDR Kr, GM_ADDR cos, GM_ADDR sin)
    {
        pipe.InitBuffer(ub, 192 * 1024);
        buf = ub.Get<uint8_t>();

        g_Kl.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(Kl), N * D);
        g_Kr.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(Kr), N * D);
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
        DataCopy(Kl, g_Kl, N * D);
        DataCopy(Kr, g_Kr, N * D);
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
        Sub(Kl, Kl, tmp, N * D);
    }

    __aicore__ inline void Store()
    {
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(g_Kl, Kl, N * D);
        DataCopy(g_Kr, Kr, N * D);
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
    GlobalTensor<float> g_Kl, g_Kr, g_cos, g_sin;
};

__global__ __aicore__ void kernel_func(GM_ADDR Kl, GM_ADDR Kr, GM_ADDR cos, GM_ADDR sin)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Kernel op(Kl, Kr, cos, sin);
    op.Main();
}

void rope_npu(std::vector<float>& Kl, std::vector<float>& Kr, std::vector<float>& cos, std::vector<float>& sin)
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    uint8_t* h_Kl = reinterpret_cast<uint8_t*>(Kl.data());
    uint8_t* h_Kr = reinterpret_cast<uint8_t*>(Kr.data());
    uint8_t* h_cos = reinterpret_cast<uint8_t*>(cos.data());
    uint8_t* h_sin = reinterpret_cast<uint8_t*>(sin.data());
    uint8_t* d_Kl = nullptr;
    uint8_t* d_Kr = nullptr;
    uint8_t* d_cos = nullptr;
    uint8_t* d_sin = nullptr;

    aclrtMalloc(reinterpret_cast<void**>(&d_Kl), N * S, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&d_Kr), N * S, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&d_cos), S, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&d_sin), S, ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(d_Kl, N * S, h_Kl, N * S, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(d_Kr, N * S, h_Kr, N * S, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(d_cos, S, h_cos, S, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(d_sin, S, h_sin, S, ACL_MEMCPY_HOST_TO_DEVICE);

    kernel_func<<<1, nullptr, stream>>>(d_Kl, d_Kr, d_cos, d_sin);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(h_Kl, N * S, d_Kl, N * S, ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(h_Kr, N * S, d_Kr, N * S, ACL_MEMCPY_DEVICE_TO_HOST);

    aclrtFree(d_Kl);
    aclrtFree(d_Kr);
    aclrtFree(d_cos);
    aclrtFree(d_sin);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}

void rope_cpu(std::vector<float>& Kl, std::vector<float>& Kr, std::vector<float>& cos, std::vector<float>& sin)
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < D; j++) {
            float real = Kl[i * D + j] * cos[j] - Kr[i * D + j] * sin[j];
            float imag = Kl[i * D + j] * sin[j] + Kr[i * D + j] * cos[j];
            Kl[i * D + j] = real;
            Kr[i * D + j] = imag;
        }
    }
}

int main()
{
    std::vector<float> cos(D);
    std::vector<float> sin(D);
    std::vector<float> Kl1(N * D);
    std::vector<float> Kr1(N * D);
    std::vector<float> Kl2(N * D);
    std::vector<float> Kr2(N * D);

    for (int i = 0; i < D; i++) {
        cos[i] = static_cast<float>(i + 1);
        sin[i] = static_cast<float>(i + 1);
    }

    for (int i = 0; i < N * D; i++) {
        Kl1[i] = Kl2[i] = static_cast<float>(i + 1);
        Kr1[i] = Kr2[i] = static_cast<float>(N * D + i + 1);
    }

    rope_npu(Kl1, Kr1, cos, sin);
    rope_cpu(Kl2, Kr2, cos, sin);

    if (Kl1 == Kl2 && Kr1 == Kr2) {
        std::cout << "Results match!" << std::endl;
    } else {
        std::cout << "Results do not match!" << std::endl;
        std::cout << "NPU output: " << std::endl;
        for (const auto& val : Kl1) {
            std::cout << val << " ";
        }
        for (const auto& val : Kr1) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
        std::cout << "CPU output: " << std::endl;
        for (const auto& val : Kl2) {
            std::cout << val << " ";
        }
        for (const auto& val : Kr2) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
