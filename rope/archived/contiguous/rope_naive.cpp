#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "kernel_operator.h"

using namespace AscendC;

constexpr int N = 48;
constexpr int SIZE = N * sizeof(float);

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

        K_left = GetBuf<float>(2 * SIZE);
        K_right = GetBuf<float>(3 * SIZE);
        tmp = GetBuf<float>(4 * SIZE);
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
        Mul(tmp, K_right, sin, N);
        Mul(K_right, K_right, cos, N);
        PipeBarrier<PIPE_V>();
        MulAddDst(K_right, K_left, sin, N);
        Mul(K_left, K_left, cos, N);
        PipeBarrier<PIPE_V>();
        Sub(K_left, K_left, tmp, N);
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
    LocalTensor<float> K_left, K_right, tmp;
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
        float real = K[i] * cos[i] - K[N + i] * sin[i];
        float imag = K[i] * sin[i] + K[N + i] * cos[i];
        K[i] = real;
        K[N + i] = imag;
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
