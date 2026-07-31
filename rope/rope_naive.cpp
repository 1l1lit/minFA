#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "kernel_operator.h"

using namespace AscendC;
using T = float;

constexpr int N = 32;
constexpr int SIZE = N * sizeof(T);
constexpr bool IS_INTERLEAVED = true;

template <class T, class U>
__aicore__ inline constexpr auto CeilDiv(T const& dividend, U const& divisor)
{
    return (dividend + divisor - 1) / divisor;
}

template <bool IS_INTERLEAVED_ = true>
class Kernel {
public:
    __aicore__ inline Kernel(GM_ADDR K, GM_ADDR cos, GM_ADDR sin)
    {
        pipe.InitBuffer(ub, 192 * 1024);
        buf = ub.Get<uint8_t>();

        g_K.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(K), 2 * N);
        g_cos.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(cos), N);
        g_sin.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(sin), N);
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
        cos = GetBuf<T>(0);
        sin = GetBuf<T>(SIZE);
        K = GetBuf<T>(2 * SIZE);

        if constexpr (IS_INTERLEAVED_) {
            K_even = GetBuf<T>(4 * SIZE);
            K_odd = GetBuf<T>(5 * SIZE);
            tmp = GetBuf<T>(2 * SIZE);
        } else {
            K_left = GetBuf<T>(2 * SIZE);
            K_right = GetBuf<T>(3 * SIZE);
            tmp = GetBuf<T>(4 * SIZE);
        }
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
        if constexpr (IS_INTERLEAVED_) {
            uint64_t rsvdCnt = 0;
            GatherMask(K_even, K, 1, false, 0, {1, CeilDiv(N, 64), 8, 0}, rsvdCnt);
            GatherMask(K_odd, K, 2, false, 0, {1, CeilDiv(N, 64), 8, 0}, rsvdCnt);
            PipeBarrier<PIPE_V>();
            Mul(tmp, K_odd, sin, N);
            Mul(K_odd, K_odd, cos, N);
            PipeBarrier<PIPE_V>();
            MulAddDst(K_odd, K_even, sin, N);
            Mul(K_even, K_even, cos, N);
            PipeBarrier<PIPE_V>();
            Sub(K_even, K_even, tmp, N);
            K = GetBuf<T>(4 * SIZE);
        } else {
            Mul(tmp, K_right, sin, N);
            Mul(K_right, K_right, cos, N);
            PipeBarrier<PIPE_V>();
            MulAddDst(K_right, K_left, sin, N);
            Mul(K_left, K_left, cos, N);
            PipeBarrier<PIPE_V>();
            Sub(K_left, K_left, tmp, N);
        }
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

    LocalTensor<T> K, cos, sin;
    LocalTensor<T> K_even, K_odd, K_left, K_right, tmp;
    GlobalTensor<T> g_K, g_cos, g_sin;
};

template <bool IS_INTERLEAVED_ = true>
__global__ __aicore__ void kernel_func(GM_ADDR K, GM_ADDR cos, GM_ADDR sin)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Kernel<IS_INTERLEAVED_> op(K, cos, sin);
    op.Main();
}

template <bool IS_INTERLEAVED_ = true>
void rope_npu(std::vector<T>& K, std::vector<T>& cos, std::vector<T>& sin)
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

    kernel_func<IS_INTERLEAVED_><<<1, nullptr, stream>>>(d_K, d_cos, d_sin);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(h_K, 2 * SIZE, d_K, 2 * SIZE, ACL_MEMCPY_DEVICE_TO_HOST);

    aclrtFree(d_K);
    aclrtFree(d_cos);
    aclrtFree(d_sin);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}

template <bool IS_INTERLEAVED_ = true>
void rope_cpu(std::vector<T>& K, std::vector<T>& cos, std::vector<T>& sin)
{
    if constexpr (IS_INTERLEAVED_) {
        for (int i = 0; i < N; i++) {
            float real = K[2 * i] * cos[i] - K[2 * i + 1] * sin[i];
            float imag = K[2 * i] * sin[i] + K[2 * i + 1] * cos[i];
            K[2 * i] = real;
            K[2 * i + 1] = imag;
        }
    } else {
        for (int i = 0; i < N; i++) {
            float real = K[i] * cos[i] - K[N + i] * sin[i];
            float imag = K[i] * sin[i] + K[N + i] * cos[i];
            K[i] = real;
            K[N + i] = imag;
        }
    }
}

int main()
{
    std::vector<T> K1(2 * N, 1.0f);
    std::vector<T> K2(2 * N, 1.0f);
    std::vector<T> cos(N, 1.0f);
    std::vector<T> sin(N, 1.0f);

    std::cout << "Input K: " << std::endl;
    for (int i = 0; i < 2 * N; i++) {
        K1[i] = static_cast<T>(i);
        K2[i] = static_cast<T>(i);
        std::cout << K1[i] << " ";
    }
    std::cout << std::endl;

    rope_npu<IS_INTERLEAVED>(K1, cos, sin);
    std::cout << "NPU output: " << std::endl;
    if constexpr (IS_INTERLEAVED) {
        for (int i = 0; i < N; i++) {
            std::cout << K1[i] << " " << K1[N + i] << " ";
        }
    } else {
        for (const auto& val : K1) {
            std::cout << val << " ";
        }
    }
    std::cout << std::endl;

    rope_cpu<IS_INTERLEAVED>(K2, cos, sin);
    std::cout << "CPU output: " << std::endl;
    for (const auto& val : K2) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}
