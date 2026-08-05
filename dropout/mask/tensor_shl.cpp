#include <iostream>
#include <vector>
#include <random>
#include <bitset>
#include "acl/acl.h"
#include "kernel_operator.h"

using namespace AscendC;

#define AsU16() template ReinterpretCast<uint16_t>()
#define AsI32() template ReinterpretCast<int32_t>()

constexpr int N = 8;
constexpr int SIZE = N * sizeof(uint32_t);

class Kernel {
public:
    __aicore__ inline Kernel(GM_ADDR x, uint32_t offset) : offset(offset)
    {
        pipe.InitBuffer(ub, 192 * 1024);
        buf = ub.Get<uint8_t>();

        g_x.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(x), N);
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
        x = GetBuf<uint32_t>(0);
        t0 = GetBuf<uint32_t>(SIZE);
        t1 = GetBuf<uint32_t>(2 * SIZE);
    }

    __aicore__ inline void Load()
    {
        DataCopy(x, g_x, N);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }

    __aicore__ inline void Playground()
    {
        // Tensor shift left
        ShiftRight(t0, x, 32 - offset, N);
        PipeBarrier<PIPE_V>();
        ShiftLeft(x, x, offset, N);
        int n = 8;
        for (int i = 0; i < n; i++) {
            t1(i) = 4 * i;
        }
        while (n < N) {
            if (2 * n < N) {
                Adds(t1[n].AsI32(), t1.AsI32(), 4 * n, n);
                PipeBarrier<PIPE_V>();
                n *= 2;
            } else {
                Adds(t1[n].AsI32(), t1.AsI32(), 4 * n, N - n);
                PipeBarrier<PIPE_V>();
                break;
            }
        }
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Gather(t1, t0, t1, 4, N - 1);
        PipeBarrier<PIPE_V>();
        Or(x.AsU16(), x.AsU16(), t1.AsU16(), 2 * (N - 1));
    }

    __aicore__ inline void Store()
    {
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(g_x, x, N);
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

    uint32_t offset;
    LocalTensor<uint32_t> x;
    LocalTensor<uint32_t> t0, t1;
    GlobalTensor<uint32_t> g_x;
};

__global__ __aicore__ void kernel_func(GM_ADDR x, uint32_t offset)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Kernel op(x, offset);
    op.Main();
}

void launch_kernel(std::vector<uint32_t>& x, uint32_t offset)
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    uint8_t* h_x = reinterpret_cast<uint8_t*>(x.data());
    uint8_t* d_x = nullptr;

    aclrtMalloc(reinterpret_cast<void**>(&d_x), SIZE, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(d_x, SIZE, h_x, SIZE, ACL_MEMCPY_HOST_TO_DEVICE);
    kernel_func<<<1, nullptr, stream>>>(d_x, offset);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(h_x, SIZE, d_x, SIZE, ACL_MEMCPY_DEVICE_TO_HOST);

    aclrtFree(d_x);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}

int main()
{
    std::mt19937 gen(42);
    std::vector<uint32_t> x(N);
    for (int i = 0; i < N; i++) {
        x[i] = gen();
        std::cout << std::bitset<32>(x[i]);
    }
    std::cout << std::endl;

    uint32_t offset = 3;
    launch_kernel(x, offset);

    for (const auto& val : x) {
        std::cout << std::bitset<32>(val);
    }
    std::cout << std::endl;

    return 0;
}
