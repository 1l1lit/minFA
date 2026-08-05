#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "aclnnop/aclnn_dropout_gen_mask.h"
#include "kernel_operator.h"

using namespace AscendC;
using T = half;

#define AsU16() template ReinterpretCast<uint16_t>()

constexpr int N = 32; // 需为 16 的倍数
constexpr int S = N * sizeof(T);

template <bool RETURN_DROPOUT_ = false>
class Kernel {
public:
    __aicore__ inline Kernel(GM_ADDR x, GM_ADDR m)
    {
        pipe.InitBuffer(ub, 192 * 1024);
        buf = ub.Get<uint8_t>();

        g_x.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x), N);
        g_m.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(m), N);
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
        x = GetBuf<T>(0);
        m = GetBuf<uint8_t>(S);

        y = GetBuf<half>(0);
        t0 = GetBuf<half>(S + N);
        t1 = GetBuf<half>(2 * S + N);
    }

    __aicore__ inline void Load()
    {
        DataCopy(x, g_x, N);
        DataCopy(m, g_m, N);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }

    __aicore__ inline void Playground()
    {
        if constexpr (RETURN_DROPOUT_) {
            Duplicate(t0, static_cast<half>(0.0f), N);
            PipeBarrier<PIPE_V>();
            Select(t0, m, t0, static_cast<half>(-0.0f), SELMODE::VSEL_TENSOR_SCALAR_MODE, N);
            PipeBarrier<PIPE_V>();
            Or(t1.AsU16(), y.AsU16(), t0.AsU16(), N);
            PipeBarrier<PIPE_V>();
            And(y.AsU16(), y.AsU16(), t0.AsU16(), N);
            PipeBarrier<PIPE_V>();
            Not(y.AsU16(), y.AsU16(), N);
            PipeBarrier<PIPE_V>();
            And(y.AsU16(), y.AsU16(), t1.AsU16(), N);
        } else {
            Select(y, m, y, static_cast<half>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, N);
        }
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

    LocalTensor<T> x;
    LocalTensor<uint8_t> m;
    LocalTensor<half> y, t0, t1;

    GlobalTensor<T> g_x;
    GlobalTensor<uint8_t> g_m;
};

template <bool RETURN_DROPOUT_ = false>
__global__ __aicore__ void kernel_func(GM_ADDR x, GM_ADDR m)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Kernel<RETURN_DROPOUT_> op(x, m);
    op.Main();
}

template <bool RETURN_DROPOUT_ = false>
void launch_kernel(std::vector<T>& x, float p, int64_t seed, int64_t offset)
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    uint8_t* h_x = reinterpret_cast<uint8_t*>(x.data());
    uint8_t* d_x = nullptr;
    aclrtMalloc(reinterpret_cast<void**>(&d_x), S, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(d_x, S, h_x, S, ACL_MEMCPY_HOST_TO_DEVICE);

    uint8_t* d_m = nullptr;
    aclrtMalloc(reinterpret_cast<void**>(&d_m), N, ACL_MEM_MALLOC_HUGE_FIRST);

    // *** generate dropout mask ***
    std::vector<int64_t> shapeElem = {N, 8};
    std::vector<int64_t> shapeByte = {N};
    std::vector<int64_t> stride = {1};

    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    aclIntArray* shape = aclCreateIntArray(shapeElem.data(), shapeElem.size());
    aclTensor* t_mask = aclCreateTensor(
        shapeByte.data(), shapeByte.size(), aclDataType::ACL_UINT8, stride.data(), 0, aclFormat::ACL_FORMAT_ND,
        shapeByte.data(), shapeByte.size(), d_m);
    aclnnDropoutGenMaskGetWorkspaceSize(shape, p, seed, offset, t_mask, &workspaceSize, &executor);
    void* workspaceAddr = nullptr;
    if (workspaceSize > 0) {
        aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    }
    aclnnDropoutGenMask(workspaceAddr, workspaceSize, executor, stream);
    aclrtSynchronizeStream(stream);
    if (workspaceSize > 0) {
        aclrtFree(workspaceAddr);
    }

    kernel_func<RETURN_DROPOUT_><<<1, nullptr, stream>>>(d_x, d_m);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(h_x, S, d_x, S, ACL_MEMCPY_DEVICE_TO_HOST);

    aclrtFree(d_x);
    aclrtFree(d_m);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
}

int main()
{
    float p = 0.7;
    int64_t seed = 37;
    int64_t offset = 42;

    std::vector<T> x(N, static_cast<T>(1.0f));
    launch_kernel<true>(x, p, seed, offset);
    for (const auto& val : x) {
        std::cout << static_cast<float>(val) << " ";
    }
    std::cout << std::endl;

    return 0;
}
