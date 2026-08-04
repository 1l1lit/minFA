#include <iostream>
#include <vector>
#include "acl/acl.h"
#include "kernel_operator.h"

using namespace AscendC;

constexpr int N = 2;
constexpr int D = 8; // D 需为 8 的倍数
constexpr int SIZE = D * sizeof(float);

class Kernel {
public:
    __aicore__ inline Kernel(GM_ADDR x, GM_ADDR y, GM_ADDR z)
    {
        pipe.InitBuffer(ub, 192 * 1024);
        buf = ub.Get<uint8_t>();

        g_x.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), N * D);
        g_y.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y), D);
        g_z.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(z), N * D);
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
        x = GetBuf<float>(0);
        y = GetBuf<float>(N * SIZE);
        z = GetBuf<float>(2 * N * SIZE);
    }

    __aicore__ inline void Load()
    {
        DataCopy(x, g_x, N * D);
        DataCopy(y, g_y, D);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
    }

    __aicore__ inline void Playground()
    {
        int offset = 0;
        while (offset < D) {
            int stride = Std::min(D - offset, 64);
            Mul(z[offset], x[offset], y[offset], stride, N, {1, 1, 1, D / 8, D / 8, 0});
            offset += stride;
        }
    }

    __aicore__ inline void Store()
    {
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(g_z, z, N * D);
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

    LocalTensor<float> x, y, z;
    GlobalTensor<float> g_x, g_y, g_z;
};

__global__ __aicore__ void kernel_func(GM_ADDR x, GM_ADDR y, GM_ADDR z)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Kernel op(x, y, z);
    op.Main();
}

std::vector<float> launch_kernel(std::vector<float>& x, std::vector<float>& y)
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    uint8_t* h_x = reinterpret_cast<uint8_t*>(x.data());
    uint8_t* h_y = reinterpret_cast<uint8_t*>(y.data());
    uint8_t* h_z = nullptr;
    uint8_t* d_x = nullptr;
    uint8_t* d_y = nullptr;
    uint8_t* d_z = nullptr;

    aclrtMallocHost(reinterpret_cast<void**>(&h_z), N * SIZE);
    aclrtMalloc(reinterpret_cast<void**>(&d_x), N * SIZE, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&d_y), SIZE, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&d_z), N * SIZE, ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(d_x, N * SIZE, h_x, N * SIZE, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(d_y, SIZE, h_y, SIZE, ACL_MEMCPY_HOST_TO_DEVICE);

    kernel_func<<<1, nullptr, stream>>>(d_x, d_y, d_z);
    aclrtSynchronizeStream(stream);

    aclrtMemcpy(h_z, N * SIZE, d_z, N * SIZE, ACL_MEMCPY_DEVICE_TO_HOST);
    std::vector<float> z(reinterpret_cast<float*>(h_z), reinterpret_cast<float*>(h_z) + N * D);

    aclrtFreeHost(h_z);
    aclrtFree(d_x);
    aclrtFree(d_y);
    aclrtFree(d_z);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    return z;
}

int main()
{
    std::vector<float> x(N * D);
    for (int i = 0; i < N * D; i++) {
        x[i] = static_cast<float>(i + 1);
    }
    std::vector<float> y(D);
    for (int i = 0; i < D; i++) {
        y[i] = static_cast<float>(i + 1);
    }
    std::vector<float> z = launch_kernel(x, y);
    for (const auto& val : z) {
        std::cout << val << " ";
    }
    std::cout << std::endl;
    return 0;
}
