#include <iostream>
#include <random>
#include <vector>
#include "acl/acl.h"
#include "kernel_operator.h"

using namespace AscendC;

#define AsU16() template ReinterpretCast<uint16_t>()
#define AsI32() template ReinterpretCast<int32_t>()

constexpr int N = 32;
constexpr int SIZE = N * sizeof(uint32_t);

class Kernel {
public:
    __aicore__ inline Kernel() = default;

    __aicore__ inline void Init(GM_ADDR c, uint64_t k, GM_ADDR o)
    {
        g_c.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(c), 4 * N);
        g_o.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(o), 4 * N);

        pipe.InitBuffer(cQ, 1, 4 * SIZE);
        pipe.InitBuffer(oQ, 1, 4 * SIZE);

        k0 = static_cast<uint32_t>(k & 0xffffffff);
        k1 = static_cast<uint32_t>(k >> 32);

        pipe.InitBuffer(tB, 2 * SIZE);
        t4 = tB.GetWithOffset<uint32_t>(N, 0);
        t5 = tB.GetWithOffset<uint32_t>(N, SIZE);
    }

    __aicore__ inline void Main()
    {
        Load();
        Compute();
        Store();
    }

private:
    __aicore__ inline void gather()
    {
        c0 = o[0];
        c1 = o[N];
        c2 = o[2 * N];
        c3 = o[3 * N];

        uint64_t rsvdCnt = 0;
        GatherMask(c0, c, 3, false, 0, {1, N / 16, 8, 0}, rsvdCnt);
        GatherMask(c1, c, 4, false, 0, {1, N / 16, 8, 0}, rsvdCnt);
        GatherMask(c2, c, 5, false, 0, {1, N / 16, 8, 0}, rsvdCnt);
        GatherMask(c3, c, 6, false, 0, {1, N / 16, 8, 0}, rsvdCnt);

        t0 = c[0];
        t1 = c[N];
        t2 = c[2 * N];
        t3 = c[3 * N];
    }

    __aicore__ inline void scatter()
    {
        uint64_t dst[16], src[16];
        for (int i = 0; i < 16; i++) {
            dst[i] = c[8 * i].GetPhyAddr();
            src[i] = o[8 * i].GetPhyAddr();
        }
        TransDataTo5HD<uint32_t>(dst, src, {false, false, 1, 0, 0});

        uint64_t rsvdCnt = 0;
        GatherMask(o[0], c, 3, false, 0, {1, N / 2, 8, 0}, rsvdCnt);
        GatherMask(o[N], c, 4, false, 0, {1, N / 2, 8, 0}, rsvdCnt);
        GatherMask(o[2 * N], c, 5, false, 0, {1, N / 2, 8, 0}, rsvdCnt);
        GatherMask(o[3 * N], c, 6, false, 0, {1, N / 2, 8, 0}, rsvdCnt);
    }

    __aicore__ inline void Xor(
        const LocalTensor<uint32_t>& dst, const LocalTensor<uint32_t>& src0, const LocalTensor<uint32_t>& src1,
        uint32_t n)
    {
        Or(dst.AsU16(), src0.AsU16(), src1.AsU16(), 2 * n); // src0, src1 与 dst 不能重叠
        PipeBarrier<PIPE_V>();
        And(src0.AsU16(), src0.AsU16(), src1.AsU16(), 2 * n);
        PipeBarrier<PIPE_V>();
        Not(src0.AsU16(), src0.AsU16(), 2 * n);
        PipeBarrier<PIPE_V>();
        And(dst.AsU16(), dst.AsU16(), src0.AsU16(), 2 * n);
    }

    __aicore__ inline void Mask16(const LocalTensor<uint32_t>& dst, const LocalTensor<uint32_t>& src, uint32_t n)
    {
        Duplicate(dst, 0xffffu, n); // dst 与 src 不能重叠
        PipeBarrier<PIPE_V>();
        And(dst.AsU16(), dst.AsU16(), src.AsU16(), 2 * n);
    }

    __aicore__ inline void mulhilo32(
        const LocalTensor<uint32_t>& h, const LocalTensor<uint32_t>& l, const LocalTensor<uint32_t>& x, uint32_t a)
    {
        Mask16(l, x, N);          // l = xl = x & 0xffff
        ShiftRight(h, x, 16u, N); // h = x = x >> 16
        PipeBarrier<PIPE_V>();
        uint32_t al = a & 0xffff;
        a = a >> 16;

        Muls(t0.AsI32(), l.AsI32(), static_cast<int32_t>(a), N);  // t0 = m1 = xl * a
        Muls(t1.AsI32(), h.AsI32(), static_cast<int32_t>(al), N); // t1 = m2 = x * al
        PipeBarrier<PIPE_V>();
        Muls(l.AsI32(), l.AsI32(), static_cast<int32_t>(al), N); // l = m0 = xl * al
        Muls(h.AsI32(), h.AsI32(), static_cast<int32_t>(a), N);  // h = m3 = x * a

        Mask16(x, t0, N); // x = m1l = m1 & 0xffff
        PipeBarrier<PIPE_V>();
        ShiftRight(t0, t0, 16u, N); // t0 = m1 = m1 >> 16
        PipeBarrier<PIPE_V>();
        Add(h.AsI32(), h.AsI32(), t0.AsI32(), N); // h = m3 + m1

        Mask16(t0, t1, N); // t0 = m2l = m2 & 0xffff
        PipeBarrier<PIPE_V>();
        ShiftRight(t1, t1, 16u, N);               // t1 = m2 = m2 >> 16
        Add(x.AsI32(), x.AsI32(), t0.AsI32(), N); // x = mm = m1l + m2l
        PipeBarrier<PIPE_V>();
        Add(h.AsI32(), h.AsI32(), t1.AsI32(), N); // h += m2

        Mask16(t0, l, N); // t0 = m0l = m0 & 0xffff
        PipeBarrier<PIPE_V>();
        ShiftRight(l, l, 16u, N); // l = m0 = m0 >> 16
        PipeBarrier<PIPE_V>();
        Add(x.AsI32(), x.AsI32(), l.AsI32(), N); // x = mm += m0
        PipeBarrier<PIPE_V>();
        ShiftLeft(l, x, 16u, N);   // l = mml = mm << 16
        ShiftRight(t1, x, 16u, N); // t1 = mm = mm >> 16
        PipeBarrier<PIPE_V>();
        Or(l.AsU16(), t0.AsU16(), l.AsU16(), 2 * N); // t0 = l = mml | m0l
        Add(h.AsI32(), h.AsI32(), t1.AsI32(), N);    // h += mm
    }

    __aicore__ inline void philox_single_round()
    {
        Duplicate(t0, k0, N);
        Duplicate(t1, k1, N);
        PipeBarrier<PIPE_V>();
        Xor(t2, t0, c1, N); // t2 = c1 ^ k0
        Xor(t3, t1, c3, N); // t3 = c3 ^ k1

        mulhilo32(t4, c3, c0, 0xd2511f53); // t4 = h0, c3 = l0
        mulhilo32(t5, c1, c2, 0xcd9e8d57); // t5 = h1, c1 = l1
        PipeBarrier<PIPE_V>();
        Xor(c2, t3, t4, N); // c2 = h0 ^ c3 ^ k1
        Xor(c0, t2, t5, N); // c0 = h1 ^ c1 ^ k0
    }

    __aicore__ inline void philox()
    {
        for (int i = 0; i < 6; i++) {
            philox_single_round();
            PipeBarrier<PIPE_V>();
            k0 += 0x9e3779b9;
            k1 += 0xbb67ae85;
        }
        philox_single_round();
        PipeBarrier<PIPE_V>();
    }

private:
    __aicore__ inline void Load()
    {
        c = cQ.AllocTensor<uint32_t>();
        DataCopy(c, g_c, 4 * N);
        cQ.EnQue(c);
    }

    __aicore__ inline void Compute()
    {
        c = cQ.DeQue<uint32_t>();
        o = oQ.AllocTensor<uint32_t>();

        gather();
        philox();
        scatter();

        oQ.EnQue(o);
        cQ.FreeTensor(c);
    }

    __aicore__ inline void Store()
    {
        o = oQ.DeQue<uint32_t>();
        DataCopy(g_o, o, 4 * N);
        oQ.FreeTensor(o);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, 1> cQ;
    TQue<TPosition::VECOUT, 1> oQ;
    TBuf<TPosition::VECCALC> tB;

    LocalTensor<uint32_t> c, o;
    uint32_t k0, k1;

    LocalTensor<uint32_t> c0, c1, c2, c3;
    LocalTensor<uint32_t> t0, t1, t2, t3, t4, t5;

    GlobalTensor<uint32_t> g_c, g_o;
};

__global__ __aicore__ void kernel_func(GM_ADDR c, uint64_t k, GM_ADDR o)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Kernel op;
    op.Init(c, k, o);
    op.Main();
}

std::vector<uint32_t> philox_npu(std::vector<uint32_t>& c, uint64_t k)
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    uint8_t* h_c = reinterpret_cast<uint8_t*>(c.data());
    uint8_t* h_o = nullptr;
    uint8_t* d_c = nullptr;
    uint8_t* d_o = nullptr;

    aclrtMallocHost(reinterpret_cast<void**>(&h_o), 4 * SIZE);
    aclrtMalloc(reinterpret_cast<void**>(&d_c), 4 * SIZE, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&d_o), 4 * SIZE, ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(d_c, 4 * SIZE, h_c, 4 * SIZE, ACL_MEMCPY_HOST_TO_DEVICE);

    kernel_func<<<1, nullptr, stream>>>(d_c, k, d_o);
    aclrtSynchronizeStream(stream);

    aclrtMemcpy(h_o, 4 * SIZE, d_o, 4 * SIZE, ACL_MEMCPY_DEVICE_TO_HOST);
    std::vector<uint32_t> o(reinterpret_cast<uint32_t*>(h_o), reinterpret_cast<uint32_t*>(h_o) + 4 * N);

    aclrtFreeHost(h_o);
    aclrtFree(d_c);
    aclrtFree(d_o);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    return o;
}

std::vector<uint32_t> philox_cpu(std::vector<uint32_t>& c, uint64_t k)
{
    for (int b = 0; b < N; b++) {
        uint32_t& c0 = c[4 * b];
        uint32_t& c1 = c[4 * b + 1];
        uint32_t& c2 = c[4 * b + 2];
        uint32_t& c3 = c[4 * b + 3];
        uint32_t k0 = k & 0xffffffff;
        uint32_t k1 = k >> 32;
        for (int i = 0; i < 6; i++) {
            uint64_t p0 = static_cast<uint64_t>(c0) * 0xd2511f53;
            uint64_t p1 = static_cast<uint64_t>(c2) * 0xcd9e8d57;
            uint32_t h0 = p0 >> 32, l0 = p0 & 0xffffffff;
            uint32_t h1 = p1 >> 32, l1 = p1 & 0xffffffff;
            c0 = h1 ^ c1 ^ k0;
            c1 = l1;
            c2 = h0 ^ c3 ^ k1;
            c3 = l0;
            k0 += 0x9e3779b9;
            k1 += 0xbb67ae85;
        }
        uint64_t p0 = static_cast<uint64_t>(c0) * 0xd2511f53;
        uint64_t p1 = static_cast<uint64_t>(c2) * 0xcd9e8d57;
        uint32_t h0 = p0 >> 32, l0 = p0 & 0xffffffff;
        uint32_t h1 = p1 >> 32, l1 = p1 & 0xffffffff;
        c0 = h1 ^ c1 ^ k0;
        c1 = l1;
        c2 = h0 ^ c3 ^ k1;
        c3 = l0;
    }
    return c;
}

int main()
{
    std::mt19937 gen(42);
    std::uniform_int_distribution<uint32_t> dis(0, 0xffffffff);

    std::vector<uint32_t> c(4 * N);
    std::cout << "Input c: " << std::endl;
    for (int i = 0; i < 4 * N; i++) {
        c[i] = dis(gen);
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;

    uint64_t k = (static_cast<uint64_t>(dis(gen)) << 32) | dis(gen);

    std::vector<uint32_t> o = philox_npu(c, k);
    std::cout << "NPU output: " << std::endl;
    for (const auto& val : o) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    std::vector<uint32_t> r = philox_cpu(c, k);
    std::cout << "CPU output: " << std::endl;
    for (const auto& val : r) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    return 0;
}