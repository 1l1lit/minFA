#include <iostream>
#include <random>
#include <vector>
#include "acl/acl.h"
#include "kernel_operator.h"

using namespace AscendC;

#define AsU8() template ReinterpretCast<uint8_t>()
#define AsU16() template ReinterpretCast<uint16_t>()
#define AsU32() template ReinterpretCast<uint32_t>()
#define AsI32() template ReinterpretCast<int32_t>()

constexpr int N = 32;                       // 需为 16 的倍数
constexpr int SIZE = 16 * N * sizeof(half); // 8 * N * sizeof(uint32_t)
constexpr float DROP = 0.5f;
constexpr half THRESH = static_cast<half>(DROP * static_cast<float>(std::numeric_limits<uint8_t>::max()));

struct PhiloxArg {
    uint32_t c0, c1, c2, c3;
    uint32_t k0, k1;
};

template <class T, class U>
__aicore__ inline constexpr auto CeilDiv(T const& dividend, U const& divisor)
{
    return (dividend + divisor - 1) / divisor;
}

template <bool RETURN_DROPOUT_ = false>
class Kernel {
public:
    __aicore__ inline Kernel(GM_ADDR src, GM_ADDR dst, PhiloxArg arg)
    {
        g_src.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(src), 16 * N);
        g_dst.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(dst), 16 * N);

        pipe.InitBuffer(srcQ, 1, SIZE);
        pipe.InitBuffer(dstQ, 1, SIZE);
        pipe.InitBuffer(tmpB, 2 * N * sizeof(uint32_t));

        f0 = arg.c0;
        f1 = arg.c1;
        f2 = arg.c2;
        f3 = arg.c3;
        k0 = arg.k0;
        k1 = arg.k1;
    };

    __aicore__ inline void Main()
    {
        Load();
        Compute();
        Store();
    }

private:
    __aicore__ inline void init()
    {
        c0 = dst.AsU32()[0];
        c1 = dst.AsU32()[N];
        c2 = dst.AsU32()[2 * N];
        c3 = dst.AsU32()[3 * N];

        CreateVecIndex(c3.AsI32(), 0, N);
        Adds(c0.AsI32(), c3.AsI32(), static_cast<int32_t>(f0), N);
        Adds(c1.AsI32(), c3.AsI32(), static_cast<int32_t>(f1), N);
        Adds(c2.AsI32(), c3.AsI32(), static_cast<int32_t>(f2), N);
        Adds(c3.AsI32(), c3.AsI32(), static_cast<int32_t>(f3), N);

        t0 = dst.AsU32()[4 * N];
        t1 = dst.AsU32()[5 * N];
        t2 = dst.AsU32()[6 * N];
        t3 = dst.AsU32()[7 * N];
        t4 = tmpB.GetWithOffset<uint32_t>(N, 0);
        t5 = tmpB.GetWithOffset<uint32_t>(N, N * sizeof(uint32_t));

        o = dst.AsU16()[8 * N];
        cmp = tmpB.GetWithOffset<uint8_t>(2 * N, 0);
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

    __aicore__ inline void gather()
    {
        uint64_t rsvdCnt = 0;
        GatherMask(o[0], c0.AsU16(), 1, false, 0, {1, CeilDiv(N, 64), 8, 0}, rsvdCnt);
        GatherMask(o[N], c0.AsU16(), 2, false, 0, {1, CeilDiv(N, 64), 8, 0}, rsvdCnt);
        GatherMask(o[2 * N], c1.AsU16(), 1, false, 0, {1, CeilDiv(N, 64), 8, 0}, rsvdCnt);
        GatherMask(o[3 * N], c1.AsU16(), 2, false, 0, {1, CeilDiv(N, 64), 8, 0}, rsvdCnt);
        GatherMask(o[4 * N], c2.AsU16(), 1, false, 0, {1, CeilDiv(N, 64), 8, 0}, rsvdCnt);
        GatherMask(o[5 * N], c2.AsU16(), 2, false, 0, {1, CeilDiv(N, 64), 8, 0}, rsvdCnt);
        GatherMask(o[6 * N], c3.AsU16(), 1, false, 0, {1, CeilDiv(N, 64), 8, 0}, rsvdCnt);
        GatherMask(o[7 * N], c3.AsU16(), 2, false, 0, {1, CeilDiv(N, 64), 8, 0}, rsvdCnt);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void dropout()
    {
        Cast(dst, o.AsU8(), RoundMode::CAST_NONE, 16 * N);
        PipeBarrier<PIPE_V>();
        CompareScalar(cmp, dst, THRESH, CMPMODE::GT, 16 * N);
        PipeBarrier<PIPE_V>();

        if constexpr (RETURN_DROPOUT_) {
            Duplicate(dst, static_cast<half>(1), 16 * N);
            Select(dst, cmp, dst, static_cast<half>(-1), SELMODE::VSEL_TENSOR_SCALAR_MODE, 16 * N);
            Mul(dst, dst, src, 16 * N);
        } else {
            Select(dst, cmp, src, static_cast<half>(0), SELMODE::VSEL_TENSOR_SCALAR_MODE, 16 * N);
        }
    }

private:
    __aicore__ inline void Load()
    {
        src = srcQ.AllocTensor<half>();
        DataCopy(src, g_src, 16 * N);
        srcQ.EnQue(src);
    }

    __aicore__ inline void Compute()
    {
        src = srcQ.DeQue<half>();
        dst = dstQ.AllocTensor<half>();

        init();
        philox();
        gather();
        dropout();

        dstQ.EnQue(dst);
        srcQ.FreeTensor(src);
    }

    __aicore__ inline void Store()
    {
        dst = dstQ.DeQue<half>();
        DataCopy(g_dst, dst, 16 * N);
        dstQ.FreeTensor(dst);
    }

private:
    TPipe pipe;
    TQue<TPosition::VECIN, 1> srcQ;
    TQue<TPosition::VECOUT, 1> dstQ;
    TBuf<TPosition::VECCALC> tmpB;

    LocalTensor<half> src, dst;
    uint32_t f0, f1, f2, f3;
    LocalTensor<uint32_t> c0, c1, c2, c3;
    uint32_t k0, k1;
    LocalTensor<uint32_t> t0, t1, t2, t3, t4, t5;
    LocalTensor<uint16_t> o;
    LocalTensor<uint8_t> cmp;

    GlobalTensor<half> g_src, g_dst;
};

template <bool RETURN_DROPOUT_ = false>
__global__ __aicore__ void kernel_func(GM_ADDR src, GM_ADDR dst, PhiloxArg arg)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    Kernel<RETURN_DROPOUT_> op(src, dst, arg);
    op.Main();
}

template <bool RETURN_DROPOUT_ = false>
std::vector<half> dropout_npu(std::vector<half>& src, PhiloxArg arg)
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);

    uint8_t* h_src = reinterpret_cast<uint8_t*>(src.data());
    uint8_t* h_dst = nullptr;
    uint8_t* d_src = nullptr;
    uint8_t* d_dst = nullptr;

    aclrtMallocHost(reinterpret_cast<void**>(&h_dst), SIZE);
    aclrtMalloc(reinterpret_cast<void**>(&d_src), SIZE, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&d_dst), SIZE, ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(d_src, SIZE, h_src, SIZE, ACL_MEMCPY_HOST_TO_DEVICE);

    kernel_func<RETURN_DROPOUT_><<<1, nullptr, stream>>>(d_src, d_dst, arg);
    aclrtSynchronizeStream(stream);

    aclrtMemcpy(h_dst, SIZE, d_dst, SIZE, ACL_MEMCPY_DEVICE_TO_HOST);
    std::vector<half> dst(reinterpret_cast<half*>(h_dst), reinterpret_cast<half*>(h_dst) + 16 * N);

    aclrtFreeHost(h_dst);
    aclrtFree(d_src);
    aclrtFree(d_dst);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    return dst;
}

template <bool RETURN_DROPOUT_ = false>
std::vector<half> dropout_cpu(std::vector<half>& src, const PhiloxArg& arg)
{
    // philox 8x16
    std::vector<uint16_t> o(8 * N);
    for (int b = 0; b < N; b++) {
        uint32_t c0 = arg.c0 + b;
        uint32_t c1 = arg.c1 + b;
        uint32_t c2 = arg.c2 + b;
        uint32_t c3 = arg.c3 + b;
        uint32_t k0 = arg.k0;
        uint32_t k1 = arg.k1;
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
        uint32_t o0 = h1 ^ c1 ^ k0;
        uint32_t o1 = l1;
        uint32_t o2 = h0 ^ c3 ^ k1;
        uint32_t o3 = l0;
        o[b] = o0 & 0xffff;
        o[N + b] = o0 >> 16;
        o[2 * N + b] = o1 & 0xffff;
        o[3 * N + b] = o1 >> 16;
        o[4 * N + b] = o2 & 0xffff;
        o[5 * N + b] = o2 >> 16;
        o[6 * N + b] = o3 & 0xffff;
        o[7 * N + b] = o3 >> 16;
    }

    // dropout
    std::vector<uint8_t> p(
        reinterpret_cast<uint8_t*>(o.data()), reinterpret_cast<uint8_t*>(o.data()) + 8 * N * sizeof(uint16_t));
    for (int i = 0; i < 16 * N; i++) {
        if (p[i] < THRESH) {
            if constexpr (RETURN_DROPOUT_) {
                src[i] = -src[i];
            } else {
                src[i] = static_cast<half>(0);
            }
        }
    }

    return src;
}

int main()
{
    std::mt19937 gen(37);
    std::vector<half> src(16 * N);
    std::cout << "Input: " << std::endl;
    for (int i = 0; i < 16 * N; i++) {
        src[i] = static_cast<half>(gen() / static_cast<float>(std::numeric_limits<uint32_t>::max()));
        std::cout << static_cast<float>(src[i]) << " ";
    }
    std::cout << std::endl;

    PhiloxArg arg;
    arg.c0 = gen();
    arg.c1 = gen();
    arg.c2 = gen();
    arg.c3 = gen();
    arg.k0 = gen();
    arg.k1 = gen();

    std::vector<half> dst = dropout_npu<true>(src, arg);
    std::cout << "NPU output: " << std::endl;
    for (const auto& val : dst) {
        std::cout << static_cast<float>(val) << " ";
    }
    std::cout << std::endl;

    std::vector<half> ref = dropout_cpu<true>(src, arg);
    std::cout << "CPU output: " << std::endl;
    for (const auto& val : ref) {
        std::cout << static_cast<float>(val) << " ";
    }
    std::cout << std::endl;

    return 0;
}