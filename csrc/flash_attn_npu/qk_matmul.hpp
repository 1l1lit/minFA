/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

#ifndef CATLASS_GEMM_BLOCK_MMAD_QK_HPP_T
#define CATLASS_GEMM_BLOCK_MMAD_QK_HPP_T

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/coord.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/helper.hpp"
#include "catlass/gemm_coord.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/gemm/tile/tile_mmad.hpp"

#include "fa_block.hpp"
#include "kernel_common.hpp"

using namespace KernelCommon;

namespace Catlass::Gemm::Block {
template <
    bool PAGED_CACHE_FLAG_, class L1TileShape_, class L0TileShape_, class AType_, class BType_, class CType_,
    class BiasType_, class TileCopy_, class TileMmad_>
struct BlockMmad<
    MmadAtlasA2FAIQKT<PAGED_CACHE_FLAG_>, L1TileShape_, L0TileShape_, AType_, BType_, CType_, BiasType_, TileCopy_,
    TileMmad_> {
public:
    using DispatchPolicy = MmadAtlasA2FAIQKT<PAGED_CACHE_FLAG_>;
    using ArchTag = Arch::AtlasA2;

    using L1TileShape = L1TileShape_;
    using L0TileShape = L0TileShape_;

    using ElementA = typename AType_::Element;
    using LayoutA = typename AType_::Layout;
    using ElementB = typename BType_::Element;
    using LayoutB = typename BType_::Layout;
    using ElementC = typename CType_::Element;
    using LayoutC = typename CType_::Layout;

    using CopyGmToL1A = typename TileCopy_::CopyGmToL1A;
    using CopyGmToL1B = typename TileCopy_::CopyGmToL1B;
    using CopyL1ToL0A = typename TileCopy_::CopyL1ToL0A;
    using CopyL1ToL0B = typename TileCopy_::CopyL1ToL0B;
    using CopyL0CToGm = typename TileCopy_::CopyL0CToGm;

    using LayoutAInL1 = typename CopyL1ToL0A::LayoutSrc;
    using LayoutBInL1 = typename CopyL1ToL0B::LayoutSrc;
    using LayoutAInL0 = typename CopyL1ToL0A::LayoutDst;
    using LayoutBInL0 = typename CopyL1ToL0B::LayoutDst;
    using LayoutCInL0 = layout::zN;

    using TileMmad = TileMmad_;
    using ElementAccumulator =
        typename Gemm::helper::ElementAccumulatorSelector<ElementA, ElementB>::ElementAccumulator;

    static constexpr uint32_t STAGES = DispatchPolicy::STAGES;
    static constexpr uint32_t L0A_SIZE = ArchTag::L0A_SIZE;
    static constexpr uint32_t L0B_SIZE = ArchTag::L0B_SIZE;
    static constexpr uint32_t L0C_SIZE = ArchTag::L0C_SIZE;
    static constexpr uint32_t L0A_PINGPONG_BUF_SIZE = L0A_SIZE / STAGES;
    static constexpr uint32_t L0B_PINGPONG_BUF_SIZE = L0B_SIZE / STAGES;
    static constexpr uint32_t L0C_PINGPONG_BUF_SIZE = L0C_SIZE / STAGES;

    static_assert(std::is_same_v<LayoutC, layout::RowMajor>, "LayoutC only support RowMajor yet!");

    CATLASS_DEVICE BlockMmad(uint32_t& l1BPingPong_, uint32_t& l0ABPingPong_, uint32_t& l0CPingPong_)
        : l1BPingPong(l1BPingPong_), l0ABPingPong(l0ABPingPong_), l0CPingPong(l0CPingPong_) {};

    CATLASS_DEVICE void init(
        Arch::Resource<ArchTag>& resource, uint32_t nDyn, uint32_t kDyn, uint32_t KStackLen = MAX_KV_STACK_LEN,
        uint32_t l1BufAddrStart = 0)
    {
        maxKStackLen = KStackLen;
        l1ATensor = resource.l1Buf.template GetBufferByByte<ElementA>(l1BufAddrStart); // Q 的 L1 空间
        for (uint32_t i = 0; i < STAGES; i++) {
            l1BTensor[i] = resource.l1Buf.template GetBufferByByte<ElementB>(
                l1BufAddrStart + L1TileShape::M * kDyn * sizeof(ElementA) +
                kDyn * nDyn * sizeof(ElementB) * i); // K 的 L1 空间
            l0ATensor[i] =
                resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_PINGPONG_BUF_SIZE * i); // Q 的 L0 空间
            l0BTensor[i] =
                resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_PINGPONG_BUF_SIZE * i); // K 的 L0 空间
            l0CTensor[i] =
                resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(L0C_PINGPONG_BUF_SIZE * i); // S 的 L0 空间
        }
        l1NDynamic = nDyn;
    }

    CATLASS_DEVICE void loadQGM(
        const AscendC::GlobalTensor<ElementA>& gA, const LayoutA& layoutA, uint32_t qSBlockSize, uint32_t qNBlockSize,
        uint32_t strideQ)
    {
        // Q: (S * N, D)
        uint32_t rowNum = layoutA.shape(0);
        uint32_t embed = layoutA.shape(1);

        // Q: GM -> L1
        uint32_t rowNumRound = RoundUp<NUM_16>(rowNum);
        auto layoutASingleNd = layoutA.GetTileLayout(MakeCoord(qNBlockSize, embed));
        LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(rowNum, embed);

        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(
            EVENT_ID2); // 等待上一轮 L1 -> L0 搬运完成，再进行 GM -> L1 搬运
        copyGmToL1A(
            l1ATensor,       // dstTensor
            gA,              // srcTensor
            layoutAInL1,     // layoutDst
            layoutASingleNd, // layoutSrc
            qSBlockSize,     // ndNum
            strideQ,         // srcNdMatrixStride
            qSBlockSize,     // dstNzNStride
            NUM_16,          // dstNzMatrixStride
            rowNumRound);    // dstNzC0Stride
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID2);
    }

    CATLASS_DEVICE void operator()(
        const AscendC::GlobalTensor<ElementB>& gB, const AscendC::GlobalTensor<ElementC>& gC,
        const AscendC::GlobalTensor<int32_t>& gBlockTable, const LayoutB& layoutB, const LayoutC& layoutC,
        GemmCoord actualOriShape, uint32_t nIdx, uint32_t blockSize, uint32_t strideK, uint32_t isLastStackTile)
    {
        uint32_t rowNum = actualOriShape.m();
        uint32_t stackSeqTile = actualOriShape.n();
        uint32_t embed = actualOriShape.k();
        LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(rowNum, embed);

        if constexpr (PAGED_CACHE_FLAG_) {
            blockStartOffset = 0;
        }

        uint32_t curBlockIdx = 0;
        uint32_t mL0Loop = CeilDiv<L0TileShape::M>(rowNum);
        uint32_t nL1Loop = CeilDiv(stackSeqTile, l1NDynamic);
        uint32_t kL0Loop = CeilDiv<L0TileShape::K>(embed);
        for (uint32_t nL1Idx = 0; nL1Idx < nL1Loop; nL1Idx++) {
            uint32_t nActual = (nL1Idx == nL1Loop - 1) ? (stackSeqTile - nL1Idx * l1NDynamic) : l1NDynamic;
            LayoutBInL1 layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(embed, nActual);

            // K: GM -> L1
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(
                l1BPingPong); // 等待 L1 -> L0 搬运完成，再进行下一轮 GM -> L1 搬运
            if constexpr (PAGED_CACHE_FLAG_) {
                uint32_t nowLen = 0;
                for (uint32_t kvL1Len = 0; kvL1Len < nActual; kvL1Len += nowLen) {
                    uint32_t nowNIdx = nIdx * maxKStackLen / blockSize + curBlockIdx;
                    nowLen = Min(blockSize - blockStartOffset, nActual - kvL1Len);
                    uint32_t gBOffset = (gBlockTable.GetValue(nowNIdx) * blockSize + blockStartOffset) * strideK;
                    auto layoutBTile = layoutB.GetTileLayout(MakeCoord(embed, nowLen));
                    MatrixCoord l1BTileCoord{0, kvL1Len};
                    auto l1BTile = l1BTensor[l1BPingPong][layoutBInL1.GetOffset(l1BTileCoord)];
                    copyGmToL1B(
                        l1BTile,      // dstTensor
                        gB[gBOffset], // srcTensor
                        layoutBInL1,  // layoutDst
                        layoutBTile); // layoutSrc

                    if (blockStartOffset + nowLen == blockSize) {
                        blockStartOffset = 0;
                        curBlockIdx++;
                    } else {
                        blockStartOffset += nowLen;
                    }
                }
            } else {
                uint32_t gBOffset = (nIdx * maxKStackLen + nL1Idx * l1NDynamic) * strideK;
                auto layoutBTile = layoutB.GetTileLayout(MakeCoord(embed, nActual));
                copyGmToL1B(
                    l1BTensor[l1BPingPong], // dstTensor
                    gB[gBOffset],           // srcTensor
                    layoutBInL1,            // layoutDst
                    layoutBTile);           // layoutSrc
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(
                l1BPingPong); // 通知 GM -> L1 搬运完成，可以进行 L1 -> L0 搬运

            for (uint32_t mL0Idx = 0; mL0Idx < mL0Loop; mL0Idx++) {
                uint32_t mL0Actual = (mL0Idx == mL0Loop - 1) ? (rowNum - mL0Idx * L0TileShape::M) : L0TileShape::M;
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(
                    l0CPingPong); // 等待上一轮 L0C -> GM 搬运完成，再进行 M 计算

                for (uint32_t kL0Idx = 0; kL0Idx < kL0Loop; kL0Idx++) {
                    uint32_t kL0Actual = (kL0Idx == kL0Loop - 1) ? (embed - kL0Idx * L0TileShape::K) : L0TileShape::K;
                    AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(
                        l0ABPingPong); // 等待上一轮 M 计算完成，再进行 L1 -> L0 搬运

                    // Q: L1 -> L0
                    LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mL0Actual, kL0Actual);
                    MatrixCoord l1ATileCoord{mL0Idx * L0TileShape::M, kL0Idx * L0TileShape::K};
                    auto l1ATile = l1ATensor[layoutAInL1.GetOffset(l1ATileCoord)];
                    copyL1ToL0A(
                        l0ATensor[l0ABPingPong], // dstTensor
                        l1ATile,                 // srcTensor
                        layoutAInL0,             // layoutDst
                        layoutAInL1);            // layoutSrc

                    // K: L1 -> L0
                    LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kL0Actual, nActual);
                    MatrixCoord l1BTileCoord{kL0Idx * L0TileShape::K, 0};
                    auto l1BTile = l1BTensor[l1BPingPong][layoutBInL1.GetOffset(l1BTileCoord)];
                    if ((mL0Idx == 0) && (kL0Idx == 0)) {
                        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(
                            l1BPingPong); // 等待 GM -> L1 搬运完成，再进行 L1 -> L0 搬运
                    }
                    copyL1ToL0B(
                        l0BTensor[l0ABPingPong], // dstTensor
                        l1BTile,                 // srcTensor
                        layoutBInL0,             // layoutDst
                        layoutBInL1);            // layoutSrc
                    if ((mL0Idx == mL0Loop - 1) && (kL0Idx == kL0Loop - 1)) {
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(
                            l1BPingPong); // 通知 L1 -> L0 搬运完成，可以进行下一轮 GM -> L1 搬运
                    }

                    // 等待 L1 -> L0 搬运完成，再进行 M 计算
                    AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

                    // Q @ K = S
                    uint32_t mL0Align = RoundUp<NUM_16>(mL0Actual);
                    tileMmad(
                        l0CTensor[l0CPingPong],  // l0CTensor
                        l0ATensor[l0ABPingPong], // l0ATensor
                        l0BTensor[l0ABPingPong], // l0BTensor
                        mL0Align,                // m
                        nActual,                 // n
                        kL0Actual,               // k
                        kL0Idx == 0);            // initC (bool)
                    AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(
                        l0ABPingPong); // 通知 M 计算完成，可以进行下一轮 L1 -> L0 搬运

                    l0ABPingPong ^= 1;
                }

                // 等待 M 计算完成，再进行 L0C -> GM 搬运
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);

                // S: L0C -> GM
                MatrixCoord gmCTileCoord{mL0Idx * L0TileShape::M, nL1Idx * l1NDynamic};
                LayoutC layoutCTile = layoutC.GetTileLayout(MakeCoord(mL0Actual, nActual));
                auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(MakeCoord(mL0Actual, nActual));
                copyL0CToGm(
                    gC[layoutC.GetOffset(gmCTileCoord)], // dstTensor
                    l0CTensor[l0CPingPong],              // srcTensor
                    layoutCTile,                         // layoutDst
                    layoutInL0C);                        // layoutSrc
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(
                    l0CPingPong); // 通知 L0C -> GM 搬运完成，可以进行下一轮 M 计算

                l0CPingPong ^= 1;
            }

            l1BPingPong ^= 1;
        }

        if (isLastStackTile) {
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(
                EVENT_ID2); // 通知 L1 -> L0 搬运完成，可以进行下一轮 GM -> L1 搬运
        }
    }

private:
    AscendC::LocalTensor<ElementA> l1ATensor;
    AscendC::LocalTensor<ElementB> l1BTensor[STAGES];
    AscendC::LocalTensor<ElementA> l0ATensor[STAGES];
    AscendC::LocalTensor<ElementB> l0BTensor[STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor[STAGES];

    TileMmad tileMmad;
    CopyGmToL1A copyGmToL1A;
    CopyGmToL1B copyGmToL1B;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL0CToGm copyL0CToGm;

    uint32_t& l1BPingPong;
    uint32_t& l0ABPingPong;
    uint32_t& l0CPingPong;

    uint32_t l1NDynamic = 0;
    uint32_t blockStartOffset = 0;
    uint32_t maxKStackLen = 0;
};
} // namespace Catlass::Gemm::Block

#endif