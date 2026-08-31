/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

#ifndef CATLASS_GEMM_BLOCK_MMAD_PV_HPP_T
#define CATLASS_GEMM_BLOCK_MMAD_PV_HPP_T

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
    MmadAtlasA2FAIPVT<PAGED_CACHE_FLAG_>, L1TileShape_, L0TileShape_, AType_, BType_, CType_, BiasType_, TileCopy_,
    TileMmad_> {
public:
    using DispatchPolicy = MmadAtlasA2FAIPVT<PAGED_CACHE_FLAG_>;
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

    CATLASS_DEVICE BlockMmad() = default;

    CATLASS_DEVICE ~BlockMmad() = default;

    CATLASS_DEVICE void init(
        Arch::Resource<ArchTag>& resource, uint32_t kDyn, uint32_t VStackLen = MAX_KV_STACK_LEN,
        uint32_t l1BufAddrStart = 0)
    {
        maxVStackLen = VStackLen;
        l1BTensor = resource.l1Buf.template GetBufferByByte<ElementB>(
            l1BufAddrStart + L1TileShape::M * kDyn * sizeof(ElementA) * STAGES); // V 的 L1 空间
        for (uint32_t i = 0; i < STAGES; i++) {
            l1ATensor[i] = resource.l1Buf.template GetBufferByByte<ElementA>(
                l1BufAddrStart + L1TileShape::M * kDyn * sizeof(ElementA) * i); // P 的 L1 空间
            l0ATensor[i] =
                resource.l0ABuf.template GetBufferByByte<ElementA>(L0A_PINGPONG_BUF_SIZE * i); // P 的 L0 空间
            l0BTensor[i] =
                resource.l0BBuf.template GetBufferByByte<ElementB>(L0B_PINGPONG_BUF_SIZE * i); // V 的 L0 空间
            l0CTensor[i] = resource.l0CBuf.template GetBufferByByte<ElementAccumulator>(
                L0C_PINGPONG_BUF_SIZE * i); // OTmp 的 L0 空间
        }
        l1KDynamic = kDyn;
    }

    CATLASS_DEVICE void operator()(
        const AscendC::GlobalTensor<ElementA>& gA, const AscendC::GlobalTensor<ElementB>& gB,
        const AscendC::GlobalTensor<ElementC>& gC, const AscendC::GlobalTensor<int32_t>& gBlockTable,
        const LayoutA& layoutA, const LayoutB& layoutB, const LayoutC& layoutC, GemmCoord actualOriShape, uint32_t nIdx,
        uint32_t blockSize, uint32_t strideV, Arch::CrossCoreFlag softmaxReady)
    {
        uint32_t rowNum = actualOriShape.m();
        uint32_t embed = actualOriShape.n();
        uint32_t stackSeqTile = actualOriShape.k();
        LayoutBInL1 layoutBInL1 = LayoutBInL1::template MakeLayout<ElementB>(stackSeqTile, embed);

        // V: GM -> L1
        AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(
            EVENT_ID4); // 等待上一轮 L1 -> L0 搬运完成，再进行 GM -> L1 搬运
        if constexpr (PAGED_CACHE_FLAG_) {
            blockStartOffset = 0;
            uint32_t blockStart = 0;
            uint32_t blockEnd = 0;
            uint32_t curBlockTotalNum = 0;
            if (stackSeqTile >= blockStart && blockSize != 0) {
                blockStart = blockSize - blockStartOffset;
                blockEnd = ((stackSeqTile - blockStart) % blockSize == 0) ? blockSize
                                                                          : (stackSeqTile - blockStart) % blockSize;
                curBlockTotalNum = CeilDiv(stackSeqTile - blockStart, blockSize) + 1;
            } else {
                blockStart = stackSeqTile;
                blockEnd = stackSeqTile + blockStartOffset;
                curBlockTotalNum = 1;
            }

            for (uint32_t curBlockIdx = 0; curBlockIdx < curBlockTotalNum; curBlockIdx++) {
                uint32_t nowNIdx = nIdx * maxVStackLen / blockSize + curBlockIdx;
                uint32_t nowLen = (curBlockIdx == curBlockTotalNum - 1) ? (blockEnd - blockStartOffset)
                                                                        : (blockSize - blockStartOffset);
                uint32_t gBOffset = (gBlockTable.GetValue(nowNIdx) * blockSize + blockStartOffset) * strideV;
                auto layoutBTile = layoutB.GetTileLayout(MakeCoord(nowLen, embed));
                uint32_t curBlockSize = (curBlockIdx > 0) ? ((curBlockIdx - 1) * blockSize + blockStart) : 0;
                MatrixCoord l1BTileCoord{curBlockSize, 0};
                auto l1BTile = l1BTensor[layoutBInL1.GetOffset(l1BTileCoord)];
                copyGmToL1B(l1BTile, gB[gBOffset], layoutBInL1, layoutBTile);

                if (blockStartOffset + nowLen == blockSize) {
                    blockStartOffset = 0;
                } else {
                    blockStartOffset += nowLen;
                }
            }
        } else {
            uint32_t gBOffset = nIdx * maxVStackLen * strideV;
            auto layoutBTile = layoutB.GetTileLayout(MakeCoord(stackSeqTile, embed));
            copyGmToL1B(l1BTensor, gB[gBOffset], layoutBInL1, layoutBTile);
        }

        // 等待 GM -> L1 搬运完成，再进行 L1 -> L0 搬运
        AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(EVENT_ID0);

        // 等待 P 就绪
        Arch::CrossCoreWaitFlag(softmaxReady);

        uint32_t mL1Loop = CeilDiv(rowNum, L1TileShape::M);
        uint32_t nL1Loop = CeilDiv(embed, L0TileShape::N);
        uint32_t kL1Loop = CeilDiv(stackSeqTile, l1KDynamic);
        for (uint32_t nL1Idx = 0; nL1Idx < nL1Loop; nL1Idx++) {
            uint32_t nL1Actual = (nL1Idx == nL1Loop - 1) ? (embed - nL1Idx * L0TileShape::N) : L0TileShape::N;

            for (uint32_t mL1Idx = 0; mL1Idx < mL1Loop; mL1Idx++) {
                uint32_t mL1Actual = (mL1Idx == mL1Loop - 1) ? (rowNum - mL1Idx * L1TileShape::M) : L1TileShape::M;
                AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(
                    l0CPingPong); // 等待上一轮 L0C -> GM 搬运完成，再进行下一轮 M 计算

                for (uint32_t kL1Idx = 0; kL1Idx < kL1Loop; kL1Idx++) {
                    uint32_t kL1Actual = (kL1Idx == kL1Loop - 1) ? (stackSeqTile - kL1Idx * l1KDynamic) : l1KDynamic;
                    AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(
                        l1APingPong); // 等待上一轮 L1 -> L0 搬运完成，再进行 GM -> L1 搬运

                    // P: GM -> L1
                    MatrixCoord gmATileCoord{mL1Idx * L1TileShape::M, kL1Idx * l1KDynamic};
                    auto gmTileA = gA[layoutA.GetOffset(gmATileCoord)];
                    LayoutAInL1 layoutAInL1 = LayoutAInL1::template MakeLayout<ElementA>(mL1Actual, kL1Actual);
                    auto layoutTileA = layoutA.GetTileLayout(MakeCoord(mL1Actual, kL1Actual));
                    copyGmToL1A(l1ATensor[l1APingPong], gmTileA, layoutAInL1, layoutTileA);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE1>(
                        l1APingPong); // 通知 GM -> L1 搬运完成，可以进行 L1 -> L0 搬运

                    uint32_t kL0Loop = CeilDiv(kL1Actual, L0TileShape::K);
                    for (uint32_t kL0Idx = 0; kL0Idx < kL0Loop; kL0Idx++) {
                        uint32_t kL0Actual =
                            (kL0Idx == kL0Loop - 1) ? (kL1Actual - kL0Idx * L0TileShape::K) : L0TileShape::K;

                        // P: L1 -> L0
                        LayoutAInL0 layoutAInL0 = LayoutAInL0::template MakeLayout<ElementA>(mL1Actual, kL0Actual);
                        MatrixCoord l1ATileCoord{0, kL0Idx * L0TileShape::K};
                        auto l1ATile = l1ATensor[l1APingPong][layoutAInL1.GetOffset(l1ATileCoord)];
                        if (kL0Idx == 0) {
                            AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE1>(
                                l1APingPong); // 等待 GM -> L1 搬运完成，再进行 L1 -> L0 搬运
                        }
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(
                            l0ABPingPong); // 等待上一轮 M 计算完成，再进行 L1 -> L0 搬运
                        copyL1ToL0A(l0ATensor[l0ABPingPong], l1ATile, layoutAInL0, layoutAInL1);
                        if (kL0Idx == kL0Loop - 1) {
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(
                                l1APingPong); // 通知 L1 -> L0 搬运完成，可以进行下一轮 GM -> L1 搬运
                        }

                        // V: L1 -> L0
                        LayoutBInL0 layoutBInL0 = LayoutBInL0::template MakeLayout<ElementB>(kL0Actual, nL1Actual);
                        MatrixCoord l1BTileCoord{
                            kL1Idx * l1KDynamic + kL0Idx * L0TileShape::K, L0TileShape::N * nL1Idx};
                        auto l1BTile = l1BTensor[layoutBInL1.GetOffset(l1BTileCoord)];
                        AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(
                            l0ABPingPong + 2); // 等待上一轮 M 计算完成，再进行 L1 -> L0 搬运
                        copyL1ToL0B(l0BTensor[l0ABPingPong], l1BTile, layoutBInL0, layoutBInL1);

                        // 等待 L1 -> L0 搬运完成，再进行 M 计算
                        AscendC::SetFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);
                        AscendC::WaitFlag<AscendC::HardEvent::MTE1_M>(EVENT_ID0);

                        // P @ V = OTmp
                        uint32_t mL0Align = RoundUp(mL1Actual, NUM_16);
                        tileMmad(
                            l0CTensor[l0CPingPong], l0ATensor[l0ABPingPong], l0BTensor[l0ABPingPong], mL0Align,
                            nL1Actual, kL0Actual, (kL1Idx == 0) && (kL0Idx == 0));

                        // 通知 M 计算完成，可以进行下一轮 L1 -> L0 搬运
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPong);
                        AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(l0ABPingPong + 2);

                        l0ABPingPong ^= 1;
                    }

                    l1APingPong ^= 1;
                }

                // 等待 M 计算完成，再进行 L0C -> GM 搬运
                AscendC::SetFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::M_FIX>(EVENT_ID0);

                // OTmp: L0C -> GM
                MatrixCoord gmCTileCoord{mL1Idx * L0TileShape::M, L0TileShape::N * nL1Idx};
                LayoutC layoutCTile = layoutC.GetTileLayout(MakeCoord(mL1Actual, nL1Actual));
                auto layoutInL0C = LayoutCInL0::MakeLayoutInL0C(MakeCoord(mL1Actual, nL1Actual));
                copyL0CToGm(gC[layoutC.GetOffset(gmCTileCoord)], l0CTensor[l0CPingPong], layoutCTile, layoutInL0C);
                AscendC::SetFlag<AscendC::HardEvent::FIX_M>(
                    l0CPingPong); // 通知 L0C -> GM 搬运完成，可以进行下一轮 M 计算

                l0CPingPong ^= 1;
            }
        }

        AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(
            EVENT_ID4); // 通知 L1 -> L0 搬运完成，可以进行下一轮 GM -> L1 搬运
    }

protected:
    AscendC::LocalTensor<ElementA> l1ATensor[STAGES];
    AscendC::LocalTensor<ElementB> l1BTensor;
    AscendC::LocalTensor<ElementA> l0ATensor[STAGES];
    AscendC::LocalTensor<ElementB> l0BTensor[STAGES];
    AscendC::LocalTensor<ElementAccumulator> l0CTensor[STAGES];

    TileMmad tileMmad;
    CopyGmToL1A copyGmToL1A;
    CopyGmToL1B copyGmToL1B;
    CopyL1ToL0A copyL1ToL0A;
    CopyL1ToL0B copyL1ToL0B;
    CopyL0CToGm copyL0CToGm;

    uint32_t l1APingPong = 0;
    uint32_t l0ABPingPong = 0;
    uint32_t l0CPingPong = 0;

    uint32_t l1KDynamic = 0;
    uint32_t blockStartOffset = 0;
    uint32_t maxVStackLen = 0;
};
} // namespace Catlass::Gemm::Block

#endif