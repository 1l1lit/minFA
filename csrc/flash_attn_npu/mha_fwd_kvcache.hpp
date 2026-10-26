/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

#ifndef MHA_FWD_KVCACHE_HPP
#define MHA_FWD_KVCACHE_HPP

#include "kernel_operator.h"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/cross_core_sync.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/layout/layout.hpp"

#include "kernel_common.hpp"
#include "tiling_data.hpp"
#include "qk_matmul.hpp"
#include "online_softmax.hpp"
#include "pv_matmul.hpp"
#include "rescale_o.hpp"
#include "fa_block.hpp"

using namespace Catlass;
using namespace KernelCommon;

template <
    class BlockMmadQK, class BlockMmadPV, class EpilogueOnlineSoftmax, class EpilogueRescaleO, bool PAGED_CACHE_FLAG_,
    MaskModeT MASK_MODE_, InputLayoutT INPUT_LAYOUT_>
class FAInferKernel {
public:
    using ArchTag = Arch::AtlasA2;

    using ElementQ = typename BlockMmadQK::ElementA;
    using LayoutQ = typename BlockMmadQK::LayoutA;
    using ElementK = typename BlockMmadQK::ElementB;
    using LayoutK = typename BlockMmadQK::LayoutB;
    using ElementS = typename BlockMmadQK::ElementC;
    using LayoutS = typename BlockMmadQK::LayoutC;

    using ElementP = typename BlockMmadPV::ElementA;
    using LayoutP = typename BlockMmadPV::LayoutA;
    using ElementV = typename BlockMmadPV::ElementB;
    using LayoutV = typename BlockMmadPV::LayoutB;

    using ElementMask = typename EpilogueOnlineSoftmax::ElementMask;
    using LayoutMask = typename EpilogueOnlineSoftmax::LayoutMask;

    using ElementOTmp = typename EpilogueRescaleO::ElementInput;
    using LayoutOTmp = typename EpilogueRescaleO::LayoutInput;
    using ElementO = typename EpilogueRescaleO::ElementOutput;
    using LayoutO = typename EpilogueRescaleO::LayoutOutput;

    using ElementLse = typename EpilogueRescaleO::ElementLse;
    using LayoutLse = typename EpilogueRescaleO::LayoutLse;
    using ElementUpdate = typename EpilogueRescaleO::ElementUpdate;
    using LayoutUpdate = typename EpilogueRescaleO::LayoutUpdate;

    CATLASS_DEVICE void PreProcess(const FAIKernelParams& params, const Arch::Resource<ArchTag>& resource)
    {
        auto fATilingData = reinterpret_cast<__gm__ FAInferTilingData*>(params.tiling);
        uint32_t mm1OutSize = fATilingData->mm1OutSize;
        uint32_t smOnlineOutSize = fATilingData->smOnlineOutSize;
        uint32_t mm2OutSize = fATilingData->mm2OutSize;
        float scaleValue = fATilingData->scaleValue;

        batch = fATilingData->batch;
        qHeads = fATilingData->numHeads;
        kvHeads = fATilingData->kvHeads;
        embed = fATilingData->embeddingSize;
        pagedBlockSize = fATilingData->blockSize;
        maxNumBlocksPerBatch = fATilingData->maxNumBlocksPerBatch;
        maxQSeqlen = fATilingData->maxQSeqlen;

        gQ.SetGlobalBuffer((__gm__ ElementQ*)params.q);
        gQ.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        gK.SetGlobalBuffer((__gm__ ElementK*)params.k);
        gV.SetGlobalBuffer((__gm__ ElementV*)params.v);

        gMask.SetGlobalBuffer((__gm__ ElementMask*)params.mask);
        gBlockTable.SetGlobalBuffer((__gm__ int32_t*)params.blockTable);
        gActualQseqlen.SetGlobalBuffer((__gm__ int32_t*)params.actualQseqlen);
        gActualKvseqlen.SetGlobalBuffer((__gm__ int32_t*)params.actualKvseqlen);

        gO.SetGlobalBuffer((__gm__ ElementO*)params.o);
        gO.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);
        gLse.SetGlobalBuffer((__gm__ ElementLse*)params.lse);
        gLse.SetL2CacheHint(AscendC::CacheMode::CACHE_MODE_DISABLE);

        gS.SetGlobalBuffer((__gm__ ElementS*)(params.workspace));
        gP.SetGlobalBuffer((__gm__ ElementP*)(params.workspace + mm1OutSize));
        gOTmp.SetGlobalBuffer((__gm__ ElementOTmp*)(params.workspace + mm1OutSize + smOnlineOutSize));
        gOUpdate.SetGlobalBuffer((__gm__ ElementOTmp*)(params.workspace + mm1OutSize + smOnlineOutSize + mm2OutSize));

        if ASCEND_IS_AIC {
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        } else {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        }

        strideQ = qHeads * embed;
        strideK = kvHeads * embed;
        strideV = kvHeads * embed;
        strideO = qHeads * embed;
        embedRound = RoundUp<HALF_PER_DATABLOCK>(embed);
        groupSize = qHeads / kvHeads;

        // LSE 输出是 N 主序 (BSND -> BNS，TND -> NT)
        if constexpr (INPUT_LAYOUT_ == InputLayoutT::BSND) {
            strideLse = maxQSeqlen;
        } else {
            strideLse = gActualQseqlen.GetValue(batch); // totalQTokens
        }

        if ASCEND_IS_AIC {
            uint32_t kDynQK = RoundUp<NUM_128>(embed);                       // K 对应 D 维度
            uint32_t nDynQK = NUM_128;                                       // N 对应 KV 的序列长度
            uint32_t kDynPV = kDynQK * nDynQK / BlockMmadPV::L1TileShape::M; // P 复用 K 的 L1 空间
            uint32_t maxQL1Size = kDynQK * Q_TILE_CEIL * sizeof(ElementQ);   // Q 的预留 L1 空间

            blockMmadQK.init(resource, nDynQK, kDynQK, MAX_KV_STACK_LEN);
            blockMmadPV.init(resource, kDynPV, MAX_KV_STACK_LEN, maxQL1Size);

            layoutKTmp = LayoutK(strideK, MAX_KV_STACK_LEN);
            layoutVTmp = LayoutV(MAX_KV_STACK_LEN, strideV);
        } else { // ASCEND_IS_AIV
            epilogueOnlineSoftmax.init(resource, scaleValue);
            epilogueRescaleO.init(resource);

            layoutMask = LayoutMask(2 * MAX_KV_STACK_LEN, 2 * MAX_KV_STACK_LEN);
        }

        coreNum = AscendC::GetBlockNum();
        if ASCEND_IS_AIC {
            coreIdx = AscendC::GetBlockIdx();
        } else {
            coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        }

        // N 方向（head）合并上限：任务数足够多时，把多个 head 合到一个 task 中，
        // 从而成倍减少 task 数与跨核握手次数；batch 不足或 paged 场景保持原行为。
        // tile 的具体大小由 GetQNBlockTile 按"行状态 buffer 容量 + KV 段对齐"统一决定：
        // GQA 段内 head 共享 K/V，整段合并成一个大 GEMM（每段只占一份行状态）；
        // MHA 每段只有 1 个 head，每个 head 都要独占 RoundUp<8>(qS) 行行状态，
        // 于是容量自然把 tile 压到 128/RoundUp<8>(qS)（s=3 时为 16）。
        if constexpr (PAGED_CACHE_FLAG_) {
            qNCap = groupSize;
        } else {
            qNCap = (batch >= coreNum) ? qHeads : groupSize;
        }
        if constexpr (INPUT_LAYOUT_ == InputLayoutT::BSND) {
            taskInfo[0].qSeqlen = maxQSeqlen;
            taskInfo[1].qSeqlen = maxQSeqlen;
            CalcNormTailTask(maxQSeqlen);
        }
    }

    CATLASS_INLINE void RunMainLoop()
    {
        uint32_t loopNum = 0;
        uint32_t kvSTail = 0;
        for (uint32_t taskIdx = coreIdx; kvSTail == 0; taskIdx += coreNum, curTaskMod ^= 1) {
            TaskInfo& curTask = taskInfo[curTaskMod];
            uint32_t& kvSIdx = curTask.kvSIdx;
            uint32_t& kvSLoop = curTask.kvSLoop;
            uint32_t& qSeqlen = curTask.qSeqlen;
            uint32_t& kvSeqlen = curTask.kvSeqlen;
            uint32_t& triUp = curTask.triUp;
            uint32_t& noSkipKvS = curTask.noSkipKvS;
            uint64_t& gmOffsetQ = curTask.gmOffsetQ;
            uint64_t& gmOffsetK = curTask.gmOffsetK;
            uint64_t& gmOffsetV = curTask.gmOffsetV;
            uint64_t& gmOffsetO = curTask.gmOffsetO;
            uint64_t& gmOffsetLse = curTask.gmOffsetLse;
            uint64_t& gmOffsetBlock = curTask.gmOffsetBlock;
            uint32_t& qSBlockSize = curTask.qSBlockSize;
            uint32_t& qNBlockSize = curTask.qNBlockSize;
            uint32_t& rowNum = curTask.rowNum;
            uint32_t& sRowPad = curTask.sRowPad;
            uint32_t& qNHeadStart = curTask.qNHeadStart;
            uint32_t& headInGroup = curTask.headInGroup;
            uint32_t& kvHeadBase = curTask.kvHeadBase;
            uint32_t& segN = curTask.segN;
            uint32_t& segNum = curTask.segNum;
            uint32_t& segRows = curTask.segRows;
            uint32_t& segRowStride = curTask.segRowStride;
            uint32_t& segL1Stride = curTask.segL1Stride;
            uint32_t& segSStride = curTask.segSStride;
            uint32_t& segOStride = curTask.segOStride;

            TaskInfo& preTask = taskInfo[curTaskMod ^ 1];
            qSeqlen = preTask.qSeqlen;
            kvSeqlen = preTask.kvSeqlen;
            while (curBatch < batch && taskIdx >= curTotalTaskNum) {
                if (curBatch > 0) {
                    preTotalTaskNum = curTotalTaskNum;
                    qBOffset += qSeqlen * strideQ;
                    if constexpr (!PAGED_CACHE_FLAG_) {
                        kBOffset += kvSeqlen * strideK;
                        vBOffset += kvSeqlen * strideV;
                    } else {
                        blockBOffset += maxNumBlocksPerBatch;
                    }
                    oBOffset += qSeqlen * strideO;
                    if constexpr (INPUT_LAYOUT_ == InputLayoutT::BSND) {
                        lseBOffset += qSeqlen * qHeads;
                    } else {
                        lseBOffset += qSeqlen;
                    }
                }

                if constexpr (INPUT_LAYOUT_ == InputLayoutT::BSND) {
                    kvSeqlen = gActualKvseqlen.GetValue(curBatch);
                } else {
                    qSeqlen = gActualQseqlen.GetValue(curBatch + 1) - gActualQseqlen.GetValue(curBatch);
                    if constexpr (!PAGED_CACHE_FLAG_) {
                        kvSeqlen = gActualKvseqlen.GetValue(curBatch + 1) - gActualKvseqlen.GetValue(curBatch);
                    }
                    CalcNormTailTask(qSeqlen);
                }

                curTotalTaskNum += normTaskNum + tailTaskNum;
                curBatch++;
            }

            if (taskIdx < curTotalTaskNum) {
                uint32_t curTaskIdx = taskIdx - preTotalTaskNum;
                bool tailTask = (curTaskIdx >= normTaskNum);
                uint32_t& qNBlockTile = tailTask ? tailQNBlockTile : normQNBlockTile;

                uint32_t qSStartIdx =
                    tailTask ? (qSeqlen - tailQSBlockTile) : ((curTaskIdx / normQNBlockNum) * normQSBlockTile);
                uint32_t qNBlockIdx = tailTask ? (curTaskIdx - normTaskNum) : (curTaskIdx % normQNBlockNum);

                qNHeadStart = qNBlockIdx * qNBlockTile;              // 本 task 覆盖的 head 起点（绝对下标）
                qSBlockSize = tailTask ? tailQSBlockTile : normQSBlockTile;
                qNBlockSize = Min(qNBlockTile, qHeads - qNHeadStart); // 本 task 覆盖的 head 数
                rowNum = qSBlockSize * qNBlockSize;                   // 多个 head 的 Q 行拼成连续的 rowNum 行
                sRowPad = 0;                                          // 段内整块 flat 处理，不再需要 per-head 行 pad
                headInGroup = qNHeadStart % groupSize;
                kvHeadBase = qNHeadStart / groupSize;

                // 统一段划分：GetQNBlockTile 保证 head 起点落在 KV 段边界（tile 为 groupSize
                // 的倍数或其因子），因此一个 task 内各段 head 数/行数/各 buffer 步长都是常量。
                // 行状态按紧凑行布局：段 s 的行状态紧接段 s-1，AIV 侧可把全部段的行当成一块处理。
                segN = Min(qNBlockSize, groupSize);             // 每段 head 数
                segNum = CeilDiv(qNBlockSize, segN);            // 段数
                segRows = qSBlockSize * segN;                   // 每段 Q 行数
                segRowStride = segRows;                         // gl/gm/dm 行状态步长（紧凑，无 pad）
                segL1Stride = RoundUp<NUM_16>(segRows) * embed; // L1 中 Q 步长（16 行对齐）
                segSStride = segRows * MAX_KV_STACK_LEN;        // S/P workspace 步长
                segOStride = segRows * embedRound;              // O_tmp workspace 步长

                gmOffsetQ = qBOffset + qSStartIdx * strideQ + qNHeadStart * embed;
                gmOffsetK = kBOffset + kvHeadBase * embed;
                gmOffsetV = vBOffset + kvHeadBase * embed;
                gmOffsetO = oBOffset + qSStartIdx * strideO + qNHeadStart * embed;
                gmOffsetLse = lseBOffset + qNHeadStart * strideLse + qSStartIdx;
                gmOffsetBlock = blockBOffset;

                if constexpr (MASK_MODE_ == MaskModeT::MASK_CAUSAL) {
                    int32_t seqDiff = static_cast<int32_t>(kvSeqlen) - static_cast<int32_t>(qSeqlen);
                    triUp = Max(static_cast<int32_t>(qSStartIdx) + seqDiff, 0); // Q 分块首行允许访问的最大 KV 下标
                    int32_t qSEndIdx = qSStartIdx + qSBlockSize;                // Q 分块的结束下标（开区间）
                    noSkipKvS = Max(qSEndIdx + seqDiff, 0);                     // task 计算的 KV 序列长度
                    noSkipKvS = Min(kvSeqlen, noSkipKvS);                       // 不超过实际的 KV 序列长度
                } else {
                    noSkipKvS = kvSeqlen; // task 计算的 KV 序列长度
                }
                kvSLoop = CeilDiv<MAX_KV_STACK_LEN>(noSkipKvS); // task 的 KV 循环分块数目

                if ASCEND_IS_AIC {
                    // 每个 KV 段一次连续搬运：[qSBlockSize, segN, embed]，段间在 L1 中按 16 行对齐
                    layoutQTmp = LayoutQ(segRows, embed);
                    for (uint32_t segIdx = 0; segIdx < segNum; ++segIdx) {
                        blockMmadQK.loadQGM(
                            gQ[gmOffsetQ + segIdx * segN * embed], layoutQTmp, qSBlockSize, segN, strideQ,
                            segIdx * segL1Stride, (segIdx == 0), (segIdx == segNum - 1));
                    }
                }
            } else {
                kvSLoop = 0;
                kvSTail = PRE_LAUNCH;
            }

            for (kvSIdx = 0; kvSIdx < kvSLoop + kvSTail; kvSIdx++, loopNum++) {
                // *** QK & SM ***
                if (kvSIdx < kvSLoop) {
                    if (kvSIdx == kvSLoop - 1) {
                        stackSeqTile = noSkipKvS - kvSIdx * MAX_KV_STACK_LEN;
                    } else {
                        stackSeqTile = MAX_KV_STACK_LEN;
                    }
                    uint32_t curStackTileMod = loopNum % DOUBLE_BUFFER;
                    uint64_t gmOffsetS = (coreIdx * DOUBLE_BUFFER + curStackTileMod) * WORKSPACE_BLOCK_SIZE_DB;
                    if ASCEND_IS_AIC {
                        // 按 KV 段做 QK：同一段内所有 head 共享同一份 K，整体作为一个大 GEMM
                        LayoutS layoutS(segRows, stackSeqTile, MAX_KV_STACK_LEN);
                        GemmCoord shapeQK{segRows, stackSeqTile, embed};
                        for (uint32_t segIdx = 0; segIdx < segNum; ++segIdx) {
                            blockMmadQK(
                                gK[gmOffsetK + segIdx * embed], gS[gmOffsetS + segIdx * segSStride],
                                gBlockTable[gmOffsetBlock], layoutKTmp, layoutS, shapeQK, kvSIdx, pagedBlockSize,
                                strideK, (kvSIdx == kvSLoop - 1) && (segIdx == segNum - 1),
                                segIdx * segL1Stride);
                        }
                        Arch::CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(qkReady);
                    } else { // ASCEND_IS_AIV
                        // online softmax 是逐行独立的：一个 task 内所有段（head/group）的 S
                        // 在 workspace 里紧凑连续，直接拼成 rowNum 行一次算完，不再按段循环。
                        LayoutS layoutS(rowNum, stackSeqTile, MAX_KV_STACK_LEN);
                        LayoutP layoutP = layoutS;
                        GemmCoord shapeQK{rowNum, stackSeqTile, embed};
                        if constexpr (MASK_MODE_ == MaskModeT::MASK_CAUSAL) {
                            uint32_t kvSStartIdx = kvSIdx * MAX_KV_STACK_LEN; // KV 分块的起始下标（闭区间）
                            uint32_t kvSEndIdx = kvSStartIdx + stackSeqTile;  // KV 分块的结束下标（开区间）
                            epilogueOnlineSoftmax(
                                gP[gmOffsetS], gS[gmOffsetS], gMask, layoutP, layoutS, layoutMask, shapeQK,
                                (kvSIdx == 0), curStackTileMod, curTaskMod, qkReady, qSBlockSize, triUp,
                                kvSStartIdx, kvSEndIdx, 0, true);
                        } else {
                            epilogueOnlineSoftmax(
                                gP[gmOffsetS], gS[gmOffsetS], gMask, layoutP, layoutS, layoutMask, shapeQK,
                                (kvSIdx == 0), curStackTileMod, curTaskMod, qkReady, qSBlockSize, 0, 0, 0, 0, true);
                        }
                        Arch::CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(softmaxReady);
                    }
                }

                // *** PV & RS ***
                if (loopNum >= PRE_LAUNCH) {
                    uint32_t nowTaskMod = (kvSIdx == 0) ? curTaskMod ^ 1 : curTaskMod;
                    TaskInfo& nowTask = taskInfo[nowTaskMod];
                    uint32_t& kvSIdx = nowTask.kvSIdx;
                    uint32_t& kvSLoop = nowTask.kvSLoop;
                    uint32_t& qSeqlen = nowTask.qSeqlen;
                    uint32_t& noSkipKvS = nowTask.noSkipKvS;
                    uint64_t& gmOffsetV = nowTask.gmOffsetV;
                    uint64_t& gmOffsetO = nowTask.gmOffsetO;
                    uint64_t& gmOffsetLse = nowTask.gmOffsetLse;
                    uint64_t& gmOffsetBlock = nowTask.gmOffsetBlock;
                    uint32_t& qSBlockSize = nowTask.qSBlockSize;
                    uint32_t& qNBlockSize = nowTask.qNBlockSize;
                    uint32_t& rowNum = nowTask.rowNum;

                    uint32_t& headInGroup = nowTask.headInGroup;
                    uint32_t& segN = nowTask.segN;
                    uint32_t& segNum = nowTask.segNum;
                    uint32_t& segRows = nowTask.segRows;
                    uint32_t& segRowStride = nowTask.segRowStride;
                    uint32_t& segSStride = nowTask.segSStride;
                    uint32_t& segOStride = nowTask.segOStride;

                    uint32_t nowKvSIdx = kvSIdx - PRE_LAUNCH;
                    if (nowKvSIdx == kvSLoop - 1) {
                        stackSeqTile = noSkipKvS - nowKvSIdx * MAX_KV_STACK_LEN;
                    } else {
                        stackSeqTile = MAX_KV_STACK_LEN;
                    }
                    uint32_t nowStackTileMod = (loopNum - PRE_LAUNCH) % DOUBLE_BUFFER;
                    uint64_t gmOffsetOTmp = (coreIdx * DOUBLE_BUFFER + nowStackTileMod) * WORKSPACE_BLOCK_SIZE_DB;

                    if ASCEND_IS_AIC {
                        LayoutP layoutP(segRows, stackSeqTile, MAX_KV_STACK_LEN);
                        LayoutOTmp layoutOTmp(segRows, embed, embedRound);
                        GemmCoord shapePV{segRows, embed, stackSeqTile};
                        for (uint32_t segIdx = 0; segIdx < segNum; ++segIdx) {
                            blockMmadPV(
                                gP[gmOffsetOTmp + segIdx * segSStride], gV[gmOffsetV + segIdx * embed],
                                gOTmp[gmOffsetOTmp + segIdx * segOStride], gBlockTable[gmOffsetBlock], layoutP,
                                layoutVTmp, layoutOTmp, shapePV, nowKvSIdx, pagedBlockSize, strideV,
                                softmaxReady, (segIdx == 0));
                        }
                        Arch::CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(pvReady);
                    } else { // ASCEND_IS_AIV
                        LayoutO layoutO(qSeqlen, strideO);
                        LayoutLse layoutLse(qHeads, strideLse);
                        uint64_t gmOffsetUpdate = coreIdx * WORKSPACE_BLOCK_SIZE_DB;
                        // 与 softmax 一致：所有段的行紧凑拼接，rescale/O 累加一次算完。
                        // O/LSE 的 GM 目标按行 head-major 展开，CopyOToGm/CopyLseToGm 内部
                        // 用 qSBlockSize 折行，可跨 head 连续写。
                        LayoutOTmp layoutOTmp(rowNum, embed, embedRound);
                        LayoutUpdate layoutUpdate = layoutOTmp;
                        GemmCoord shapePV{rowNum, embed, stackSeqTile};
                        epilogueRescaleO(
                            gO[gmOffsetO], gOTmp[gmOffsetOTmp], gOUpdate[gmOffsetUpdate], gLse[gmOffsetLse],
                            layoutO, layoutOTmp, layoutUpdate, layoutLse, shapePV, qSBlockSize, qNBlockSize,
                            (nowKvSIdx == 0), (nowKvSIdx == kvSLoop - 1), nowStackTileMod, nowTaskMod, pvReady,
                            0, true);
                    }
                }
            }
        }
    }

    CATLASS_DEVICE void PostProcess()
    {
        if ASCEND_IS_AIC {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
        } else {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    CATLASS_DEVICE void CalcNormTailTask(uint32_t qSeqlen)
    {
        normQSBlockTile = GetQSBlockTile();
        normQSBlockNum = qSeqlen / normQSBlockTile;
        if (normQSBlockNum > 0) {
            normQNBlockTile = GetQNBlockTile(qSeqlen, groupSize, qNCap, qHeads);
            normQNGroupBlockNum = 1;
            normQNBlockNum = CeilDiv(qHeads, normQNBlockTile);
        }

        tailQSBlockTile = qSeqlen % normQSBlockTile;
        if (tailQSBlockTile > 0) {
            tailQSBlockNum = 1;
            tailQNBlockTile = GetQNBlockTile(tailQSBlockTile, groupSize, qNCap, qHeads);
            tailQNGroupBlockNum = 1;
            tailQNBlockNum = CeilDiv(qHeads, tailQNBlockTile);
        } else {
            tailQSBlockNum = 0;
        }

        normTaskNum = normQNBlockNum * normQSBlockNum;
        tailTaskNum = tailQNBlockNum * tailQSBlockNum;
    }

    CATLASS_DEVICE void operator()(const FAIKernelParams& params, const Arch::Resource<ArchTag>& resource)
    {
        PreProcess(params, resource);
        RunMainLoop();
        PostProcess();
    }

private:
    uint32_t coreNum, coreIdx;
    uint32_t batch, qHeads, kvHeads, embed;
    uint32_t pagedBlockSize, maxNumBlocksPerBatch, maxQSeqlen;

    AscendC::GlobalTensor<ElementQ> gQ;
    AscendC::GlobalTensor<ElementK> gK;
    AscendC::GlobalTensor<ElementV> gV;
    AscendC::GlobalTensor<ElementMask> gMask;
    AscendC::GlobalTensor<int32_t> gBlockTable;
    AscendC::GlobalTensor<int32_t> gActualQseqlen;
    AscendC::GlobalTensor<int32_t> gActualKvseqlen;
    AscendC::GlobalTensor<ElementO> gO;
    AscendC::GlobalTensor<ElementLse> gLse;
    AscendC::GlobalTensor<ElementS> gS;
    AscendC::GlobalTensor<ElementP> gP;
    AscendC::GlobalTensor<ElementOTmp> gOTmp;
    AscendC::GlobalTensor<ElementOTmp> gOUpdate;

    uint32_t strideQ, strideK, strideV, strideO, strideLse;
    uint32_t embedRound, groupSize, stackSeqTile;
    uint32_t qNCap = 1;

    uint32_t l1PingPong = 0;
    uint32_t l0ABPingPong = 0;
    uint32_t l0CPingPong = 0;

    BlockMmadQK blockMmadQK{l1PingPong, l0ABPingPong, l0CPingPong};
    EpilogueOnlineSoftmax epilogueOnlineSoftmax;
    BlockMmadPV blockMmadPV{l1PingPong, l0ABPingPong, l0CPingPong};
    EpilogueRescaleO epilogueRescaleO;

    LayoutQ layoutQTmp;
    LayoutK layoutKTmp;
    LayoutV layoutVTmp;
    LayoutMask layoutMask;

    uint64_t qBOffset = 0;
    uint64_t kBOffset = 0;
    uint64_t vBOffset = 0;
    uint64_t oBOffset = 0;
    uint64_t lseBOffset = 0;
    uint64_t blockBOffset = 0;

    uint32_t preTotalTaskNum = 0;
    uint32_t curTotalTaskNum = 0;
    uint32_t curBatch = 0;
    uint32_t normQSBlockTile, normQSBlockNum, normQNBlockTile, normQNGroupBlockNum, normQNBlockNum, normTaskNum;
    uint32_t tailQSBlockTile, tailQSBlockNum, tailQNBlockTile, tailQNGroupBlockNum, tailQNBlockNum, tailTaskNum;

    struct TaskInfo {
        uint32_t kvSIdx, kvSLoop;
        uint32_t qSeqlen, kvSeqlen, triUp, noSkipKvS;
        uint64_t gmOffsetQ, gmOffsetK, gmOffsetV, gmOffsetO, gmOffsetLse, gmOffsetBlock;
        uint32_t qSBlockSize, qNBlockSize, rowNum, sRowPad;
        uint32_t qNHeadStart, headInGroup, kvHeadBase;   // head 分块信息
        // 统一 KV 段划分（一个 task 内为常量，段内循环只做乘法）
        uint32_t segN, segNum, segRows, segRowStride, segL1Stride, segSStride, segOStride;
    };
    TaskInfo taskInfo[2];
    uint32_t curTaskMod = 0;

    Arch::CrossCoreFlag qkReady{QK_READY_ID};
    Arch::CrossCoreFlag softmaxReady{SOFTMAX_READY_ID};
    Arch::CrossCoreFlag pvReady{PV_READY_ID};
};

template <typename INPUT_DTYPE_, bool PAGED_CACHE_FLAG_, MaskModeT MASK_MODE_, InputLayoutT INPUT_LAYOUT_>
CATLASS_GLOBAL void FAInfer(
    uint64_t fftsAddr, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mask, GM_ADDR blockTables, GM_ADDR o, GM_ADDR lse,
    GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::SetSyncBaseAddr(fftsAddr);

    using ArchTag = Arch::AtlasA2;
    Arch::Resource<ArchTag> resource;

    using ElementQ = INPUT_DTYPE_;
    using LayoutQ = layout::RowMajor;
    using ElementK = INPUT_DTYPE_;
    using LayoutK = layout::ColumnMajor;
    using ElementV = INPUT_DTYPE_;
    using LayoutV = layout::RowMajor;

    using ElementS = float;
    using LayoutS = layout::RowMajor;
    using ElementP = INPUT_DTYPE_;
    using LayoutP = layout::RowMajor;
    using ElementO = INPUT_DTYPE_;
    using LayoutO = layout::RowMajor;

    using ElementLse = float;
    using LayoutLse = layout::RowMajor;
    using ElementMask = uint8_t;
    using LayoutMask = layout::RowMajor;
    using ElementOTmp = float;
    using LayoutOTmp = layout::RowMajor;
    using ElementUpdate = float;
    using LayoutUpdate = layout::RowMajor;

    using DispatchPolicyQK = Gemm::MmadAtlasA2FAIQKT<PAGED_CACHE_FLAG_>;
    using L1TileShapeQK = GemmShape<Q_TILE_CEIL, 128, 128>;
    using L0TileShapeQK = GemmShape<128, 128, 128>;
    using QType = Gemm::GemmType<ElementQ, LayoutQ>;
    using KType = Gemm::GemmType<ElementK, LayoutK>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using BlockMmadQK = Gemm::Block::BlockMmad<DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK, QType, KType, SType>;

    using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueAtlasA2OnlineSoftmaxT<MASK_MODE_>;
    using PType = Gemm::GemmType<ElementP, LayoutP>;
    using MaskType = Gemm::GemmType<ElementMask, LayoutMask>;
    using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType, MaskType>;

    using DispatchPolicyPV = Gemm::MmadAtlasA2FAIPVT<PAGED_CACHE_FLAG_>;
    using L1TileShapePV = GemmShape<128, 128, 256>;
    using L0TileShapePV = GemmShape<128, 128, 128>;
    using VType = Gemm::GemmType<ElementV, LayoutV>;
    using OTmpType = Gemm::GemmType<ElementOTmp, LayoutOTmp>;
    using BlockMmadPV = Gemm::Block::BlockMmad<DispatchPolicyPV, L1TileShapePV, L0TileShapePV, PType, VType, OTmpType>;

    using DispatchPolicyRescaleO = Epilogue::EpilogueAtlasA2RescaleOT;
    using OType = Gemm::GemmType<ElementO, LayoutO>;
    using OUpdateType = Gemm::GemmType<ElementUpdate, LayoutUpdate>;
    using LseType = Gemm::GemmType<ElementLse, LayoutLse>;
    using EpilogueRescaleO =
        Epilogue::Block::BlockEpilogue<DispatchPolicyRescaleO, OType, OTmpType, OUpdateType, LseType>;

    using FAInferKernel = FAInferKernel<
        BlockMmadQK, BlockMmadPV, EpilogueOnlineSoftmax, EpilogueRescaleO, PAGED_CACHE_FLAG_, MASK_MODE_,
        INPUT_LAYOUT_>;
    FAIKernelParams params{q, k, v, mask, blockTables, actualQseqlen, actualKvseqlen, o, lse, workspace, tiling};
    FAInferKernel fAInfer;
    fAInfer(params, resource);
}

#endif // MHA_FWD_KVCACHE_HPP