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
    MaskType MASK_TYPE_, InputLayout INPUT_LAYOUT_>
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

    CATLASS_DEVICE FAInferKernel() = default;

    CATLASS_DEVICE void operator()(FAIKernelParams const& params)
    {
        auto fATilingData = reinterpret_cast<__gm__ FAInferTilingData*>(params.tiling);
        uint64_t mm1OutSize = fATilingData->mm1OutSize;
        uint64_t smOnlineOutSize = fATilingData->smOnlineOutSize;
        uint64_t mm2OutSize = fATilingData->mm2OutSize;
        uint32_t batch = fATilingData->batch;
        uint32_t qHeads = fATilingData->numHeads;
        uint32_t kvHeads = fATilingData->kvHeads;
        uint32_t embed = fATilingData->embeddingSize;
        uint32_t pagedBlockSize = fATilingData->blockSize;
        uint32_t maxNumBlocksPerBatch = fATilingData->maxNumBlocksPerBatch;
        uint32_t maxQSeqlen = fATilingData->maxQSeqlen;
        float scaleValue = fATilingData->scaleValue;

        AscendC::GlobalTensor<ElementQ> gQ;
        gQ.SetGlobalBuffer((__gm__ ElementQ*)params.q);
        AscendC::GlobalTensor<ElementK> gK;
        gK.SetGlobalBuffer((__gm__ ElementK*)params.k);
        AscendC::GlobalTensor<ElementV> gV;
        gV.SetGlobalBuffer((__gm__ ElementV*)params.v);

        AscendC::GlobalTensor<ElementMask> gMask;
        gMask.SetGlobalBuffer((__gm__ ElementMask*)params.mask);
        AscendC::GlobalTensor<int32_t> gBlockTable;
        gBlockTable.SetGlobalBuffer((__gm__ int32_t*)(params.blockTable));
        AscendC::GlobalTensor<int32_t> gActualQseqlen;
        gActualQseqlen.SetGlobalBuffer((__gm__ int32_t*)params.actualQseqlen);
        AscendC::GlobalTensor<int32_t> gActualKvseqlen;
        gActualKvseqlen.SetGlobalBuffer((__gm__ int32_t*)params.actualKvseqlen);

        AscendC::GlobalTensor<ElementO> gO;
        gO.SetGlobalBuffer((__gm__ ElementO*)params.o);
        AscendC::GlobalTensor<ElementLse> gLse;
        gLse.SetGlobalBuffer((__gm__ ElementLse*)params.lse);

        AscendC::GlobalTensor<ElementS> gS;
        gS.SetGlobalBuffer((__gm__ ElementS*)(params.workspace));
        AscendC::GlobalTensor<ElementP> gP;
        gP.SetGlobalBuffer((__gm__ ElementP*)(params.workspace + mm1OutSize));
        AscendC::GlobalTensor<ElementOTmp> gOTmp;
        gOTmp.SetGlobalBuffer((__gm__ ElementOTmp*)(params.workspace + mm1OutSize + smOnlineOutSize));
        AscendC::GlobalTensor<ElementOTmp> gOUpdate;
        gOUpdate.SetGlobalBuffer((__gm__ ElementOTmp*)(params.workspace + mm1OutSize + smOnlineOutSize + mm2OutSize));

        if ASCEND_IS_AIC {
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        } else {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        }

        if ASCEND_IS_AIC {
            uint32_t kDynNum = RoundUp(embed, NUM_128);                           // K 对应 D 维度
            uint32_t nDynNum = NUM_128;                                           // N 对应 KV 的序列长度
            uint32_t kPVDynNum = nDynNum * kDynNum / BlockMmadPV::L1TileShape::M; // P 复用 K 的 L1 空间
            uint32_t maxQL1Size = kDynNum * Q_TILE_CEIL * sizeof(ElementQ);       // Q 的预留 L1 空间

            blockMmadQK.init(resource, nDynNum, kDynNum, MAX_KV_STACK_LEN);
            blockMmadPV.init(resource, kPVDynNum, MAX_KV_STACK_LEN, maxQL1Size);
        } else { // ASCEND_IS_AIV
            epilogueOnlineSoftmax.init(resource, scaleValue);
            epilogueRescaleO.init(resource);
        }

        uint32_t strideQ = qHeads * embed;
        uint32_t strideK = kvHeads * embed;
        uint32_t strideV = kvHeads * embed;
        uint32_t strideO = qHeads * embed;
        uint32_t embedRound = RoundUp(embed, NUM_16);
        uint32_t groupSize = qHeads / kvHeads;
        uint32_t stackSeqTile = MAX_KV_STACK_LEN;

        uint64_t qBOffset = 0;
        uint64_t kBOffset = 0;
        uint64_t vBOffset = 0;
        uint64_t oBOffset = 0;
        uint64_t lseBOffset = 0;
        uint64_t blockBOffset = 0;

        uint32_t preTotalTaskNum = 0;
        uint32_t curTotalTaskNum = 0;
        uint32_t curBatch = 0;
        uint32_t curQNBlockTile = 0;
        uint32_t qNBlockNumPerGroup = 0;
        uint32_t curQNBlockNum = 0;
        uint32_t curQSBlockTile = 0;
        uint32_t curQSBlockNum = 0;

        uint32_t totalQTokens;
        if constexpr (INPUT_LAYOUT_ == InputLayout::TND) {
            totalQTokens = gActualQseqlen.GetValue(batch);
        } else {
            totalQTokens = maxQSeqlen; // Token per batch for BSND
        }

        uint32_t coreNum = AscendC::GetBlockNum();
        uint32_t coreIdx;
        if ASCEND_IS_AIC {
            coreIdx = AscendC::GetBlockIdx();
        } else {
            coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        }

        uint32_t loopNum = 0;
        bool launchNextTask = true;
        for (uint32_t taskIdx = coreIdx; launchNextTask; taskIdx += coreNum, SwapTask()) {
            for (launchNextTask = false; kvSIdx < kvSLoop + PRE_LAUNCH; kvSIdx++, loopNum++) {
                // *** PRE LAUNCH ***
                if (kvSIdx == kvSLoop) {
                    SwapTask();
                    qSeqlen = preTaskInfo.qSeqlen;
                    kvSeqlen = preTaskInfo.kvSeqlen;
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
                            // LSE 输出是 N 主序 (BSND -> BNS，TND -> NT)
                            if constexpr (INPUT_LAYOUT_ == InputLayout::TND) {
                                lseBOffset += qSeqlen;
                            } else {
                                lseBOffset += qSeqlen * qHeads;
                            }
                        }

                        qSeqlen = maxQSeqlen;
                        kvSeqlen = gActualKvseqlen.GetValue(curBatch);
                        if constexpr (INPUT_LAYOUT_ == InputLayout::TND) {
                            qSeqlen = gActualQseqlen.GetValue(curBatch + 1) - gActualQseqlen.GetValue(curBatch);
                            if constexpr (!PAGED_CACHE_FLAG_) {
                                kvSeqlen = gActualKvseqlen.GetValue(curBatch + 1) - gActualKvseqlen.GetValue(curBatch);
                            }
                        }
                        curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);     // Q 的 N 维分块大小
                        qNBlockNumPerGroup = CeilDiv(groupSize, curQNBlockTile); // 每个 group 的 N 维分块数量，向上取整
                        curQNBlockNum = qNBlockNumPerGroup * kvHeads;            // Q 的 N 维分块数量
                        curQSBlockTile = GetQSBlockTile();                       // Q 的 S 维分块大小
                        curQSBlockNum = CeilDiv(qSeqlen, curQSBlockTile);        // Q 的 S 维分块数量，向上取整
                        curTotalTaskNum += curQNBlockNum * curQSBlockNum;        // 新增当前 batch 的 Q 任务数量
                        curBatch++;
                    }

                    if (taskIdx < curTotalTaskNum) {
                        uint32_t taskIdxCurBatch = taskIdx - preTotalTaskNum;
                        uint32_t qSBlockIdx = taskIdxCurBatch / curQNBlockNum;         // task 的 S 维分块索引
                        uint32_t qNBlockIdx = taskIdxCurBatch % curQNBlockNum;         // task 的 N 维分块索引
                        uint32_t qSStartIdx = qSBlockIdx * curQSBlockTile;             // Q 的 S 维元素起始索引
                        uint32_t qNBlockIdxCurGroup = qNBlockIdx % qNBlockNumPerGroup; // Q 的 group 内 N 维分块索引
                        uint32_t kvNIdx = qNBlockIdx / qNBlockNumPerGroup;             // Q 对应 KV 的 N 维分块索引
                        uint32_t qNStartIdx =
                            kvNIdx * groupSize + qNBlockIdxCurGroup * curQNBlockTile; // Q 的 N 维元素起始索引

                        gmOffsetQ = qBOffset +             // B 偏移
                                    qSStartIdx * strideQ + // S 偏移
                                    qNStartIdx * embed;    // N 偏移

                        gmOffsetK = kBOffset +      // B 偏移
                                    kvNIdx * embed; // N 偏移，在后面遍历 S 维

                        gmOffsetV = vBOffset +      // B 偏移
                                    kvNIdx * embed; // N 偏移，在后面遍历 S 维

                        gmOffsetO = oBOffset +             // B 偏移
                                    qSStartIdx * strideO + // S 偏移
                                    qNStartIdx * embed;    // N 偏移

                        gmOffsetLse = lseBOffset +                // B 偏移
                                      qNStartIdx * totalQTokens + // N 偏移
                                      qSStartIdx;                 // S 偏移

                        gmOffsetBlock = blockBOffset;

                        qSBlockSize = (qSBlockIdx == curQSBlockNum - 1) ? (qSeqlen - qSStartIdx) : curQSBlockTile;
                        qNBlockSize = (qNBlockIdxCurGroup == qNBlockNumPerGroup - 1)
                                          ? (groupSize - qNBlockIdxCurGroup * curQNBlockTile)
                                          : curQNBlockTile;
                        rowNum = qSBlockSize * qNBlockSize; // curTask 的 Q 行数

                        noSkipKvS = kvSeqlen;                      // curTask 计算的 KV 序列长度
                        if (MASK_TYPE_ == MaskType::MASK_CAUSAL) { // 因果掩码，Q 只能看到之前的 KV
                            uint32_t diffS = kvSeqlen > qSeqlen ? kvSeqlen - qSeqlen : 0;
                            noSkipKvS = (qSBlockIdx + 1) * curQSBlockTile + // 当前分块最后一个 Q 的 S 维索引
                                        diffS;                              // Q 与 KV 的 S 维差值
                            noSkipKvS = Min(kvSeqlen, noSkipKvS);           // 不超过实际的 KV 序列长度
                        }
                        kvSLoop = CeilDiv(noSkipKvS, MAX_KV_STACK_LEN); // curTask 的 KV 循环分块数目

                        if ASCEND_IS_AIC {
                            layoutQTmp = LayoutQ(rowNum, embed);             // (S * N, D)
                            layoutKTmp = LayoutK(strideK, MAX_KV_STACK_LEN); // (N * D, S)
                            layoutVTmp = LayoutV(MAX_KV_STACK_LEN, strideV); // (S, N * D)
                            if (loopNum == PRE_LAUNCH) {
                                AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
                                AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
                            }
                            blockMmadQK.loadQGM(gQ[gmOffsetQ], layoutQTmp, qSBlockSize, qNBlockSize, qHeads);
                        }

                        kvSIdx = 0;
                        launchNextTask = true;
                    } else {
                        SwapTask();
                    }
                }

                // *** QK & SM ***
                if (kvSIdx < kvSLoop || launchNextTask) {
                    if (kvSIdx == kvSLoop - 1) {
                        stackSeqTile = noSkipKvS - kvSIdx * MAX_KV_STACK_LEN;
                    } else {
                        stackSeqTile = MAX_KV_STACK_LEN;
                    }
                    uint32_t curStackTileMod = loopNum % DOUBLE_BUFFER;
                    uint64_t gmOffsetS = (coreIdx * DOUBLE_BUFFER + curStackTileMod) * WORKSPACE_BLOCK_SIZE_DB;
                    GemmCoord actualBlockShapeQK{rowNum, stackSeqTile, embed};
                    LayoutS layoutS(rowNum, stackSeqTile, MAX_KV_STACK_LEN);

                    if ASCEND_IS_AIC {
                        if constexpr (PAGED_CACHE_FLAG_) {
                            blockMmadQK(
                                gK[gmOffsetK], gS[gmOffsetS], gBlockTable[gmOffsetBlock], layoutKTmp, layoutS,
                                actualBlockShapeQK, kvSIdx, pagedBlockSize, strideK);
                        } else {
                            blockMmadQK(
                                gK[gmOffsetK], gS[gmOffsetS], gBlockTable, layoutKTmp, layoutS, actualBlockShapeQK,
                                kvSIdx, pagedBlockSize, strideK);
                        }
                        if (loopNum == 0) {
                            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
                            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
                        }
                        Arch::CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(qkReady);
                    } else { // ASCEND_IS_AIV
                        LayoutP layoutP(rowNum, stackSeqTile, MAX_KV_STACK_LEN);
                        LayoutMask layoutMask(MASK_LEN, MASK_LEN);
                        uint64_t gmOffsetP = gmOffsetS;
                        uint32_t kvSStartIdx = kvSIdx * MAX_KV_STACK_LEN;
                        uint32_t kvSEndIdx = kvSStartIdx + stackSeqTile;
                        epilogueOnlineSoftmax.set_curTaskMod(curTaskMod);
                        if constexpr (MASK_TYPE_ == MaskType::MASK_CAUSAL) {
                            uint32_t triUp = noSkipKvS - qSBlockSize;
                            if (triUp < kvSEndIdx - 1) { // KV 分块内需要掩码
                                epilogueOnlineSoftmax(
                                    gP[gmOffsetP], gS[gmOffsetS], gMask, layoutP, layoutS, layoutMask,
                                    actualBlockShapeQK, (kvSIdx == 0), qSBlockSize, qNBlockSize, curStackTileMod,
                                    qkReady, triUp, noSkipKvS, kvSStartIdx, kvSEndIdx);
                            } else { // KV 分块内无需掩码
                                uint32_t noMaskStackSeqNum = (triUp + 1) / MAX_KV_STACK_LEN;
                                Arch::CrossCoreWaitFlag(qkReady);
                                epilogueOnlineSoftmax(
                                    gP[gmOffsetP], gS[gmOffsetS], layoutP, layoutS, actualBlockShapeQK, (kvSIdx == 0),
                                    (kvSIdx == noMaskStackSeqNum - 1), qSBlockSize, qNBlockSize, curStackTileMod);
                            }
                        } else {
                            Arch::CrossCoreWaitFlag(qkReady);
                            epilogueOnlineSoftmax(
                                gP[gmOffsetP], gS[gmOffsetS], layoutP, layoutS, actualBlockShapeQK, (kvSIdx == 0), 0,
                                qSBlockSize, qNBlockSize, curStackTileMod);
                        }
                        Arch::CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(softmaxReady);
                    }

                    if (launchNextTask) {
                        kvSIdx++;
                        SwapTask();
                    }
                }

                // *** PV & RS ***
                if (loopNum >= PRE_LAUNCH) {
                    uint32_t nowKvSIdx = kvSIdx - PRE_LAUNCH;
                    if (nowKvSIdx == kvSLoop - 1) {
                        stackSeqTile = noSkipKvS - nowKvSIdx * MAX_KV_STACK_LEN;
                    } else {
                        stackSeqTile = MAX_KV_STACK_LEN;
                    }
                    uint32_t curStackTileMod = (loopNum - PRE_LAUNCH) % DOUBLE_BUFFER;
                    uint64_t gmOffsetOTmp = (coreIdx * DOUBLE_BUFFER + curStackTileMod) * WORKSPACE_BLOCK_SIZE_DB;
                    GemmCoord actualBlockShapePV{rowNum, embed, stackSeqTile};
                    LayoutOTmp layoutOTmp(rowNum, embed, embedRound);

                    if ASCEND_IS_AIC {
                        LayoutP layoutP(rowNum, stackSeqTile, MAX_KV_STACK_LEN);
                        uint64_t gmOffsetP = gmOffsetOTmp;
                        if constexpr (PAGED_CACHE_FLAG_) {
                            blockMmadPV(
                                gP[gmOffsetP], gV[gmOffsetV], gOTmp[gmOffsetOTmp], gBlockTable[gmOffsetBlock], layoutP,
                                layoutVTmp, layoutOTmp, actualBlockShapePV, nowKvSIdx, pagedBlockSize, strideV,
                                softmaxReady);
                        } else {
                            blockMmadPV(
                                gP[gmOffsetP], gV[gmOffsetV], gOTmp[gmOffsetOTmp], gBlockTable, layoutP, layoutVTmp,
                                layoutOTmp, actualBlockShapePV, nowKvSIdx, pagedBlockSize, strideV, softmaxReady);
                        }
                        Arch::CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(pvReady);
                    } else { // ASCEND_IS_AIV
                        LayoutO layoutO(qSeqlen, qHeads * embed);
                        LayoutUpdate layoutUpdate(rowNum, embed, embedRound);
                        LayoutLse layoutLse(qHeads, totalQTokens);
                        uint64_t gmOffsetUpdate = coreIdx * WORKSPACE_BLOCK_SIZE_DB;
                        epilogueRescaleO.set_curTaskMod(curTaskMod);

                        Arch::CrossCoreWaitFlag(pvReady);
                        epilogueRescaleO(
                            gO[gmOffsetO], gOTmp[gmOffsetOTmp], gOUpdate[gmOffsetUpdate], gLse[gmOffsetLse], layoutO,
                            layoutOTmp, layoutUpdate, layoutLse, actualBlockShapePV, qSBlockSize, qNBlockSize,
                            (nowKvSIdx == 0), (nowKvSIdx == kvSLoop - 1), curStackTileMod);
                    }
                }
            }
        }

        if ASCEND_IS_AIC {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
        } else {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    uint32_t kvSIdx = 0, kvSLoop = 0;
    uint32_t qSeqlen, kvSeqlen, noSkipKvS;
    uint64_t gmOffsetQ, gmOffsetK, gmOffsetV, gmOffsetO, gmOffsetLse, gmOffsetBlock;
    uint32_t qSBlockSize, qNBlockSize, rowNum;

    struct TaskInfo {
        uint32_t kvSIdx = 0, kvSLoop = 0;
        uint32_t qSeqlen, kvSeqlen, noSkipKvS;
        uint64_t gmOffsetQ, gmOffsetK, gmOffsetV, gmOffsetO, gmOffsetLse, gmOffsetBlock;
        uint32_t qSBlockSize, qNBlockSize, rowNum;
    };

    TaskInfo preTaskInfo;
    uint32_t curTaskMod = 0;

    template <typename T>
    CATLASS_DEVICE void Swap(T& a, T& b)
    {
        T tmp = a;
        a = b;
        b = tmp;
    }

    CATLASS_DEVICE void SwapTask()
    {
        Swap(preTaskInfo.kvSIdx, kvSIdx);
        Swap(preTaskInfo.kvSLoop, kvSLoop);
        Swap(preTaskInfo.qSeqlen, qSeqlen);
        Swap(preTaskInfo.kvSeqlen, kvSeqlen);
        Swap(preTaskInfo.noSkipKvS, noSkipKvS);
        Swap(preTaskInfo.gmOffsetQ, gmOffsetQ);
        Swap(preTaskInfo.gmOffsetK, gmOffsetK);
        Swap(preTaskInfo.gmOffsetV, gmOffsetV);
        Swap(preTaskInfo.gmOffsetO, gmOffsetO);
        Swap(preTaskInfo.gmOffsetLse, gmOffsetLse);
        Swap(preTaskInfo.gmOffsetBlock, gmOffsetBlock);
        Swap(preTaskInfo.qSBlockSize, qSBlockSize);
        Swap(preTaskInfo.qNBlockSize, qNBlockSize);
        Swap(preTaskInfo.rowNum, rowNum);
        curTaskMod ^= 1;
    }

private:
    Arch::Resource<ArchTag> resource;

    BlockMmadQK blockMmadQK;
    BlockMmadPV blockMmadPV;
    EpilogueOnlineSoftmax epilogueOnlineSoftmax;
    EpilogueRescaleO epilogueRescaleO;

    LayoutQ layoutQTmp;
    LayoutK layoutKTmp;
    LayoutV layoutVTmp;

    Arch::CrossCoreFlag qkReady{QK_READY_ID};
    Arch::CrossCoreFlag softmaxReady{SOFTMAX_READY_ID};
    Arch::CrossCoreFlag pvReady{PV_READY_ID};
};

template <typename INPUT_DTYPE_, bool PAGED_CACHE_FLAG_, MaskType MASK_TYPE_, InputLayout INPUT_LAYOUT_>
CATLASS_GLOBAL void FAInfer(
    uint64_t fftsAddr, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR mask, GM_ADDR blockTables, GM_ADDR o, GM_ADDR lse,
    GM_ADDR actualQseqlen, GM_ADDR actualKvseqlen, GM_ADDR workspace, GM_ADDR tiling)
{
    AscendC::SetSyncBaseAddr(fftsAddr);

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

    using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueAtlasA2OnlineSoftmaxT;
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
        BlockMmadQK, BlockMmadPV, EpilogueOnlineSoftmax, EpilogueRescaleO, PAGED_CACHE_FLAG_, MASK_TYPE_,
        INPUT_LAYOUT_>;
    FAIKernelParams params{q, k, v, mask, blockTables, actualQseqlen, actualKvseqlen, o, lse, workspace, tiling};
    FAInferKernel fAInfer;
    fAInfer(params);
}

#endif // MHA_FWD_KVCACHE_HPP