/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

#ifndef MHA_FWD_KVCACHE_HPP
#define MHA_FWD_KVCACHE_HPP

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

#include "kernel_operator.h"

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
        uint32_t embedV = fATilingData->embeddingSizeV;
        uint32_t pagedBlockSize = fATilingData->blockSize;
        uint32_t maxNumBlocksPerBatch = fATilingData->maxNumBlocksPerBatch;
        uint32_t maxQSeqlen = fATilingData->maxQSeqlen;
        uint32_t maxKvSeqlen = fATilingData->maxKvSeqlen;
        uint32_t maskType = fATilingData->maskType;
        float scaleValue = fATilingData->scaleValue;

        AscendC::GlobalTensor<ElementQ> gQ;
        gQ.SetGlobalBuffer((__gm__ ElementQ*)params.q);
        AscendC::GlobalTensor<ElementK> gK;
        gK.SetGlobalBuffer((__gm__ ElementK*)params.k);
        AscendC::GlobalTensor<ElementP> gP;
        gP.SetGlobalBuffer((__gm__ ElementP*)fATilingData->pDevice);
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

        AscendC::GlobalTensor<ElementS> gSTmp;
        gSTmp.SetGlobalBuffer((__gm__ ElementS*)(params.workSpace));
        AscendC::GlobalTensor<ElementP> gPTmp;
        gPTmp.SetGlobalBuffer((__gm__ ElementP*)(params.workSpace + mm1OutSize));
        AscendC::GlobalTensor<ElementOTmp> gOTmp;
        gOTmp.SetGlobalBuffer((__gm__ ElementOTmp*)(params.workSpace + mm1OutSize + smOnlineOutSize));
        AscendC::GlobalTensor<ElementOTmp> gOUpdate;
        gOUpdate.SetGlobalBuffer((__gm__ ElementOTmp*)(params.workSpace + mm1OutSize + smOnlineOutSize + mm2OutSize));

        if ASCEND_IS_AIC {
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
            AscendC::SetFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
            AscendC::SetFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
        } else {
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
        }

        if ASCEND_IS_AIC {
            uint32_t kDynNum = RoundUp(embed, NUM_128); // K 对应 embed 维度，不能太小
            kDynNum = Max(kDynNum, NUM_256);            // 至少为 256

            uint32_t maxQL1Size = kDynNum * Q_TILE_CEIL * sizeof(ElementQ);     // Q 的预留 L1 空间
            uint32_t maxVL1Size = embedV * MAX_KV_STACK_LEN * sizeof(ElementV); // V 的预留 L1 空间
            uint32_t maxKPL1Size = ArchTag::L1_SIZE - maxQL1Size - maxVL1Size;  // 剩余的 L1 空间用于 K 和 P 的复用
            uint32_t maxNDynNum = RoundDown(
                maxKPL1Size / sizeof(ElementK) / kDynNum / DOUBLE_BUFFER, NUM_32); // N 对应 K 和 V 的序列长度，不能过大
            uint32_t nDynNum = Min(maxNDynNum, L1_MAX_N_NUM);                      // 最大为 L1_MAX_N_NUM
            nDynNum = L1_MAX_N_NUM % nDynNum != 0 ? RoundDown((nDynNum - 1), NUM_32) : nDynNum;

            uint32_t kPVDynNum = nDynNum * kDynNum / BlockMmadPV::L1TileShape::M; // P 复用 K 的 L1 空间
            uint32_t qkL1Size = BlockMmadQK::L1TileShape::M * kDynNum * sizeof(ElementQ);

            blockMmadQK.init(resource, nDynNum, kDynNum, MAX_KV_STACK_LEN);
            blockMmadPV.init(resource, kPVDynNum, MAX_KV_STACK_LEN, qkL1Size);
        } else { // ASCEND_IS_AIV
            epilogueOnlineSoftmax.init(resource, scaleValue);
            epilogueRescaleO.init(resource);
        }

        uint64_t strideQ = static_cast<uint64_t>(qHeads * embed);
        uint64_t strideK = static_cast<uint64_t>(kvHeads * embed);
        uint64_t strideP = static_cast<uint64_t>(maxQSeqlen * maxKvSeqlen);
        uint64_t strideV = static_cast<uint64_t>(kvHeads * embedV);
        uint64_t strideO = static_cast<uint64_t>(qHeads * embedV);
        uint32_t embedRound = RoundUp(embed, BLOCK_SIZE);
        uint32_t embedRoundV = RoundUp(embedV, BLOCK_SIZE);
        uint32_t groupSize = qHeads / kvHeads;

        uint64_t qBOffset = 0;
        uint64_t kBOffset = 0;
        uint64_t pBOffset = 0;
        uint64_t vBOffset = 0;
        uint64_t oBOffset = 0;
        uint64_t lseBOffset = 0;
        uint64_t blockBOffset = 0;

        uint32_t preTotalTaskNum = 0;
        uint32_t curTotalTaskNum = 0;
        uint32_t curBatch = 0;
        uint32_t qSeqlen = 0;
        uint32_t kvSeqlen = 0;
        uint32_t curQNBlockTile = 0;
        uint32_t qNBlockNumPerGroup = 0;
        uint32_t curQNBlockNum = 0;
        uint32_t curQSBlockTile = 0;
        uint32_t curQSBlockNum = 0;

        uint32_t totalQTokens = 0;
        if constexpr (INPUT_LAYOUT_ == InputLayout::TND) {
            totalQTokens = static_cast<uint32_t>(gActualQseqlen.GetValue(batch));
        } else {
            totalQTokens = maxQSeqlen; // Token per batch for BSND
        }

        uint32_t coreNum = AscendC::GetBlockNum();
        if ASCEND_IS_AIC {
            coreIdx = AscendC::GetBlockIdx();
        } else {
            coreIdx = AscendC::GetBlockIdx() / AscendC::GetSubBlockNum();
        }

        for (uint32_t taskIdx = coreIdx; /* 通过 if break 出循环 */; taskIdx += coreNum) {
            while (curBatch < batch && taskIdx >= curTotalTaskNum) {
                if (curBatch > 0) {
                    preTotalTaskNum = curTotalTaskNum;
                    qBOffset += qSeqlen * strideQ;
                    if constexpr (!PAGED_CACHE_FLAG_) {
                        kBOffset += static_cast<uint64_t>(kvSeqlen * strideK);
                        vBOffset += static_cast<uint64_t>(kvSeqlen * strideV);
                    } else {
                        blockBOffset += static_cast<uint64_t>(maxNumBlocksPerBatch);
                    }
                    pBOffset += static_cast<uint64_t>(qHeads * strideP);
                    oBOffset += static_cast<uint64_t>(qSeqlen * strideO);
                    // LSE 输出是 N 主序 (BSND -> BNS，TND -> NT)
                    if constexpr (INPUT_LAYOUT_ == InputLayout::TND) {
                        lseBOffset += static_cast<uint64_t>(qSeqlen);
                    } else {
                        lseBOffset += static_cast<uint64_t>(qSeqlen * qHeads);
                    }
                }

                qSeqlen = maxQSeqlen;
                kvSeqlen = static_cast<uint32_t>(gActualKvseqlen.GetValue(curBatch));
                if constexpr (INPUT_LAYOUT_ == InputLayout::TND) {
                    qSeqlen = static_cast<uint32_t>(
                        gActualQseqlen.GetValue(curBatch + 1) - gActualQseqlen.GetValue(curBatch));
                    if constexpr (!PAGED_CACHE_FLAG_) {
                        kvSeqlen = static_cast<uint32_t>(
                            gActualKvseqlen.GetValue(curBatch + 1) - gActualKvseqlen.GetValue(curBatch));
                    }
                }
                curQNBlockTile = GetQNBlockTile(qSeqlen, groupSize);     // Q 的 N 维分块大小
                qNBlockNumPerGroup = CeilDiv(groupSize, curQNBlockTile); // 每个 group 的 N 维分块数量，向上取整
                curQNBlockNum = qNBlockNumPerGroup * kvHeads;            // Q 的 N 维分块数量
                curQSBlockTile = GetQSBlockTile(kvSeqlen);               // Q 的 S 维分块大小
                curQSBlockNum = CeilDiv(qSeqlen, curQSBlockTile);        // Q 的 S 维分块数量，向上取整
                curTotalTaskNum += curQNBlockNum * curQSBlockNum;        // 新增当前 batch 的 Q 任务数量
                curBatch++;
            }

            if (curBatch >= batch &&          // 到达最后一个 batch，此时 curTotalTaskNum 即为 totalTaskNum
                taskIdx >= curTotalTaskNum) { // taskIdx 超过 totalTaskNum，退出 taskIdx 的 for 循环
                break;
            }

            uint32_t taskIdxCurBatch = taskIdx - preTotalTaskNum;
            uint32_t qSBlockIdx = taskIdxCurBatch / curQNBlockNum;         // task 的 S 维分块索引
            uint32_t qNBlockIdx = taskIdxCurBatch % curQNBlockNum;         // task 的 N 维分块索引
            uint32_t qNBlockIdxCurGroup = qNBlockIdx % qNBlockNumPerGroup; // Q 的 group 内 N 维分块索引
            uint32_t kvNIdx = qNBlockIdx / qNBlockNumPerGroup;             // Q 对应 KV 的 N 维分块索引
            uint32_t qNStartIdx = kvNIdx * groupSize + qNBlockIdxCurGroup * curQNBlockTile; // Q 的 N 维元素起始索引

            uint64_t gmOffsetQ = qBOffset +                                                     // B 偏移
                                 static_cast<uint64_t>(qSBlockIdx * curQSBlockTile) * strideQ + // S 偏移
                                 static_cast<uint64_t>(qNStartIdx * embed);                     // N 偏移

            uint64_t gmOffsetK = kBOffset +                             // B 偏移
                                 static_cast<uint64_t>(kvNIdx * embed); // N 偏移，在后面遍历 S 维

            uint64_t gmOffsetP =
                pBOffset +                                                        // B 偏移
                static_cast<uint64_t>(qNStartIdx) * strideP +                     // N 偏移
                static_cast<uint64_t>(qSBlockIdx * curQSBlockTile * maxKvSeqlen); // S(Q) 偏移，在后面遍历 S(K) 维

            uint64_t gmOffsetV = vBOffset +                              // B 偏移
                                 static_cast<uint64_t>(kvNIdx * embedV); // N 偏移，在后面遍历 S 维

            uint64_t gmOffsetO = oBOffset +                                                     // B 偏移
                                 static_cast<uint64_t>(qSBlockIdx * curQSBlockTile) * strideO + // S 偏移
                                 static_cast<uint64_t>(qNStartIdx * embedV);                    // N 偏移

            uint64_t gmOffsetLse = lseBOffset +                                        // B 偏移
                                   static_cast<uint64_t>(qNStartIdx * totalQTokens) +  // N 偏移
                                   static_cast<uint64_t>(qSBlockIdx * curQSBlockTile); // S 偏移

            uint32_t qSBlockSize = (qSBlockIdx == curQSBlockNum - 1) // 处理尾块
                                       ? (qSeqlen - qSBlockIdx * curQSBlockTile)
                                       : curQSBlockTile;
            uint32_t qNBlockSize = (qNBlockIdxCurGroup == qNBlockNumPerGroup - 1)
                                       ? (groupSize - qNBlockIdxCurGroup * curQNBlockTile)
                                       : curQNBlockTile;
            uint32_t rowNum = qSBlockSize * qNBlockSize; // task 的 Q 行数

            uint32_t noSkipKvS = kvSeqlen; // task 计算的 KV 序列长度
            if (maskType != 0) {           // 因果掩码，Q 只能看到之前的 KV
                uint32_t diffS = kvSeqlen > qSeqlen ? kvSeqlen - qSeqlen : 0;
                noSkipKvS = (qSBlockIdx + 1) * curQSBlockTile + // 当前分块最后一个 Q 的 S 维索引
                            diffS;                              // Q 与 KV 的 S 维差值
                noSkipKvS = Min(kvSeqlen, noSkipKvS);           // 不超过实际的 KV 序列长度
            }
            uint32_t kvSLoopNumTotal = CeilDiv(noSkipKvS, MAX_KV_STACK_LEN); // task 的 KV 循环分块数目
            uint32_t stackSeqTile = MAX_KV_STACK_LEN;                        // 每次循环处理的 KV 序列长度

            if ASCEND_IS_AIC {
                layoutQTmp = LayoutQ(rowNum, embed);         // (S * N, D)
                layoutKTmp = LayoutK(strideK, stackSeqTile); // (N * D, S)
                layoutVTmp = LayoutV(stackSeqTile, strideV); // (S, N * D)
                blockMmadQK.loadQGM(gQ[gmOffsetQ], layoutQTmp, qSBlockSize, qNBlockSize, qHeads);
            }

            for (uint32_t kvSIdx = 0; kvSIdx < kvSLoopNumTotal + PRE_LAUNCH; kvSIdx++) {
                if (kvSIdx < kvSLoopNumTotal) {
                    if (kvSIdx == kvSLoopNumTotal - 1) {
                        stackSeqTile = noSkipKvS - kvSIdx * MAX_KV_STACK_LEN;
                    } else {
                        stackSeqTile = MAX_KV_STACK_LEN;
                    }
                    uint32_t curStackTileMod = kvSIdx % (PRE_LAUNCH + 1); // 环形复用 GM 空间
                    uint64_t gmOffsetSTmp =
                        static_cast<uint64_t>((coreIdx * (PRE_LAUNCH + 1) + curStackTileMod) * WORKSPACE_BLOCK_SIZE_DB);
                    GemmCoord actualBlockShapeQK{rowNum, stackSeqTile, embed};
                    LayoutS layoutSTmp(rowNum, stackSeqTile, MAX_KV_STACK_LEN);

                    if ASCEND_IS_AIC {
                        if constexpr (PAGED_CACHE_FLAG_) {
                            blockMmadQK(
                                gK[gmOffsetK], gSTmp[gmOffsetSTmp], gBlockTable[blockBOffset], layoutKTmp, layoutSTmp,
                                actualBlockShapeQK, kvSIdx, pagedBlockSize, strideK);
                        } else {
                            blockMmadQK(
                                gK[gmOffsetK], gSTmp[gmOffsetSTmp], gBlockTable, layoutKTmp, layoutSTmp,
                                actualBlockShapeQK, kvSIdx, pagedBlockSize, strideK);
                        }
                        Arch::CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(qkReady);
                    } else { // ASCEND_IS_AIV
                        LayoutP layoutPTmp(rowNum, stackSeqTile, MAX_KV_STACK_LEN);
                        LayoutMask layoutMask(COMP_TRIU_MASK_DIM_LEN, COMP_TRIU_MASK_DIM_LEN);
                        uint64_t gmOffsetPTmp = gmOffsetSTmp;
                        uint32_t kvSStartIdx = kvSIdx * MAX_KV_STACK_LEN;
                        uint32_t kvSEndIdx = kvSStartIdx + stackSeqTile;
                        if constexpr (MASK_TYPE_ == MaskType::MASK_CAUSAL) {
                            uint32_t triUp = noSkipKvS - qSBlockSize; // 第一个需要部分掩码的 KV 索引
                            uint32_t triDown = noSkipKvS;             // 第一个完全被掩码的 KV 索引
                            if (triUp < kvSEndIdx - 1) {
                                // KV 分块内需要掩码
                                epilogueOnlineSoftmax(
                                    gPTmp[gmOffsetPTmp], gSTmp[gmOffsetSTmp], gMask, gP, gmOffsetP + kvSStartIdx,
                                    layoutPTmp, layoutSTmp, layoutMask, actualBlockShapeQK, (kvSIdx == 0), qSBlockSize,
                                    qNBlockSize, curStackTileMod, qkReady, triUp, triDown, kvSStartIdx, kvSEndIdx,
                                    maxKvSeqlen, strideP);
                            } else {
                                // KV 分块内无需掩码
                                uint32_t noMaskStackSeqNum = (triUp + 1) / MAX_KV_STACK_LEN;
                                Arch::CrossCoreWaitFlag(qkReady);
                                epilogueOnlineSoftmax(
                                    gPTmp[gmOffsetPTmp], gSTmp[gmOffsetSTmp], gP, gmOffsetP + kvSStartIdx, layoutPTmp,
                                    layoutSTmp, actualBlockShapeQK, (kvSIdx == 0), (kvSIdx == noMaskStackSeqNum - 1),
                                    curStackTileMod, qSBlockSize, maxKvSeqlen, strideP);
                            }
                        } else {
                            Arch::CrossCoreWaitFlag(qkReady);
                            epilogueOnlineSoftmax(
                                gPTmp[gmOffsetPTmp], gSTmp[gmOffsetSTmp], gP, gmOffsetP + kvSStartIdx, layoutPTmp,
                                layoutSTmp, actualBlockShapeQK, (kvSIdx == 0), 0, curStackTileMod, qSBlockSize,
                                maxKvSeqlen, strideP);
                        }
                        Arch::CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(softmaxReady);
                    }
                }

                if (kvSIdx >= PRE_LAUNCH) {
                    uint32_t nowkvSIdx = kvSIdx - PRE_LAUNCH;
                    if (nowkvSIdx == kvSLoopNumTotal - 1) {
                        stackSeqTile = noSkipKvS - nowkvSIdx * MAX_KV_STACK_LEN;
                    } else {
                        stackSeqTile = MAX_KV_STACK_LEN;
                    }
                    uint32_t curStackTileMod = nowkvSIdx % (PRE_LAUNCH + 1);
                    uint64_t gmOffsetOTmp =
                        static_cast<uint64_t>((coreIdx * (PRE_LAUNCH + 1) + curStackTileMod) * WORKSPACE_BLOCK_SIZE_DB);
                    GemmCoord actualBlockShapePV{rowNum, embedV, stackSeqTile};
                    LayoutOTmp layoutOTmp(rowNum, embedV, embedRoundV);

                    if ASCEND_IS_AIC {
                        LayoutP layoutPTmp(rowNum, stackSeqTile, MAX_KV_STACK_LEN);
                        uint64_t gmOffsetPTmp = gmOffsetOTmp;
                        if constexpr (PAGED_CACHE_FLAG_) {
                            blockMmadPV(
                                gPTmp[gmOffsetPTmp], gV[gmOffsetV], gOTmp[gmOffsetOTmp], gBlockTable[blockBOffset],
                                layoutPTmp, layoutVTmp, layoutOTmp, actualBlockShapePV, nowkvSIdx, pagedBlockSize,
                                strideV, softmaxReady);
                        } else {
                            blockMmadPV(
                                gPTmp[gmOffsetPTmp], gV[gmOffsetV], gOTmp[gmOffsetOTmp], gBlockTable, layoutPTmp,
                                layoutVTmp, layoutOTmp, actualBlockShapePV, nowkvSIdx, pagedBlockSize, strideV,
                                softmaxReady);
                        }
                        Arch::CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(pvReady);
                    } else { // ASCEND_IS_AIV
                        LayoutO layoutO(qSeqlen, qHeads * embed);
                        LayoutUpdate layoutUpdate(rowNum, embed, embedRound);
                        LayoutLse layoutLse(qHeads, totalQTokens);
                        uint64_t gmOffsetUpdate = static_cast<uint64_t>(coreIdx * WORKSPACE_BLOCK_SIZE_DB);

                        Arch::CrossCoreWaitFlag(pvReady);
                        epilogueRescaleO(
                            gO[gmOffsetO], gOTmp[gmOffsetOTmp], gOUpdate[gmOffsetUpdate], gLse[gmOffsetLse], layoutO,
                            layoutOTmp, layoutUpdate, layoutLse, actualBlockShapePV, qSBlockSize, qNBlockSize,
                            (nowkvSIdx == 0), (nowkvSIdx == kvSLoopNumTotal - 1), curStackTileMod);
                    }
                }
            }
        }

        if ASCEND_IS_AIC {
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::M_MTE1>(EVENT_ID7);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::FIX_M>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::MTE1_MTE2>(EVENT_ID7);
        } else {
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID3);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID4);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID5);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID6);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID1);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(EVENT_ID3);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

private:
    Arch::Resource<ArchTag> resource;
    uint32_t coreIdx;

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

template <
    typename INPUT_DTYPE_, bool PAGED_CACHE_FLAG_, MaskType MASK_TYPE_, InputLayout INPUT_LAYOUT_, bool RETURN_SOFTMAX_>
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

    using DispatchPolicyQK = Gemm::MmadAtlasA2FAIQKT<PAGED_CACHE_FLAG_, false>;
    using L1TileShapeQK = GemmShape<Q_TILE_CEIL, 128, 128>;
    using L0TileShapeQK = GemmShape<128, 128, 128>;
    using QType = Gemm::GemmType<ElementQ, LayoutQ>;
    using KType = Gemm::GemmType<ElementK, LayoutK>;
    using SType = Gemm::GemmType<ElementS, LayoutS>;
    using BlockMmadQK = Gemm::Block::BlockMmad<DispatchPolicyQK, L1TileShapeQK, L0TileShapeQK, QType, KType, SType>;

    using DispatchPolicyOnlineSoftmax = Epilogue::EpilogueAtlasA2OnlineSoftmaxT<RETURN_SOFTMAX_>;
    using PType = Gemm::GemmType<ElementP, LayoutP>;
    using maskType = Gemm::GemmType<ElementMask, LayoutMask>;
    using EpilogueOnlineSoftmax = Epilogue::Block::BlockEpilogue<DispatchPolicyOnlineSoftmax, PType, SType, maskType>;

    using DispatchPolicyPV = Gemm::MmadAtlasA2FAIPVT<PAGED_CACHE_FLAG_, false>;
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