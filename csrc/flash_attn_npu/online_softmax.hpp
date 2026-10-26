/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_NO_MASK_HPP_T
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_ONLINE_SOFTMAX_NO_MASK_HPP_T

#include "catlass/catlass.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_copy.hpp"
#include "catlass/gemm_coord.hpp"

#include "fa_block.hpp"
#include "kernel_common.hpp"

using namespace AscendC;
using namespace KernelCommon;

namespace Catlass::Epilogue::Block {
template <class OutputType_, class InputType_, class MaskType_, MaskModeT MASK_MODE_>
class BlockEpilogue<EpilogueAtlasA2OnlineSoftmaxT<MASK_MODE_>, OutputType_, InputType_, MaskType_> {
public:
    using ArchTag = Arch::AtlasA2;

    using ElementOutput = typename OutputType_::Element;
    using ElementInput = typename InputType_::Element;
    using ElementMask = typename MaskType_::Element;

    using LayoutOutput = typename OutputType_::Layout;
    using LayoutInput = typename InputType_::Layout;
    using LayoutMask = typename MaskType_::Layout;

    CATLASS_DEVICE void init(const Arch::Resource<ArchTag>& resource, float scaleVal_)
    {
        scaleVal = scaleVal_;

        constexpr uint32_t LS_OFFSET = 0;
        constexpr uint32_t LP_OFFSET = 4 * UB_BLK_SIZE;
        constexpr uint32_t MASK_OFFSET = 6 * UB_BLK_SIZE;
        constexpr uint32_t MASK16_OFFSET = 7 * UB_BLK_SIZE;

        constexpr uint32_t GM_OFFSET = 10 * UB_BLK_SIZE;
        constexpr uint32_t GL_OFFSET = 10 * UB_BLK_SIZE + 4 * UB_RPT_SIZE;

        constexpr uint32_t LM_OFFSET = 11 * UB_BLK_SIZE;
        constexpr uint32_t HM_OFFSET = 11 * UB_BLK_SIZE + 2 * UB_RPT_SIZE;
        constexpr uint32_t DM_OFFSET = 11 * UB_BLK_SIZE + 4 * UB_RPT_SIZE;
        constexpr uint32_t LL_OFFSET = 11 * UB_BLK_SIZE + 8 * UB_RPT_SIZE;
        constexpr uint32_t TB_OFFSET = 11 * UB_BLK_SIZE + 11 * UB_VEC_SIZE + 2 * UB_RPT_SIZE;
        constexpr uint32_t TV_OFFSET = 11 * UB_BLK_SIZE + 12 * UB_VEC_SIZE;

        ls = resource.ubBuf.template GetBufferByByte<float>(LS_OFFSET);
        lp = resource.ubBuf.template GetBufferByByte<ElementOutput>(LP_OFFSET);
        mask = resource.ubBuf.template GetBufferByByte<ElementMask>(MASK_OFFSET);
        mask32 = resource.ubBuf.template GetBufferByByte<float>(MASK_OFFSET);
        mask16 = resource.ubBuf.template GetBufferByByte<half>(MASK16_OFFSET);
        gm = resource.ubBuf.template GetBufferByByte<float>(GM_OFFSET);
        gl = resource.ubBuf.template GetBufferByByte<float>(GL_OFFSET);
        lm = resource.ubBuf.template GetBufferByByte<float>(LM_OFFSET);
        hm = resource.ubBuf.template GetBufferByByte<float>(HM_OFFSET);
        dm = resource.ubBuf.template GetBufferByByte<float>(DM_OFFSET);
        ll = resource.ubBuf.template GetBufferByByte<float>(LL_OFFSET);
        tb = resource.ubBuf.template GetBufferByByte<float>(TB_OFFSET);
        tv = resource.ubBuf.template GetBufferByByte<float>(TV_OFFSET);
    }

    CATLASS_DEVICE void RowMaxFull(const LocalTensor<float>& dst, const LocalTensor<float>& src, uint32_t rowNum)
    {
        BlockReduceMax<float, false>(tv, src, FLT_PER_DATABLOCK * rowNum, 0, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        BlockReduceMax<float, false>(tb, tv, rowNum, 0, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        BlockReduceMax<float, false>(dst, tb, CeilDiv<FLT_PER_DATABLOCK>(rowNum), 0, 1, 1, 8);
        PipeBarrier<PIPE_V>();
    }

    CATLASS_DEVICE void RowMaxNorm(
        const LocalTensor<float>& dst, const LocalTensor<float>& src, uint32_t rowNum, uint32_t colNum)
    {
        uint32_t colRep = colNum / FLT_PER_REPEAT;
        BlockReduceMax<float, false>(tv, src, colRep * rowNum, 0, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        SetVecMaskLen(FLT_PER_DATABLOCK * colRep);
        BlockReduceMax<float, false>(tb, tv, rowNum, 0, 1, 1, colRep);
        PipeBarrier<PIPE_V>();
        SetBlockReduceMaskLen(colRep);
        BlockReduceMax<float, false>(dst, tb, CeilDiv<FLT_PER_DATABLOCK>(rowNum), 0, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        ResetMask();
    }

    CATLASS_DEVICE void RowMaxTail(
        const LocalTensor<float>& dst, const LocalTensor<float>& src, uint32_t rowNum, uint32_t colNum,
        uint32_t colNumRound)
    {
        if (colNum >= FLT_PER_REPEAT) {
            BlockReduceMax<float, false>(tv, src, rowNum, 0, 1, 1, colNumRound / FLT_PER_DATABLOCK);
            PipeBarrier<PIPE_V>();
            BlockReduceMax<float, false>(dst, tv, CeilDiv<FLT_PER_DATABLOCK>(rowNum), 0, 1, 1, 8);
            PipeBarrier<PIPE_V>();

            for (uint32_t idx = FLT_PER_REPEAT; idx < RoundDown<FLT_PER_REPEAT>(colNum); idx += FLT_PER_REPEAT) {
                BlockReduceMax<float, false>(tv, src[idx], rowNum, 0, 1, 1, colNumRound / FLT_PER_DATABLOCK);
                PipeBarrier<PIPE_V>();
                BlockReduceMax<float, false>(tb, tv, CeilDiv<FLT_PER_DATABLOCK>(rowNum), 0, 1, 1, 8);
                PipeBarrier<PIPE_V>();
                SetVecMaskLen(rowNum);
                Max<float, false>(dst, dst, tb, MASK_NUL, 1, BinaryRepeatParams());
                PipeBarrier<PIPE_V>();
                ResetMask();
            }
        }

        if (colNum % FLT_PER_REPEAT > 0) {
            SetVecMaskLen(colNum % FLT_PER_REPEAT);
            BlockReduceMax<float, false>(
                tv, src[RoundDown<FLT_PER_REPEAT>(colNum)], rowNum, 0, 1, 1, colNumRound / FLT_PER_DATABLOCK);
            PipeBarrier<PIPE_V>();

            SetBlockReduceMaskLen(CeilDiv<FLT_PER_DATABLOCK>(colNum % FLT_PER_REPEAT));
            if (colNum < FLT_PER_REPEAT) {
                BlockReduceMax<float, false>(dst, tv, CeilDiv<FLT_PER_DATABLOCK>(rowNum), 0, 1, 1, 8);
                PipeBarrier<PIPE_V>();
            } else {
                BlockReduceMax<float, false>(tb, tv, CeilDiv<FLT_PER_DATABLOCK>(rowNum), 0, 1, 1, 8);
                PipeBarrier<PIPE_V>();
                SetVecMaskLen(rowNum);
                Max<float, false>(dst, dst, tb, MASK_NUL, 1, BinaryRepeatParams());
                PipeBarrier<PIPE_V>();
            }
            ResetMask();
        }
    }

    CATLASS_DEVICE void CalcLocalRowMax(uint32_t rowOffset, uint32_t rowNum, uint32_t colNum, uint32_t colNumRound)
    {
        if (colNum == FLT_PER_REPEAT * FLT_PER_DATABLOCK) {
            RowMaxFull(lm[rowOffset], ls[pingPongOffset], rowNum);
        } else if (colNum > FLT_PER_REPEAT && colNum % FLT_PER_REPEAT == 0) {
            RowMaxNorm(lm[rowOffset], ls[pingPongOffset], rowNum, colNum);
        } else {
            RowMaxTail(lm[rowOffset], ls[pingPongOffset], rowNum, colNum, colNumRound);
        }
    }

    CATLASS_DEVICE void UpdateGlobalRowMax(
        uint32_t rowNum, uint32_t dmOffset, uint32_t rowOffset, uint32_t isFirstStackTile)
    {
        if (isFirstStackTile) {
            DataCopy(hm[rowOffset], lm[rowOffset], DataCopyParams(1, CeilDiv<FLT_PER_DATABLOCK>(rowNum), 0, 0));
            PipeBarrier<PIPE_V>();
        } else {
            SetVecMaskLen(rowNum);
            // hm = max(lm, gm)
            Max<float, false>(
                hm[rowOffset], lm[rowOffset], gm[taskOffset][rowOffset], MASK_NUL, 1, BinaryRepeatParams());
            PipeBarrier<PIPE_V>();
            // dm = gm - hm
            Sub<float, false>(
                dm[dmOffset], gm[taskOffset][rowOffset], hm[rowOffset], MASK_NUL, 1, BinaryRepeatParams());
            PipeBarrier<PIPE_V>();
            // dm = exp(dm)
            Exp<float, false>(dm[dmOffset], dm[dmOffset], MASK_NUL, 1, UnaryRepeatParams());
            PipeBarrier<PIPE_V>();
            ResetMask();
        }

        // gm = hm
        DataCopy(gm[taskOffset][rowOffset], hm[rowOffset],
                 DataCopyParams(1, CeilDiv<FLT_PER_DATABLOCK>(rowNum), 0, 0));
        PipeBarrier<PIPE_V>();
    }

    CATLASS_DEVICE void RowSumFull(const LocalTensor<float>& dst, const LocalTensor<float>& src, uint32_t rowNum)
    {
        BlockReduceSum<float, false>(tv, src, FLT_PER_DATABLOCK * rowNum, 0, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        BlockReduceSum<float, false>(tb, tv, rowNum, 0, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        BlockReduceSum<float, false>(dst, tb, CeilDiv<FLT_PER_DATABLOCK>(rowNum), 0, 1, 1, 8);
        PipeBarrier<PIPE_V>();
    }

    CATLASS_DEVICE void RowSumNorm(
        const LocalTensor<float>& dst, const LocalTensor<float>& src, uint32_t rowNum, uint32_t colNum)
    {
        uint32_t colRep = colNum / FLT_PER_REPEAT;
        BlockReduceSum<float, false>(tv, src, colRep * rowNum, 0, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        SetVecMaskLen(FLT_PER_DATABLOCK * colRep);
        BlockReduceSum<float, false>(tb, tv, rowNum, 0, 1, 1, colRep);
        PipeBarrier<PIPE_V>();
        SetBlockReduceMaskLen(colRep);
        BlockReduceSum<float, false>(dst, tb, CeilDiv<FLT_PER_DATABLOCK>(rowNum), 0, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        ResetMask();
    }

    CATLASS_DEVICE void RowSumTail(
        const LocalTensor<float>& dst, const LocalTensor<float>& src, uint32_t rowNum, uint32_t colNum,
        uint32_t colNumRound)
    {
        if (colNum >= FLT_PER_REPEAT) {
            BlockReduceSum<float, false>(tv, src, rowNum, 0, 1, 1, colNumRound / FLT_PER_DATABLOCK);
            PipeBarrier<PIPE_V>();
            BlockReduceSum<float, false>(dst, tv, CeilDiv<FLT_PER_DATABLOCK>(rowNum), 0, 1, 1, 8);
            PipeBarrier<PIPE_V>();

            for (uint32_t idx = FLT_PER_REPEAT; idx < RoundDown<FLT_PER_REPEAT>(colNum); idx += FLT_PER_REPEAT) {
                BlockReduceSum<float, false>(tv, src[idx], rowNum, 0, 1, 1, colNumRound / FLT_PER_DATABLOCK);
                PipeBarrier<PIPE_V>();
                BlockReduceSum<float, false>(tb, tv, CeilDiv<FLT_PER_DATABLOCK>(rowNum), 0, 1, 1, 8);
                PipeBarrier<PIPE_V>();
                SetVecMaskLen(rowNum);
                Add<float, false>(dst, dst, tb, MASK_NUL, 1, BinaryRepeatParams());
                PipeBarrier<PIPE_V>();
                ResetMask();
            }
        }

        if (colNum % FLT_PER_REPEAT > 0) {
            SetVecMaskLen(colNum % FLT_PER_REPEAT);
            BlockReduceSum<float, false>(
                tv, src[RoundDown<FLT_PER_REPEAT>(colNum)], rowNum, 0, 1, 1, colNumRound / FLT_PER_DATABLOCK);
            PipeBarrier<PIPE_V>();

            SetBlockReduceMaskLen(CeilDiv<FLT_PER_DATABLOCK>(colNum % FLT_PER_REPEAT));
            if (colNum < FLT_PER_REPEAT) {
                BlockReduceSum<float, false>(dst, tv, CeilDiv<FLT_PER_DATABLOCK>(rowNum), 0, 1, 1, 8);
                PipeBarrier<PIPE_V>();
            } else {
                BlockReduceSum<float, false>(tb, tv, CeilDiv<FLT_PER_DATABLOCK>(rowNum), 0, 1, 1, 8);
                PipeBarrier<PIPE_V>();
                SetVecMaskLen(rowNum);
                Add<float, false>(dst, dst, tb, MASK_NUL, 1, BinaryRepeatParams());
                PipeBarrier<PIPE_V>();
            }
            ResetMask();
        }
    }

    CATLASS_DEVICE void CalcLocalRowSum(uint32_t rowOffset, uint32_t rowNum, uint32_t colNum, uint32_t colNumRound)
    {
        if (colNum == FLT_PER_REPEAT * FLT_PER_DATABLOCK) {
            RowSumFull(ll[rowOffset], ls[pingPongOffset], rowNum);
        } else if (colNum > FLT_PER_REPEAT && colNum % FLT_PER_REPEAT == 0) {
            RowSumNorm(ll[rowOffset], ls[pingPongOffset], rowNum, colNum);
        } else {
            RowSumTail(ll[rowOffset], ls[pingPongOffset], rowNum, colNum, colNumRound);
        }
    }

    CATLASS_DEVICE void UpdateGlobalRowSum(
        uint32_t rowNum, uint32_t dmOffset, uint32_t rowOffset, uint32_t isFirstStackTile)
    {
        if (isFirstStackTile) {
            // gl = ll
            DataCopy(
                gl[taskOffset][rowOffset], ll[rowOffset], DataCopyParams(1, CeilDiv<FLT_PER_DATABLOCK>(rowNum), 0, 0));
            PipeBarrier<PIPE_V>();
        } else {
            SetVecMaskLen(rowNum);
            // gl = dm * gl + ll
            FusedMulAdd<float, false>(
                gl[taskOffset][rowOffset], dm[dmOffset], ll[rowOffset], MASK_NUL, 1, BinaryRepeatParams());
            PipeBarrier<PIPE_V>();
            ResetMask();
        }
    }

    CATLASS_DEVICE void CopyMaskToUb(
        const GlobalTensor<ElementMask>& gMask, uint32_t rowNum, uint32_t maskColNum, uint32_t maskStride,
        uint32_t rowOffset, uint32_t qSOffset, uint32_t qSBlockSize)
    {
        uint32_t qSBlockIdx = (qSOffset + rowOffset) % qSBlockSize;
        uint32_t rowsDone = 0;
        while (rowsDone < rowNum) {
            uint32_t rowsCurHead = Min(rowNum - rowsDone, qSBlockSize - qSBlockIdx);
            DataCopy(
                mask[rowsDone * maskColNum], gMask[qSBlockIdx * maskStride],
                DataCopyParams(rowsCurHead, maskColNum / NUM_32, (maskStride - maskColNum) / NUM_32, 0));
            rowsDone += rowsCurHead;
            qSBlockIdx = 0;
        }
    }

    CATLASS_DEVICE void ApplyMask(uint32_t rowNum, uint32_t colNumRound, uint32_t maskColNum, uint32_t maskColOffset)
    {
        Cast<half, ElementMask, false>(
            mask16, mask, RoundMode::CAST_NONE, MASK_NUL, CeilDiv<HALF_PER_REPEAT>(rowNum * colNumRound),
            UnaryRepeatParams(1, 1, 8, 4));
        PipeBarrier<PIPE_V>();

        Cast<float, half, false>(
            mask32, mask16, RoundMode::CAST_NONE, MASK_NUL, CeilDiv<FLT_PER_REPEAT>(rowNum * colNumRound),
            UnaryRepeatParams(1, 1, 8, 4));
        PipeBarrier<PIPE_V>();

        if (maskColNum == colNumRound) {
            Axpy<float, float, false>(
                ls[pingPongOffset], mask32, -3e38f, MASK_NUL, CeilDiv<FLT_PER_REPEAT>(rowNum * colNumRound),
                UnaryRepeatParams());
        } else {
            for (uint32_t idx = 0; idx < RoundDown<FLT_PER_REPEAT>(maskColNum); idx += FLT_PER_REPEAT) {
                Axpy<float, float, false>(
                    ls[pingPongOffset][maskColOffset + idx], mask32[idx], -3e38f, MASK_NUL, rowNum,
                    UnaryRepeatParams(1, 1, colNumRound / FLT_PER_DATABLOCK, maskColNum / FLT_PER_DATABLOCK));
            }

            if (maskColNum % FLT_PER_REPEAT > 0) {
                SetVecMaskLen(maskColNum % FLT_PER_REPEAT);
                Axpy<float, float, false>(
                    ls[pingPongOffset][maskColOffset + RoundDown<FLT_PER_REPEAT>(maskColNum)],
                    mask32[RoundDown<FLT_PER_REPEAT>(maskColNum)], -3e38f, MASK_NUL, rowNum,
                    UnaryRepeatParams(1, 1, colNumRound / FLT_PER_DATABLOCK, maskColNum / FLT_PER_DATABLOCK));
                ResetMask();
            }
        }
        PipeBarrier<PIPE_V>();
    }

    CATLASS_DEVICE void CalcExp(uint32_t rowNum, uint32_t colNum, uint32_t colNumRound, uint32_t rowOffset)
    {
        // hm_block = expand_to_block(hm), 存放于 tv
        Brcb(tv, hm[rowOffset], CeilDiv<FLT_PER_DATABLOCK>(rowNum), BrcbRepeatParams());
        PipeBarrier<PIPE_V>();

        // ls = ls - hm_block
        for (uint32_t idx = 0; idx < RoundDown<FLT_PER_REPEAT>(colNum); idx += FLT_PER_REPEAT) {
            Sub<float, false>(
                ls[pingPongOffset][idx], ls[pingPongOffset][idx], tv, MASK_NUL, rowNum,
                BinaryRepeatParams(1, 1, 0, colNumRound / FLT_PER_DATABLOCK, colNumRound / FLT_PER_DATABLOCK, 1));
        }
        if (colNum % FLT_PER_REPEAT > 0) {
            SetVecMaskLen(colNum % FLT_PER_REPEAT);
            Sub<float, false>(
                ls[pingPongOffset][RoundDown<FLT_PER_REPEAT>(colNum)],
                ls[pingPongOffset][RoundDown<FLT_PER_REPEAT>(colNum)], tv, MASK_NUL, rowNum,
                BinaryRepeatParams(1, 1, 0, colNumRound / FLT_PER_DATABLOCK, colNumRound / FLT_PER_DATABLOCK, 1));
            ResetMask();
        }
        PipeBarrier<PIPE_V>();

        // ls = exp(ls)
        Exp<float, false>(
            ls[pingPongOffset], ls[pingPongOffset], MASK_NUL, CeilDiv<FLT_PER_REPEAT>(rowNum * colNumRound),
            UnaryRepeatParams());
        PipeBarrier<PIPE_V>();
    }

    CATLASS_DEVICE void ScaleS(uint32_t rowNum, uint32_t colNumRound)
    {
        Muls<float, false>(
            ls[pingPongOffset], ls[pingPongOffset], scaleVal, MASK_NUL, CeilDiv<FLT_PER_REPEAT>(rowNum * colNumRound),
            UnaryRepeatParams());
        PipeBarrier<PIPE_V>();
    }

    CATLASS_DEVICE void SubCoreCompute(
        const GlobalTensor<ElementOutput>& gOutput, const LayoutOutput& layoutOutput, uint32_t rowOffset,
        uint32_t isFirstStackTile, uint32_t isFirstRowLoop, uint32_t colNumRound, uint32_t curStackTileMod,
        uint32_t rowBase)
    {
        uint32_t rowNum = layoutOutput.shape(0);
        uint32_t colNum = layoutOutput.shape(1);
        uint32_t colStride = layoutOutput.stride(0);
        uint32_t dmOffset = curStackTileMod * MAX_ROW_NUM_SUB_CORE + rowBase + rowOffset;

        CalcLocalRowMax(rowOffset, rowNum, colNum, colNumRound);
        UpdateGlobalRowMax(rowNum, dmOffset, rowOffset, isFirstStackTile);

        CalcExp(rowNum, colNum, colNumRound, rowOffset);

        // lp = castfp32to16(ls)
        RoundMode roundMode = std::is_same_v<ElementOutput, bfloat16_t> ? RoundMode::CAST_RINT : RoundMode::CAST_NONE;
        WaitFlag<HardEvent::MTE3_V>(pingPong);
        Cast<ElementOutput, float, false>(
            lp[pingPongOffset], ls[pingPongOffset], roundMode, MASK_NUL, CeilDiv<FLT_PER_REPEAT>(rowNum * colNumRound),
            UnaryRepeatParams(1, 1, 4, 8));
        SetFlag<HardEvent::V_MTE3>(pingPong);

        CalcLocalRowSum(rowOffset, rowNum, colNum, colNumRound);
        SetFlag<HardEvent::V_MTE2>(pingPong);

        WaitFlag<HardEvent::V_MTE3>(pingPong);
        DataCopy(
            gOutput, lp[pingPongOffset],
            DataCopyParams(
                rowNum, colNumRound / HALF_PER_DATABLOCK, 0, (colStride - colNumRound) / HALF_PER_DATABLOCK));
        SetFlag<HardEvent::MTE3_V>(pingPong);

        UpdateGlobalRowSum(rowNum, dmOffset, rowOffset, isFirstStackTile);
    }

    CATLASS_DEVICE void operator()(
        const GlobalTensor<ElementOutput>& gOutput, const GlobalTensor<ElementInput>& gInput,
        const GlobalTensor<ElementMask>& gMask, const LayoutOutput& layoutOutput, const LayoutInput& layoutInput,
        const LayoutInput& layoutMask, GemmCoord actualBlockShape, uint32_t isFirstStackTile, uint32_t curStackTileMod,
        uint32_t curTaskMod, Arch::CrossCoreFlag qkReady, uint32_t qSBlockSize, uint32_t triUp, uint32_t kvSStartIdx,
        uint32_t kvSEndIdx, uint32_t rowBase = 0, bool isFirstHead = true)
    {
        uint32_t rowNum = actualBlockShape.m();
        uint32_t colNum = actualBlockShape.n();
        uint32_t colNumRound = RoundUp<NUM_32>(colNum);
        uint32_t colStride = layoutInput.stride(0);

        uint32_t core1 = GetSubBlockIdx();
        uint32_t rowSplit = rowNum / NUM_2;
        uint32_t coreRow = core1 ? (rowNum - rowSplit) : rowSplit;

        if (coreRow == 0) {
            if (isFirstHead) {
                Arch::CrossCoreWaitFlag(qkReady);
            }
            return;
        }

        uint32_t coreRowOffset = core1 ? rowSplit : 0;
        uint32_t rowNumTile = RoundDown<FLT_PER_DATABLOCK>(MAX_UB_ELEM_NUM / colNumRound);
        rowNumTile = Min(rowNumTile, FLT_PER_REPEAT);
        uint32_t rowLoop = CeilDiv(coreRow, rowNumTile);

        // *** CAUSAL MASK ***
        bool applyMask = false;
        uint32_t coreQSOffset, gmMaskRow, gmMaskCol;
        uint32_t maskColOffset, maskColNum, maskStride;

        if constexpr (MASK_MODE_ == MaskModeT::MASK_CAUSAL) {
            applyMask = (triUp < kvSEndIdx - 1);
            if (applyMask) { // Q 分块首行可访问的最大 KV 下标 < KV 分块的结束下标
                coreQSOffset = core1 ? (rowSplit % qSBlockSize) : 0;
                maskStride = layoutMask.stride(0);

                if (triUp < kvSStartIdx) { // Q 分块首行全量掩码
                    gmMaskRow = 0;
                    gmMaskCol = kvSStartIdx - triUp;
                    maskColOffset = 0;
                    maskColNum = colNumRound;
                } else { // Q 分块首行部分掩码
                    uint32_t triUpRound = RoundDown<NUM_32>(triUp);
                    gmMaskRow = triUp - triUpRound;
                    gmMaskCol = 0;
                    maskColOffset = triUpRound - kvSStartIdx;
                    maskColNum = colNumRound - maskColOffset;
                }

                WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
                CopyMaskToUb(
                    gMask[layoutMask.GetOffset(MatrixCoord(gmMaskRow, gmMaskCol))], Min(rowNumTile, coreRow),
                    maskColNum, maskStride, 0, coreQSOffset, qSBlockSize);
                SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
            }
        }

        taskOffset = curTaskMod * MAX_ROW_NUM_SUB_CORE + rowBase;
        if (isFirstHead) {
            Arch::CrossCoreWaitFlag(qkReady);
        }

        for (uint32_t rowIdx = 0; rowIdx < rowLoop + PRE_LAUNCH; rowIdx++) {
            uint32_t rowOffset = rowIdx * rowNumTile;
            uint32_t rowOffsetActual = coreRowOffset + rowOffset;
            uint32_t rowNumCurLoop = (rowIdx == rowLoop - 1) ? (coreRow - rowOffset) : rowNumTile;

            if (rowIdx < rowLoop) {
                WaitFlag<HardEvent::V_MTE2>(pingPong);
                DataCopy(
                    ls[pingPongOffset], gInput[layoutInput.GetOffset(MatrixCoord(rowOffsetActual, 0))],
                    DataCopyParams(
                        rowNumCurLoop, colNumRound / FLT_PER_DATABLOCK, (colStride - colNumRound) / FLT_PER_DATABLOCK,
                        0));
                SetFlag<HardEvent::MTE2_V>(pingPong);
            }

            pingPong ^= 1;
            pingPongOffset = pingPong ? MAX_UB_ELEM_NUM : 0;

            if (rowIdx >= PRE_LAUNCH) {
                uint32_t nowRowIdx = rowIdx - PRE_LAUNCH;
                uint32_t nowRowOffset = nowRowIdx * rowNumTile;
                uint32_t rowNumNowLoop = (nowRowIdx == rowLoop - 1) ? (coreRow - nowRowOffset) : rowNumTile;

                WaitFlag<HardEvent::MTE2_V>(pingPong);
                ScaleS(rowNumNowLoop, colNumRound);

                if constexpr (MASK_MODE_ == MaskModeT::MASK_CAUSAL) {
                    if (applyMask) {
                        WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
                        ApplyMask(rowNumNowLoop, colNumRound, maskColNum, maskColOffset);
                        SetFlag<HardEvent::V_MTE2>(EVENT_ID2);

                        if (rowIdx < rowLoop) {
                            WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
                            CopyMaskToUb(
                                gMask[layoutMask.GetOffset(MatrixCoord(gmMaskRow, gmMaskCol))], rowNumCurLoop,
                                maskColNum, maskStride, rowOffset, coreQSOffset, qSBlockSize);
                            SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
                        }
                    }
                }

                uint32_t rowOffsetActual = coreRowOffset + nowRowOffset;
                SubCoreCompute(
                    gOutput[layoutOutput.GetOffset(MatrixCoord(rowOffsetActual, 0))],
                    layoutOutput.GetTileLayout(MatrixCoord(rowNumNowLoop, colNum)), nowRowOffset, isFirstStackTile,
                    (nowRowIdx == 0), colNumRound, curStackTileMod, rowBase);
            }
        }

        pingPong ^= 1;
        pingPongOffset = pingPong ? MAX_UB_ELEM_NUM : 0;
    }

private:
    float scaleVal;
    uint32_t taskOffset;

    uint32_t pingPong = 0;
    uint32_t pingPongOffset = 0;

    LocalTensor<float> ls;
    LocalTensor<ElementOutput> lp;
    LocalTensor<ElementMask> mask;
    LocalTensor<float> mask32;
    LocalTensor<half> mask16;
    LocalTensor<float> gm;
    LocalTensor<float> gl;
    LocalTensor<float> lm;
    LocalTensor<float> hm;
    LocalTensor<float> dm;
    LocalTensor<float> ll;
    LocalTensor<float> tb;
    LocalTensor<float> tv;
};
} // namespace Catlass::Epilogue::Block

#endif