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
template <class OutputType_, class InputType_, class MaskType_>
class BlockEpilogue<EpilogueAtlasA2OnlineSoftmaxT, OutputType_, InputType_, MaskType_> {
public:
    using ArchTag = Arch::AtlasA2;

    using ElementOutput = typename OutputType_::Element;
    using ElementInput = typename InputType_::Element;
    using ElementMask = typename MaskType_::Element;

    using LayoutOutput = typename OutputType_::Layout;
    using LayoutInput = typename InputType_::Layout;
    using LayoutMask = typename MaskType_::Layout;

    CATLASS_DEVICE BlockEpilogue() = default;

    CATLASS_DEVICE ~BlockEpilogue() = default;

    CATLASS_DEVICE void init(Arch::Resource<ArchTag>& resource, float scaleVal_)
    {
        scaleVal = scaleVal_;

        constexpr uint32_t LS_OFFSET = 0;
        constexpr uint32_t LP_OFFSET = 4 * UB_BLK_SIZE;
        constexpr uint32_t MASK_OFFSET = 4 * UB_BLK_SIZE;
        constexpr uint32_t MASK16_OFFSET = 5 * UB_BLK_SIZE;

        constexpr uint32_t GM_OFFSET = 10 * UB_BLK_SIZE;
        constexpr uint32_t GL_OFFSET = 10 * UB_BLK_SIZE + UB_RPT_SIZE;

        constexpr uint32_t LM_OFFSET = 11 * UB_BLK_SIZE;
        constexpr uint32_t HM_OFFSET = 11 * UB_BLK_SIZE + UB_RPT_SIZE;
        constexpr uint32_t DM_OFFSET = 11 * UB_BLK_SIZE + 2 * UB_RPT_SIZE;
        constexpr uint32_t LL_OFFSET = 11 * UB_BLK_SIZE + UB_VEC_SIZE;
        constexpr uint32_t TB_OFFSET = 11 * UB_BLK_SIZE + 10 * UB_VEC_SIZE;
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

    CATLASS_DEVICE void set_curTaskMod(uint32_t curTaskMod)
    {
        taskEvent = curTaskMod + 2;
        taskOffset = curTaskMod * (2 * UB_RPT_SIZE);
    }

    CATLASS_DEVICE void RowMax512(const LocalTensor<float>& dst, const LocalTensor<float>& src, uint32_t rowNum)
    {
        uint32_t repeatTime = 512 * RoundUp(rowNum, FLT_PER_DATABLOCK) / FLT_PER_REPEAT;
        BlockReduceMax<float, false>(tv, src, repeatTime, 0, 1, 1, 8);
        PipeBarrier<PIPE_V>();

        repeatTime /= FLT_PER_DATABLOCK;
        BlockReduceMax<float, false>(tb, tv, repeatTime, 0, 1, 1, 8);
        PipeBarrier<PIPE_V>();

        repeatTime /= FLT_PER_DATABLOCK;
        BlockReduceMax<float, false>(dst, tb, repeatTime, 0, 1, 1, 8);
        PipeBarrier<PIPE_V>();
    }

    CATLASS_DEVICE void RowMax256(const LocalTensor<float>& dst, const LocalTensor<float>& src, uint32_t rowNum)
    {
        uint32_t repeatTime = 256 * RoundUp(rowNum, FLT_PER_DATABLOCK) / FLT_PER_REPEAT;
        BlockReduceMax<float, false>(tv, src, repeatTime, 0, 1, 1, 8);
        PipeBarrier<PIPE_V>();

        repeatTime /= FLT_PER_DATABLOCK / 2;
        SetVecMaskLen(FLT_PER_REPEAT / 2);
        BlockReduceMax<float, false>(tb, tv, repeatTime, 0, 1, 1, 4 /* srcRepStride */);
        PipeBarrier<PIPE_V>();

        repeatTime = CeilDiv(repeatTime, FLT_PER_DATABLOCK);
        SetBlockReduceMaskLen(4);
        BlockReduceMax<float, false>(dst, tb, repeatTime, 0, 1, 1, 8);
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

            BlockReduceMax<float, false>(dst, tv, CeilDiv(rowNum, FLT_PER_DATABLOCK), 0, 1, 1, 8);
            PipeBarrier<PIPE_V>();

            for (uint32_t idx = FLT_PER_REPEAT; idx < RoundDown(colNum, FLT_PER_REPEAT); idx += FLT_PER_REPEAT) {
                BlockReduceMax<float, false>(tv, src[idx], rowNum, 0, 1, 1, colNumRound / FLT_PER_DATABLOCK);
                PipeBarrier<PIPE_V>();

                BlockReduceMax<float, false>(tb, tv, CeilDiv(rowNum, FLT_PER_DATABLOCK), 0, 1, 1, 8);
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
                tv, src[RoundDown(colNum, FLT_PER_REPEAT)], rowNum, 0, 1, 1, colNumRound / FLT_PER_DATABLOCK);
            PipeBarrier<PIPE_V>();

            SetBlockReduceMaskLen(CeilDiv(colNum % FLT_PER_REPEAT, FLT_PER_DATABLOCK));
            if (colNum < FLT_PER_REPEAT) {
                BlockReduceMax<float, false>(dst, tv, CeilDiv(rowNum, FLT_PER_DATABLOCK), 0, 1, 1, 8);
                PipeBarrier<PIPE_V>();
            } else {
                BlockReduceMax<float, false>(tb, tv, CeilDiv(rowNum, FLT_PER_DATABLOCK), 0, 1, 1, 8);
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
        if (colNum == 512) {
            RowMax512(lm[rowOffset], ls[pingPongOffset], rowNum);
        } else if (colNum == 256) {
            RowMax256(lm[rowOffset], ls[pingPongOffset], rowNum);
        } else {
            RowMaxTail(lm[rowOffset], ls[pingPongOffset], rowNum, colNum, colNumRound);
        }
    }

    CATLASS_DEVICE void UpdateGlobalRowMax(
        uint32_t rowNum, uint32_t dmOffset, uint32_t rowOffset, uint32_t isFirstStackTile)
    {
        if (isFirstStackTile) {
            DataCopy(hm[rowOffset], lm[rowOffset], DataCopyParams(1, CeilDiv(rowNum, FLT_PER_DATABLOCK), 0, 0));
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
        DataCopy(gm[taskOffset][rowOffset], hm[rowOffset], DataCopyParams(1, CeilDiv(rowNum, FLT_PER_DATABLOCK), 0, 0));
        PipeBarrier<PIPE_V>();
    }

    CATLASS_DEVICE void RowSum512(const LocalTensor<float>& dst, const LocalTensor<float>& src, uint32_t rowNum)
    {
        uint32_t repeatTime = 512 * RoundUp(rowNum, FLT_PER_DATABLOCK) / FLT_PER_REPEAT;
        BlockReduceSum<float, false>(tv, src, repeatTime, 0, 1, 1, 8);
        PipeBarrier<PIPE_V>();

        repeatTime /= FLT_PER_DATABLOCK;
        BlockReduceSum<float, false>(tb, tv, repeatTime, 0, 1, 1, 8);
        PipeBarrier<PIPE_V>();

        repeatTime /= FLT_PER_DATABLOCK;
        BlockReduceSum<float, false>(dst, tb, repeatTime, 0, 1, 1, 8);
        PipeBarrier<PIPE_V>();
    }

    CATLASS_DEVICE void RowSum256(const LocalTensor<float>& dst, const LocalTensor<float>& src, uint32_t rowNum)
    {
        uint32_t repeatTime = 256 * RoundUp(rowNum, FLT_PER_DATABLOCK) / FLT_PER_REPEAT;
        BlockReduceSum<float, false>(tv, src, repeatTime, 0, 1, 1, 8);
        PipeBarrier<PIPE_V>();

        repeatTime /= FLT_PER_DATABLOCK / 2;
        SetVecMaskLen(FLT_PER_REPEAT / 2);
        BlockReduceSum<float, false>(tb, tv, repeatTime, 0, 1, 1, 4);
        PipeBarrier<PIPE_V>();

        repeatTime = CeilDiv(repeatTime, FLT_PER_DATABLOCK);
        SetBlockReduceMaskLen(4);
        BlockReduceSum<float, false>(dst, tb, repeatTime, 0, 1, 1, 8);
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

            BlockReduceSum<float, false>(dst, tv, CeilDiv(rowNum, FLT_PER_DATABLOCK), 0, 1, 1, 8);
            PipeBarrier<PIPE_V>();

            for (uint32_t idx = FLT_PER_REPEAT; idx < RoundDown(colNum, FLT_PER_REPEAT); idx += FLT_PER_REPEAT) {
                BlockReduceSum<float, false>(tv, src[idx], rowNum, 0, 1, 1, colNumRound / FLT_PER_DATABLOCK);
                PipeBarrier<PIPE_V>();

                BlockReduceSum<float, false>(tb, tv, CeilDiv(rowNum, FLT_PER_DATABLOCK), 0, 1, 1, 8);
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
                tv, src[RoundDown(colNum, FLT_PER_REPEAT)], rowNum, 0, 1, 1, colNumRound / FLT_PER_DATABLOCK);
            PipeBarrier<PIPE_V>();

            SetBlockReduceMaskLen(CeilDiv(colNum % FLT_PER_REPEAT, FLT_PER_DATABLOCK));
            if (colNum < FLT_PER_REPEAT) {
                BlockReduceSum<float, false>(dst, tv, CeilDiv(rowNum, FLT_PER_DATABLOCK), 0, 1, 1, 8);
                PipeBarrier<PIPE_V>();
            } else {
                BlockReduceSum<float, false>(tb, tv, CeilDiv(rowNum, FLT_PER_DATABLOCK), 0, 1, 1, 8);
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
        if (colNum == 512) {
            RowSum512(ll[rowOffset], ls[pingPongOffset], rowNum);
        } else if (colNum == 256) {
            RowSum256(ll[rowOffset], ls[pingPongOffset], rowNum);
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
                gl[taskOffset][rowOffset], ll[rowOffset], DataCopyParams(1, CeilDiv(rowNum, FLT_PER_DATABLOCK), 0, 0));
            PipeBarrier<PIPE_V>();
        } else {
            SetVecMaskLen(rowNum);
            // gl = dm * gl
            Mul<float, false>(
                gl[taskOffset][rowOffset], dm[dmOffset], gl[taskOffset][rowOffset], MASK_NUL, 1, BinaryRepeatParams());
            PipeBarrier<PIPE_V>();
            // gl = gl + ll
            Add<float, false>(
                gl[taskOffset][rowOffset], gl[taskOffset][rowOffset], ll[rowOffset], MASK_NUL, 1, BinaryRepeatParams());
            PipeBarrier<PIPE_V>();
            ResetMask();
        }
    }

    CATLASS_DEVICE void CopyMaskToUb(
        const GlobalTensor<ElementMask>& gMask, uint32_t rowNum, uint32_t maskColRound, uint32_t maskStride,
        uint32_t rowOffset, uint32_t qSBlockSize)
    {
        uint32_t qSIdxCurBlock = rowOffset % qSBlockSize;
        uint32_t rowsDone = 0;
        while (rowsDone < rowNum) {
            uint32_t rowsCurHead = Min(rowNum - rowsDone, qSBlockSize - qSIdxCurBlock);
            DataCopy(
                mask[rowsDone * maskColRound], gMask[qSIdxCurBlock * maskStride],
                DataCopyParams(rowsCurHead, maskColRound / NUM_32, (maskStride - maskColRound) / NUM_32, 0));
            rowsDone += rowsCurHead;
            qSIdxCurBlock = 0;
        }
    }

    CATLASS_DEVICE void ApplyMask(uint32_t rowNum, uint32_t colNumRound, uint32_t maskColRound, uint32_t addMaskOffset)
    {
        Muls<float, false>(
            mask32, mask32, -3e38f /* -FLT_MAX */, MASK_NUL, CeilDiv(rowNum * maskColRound, FLT_PER_REPEAT),
            UnaryRepeatParams());
        PipeBarrier<PIPE_V>();

        if (maskColRound == colNumRound) {
            Add<float, false>(
                ls[pingPongOffset], ls[pingPongOffset], mask32, MASK_NUL,
                CeilDiv(rowNum * maskColRound, FLT_PER_REPEAT), BinaryRepeatParams());
        } else {
            for (uint32_t idx = 0; idx < RoundDown(maskColRound, FLT_PER_REPEAT); idx += FLT_PER_REPEAT) {
                Add<float, false>(
                    ls[pingPongOffset][addMaskOffset + idx], ls[pingPongOffset][addMaskOffset + idx], mask32[idx],
                    MASK_NUL, rowNum,
                    BinaryRepeatParams(
                        1, 1, 1, colNumRound / FLT_PER_DATABLOCK, colNumRound / FLT_PER_DATABLOCK,
                        maskColRound / FLT_PER_DATABLOCK));
            }

            if (maskColRound % FLT_PER_REPEAT > 0) {
                SetVecMaskLen(maskColRound % FLT_PER_REPEAT);
                Add<float, false>(
                    ls[pingPongOffset][addMaskOffset + RoundDown(maskColRound, FLT_PER_REPEAT)],
                    ls[pingPongOffset][addMaskOffset + RoundDown(maskColRound, FLT_PER_REPEAT)],
                    mask32[RoundDown(maskColRound, FLT_PER_REPEAT)], MASK_NUL, rowNum,
                    BinaryRepeatParams(
                        1, 1, 1, colNumRound / FLT_PER_DATABLOCK, colNumRound / FLT_PER_DATABLOCK,
                        maskColRound / FLT_PER_DATABLOCK));
                ResetMask();
            }
        }
        PipeBarrier<PIPE_V>();
    }

    CATLASS_DEVICE void CalcExp(uint32_t rowNum, uint32_t colNum, uint32_t colNumRound, uint32_t rowOffset)
    {
        // hm_block = expand_to_block(hm), 存放于 tv
        Brcb(tv, hm[rowOffset], CeilDiv(rowNum, FLT_PER_DATABLOCK), BrcbRepeatParams());
        PipeBarrier<PIPE_V>();

        // ls = ls - hm_block
        for (uint32_t idx = 0; idx < RoundDown(colNum, FLT_PER_REPEAT); idx += FLT_PER_REPEAT) {
            Sub<float, false>(
                ls[pingPongOffset][idx], ls[pingPongOffset][idx], tv, MASK_NUL, rowNum,
                BinaryRepeatParams(1, 1, 0, colNumRound / FLT_PER_DATABLOCK, colNumRound / FLT_PER_DATABLOCK, 1));
        }

        if (colNum % FLT_PER_REPEAT > 0) {
            SetVecMaskLen(colNum % FLT_PER_REPEAT);
            Sub<float, false>(
                ls[pingPongOffset][RoundDown(colNum, FLT_PER_REPEAT)],
                ls[pingPongOffset][RoundDown(colNum, FLT_PER_REPEAT)], tv, MASK_NUL, rowNum,
                BinaryRepeatParams(1, 1, 0, colNumRound / FLT_PER_DATABLOCK, colNumRound / FLT_PER_DATABLOCK, 1));
            ResetMask();
        }
        PipeBarrier<PIPE_V>();

        // ls = exp(ls)
        Exp<float, false>(
            ls[pingPongOffset], ls[pingPongOffset], MASK_NUL, CeilDiv(rowNum * colNumRound, FLT_PER_REPEAT),
            UnaryRepeatParams());
        PipeBarrier<PIPE_V>();
    }

    CATLASS_DEVICE void ScaleS(uint32_t rowNum, uint32_t colNumRound)
    {
        Muls<float, false>(
            ls[pingPongOffset], ls[pingPongOffset], scaleVal, MASK_NUL, CeilDiv(rowNum * colNumRound, FLT_PER_REPEAT),
            UnaryRepeatParams());
        PipeBarrier<PIPE_V>();
    }

    template <bool DO_TRIU_MASK_>
    CATLASS_DEVICE void SubCoreCompute(
        const GlobalTensor<ElementOutput>& gOutput, const LayoutOutput& layoutOutput, uint32_t rowOffset,
        uint32_t isFirstStackTile, uint32_t isLastNoMaskStackTile, uint32_t isFirstRowLoop, uint32_t isLastRowLoop,
        uint32_t colNumRound, uint32_t curStackTileMod)
    {
        uint32_t rowNum = layoutOutput.shape(0);
        uint32_t colNum = layoutOutput.shape(1);
        uint32_t colStride = layoutOutput.stride(0);
        uint32_t dmOffset = curStackTileMod * MAX_ROW_NUM_SUB_CORE + rowOffset;

        if (isFirstStackTile && isFirstRowLoop) {
            WaitFlag<HardEvent::MTE3_V>(taskEvent); // 等待上个 task 的 RS 完成
        }

        CalcLocalRowMax(rowOffset, rowNum, colNum, colNumRound);
        UpdateGlobalRowMax(rowNum, dmOffset, rowOffset, isFirstStackTile);

        CalcExp(rowNum, colNum, colNumRound, rowOffset);

        if constexpr (!DO_TRIU_MASK_) {
            WaitFlag<HardEvent::MTE3_V>(pingPong);
        }

        // lp = castfp32to16(ls)
        RoundMode roundMode = std::is_same_v<ElementOutput, bfloat16_t> ? RoundMode::CAST_RINT : RoundMode::CAST_NONE;
        Cast<ElementOutput, float, false>(
            lp[pingPongOffset], ls[pingPongOffset], roundMode, MASK_NUL, CeilDiv(rowNum * colNumRound, FLT_PER_REPEAT),
            UnaryRepeatParams(1, 1, 4, 8));
        SetFlag<HardEvent::V_MTE3>(pingPong);

        CalcLocalRowSum(rowOffset, rowNum, colNum, colNumRound);
        SetFlag<HardEvent::V_MTE2>(pingPong);

        WaitFlag<HardEvent::V_MTE3>(pingPong);
        DataCopy(
            gOutput, lp[pingPongOffset],
            DataCopyParams(
                rowNum, colNumRound / HALF_PER_DATABLOCK, 0, (colStride - colNumRound) / HALF_PER_DATABLOCK));

        if constexpr (!DO_TRIU_MASK_) {
            SetFlag<HardEvent::MTE3_V>(pingPong);
            if (isLastNoMaskStackTile && isLastRowLoop) {
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
        } else {
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }

        UpdateGlobalRowSum(rowNum, dmOffset, rowOffset, isFirstStackTile);
    }

    CATLASS_DEVICE void operator()(
        const GlobalTensor<ElementOutput>& gOutput, const GlobalTensor<ElementInput>& gInput,
        const LayoutOutput& layoutOutput, const LayoutInput& layoutInput, GemmCoord actualBlockShape,
        uint32_t isFirstStackTile, uint32_t isLastNoMaskStackTile, uint32_t qSBlockSize, uint32_t qNBlockSize,
        uint32_t curStackTileMod)
    {
        uint32_t rowNum = actualBlockShape.m();
        uint32_t colNum = actualBlockShape.n();
        uint32_t colNumRound = RoundUp(colNum, NUM_32);
        uint32_t colStride = layoutInput.stride(0);

        uint32_t coreIdx = GetSubBlockIdx();
        uint32_t coreRowSplit = (qNBlockSize == 1) ? (qSBlockSize / 2) : (qSBlockSize * (qNBlockSize / 2));
        uint32_t coreRowActual = (coreIdx == 1) ? (rowNum - coreRowSplit) : coreRowSplit;
        uint32_t coreRowOffset = coreIdx * coreRowSplit;

        if (coreRowActual == 0) {
            return;
        }

        uint32_t rowNumTile = RoundDown(MAX_UB_ELEM_NUM / colNumRound, FLT_PER_DATABLOCK);
        rowNumTile = Min(rowNumTile, FLT_PER_REPEAT);
        uint32_t rowLoop = CeilDiv(coreRowActual, rowNumTile);

        for (uint32_t rowIdx = 0; rowIdx < rowLoop + PRE_LAUNCH; rowIdx++) {
            if (rowIdx < rowLoop) {
                uint32_t rowOffset = rowIdx * rowNumTile;
                uint32_t gmRowOffset = rowOffset + coreRowOffset;
                uint32_t rowNumCurLoop = (rowIdx == rowLoop - 1) ? (coreRowActual - rowOffset) : rowNumTile;

                WaitFlag<HardEvent::V_MTE2>(pingPong);
                DataCopy(
                    ls[pingPongOffset], gInput[layoutInput.GetOffset(MatrixCoord(gmRowOffset, 0))],
                    DataCopyParams(
                        rowNumCurLoop, colNumRound / FLT_PER_DATABLOCK, (colStride - colNumRound) / FLT_PER_DATABLOCK,
                        0));
                SetFlag<HardEvent::MTE2_V>(pingPong);
            }

            pingPong ^= 1;
            pingPongOffset = pingPong ? MAX_UB_ELEM_NUM : 0;

            if (rowIdx >= PRE_LAUNCH) {
                uint32_t nowRowIdx = rowIdx - PRE_LAUNCH;
                uint32_t rowOffset = nowRowIdx * rowNumTile;
                uint32_t gmRowOffset = rowOffset + coreRowOffset;
                uint32_t rowNumCurLoop = (nowRowIdx == rowLoop - 1) ? (coreRowActual - rowOffset) : rowNumTile;

                WaitFlag<HardEvent::MTE2_V>(pingPong);
                ScaleS(rowNumCurLoop, colNumRound);

                SubCoreCompute<false>(
                    gOutput[layoutOutput.GetOffset(MatrixCoord(gmRowOffset, 0))],
                    layoutOutput.GetTileLayout(MatrixCoord(rowNumCurLoop, colNum)), rowOffset, isFirstStackTile,
                    isLastNoMaskStackTile, (nowRowIdx == 0), (nowRowIdx == rowLoop - 1), colNumRound, curStackTileMod);
            }
        }
    }

    CATLASS_DEVICE void operator()(
        const GlobalTensor<ElementOutput>& gOutput, const GlobalTensor<ElementInput>& gInput,
        const GlobalTensor<ElementMask>& gMask, const LayoutOutput& layoutOutput, const LayoutInput& layoutInput,
        const LayoutInput& layoutMask, GemmCoord actualBlockShape, uint32_t isFirstStackTile, uint32_t qSBlockSize,
        uint32_t qNBlockSize, uint32_t curStackTileMod, Arch::CrossCoreFlag qkReady, uint32_t triUp, uint32_t triDown,
        uint32_t kvSStartIdx, uint32_t kvSEndIdx)
    {
        uint32_t rowNum = actualBlockShape.m();
        uint32_t colNum = actualBlockShape.n();
        uint32_t colNumRound = RoundUp(colNum, NUM_32);
        uint32_t colStride = layoutInput.stride(0);
        uint32_t maskStride = layoutMask.stride(0);

        uint32_t coreIdx = GetSubBlockIdx();
        uint32_t coreRowSplit = (qNBlockSize == 1) ? (qSBlockSize / 2) : (qSBlockSize * (qNBlockSize / 2));
        uint32_t coreRowActual = (coreIdx == 1) ? (rowNum - coreRowSplit) : coreRowSplit;
        uint32_t coreRowOffset = coreIdx * coreRowSplit;

        if (coreRowActual == 0) {
            Arch::CrossCoreWaitFlag(qkReady);
            return;
        }

        // calc mask shift in GM
        uint32_t gmMaskRowOffset;
        uint32_t gmMaskColOffset;
        uint32_t maskCol;
        uint32_t addMaskOffset;
        if (triUp >= kvSStartIdx) {
            uint32_t triUpRound = RoundDown(triUp, NUM_32);
            gmMaskRowOffset = triUp - triUpRound + ((qNBlockSize == 1) ? coreRowOffset : 0);
            gmMaskColOffset = 0;
            maskCol = kvSEndIdx - triUpRound;
            addMaskOffset = triUpRound - kvSStartIdx;
        } else {
            gmMaskRowOffset = (qNBlockSize == 1) ? coreRowOffset : 0;
            gmMaskColOffset = kvSStartIdx - triUp;
            maskCol = colNum;
            addMaskOffset = 0;
        }
        uint32_t maskColRound = RoundUp(maskCol, NUM_32);

        uint32_t rowNumTile = RoundDown(MAX_UB_ELEM_NUM / colNumRound, FLT_PER_DATABLOCK);
        rowNumTile = Min(rowNumTile, FLT_PER_REPEAT);
        uint32_t rowLoop = CeilDiv(coreRowActual, rowNumTile);

        for (uint32_t rowIdx = 0; rowIdx < rowLoop + PRE_LAUNCH; rowIdx++) {
            if (rowIdx < rowLoop) {
                uint32_t rowOffset = rowIdx * rowNumTile;
                uint32_t gmRowOffset = rowOffset + coreRowOffset;
                uint32_t rowNumCurLoop = (rowIdx == rowLoop - 1) ? (coreRowActual - rowOffset) : rowNumTile;

                // loop 0 mask load before cross core sync
                if (rowIdx == 0) {
                    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                    CopyMaskToUb(
                        gMask[layoutMask.GetOffset(MatrixCoord(gmMaskRowOffset, gmMaskColOffset))], rowNumCurLoop,
                        maskColRound, maskStride, rowOffset, qSBlockSize);
                    SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
                    Arch::CrossCoreWaitFlag(qkReady);
                }

                WaitFlag<HardEvent::V_MTE2>(pingPong);
                DataCopy(
                    ls[pingPongOffset], gInput[layoutInput.GetOffset(MatrixCoord(gmRowOffset, 0))],
                    DataCopyParams(
                        rowNumCurLoop, colNumRound / FLT_PER_DATABLOCK, (colStride - colNumRound) / FLT_PER_DATABLOCK,
                        0));
                SetFlag<HardEvent::MTE2_V>(pingPong);
            }

            pingPong ^= 1;
            pingPongOffset = pingPong ? MAX_UB_ELEM_NUM : 0;

            if (rowIdx >= PRE_LAUNCH) {
                uint32_t nowRowIdx = rowIdx - PRE_LAUNCH;
                uint32_t rowOffset = nowRowIdx * rowNumTile;
                uint32_t gmRowOffset = rowOffset + coreRowOffset;
                uint32_t rowNumCurLoop = (nowRowIdx == rowLoop - 1) ? (coreRowActual - rowOffset) : rowNumTile;

                if (nowRowIdx > 0) {
                    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                    CopyMaskToUb(
                        gMask[layoutMask.GetOffset(MatrixCoord(gmMaskRowOffset, gmMaskColOffset))], rowNumCurLoop,
                        maskColRound, maskStride, rowOffset, qSBlockSize);
                    SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
                }

                WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
                Cast<half, ElementMask, false>(
                    mask16, mask, RoundMode::CAST_NONE, MASK_NUL, CeilDiv(rowNumCurLoop * colNumRound, HALF_PER_REPEAT),
                    UnaryRepeatParams(1, 1, 8, 4));
                PipeBarrier<PIPE_V>();
                Cast<float, half, false>(
                    mask32, mask16, RoundMode::CAST_NONE, MASK_NUL,
                    CeilDiv(rowNumCurLoop * colNumRound, FLT_PER_REPEAT), UnaryRepeatParams(1, 1, 8, 4));
                PipeBarrier<PIPE_V>();

                WaitFlag<HardEvent::MTE2_V>(pingPong);
                ScaleS(rowNumCurLoop, colNumRound);
                ApplyMask(rowNumCurLoop, colNumRound, maskColRound, addMaskOffset);

                SubCoreCompute<true>(
                    gOutput[layoutOutput.GetOffset(MatrixCoord(gmRowOffset, 0))],
                    layoutOutput.GetTileLayout(MatrixCoord(rowNumCurLoop, colNum)), rowOffset, isFirstStackTile, 0,
                    (nowRowIdx == 0), (nowRowIdx == rowLoop - 1), colNumRound, curStackTileMod);
            }
        }
    }

private:
    float scaleVal;
    uint32_t taskEvent;
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