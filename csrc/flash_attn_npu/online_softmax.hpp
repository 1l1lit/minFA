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

    static constexpr uint32_t MAX_UB_S_ELEM_NUM = 8192;
    static constexpr uint32_t MAX_ROW_NUM_SUB_CORE = 256;

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
        constexpr uint32_t GL_OFFSET = 10 * UB_BLK_SIZE + UB_VEC_SIZE;
        constexpr uint32_t DM_OFFSET = 10 * UB_BLK_SIZE + 2 * UB_VEC_SIZE;

        constexpr uint32_t LM_OFFSET = 10 * UB_BLK_SIZE + 8 * UB_VEC_SIZE;
        constexpr uint32_t HM_OFFSET = 10 * UB_BLK_SIZE + 9 * UB_VEC_SIZE;
        constexpr uint32_t LL_OFFSET = 10 * UB_BLK_SIZE + 10 * UB_VEC_SIZE;

        constexpr uint32_t TV_OFFSET = 11 * UB_BLK_SIZE + 8 * UB_VEC_SIZE;
        constexpr uint32_t TVB_OFFSET = 11 * UB_BLK_SIZE + 12 * UB_VEC_SIZE;

        ls = resource.ubBuf.template GetBufferByByte<float>(LS_OFFSET);
        lp = resource.ubBuf.template GetBufferByByte<ElementOutput>(LP_OFFSET);
        mask = resource.ubBuf.template GetBufferByByte<ElementMask>(MASK_OFFSET);
        mask32 = resource.ubBuf.template GetBufferByByte<float>(MASK_OFFSET);
        mask16 = resource.ubBuf.template GetBufferByByte<half>(MASK16_OFFSET);
        gm = resource.ubBuf.template GetBufferByByte<float>(GM_OFFSET);
        gl = resource.ubBuf.template GetBufferByByte<float>(GL_OFFSET);
        dm = resource.ubBuf.template GetBufferByByte<float>(DM_OFFSET);
        lm = resource.ubBuf.template GetBufferByByte<float>(LM_OFFSET);
        hm = resource.ubBuf.template GetBufferByByte<float>(HM_OFFSET);
        ll = resource.ubBuf.template GetBufferByByte<float>(LL_OFFSET);
        tv = resource.ubBuf.template GetBufferByByte<float>(TV_OFFSET);
        tvb = resource.ubBuf.template GetBufferByByte<float>(TVB_OFFSET);
    }

    CATLASS_DEVICE void set_curTaskMod(uint32_t curTaskMod)
    {
        taskEvent = curTaskMod + 2;
        taskOffset = curTaskMod * ((4 * UB_VEC_SIZE) / sizeof(float));
    }

    CATLASS_DEVICE void RowMax512(const LocalTensor<float>& dst, const LocalTensor<float>& src, uint32_t rowNumRound)
    {
        uint32_t repeatTime = rowNumRound * 512 / FLT_PER_REPEAT;
        BlockReduceMax<float, false>(tv, src, repeatTime, 0, 1, 1, 8);
        PipeBarrier<PIPE_V>();

        repeatTime /= DATABLOCK_PER_REPEAT;
        BlockReduceMax<float, false>(tvb, tv, repeatTime, 0, 1, 1, 8);
        PipeBarrier<PIPE_V>();

        repeatTime /= DATABLOCK_PER_REPEAT;
        BlockReduceMax<float, false>(dst, tvb, repeatTime, 0, 1, 1, 8);
        PipeBarrier<PIPE_V>();
    }

    CATLASS_DEVICE void RowMax256(const LocalTensor<float>& dst, const LocalTensor<float>& src, uint32_t rowNumRound)
    {
        BlockReduceMax<float, false>(tv, src, rowNumRound * 256 / FLT_PER_REPEAT, 0, 1, 1, 8);
        PipeBarrier<PIPE_V>();

        SetVecMaskLen(32); // FLT_PER_REPEAT / 2
        BlockReduceMax<float, false>(tvb, tv, rowNumRound, 0, 1, 1, 4 /* srcRepStride */);
        PipeBarrier<PIPE_V>();

        SetBlockReduceMaskLen(4);
        BlockReduceMax<float, false>(dst, tvb, CeilDiv(rowNumRound, FLT_PER_DATABLOCK), 0, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        ResetMask();
    }

    CATLASS_DEVICE void RowMaxTail(
        const LocalTensor<float>& dst, const LocalTensor<float>& src, uint32_t rowNumRound, uint32_t colNum,
        uint32_t colNumRound)
    {
        if (colNum >= FLT_PER_REPEAT) {
            BlockReduceMax<float, false>(tv, src, rowNumRound, 0, 1, 1, colNumRound / DATABLOCK_PER_REPEAT);
            PipeBarrier<PIPE_V>();

            BlockReduceMax<float, false>(dst, tv, CeilDiv(rowNumRound, FLT_PER_DATABLOCK), 0, 1, 1, 8);
            PipeBarrier<PIPE_V>();

            for (uint32_t idx = FLT_PER_REPEAT; idx < RoundDown(colNum, FLT_PER_REPEAT); idx += FLT_PER_REPEAT) {
                BlockReduceMax<float, false>(tv, src[idx], rowNumRound, 0, 1, 1, colNumRound / DATABLOCK_PER_REPEAT);
                PipeBarrier<PIPE_V>();

                BlockReduceMax<float, false>(tvb, tv, CeilDiv(rowNumRound, FLT_PER_DATABLOCK), 0, 1, 1, 8);
                PipeBarrier<PIPE_V>();

                SetVecMaskLen(rowNumRound);
                Max<float, false>(dst, dst, tvb, MASK_NUL, 1, BinaryRepeatParams());
                PipeBarrier<PIPE_V>();
                ResetMask();
            }
        }

        if (colNum % FLT_PER_REPEAT > 0) {
            SetVecMaskLen(colNum % FLT_PER_REPEAT);
            BlockReduceMax<float, false>(
                tv, src[RoundDown(colNum, FLT_PER_REPEAT)], rowNumRound, 0, 1, 1, colNumRound / DATABLOCK_PER_REPEAT);
            PipeBarrier<PIPE_V>();

            SetBlockReduceMaskLen(CeilDiv(colNum % FLT_PER_REPEAT, DATABLOCK_PER_REPEAT));
            if (colNum < FLT_PER_REPEAT) {
                BlockReduceMax<float, false>(dst, tv, CeilDiv(rowNumRound, FLT_PER_DATABLOCK), 0, 1, 1, 8);
                PipeBarrier<PIPE_V>();
            } else {
                BlockReduceMax<float, false>(tvb, tv, CeilDiv(rowNumRound, FLT_PER_DATABLOCK), 0, 1, 1, 8);
                PipeBarrier<PIPE_V>();

                SetVecMaskLen(rowNumRound);
                Max<float, false>(dst, dst, tvb, MASK_NUL, 1, BinaryRepeatParams());
                PipeBarrier<PIPE_V>();
            }
            ResetMask();
        }
    }

    CATLASS_DEVICE void CalcLocalRowMax(
        uint32_t coreRowOffset, uint32_t rowNumRound, uint32_t colNum, uint32_t colNumRound)
    {
        if (colNum == 512) {
            RowMax512(lm[coreRowOffset], ls[pingPongOffset], rowNumRound);
        } else if (colNum == 256) {
            RowMax256(lm[coreRowOffset], ls[pingPongOffset], rowNumRound);
        } else {
            RowMaxTail(lm[coreRowOffset], ls[pingPongOffset], rowNumRound, colNum, colNumRound);
        }
    }

    CATLASS_DEVICE void UpdateGlobalRowMax(
        uint32_t rowNum, uint32_t rowNumRound, uint32_t dmOffset, uint32_t coreRowOffset, uint32_t isFirstStackTile)
    {
        DataCopyParams copyParams(1, rowNumRound / DATABLOCK_PER_REPEAT /* blockLen */, 0, 0);

        if (isFirstStackTile) {
            DataCopy(hm[coreRowOffset], lm[coreRowOffset], copyParams);
            PipeBarrier<PIPE_V>();
        } else {
            SetVecMaskLen(rowNum);
            // hm = max(lm, gm)
            Max<float, false>(
                hm[coreRowOffset], lm[coreRowOffset], gm[taskOffset][coreRowOffset], MASK_NUL, 1, BinaryRepeatParams());
            PipeBarrier<PIPE_V>();
            // dm = gm - hm
            Sub<float, false>(
                dm[taskOffset][dmOffset], gm[taskOffset][coreRowOffset], hm[coreRowOffset], MASK_NUL, 1,
                BinaryRepeatParams());
            PipeBarrier<PIPE_V>();
            // dm = exp(dm)
            Exp<float, false>(dm[taskOffset][dmOffset], dm[taskOffset][dmOffset], MASK_NUL, 1, UnaryRepeatParams());
            PipeBarrier<PIPE_V>();
            ResetMask();
        }

        // gm = hm
        DataCopy(gm[taskOffset][coreRowOffset], hm[coreRowOffset], copyParams);
        PipeBarrier<PIPE_V>();
    }

    CATLASS_DEVICE void RowSum512(const LocalTensor<float>& dst, const LocalTensor<float>& src, uint32_t rowNumRound)
    {
        uint32_t repeatTime = rowNumRound * 512 / FLT_PER_REPEAT;
        BlockReduceSum<float, false>(tv, src, repeatTime, 0, 1, 1, 8);
        PipeBarrier<PIPE_V>();

        repeatTime /= DATABLOCK_PER_REPEAT;
        BlockReduceSum<float, false>(tvb, tv, repeatTime, 0, 1, 1, 8);
        PipeBarrier<PIPE_V>();

        repeatTime /= DATABLOCK_PER_REPEAT;
        BlockReduceSum<float, false>(dst, tvb, repeatTime, 0, 1, 1, 8);
        PipeBarrier<PIPE_V>();
    }

    CATLASS_DEVICE void RowSum256(const LocalTensor<float>& dst, const LocalTensor<float>& src, uint32_t rowNumRound)
    {
        BlockReduceSum<float, false>(tv, src, rowNumRound * 256 / FLT_PER_REPEAT, 0, 1, 1, 8);
        PipeBarrier<PIPE_V>();

        SetVecMaskLen(32);
        BlockReduceSum<float, false>(tvb, tv, rowNumRound, 0, 1, 1, 4);
        PipeBarrier<PIPE_V>();

        SetBlockReduceMaskLen(4);
        BlockReduceSum<float, false>(dst, tvb, CeilDiv(rowNumRound, FLT_PER_DATABLOCK), 0, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        ResetMask();
    }

    CATLASS_DEVICE void RowSumTail(
        const LocalTensor<float>& dst, const LocalTensor<float>& src, uint32_t rowNumRound, uint32_t colNum,
        uint32_t colNumRound)
    {
        if (colNum >= FLT_PER_REPEAT) {
            BlockReduceSum<float, false>(tv, src, rowNumRound, 0, 1, 1, colNumRound / DATABLOCK_PER_REPEAT);
            PipeBarrier<PIPE_V>();

            BlockReduceSum<float, false>(dst, tv, CeilDiv(rowNumRound, FLT_PER_DATABLOCK), 0, 1, 1, 8);
            PipeBarrier<PIPE_V>();

            for (uint32_t idx = FLT_PER_REPEAT; idx < RoundDown(colNum, FLT_PER_REPEAT); idx += FLT_PER_REPEAT) {
                BlockReduceSum<float, false>(tv, src[idx], rowNumRound, 0, 1, 1, colNumRound / DATABLOCK_PER_REPEAT);
                PipeBarrier<PIPE_V>();

                BlockReduceSum<float, false>(tvb, tv, CeilDiv(rowNumRound, FLT_PER_DATABLOCK), 0, 1, 1, 8);
                PipeBarrier<PIPE_V>();

                SetVecMaskLen(rowNumRound);
                Add<float, false>(dst, dst, tvb, MASK_NUL, 1, BinaryRepeatParams());
                PipeBarrier<PIPE_V>();
                ResetMask();
            }
        }

        if (colNum % FLT_PER_REPEAT > 0) {
            SetVecMaskLen(colNum % FLT_PER_REPEAT);
            BlockReduceSum<float, false>(
                tv, src[RoundDown(colNum, FLT_PER_REPEAT)], rowNumRound, 0, 1, 1, colNumRound / DATABLOCK_PER_REPEAT);
            PipeBarrier<PIPE_V>();

            SetBlockReduceMaskLen(CeilDiv(colNum % FLT_PER_REPEAT, DATABLOCK_PER_REPEAT));
            if (colNum < FLT_PER_REPEAT) {
                BlockReduceSum<float, false>(dst, tv, CeilDiv(rowNumRound, FLT_PER_DATABLOCK), 0, 1, 1, 8);
                PipeBarrier<PIPE_V>();
            } else {
                BlockReduceSum<float, false>(tvb, tv, CeilDiv(rowNumRound, FLT_PER_DATABLOCK), 0, 1, 1, 8);
                PipeBarrier<PIPE_V>();

                SetVecMaskLen(rowNumRound);
                Add<float, false>(dst, dst, tvb, MASK_NUL, 1, BinaryRepeatParams());
                PipeBarrier<PIPE_V>();
            }
            ResetMask();
        }
    }

    CATLASS_DEVICE void CalcLocalRowSum(
        uint32_t coreRowOffset, uint32_t rowNumRound, uint32_t colNum, uint32_t colNumRound)
    {
        if (colNum == 512) {
            RowSum512(ll[coreRowOffset], ls[pingPongOffset], rowNumRound);
        } else if (colNum == 256) {
            RowSum256(ll[coreRowOffset], ls[pingPongOffset], rowNumRound);
        } else {
            RowSumTail(ll[coreRowOffset], ls[pingPongOffset], rowNumRound, colNum, colNumRound);
        }
    }

    CATLASS_DEVICE void UpdateGlobalRowSum(
        uint32_t rowNum, uint32_t rowNumRound, uint32_t dmOffset, uint32_t coreRowOffset, uint32_t isFirstStackTile)
    {
        if (isFirstStackTile) {
            // gl = ll
            DataCopy(
                gl[taskOffset][coreRowOffset], ll[coreRowOffset],
                DataCopyParams(1, rowNumRound / FLT_PER_DATABLOCK, 0, 0));
            PipeBarrier<PIPE_V>();
        } else {
            SetVecMaskLen(rowNum);
            // gl = dm * gl
            Mul<float, false>(
                gl[taskOffset][coreRowOffset], dm[taskOffset][dmOffset], gl[taskOffset][coreRowOffset], MASK_NUL, 1,
                BinaryRepeatParams());
            PipeBarrier<PIPE_V>();
            // gl = gl + ll
            Add<float, false>(
                gl[taskOffset][coreRowOffset], gl[taskOffset][coreRowOffset], ll[coreRowOffset], MASK_NUL, 1,
                BinaryRepeatParams());
            PipeBarrier<PIPE_V>();
            ResetMask();
        }
    }

    CATLASS_DEVICE void CopyMaskToUb(
        const GlobalTensor<ElementMask>& gMask, uint32_t colNum, uint32_t colNumRound, uint32_t maskStride,
        uint32_t tokenNumPerHead, uint32_t proTokenIdx, uint32_t proTokenNum, uint32_t integralHeadNum,
        uint32_t epiTokenNum)
    {
        uint32_t blockLen = colNum * sizeof(ElementMask);
        uint32_t srcStride = (maskStride - colNum) * sizeof(ElementMask);
        uint32_t innerRowOffset = 0;

        if (proTokenNum != 0) {
            DataCopyPad(
                mask[innerRowOffset], gMask[proTokenIdx * maskStride],
                DataCopyExtParams(proTokenNum, blockLen, srcStride, 0, 0), DataCopyPadExtParams<ElementMask>());
            innerRowOffset += proTokenNum * colNumRound;
        }

        for (uint32_t i = 0; i < integralHeadNum; i++) {
            DataCopyPad(
                mask[innerRowOffset], gMask, DataCopyExtParams(tokenNumPerHead, blockLen, srcStride, 0, 0),
                DataCopyPadExtParams<ElementMask>());
            innerRowOffset += tokenNumPerHead * colNumRound;
        }

        if (epiTokenNum != 0) {
            DataCopyPad(
                mask[innerRowOffset], gMask, DataCopyExtParams(epiTokenNum, blockLen, srcStride, 0, 0),
                DataCopyPadExtParams<ElementMask>());
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
                        1, 1, 1, colNumRound / DATABLOCK_PER_REPEAT, colNumRound / DATABLOCK_PER_REPEAT,
                        maskColRound / DATABLOCK_PER_REPEAT));
            }

            if (maskColRound % FLT_PER_REPEAT > 0) {
                SetVecMaskLen(maskColRound % FLT_PER_REPEAT);
                Add<float, false>(
                    ls[pingPongOffset][addMaskOffset + RoundDown(maskColRound, FLT_PER_REPEAT)],
                    ls[pingPongOffset][addMaskOffset + RoundDown(maskColRound, FLT_PER_REPEAT)],
                    mask32[RoundDown(maskColRound, FLT_PER_REPEAT)], MASK_NUL, rowNum,
                    BinaryRepeatParams(
                        1, 1, 1, colNumRound / DATABLOCK_PER_REPEAT, colNumRound / DATABLOCK_PER_REPEAT,
                        maskColRound / DATABLOCK_PER_REPEAT));
                ResetMask();
            }
        }
        PipeBarrier<PIPE_V>();
    }

    CATLASS_DEVICE void CalcExp(
        uint32_t rowNum, uint32_t rowNumRound, uint32_t colNum, uint32_t colNumRound, uint32_t coreRowOffset)
    {
        // hm_block = expand_to_block(hm), 存放于 tv
        Brcb(tv, hm[coreRowOffset], rowNumRound / DATABLOCK_PER_REPEAT, BrcbRepeatParams());
        PipeBarrier<PIPE_V>();

        // ls = ls - hm_block
        for (uint32_t idx = 0; idx < RoundDown(colNum, FLT_PER_REPEAT); idx += FLT_PER_REPEAT) {
            Sub<float, false>(
                ls[pingPongOffset][idx], ls[pingPongOffset][idx], tv, MASK_NUL, rowNum,
                BinaryRepeatParams(1, 1, 0, colNumRound / DATABLOCK_PER_REPEAT, colNumRound / DATABLOCK_PER_REPEAT, 1));
        }

        if (colNum % FLT_PER_REPEAT > 0) {
            SetVecMaskLen(colNum % FLT_PER_REPEAT);
            Sub<float, false>(
                ls[pingPongOffset][RoundDown(colNum, FLT_PER_REPEAT)],
                ls[pingPongOffset][RoundDown(colNum, FLT_PER_REPEAT)], tv, MASK_NUL, rowNum,
                BinaryRepeatParams(1, 1, 0, colNumRound / DATABLOCK_PER_REPEAT, colNumRound / DATABLOCK_PER_REPEAT, 1));
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
        uint32_t repeatTimes = CeilDiv(rowNum * colNumRound, FLT_PER_REPEAT);
        Muls<float, false>(
            ls[pingPongOffset], ls[pingPongOffset], scaleVal, MASK_NUL, repeatTimes, UnaryRepeatParams());
        PipeBarrier<PIPE_V>();
    }

    template <bool doTriUMask>
    CATLASS_DEVICE void SubCoreCompute(
        const GlobalTensor<ElementOutput>& gOutput, const LayoutOutput& layoutOutput, uint32_t coreRowOffset,
        uint32_t isFirstStackTile, uint32_t isLastNoMaskStackTile, uint32_t isFirstRowLoop, uint32_t isLastRowLoop,
        uint32_t colNumRound, uint32_t curStackTileMod)
    {
        uint32_t rowNumLoop = layoutOutput.shape(0);
        uint32_t rowNumLoopRound = RoundUp(rowNumLoop, DATABLOCK_PER_REPEAT);
        uint32_t colNum = layoutOutput.shape(1);
        uint32_t colNumPad = layoutOutput.stride(0);
        uint32_t dmOffsetCurCycle = curStackTileMod * MAX_ROW_NUM_SUB_CORE + coreRowOffset;

        if (isFirstStackTile && isFirstRowLoop) {
            WaitFlag<HardEvent::MTE3_V>(taskEvent); // 等待上个 task 的 RS 完成
        }

        CalcLocalRowMax(coreRowOffset, rowNumLoopRound, colNum, colNumRound);
        UpdateGlobalRowMax(rowNumLoop, rowNumLoopRound, dmOffsetCurCycle, coreRowOffset, isFirstStackTile);

        CalcExp(rowNumLoop, rowNumLoopRound, colNum, colNumRound, coreRowOffset);

        if constexpr (!doTriUMask) {
            WaitFlag<HardEvent::MTE3_V>(pingPong);
        }

        // lp = castfp32to16(ls)
        RoundMode roundMode = std::is_same_v<ElementOutput, bfloat16_t> ? RoundMode::CAST_RINT : RoundMode::CAST_NONE;
        Cast<ElementOutput, float, false>(
            lp[pingPongOffset], ls[pingPongOffset], roundMode, MASK_NUL,
            CeilDiv(rowNumLoop * colNumRound, FLT_PER_REPEAT), UnaryRepeatParams(1, 1, 4, 8));
        SetFlag<HardEvent::V_MTE3>(pingPong);

        CalcLocalRowSum(coreRowOffset, rowNumLoopRound, colNum, colNumRound);
        SetFlag<HardEvent::V_MTE2>(pingPong);

        WaitFlag<HardEvent::V_MTE3>(pingPong);
        DataCopy(
            gOutput, lp[pingPongOffset],
            DataCopyParams(rowNumLoop, colNumRound / NUM_16, 0, (colNumPad - colNumRound) / NUM_16));

        if constexpr (!doTriUMask) {
            SetFlag<HardEvent::MTE3_V>(pingPong);
            if (isLastNoMaskStackTile && isLastRowLoop) {
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
        } else {
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }

        UpdateGlobalRowSum(rowNumLoop, rowNumLoopRound, dmOffsetCurCycle, coreRowOffset, isFirstStackTile);
    }

    CATLASS_DEVICE void operator()(
        const GlobalTensor<ElementOutput>& gOutput, const GlobalTensor<ElementInput>& gInput,
        const LayoutOutput& layoutOutput, const LayoutInput& layoutInput, GemmCoord actualBlockShape,
        uint32_t isFirstStackTile, uint32_t isLastNoMaskStackTile, uint32_t curStackTileMod)
    {
        uint32_t rowNum = actualBlockShape.m();
        uint32_t colNum = actualBlockShape.n();
        uint32_t colNumRound = RoundUp(colNum, NUM_32);
        uint32_t colNumPad = layoutInput.stride(0);

        uint32_t coreIdx = GetSubBlockIdx();
        uint32_t coreRowActual = (coreIdx == 1) ? (rowNum - rowNum / 2) : (rowNum / 2);
        uint32_t coreRowOffset = coreIdx * (rowNum / 2);

        uint32_t rowNumTile = RoundDown(MAX_UB_S_ELEM_NUM / colNumRound, DATABLOCK_PER_REPEAT);
        rowNumTile = Min(rowNumTile, FLT_PER_REPEAT);
        uint32_t rowLoop = CeilDiv(coreRowActual, rowNumTile);

        for (uint32_t rowIdx = 0; rowIdx < rowLoop + PRE_LAUNCH; rowIdx++) {
            if (rowIdx < rowLoop) {
                uint32_t rowOffset = rowIdx * rowNumTile;
                uint32_t gRowOffset = rowOffset + coreRowOffset;
                uint32_t rowNumLoop = (rowIdx == rowLoop - 1) ? (coreRowActual - rowOffset) : rowNumTile;

                WaitFlag<HardEvent::V_MTE2>(pingPong);
                DataCopy(
                    ls[pingPongOffset], gInput[layoutInput.GetOffset(MatrixCoord(gRowOffset, 0))],
                    DataCopyParams(
                        rowNumLoop, colNumRound / FLT_PER_DATABLOCK, (colNumPad - colNumRound) / FLT_PER_DATABLOCK, 0));
                SetFlag<HardEvent::MTE2_V>(pingPong);
            }

            pingPong ^= 1;
            pingPongOffset = pingPong ? MAX_UB_S_ELEM_NUM : 0;

            if (rowIdx >= PRE_LAUNCH) {
                uint32_t nowRowIdx = rowIdx - PRE_LAUNCH;
                uint32_t rowOffset = nowRowIdx * rowNumTile;
                uint32_t gRowOffset = rowOffset + coreRowOffset;
                uint32_t rowNumLoop = (nowRowIdx == rowLoop - 1) ? (coreRowActual - rowOffset) : rowNumTile;

                WaitFlag<HardEvent::MTE2_V>(pingPong);
                ScaleS(rowNumLoop, colNumRound);

                SubCoreCompute<false>(
                    gOutput[layoutOutput.GetOffset(MatrixCoord(gRowOffset, 0))],
                    layoutOutput.GetTileLayout(MatrixCoord(rowNumLoop, colNum)), rowOffset, isFirstStackTile,
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
        uint32_t colNumPad = layoutInput.stride(0);
        uint32_t maskStride = layoutMask.stride(0);

        uint32_t coreIdx = GetSubBlockIdx();
        uint32_t coreRowActual = (coreIdx == 1) ? (rowNum - rowNum / 2) : rowNum / 2;
        uint32_t coreRowOffset = coreIdx * (rowNum / 2);

        uint32_t tokenNumPerHeadThisSubBlock = Min(qSBlockSize, coreRowActual);
        uint32_t maskOffsetThisSubBlock = (qNBlockSize == 1) ? coreRowOffset : 0;

        // calc mask shift in GM
        uint32_t gmOffsetMaskRow;
        uint32_t gmOffsetMaskCol;
        uint32_t maskCol;
        uint32_t ubAddMaskOffset;
        if (triUp >= kvSStartIdx) {
            uint32_t triUpRoundDown = RoundDown(triUp, NUM_32);
            gmOffsetMaskRow = triUp - triUpRoundDown;
            gmOffsetMaskCol = 0;
            maskCol = kvSEndIdx - triUpRoundDown;
            ubAddMaskOffset = triUpRoundDown - kvSStartIdx;
        } else {
            gmOffsetMaskRow = 0;
            gmOffsetMaskCol = kvSStartIdx - triUp;
            maskCol = colNum;
            ubAddMaskOffset = 0;
        }
        uint32_t maskColRound = RoundUp(maskCol, NUM_32);

        auto gMaskThisSubBlock =
            gMask[layoutMask.GetOffset(MatrixCoord(gmOffsetMaskRow + maskOffsetThisSubBlock, gmOffsetMaskCol))];
        auto layoutMaskThisSubBlock = layoutMask;

        uint32_t maxRowNumPerLoop = MAX_UB_S_ELEM_NUM / colNumRound;
        uint32_t rowNumTile = RoundDown(maxRowNumPerLoop, DATABLOCK_PER_REPEAT);
        rowNumTile = Min(rowNumTile, FLT_PER_REPEAT);
        uint32_t rowLoop = CeilDiv(coreRowActual, rowNumTile);

        if (coreRowActual == 0) {
            Arch::CrossCoreWaitFlag(qkReady);
            return;
        }

        for (uint32_t rowIdx = 0; rowIdx < rowLoop + PRE_LAUNCH; rowIdx++) {
            if (rowIdx < rowLoop) {
                uint32_t rowOffset = rowIdx * rowNumTile;
                uint32_t gRowOffset = rowOffset + coreRowOffset;
                uint32_t rowNumLoop = (rowIdx == rowLoop - 1) ? (coreRowActual - rowOffset) : rowNumTile;

                // loop 0 mask load before cross core sync
                if (rowIdx == 0) {
                    // the token idx of the start token of the prologue part
                    uint32_t proTokenIdx = rowOffset % tokenNumPerHeadThisSubBlock;
                    // the token num of the prologue part
                    uint32_t proTokenNum =
                        Min(rowNumLoop, (tokenNumPerHeadThisSubBlock - proTokenIdx)) % tokenNumPerHeadThisSubBlock;
                    // the token num of the epilogue part
                    uint32_t integralHeadNum = (rowNumLoop - proTokenNum) / tokenNumPerHeadThisSubBlock;
                    // the number of integral heads within a cycle
                    uint32_t epiTokenNum = rowNumLoop - proTokenNum - integralHeadNum * tokenNumPerHeadThisSubBlock;
                    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                    CopyMaskToUb(
                        gMaskThisSubBlock, maskCol, maskColRound, maskStride, tokenNumPerHeadThisSubBlock, proTokenIdx,
                        proTokenNum, integralHeadNum, epiTokenNum);
                    SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
                    Arch::CrossCoreWaitFlag(qkReady);
                }

                WaitFlag<HardEvent::V_MTE2>(pingPong);
                DataCopy(
                    ls[pingPongOffset], gInput[layoutInput.GetOffset(MatrixCoord(gRowOffset, 0))],
                    DataCopyParams(
                        rowNumLoop, colNumRound / DATABLOCK_PER_REPEAT,
                        (colNumPad - colNumRound) / DATABLOCK_PER_REPEAT, 0));
                SetFlag<HardEvent::MTE2_V>(pingPong);
            }

            pingPong ^= 1;
            pingPongOffset = pingPong ? MAX_UB_S_ELEM_NUM : 0;

            if (rowIdx >= PRE_LAUNCH) {
                uint32_t nowRowIdx = rowIdx - PRE_LAUNCH;
                uint32_t rowOffset = nowRowIdx * rowNumTile;
                uint32_t rowNumLoop = (nowRowIdx == rowLoop - 1) ? (coreRowActual - rowOffset) : rowNumTile;

                if (nowRowIdx > 0) {
                    // the token idx of the start token of the prologue part
                    uint32_t proTokenIdx = rowOffset % tokenNumPerHeadThisSubBlock;
                    // the token num of the prologue part
                    uint32_t proTokenNum =
                        Min(rowNumLoop, (tokenNumPerHeadThisSubBlock - proTokenIdx)) % tokenNumPerHeadThisSubBlock;
                    // the token num of the epilogue part
                    uint32_t integralHeadNum = (rowNumLoop - proTokenNum) / tokenNumPerHeadThisSubBlock;
                    // the number of integral heads within a cycle
                    uint32_t epiTokenNum = rowNumLoop - proTokenNum - integralHeadNum * tokenNumPerHeadThisSubBlock;
                    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                    CopyMaskToUb(
                        gMaskThisSubBlock, maskCol, maskColRound, maskStride, tokenNumPerHeadThisSubBlock, proTokenIdx,
                        proTokenNum, integralHeadNum, epiTokenNum);
                    SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
                }

                WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
                Cast<half, ElementMask, false>(
                    mask16, mask, RoundMode::CAST_NONE, MASK_NUL, CeilDiv(rowNumLoop * colNumRound, HALF_PER_REPEAT),
                    UnaryRepeatParams(1, 1, 8, 4));
                PipeBarrier<PIPE_V>();
                Cast<float, half, false>(
                    mask32, mask16, RoundMode::CAST_NONE, MASK_NUL, CeilDiv(rowNumLoop * colNumRound, FLT_PER_REPEAT),
                    UnaryRepeatParams(1, 1, 8, 4));
                PipeBarrier<PIPE_V>();

                WaitFlag<HardEvent::MTE2_V>(pingPong);
                ScaleS(rowNumLoop, colNumRound);

                ApplyMask(rowNumLoop, colNumRound, maskColRound, ubAddMaskOffset);

                // online softmax vectorized compute
                uint32_t gRowOffset = rowOffset + coreRowOffset;
                SubCoreCompute<true>(
                    gOutput[layoutOutput.GetOffset(MatrixCoord(gRowOffset, 0))],
                    layoutOutput.GetTileLayout(MatrixCoord(rowNumLoop, colNum)), rowOffset, isFirstStackTile, 0,
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
    LocalTensor<float> dm;
    LocalTensor<float> lm;
    LocalTensor<float> hm;
    LocalTensor<float> ll;
    LocalTensor<float> tv;
    LocalTensor<float> tvb;
};
} // namespace Catlass::Epilogue::Block

#endif