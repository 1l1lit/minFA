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
template <class OutputType_, class InputType_, class MaskType_, LseModeT LSE_MODE_, bool HAS_SOFTCAP_>
class BlockEpilogue<EpilogueAtlasA2OnlineSoftmaxT<LSE_MODE_, HAS_SOFTCAP_>, OutputType_, InputType_, MaskType_> {
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

    CATLASS_DEVICE void init(Arch::Resource<ArchTag>& resource, float scaleVal_, float softcapVal_)
    {
        scaleVal = scaleVal_;
        softcapVal = softcapVal_;

        constexpr uint32_t LS_OFFSET = 0;
        constexpr uint32_t LP_OFFSET = 4 * UB_BLK_SIZE;
        constexpr uint32_t MASK_OFFSET = 4 * UB_BLK_SIZE;
        constexpr uint32_t MASK16_OFFSET = 5 * UB_BLK_SIZE;

        constexpr uint32_t TV_OFFSET = 10 * UB_BLK_SIZE;
        constexpr uint32_t TVB_OFFSET = 10 * UB_BLK_SIZE + 4 * UB_VEC_SIZE;
        constexpr uint32_t SC_OFFSET = 10 * UB_BLK_SIZE + 8 * UB_VEC_SIZE;
        constexpr uint32_t LM_OFFSET = 10 * UB_BLK_SIZE + 8 * UB_VEC_SIZE;
        constexpr uint32_t HM_OFFSET = 10 * UB_BLK_SIZE + 9 * UB_VEC_SIZE;
        constexpr uint32_t GM_OFFSET = 10 * UB_BLK_SIZE + 10 * UB_VEC_SIZE;
        constexpr uint32_t LL_OFFSET = 10 * UB_BLK_SIZE + 11 * UB_VEC_SIZE;
        constexpr uint32_t GL_OFFSET = 10 * UB_BLK_SIZE + 12 * UB_VEC_SIZE;
        constexpr uint32_t DM_OFFSET = 10 * UB_BLK_SIZE + 13 * UB_VEC_SIZE;

        ls = resource.ubBuf.template GetBufferByByte<float>(LS_OFFSET);
        lp = resource.ubBuf.template GetBufferByByte<ElementOutput>(LP_OFFSET);
        mask = resource.ubBuf.template GetBufferByByte<ElementMask>(MASK_OFFSET);
        mask32 = resource.ubBuf.template GetBufferByByte<float>(MASK_OFFSET);
        mask16 = resource.ubBuf.template GetBufferByByte<half>(MASK16_OFFSET);
        tv = resource.ubBuf.template GetBufferByByte<float>(TV_OFFSET);
        tvb = resource.ubBuf.template GetBufferByByte<float>(TVB_OFFSET);
        sc = resource.ubBuf.template GetBufferByByte<float>(SC_OFFSET);
        lm = resource.ubBuf.template GetBufferByByte<float>(LM_OFFSET);
        hm = resource.ubBuf.template GetBufferByByte<float>(HM_OFFSET);
        gm = resource.ubBuf.template GetBufferByByte<float>(GM_OFFSET);
        ll = resource.ubBuf.template GetBufferByByte<float>(LL_OFFSET);
        gl = resource.ubBuf.template GetBufferByByte<float>(GL_OFFSET);
        dm = resource.ubBuf.template GetBufferByByte<float>(DM_OFFSET);
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
        const LocalTensor<float>& dst, const LocalTensor<float>& src, uint32_t rowNumRound, uint32_t elemNum,
        uint32_t elemNumRound)
    {
        if (elemNum >= FLT_PER_REPEAT) {
            BlockReduceMax<float, false>(tv, src, rowNumRound, 0, 1, 1, elemNumRound / DATABLOCK_PER_REPEAT);
            PipeBarrier<PIPE_V>();

            BlockReduceMax<float, false>(dst, tv, CeilDiv(rowNumRound, FLT_PER_DATABLOCK), 0, 1, 1, 8);
            PipeBarrier<PIPE_V>();

            for (uint32_t idx = FLT_PER_REPEAT; idx < RoundDown(elemNum, FLT_PER_REPEAT); idx += FLT_PER_REPEAT) {
                BlockReduceMax<float, false>(tv, src[idx], rowNumRound, 0, 1, 1, elemNumRound / DATABLOCK_PER_REPEAT);
                PipeBarrier<PIPE_V>();

                BlockReduceMax<float, false>(tvb, tv, CeilDiv(rowNumRound, FLT_PER_DATABLOCK), 0, 1, 1, 8);
                PipeBarrier<PIPE_V>();

                SetVecMaskLen(rowNumRound);
                Max<float, false>(dst, dst, tvb, MASK_NUL, 1, BinaryRepeatParams());
                PipeBarrier<PIPE_V>();
                ResetMask();
            }
        }

        if (elemNum % FLT_PER_REPEAT > 0) {
            SetVecMaskLen(elemNum % FLT_PER_REPEAT);
            BlockReduceMax<float, false>(
                tv, src[RoundDown(elemNum, FLT_PER_REPEAT)], rowNumRound, 0, 1, 1, elemNumRound / DATABLOCK_PER_REPEAT);
            PipeBarrier<PIPE_V>();

            SetBlockReduceMaskLen(CeilDiv(elemNum % FLT_PER_REPEAT, DATABLOCK_PER_REPEAT));
            if (elemNum < FLT_PER_REPEAT) {
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
        uint32_t sOffset, uint32_t rowOffset, uint32_t rowNumRound, uint32_t colNum, uint32_t colNumRound)
    {
        if (colNum == 512) {
            RowMax512(lm[rowOffset], ls[sOffset], rowNumRound);
        } else if (colNum == 256) {
            RowMax256(lm[rowOffset], ls[sOffset], rowNumRound);
        } else {
            RowMaxTail(lm[rowOffset], ls[sOffset], rowNumRound, colNum, colNumRound);
        }
    }

    CATLASS_DEVICE void UpdateGlobalRowMax(
        uint32_t rowNum, uint32_t rowNumRound, uint32_t dmOffset, uint32_t rowOffset, uint32_t isFirstStackTile)
    {
        DataCopyParams copyParams(1, rowNumRound / DATABLOCK_PER_REPEAT /* blockLen */, 0, 0);

        if (isFirstStackTile) {
            DataCopy(hm[rowOffset], lm[rowOffset], copyParams);
            PipeBarrier<PIPE_V>();
        } else {
            SetVecMaskLen(rowNum);
            // hm = max(lm, gm)
            Max<float, false>(hm[rowOffset], lm[rowOffset], gm[rowOffset], MASK_NUL, 1, BinaryRepeatParams());
            PipeBarrier<PIPE_V>();
            // dm = gm - hm
            Sub<float, false>(dm[dmOffset], gm[rowOffset], hm[rowOffset], MASK_NUL, 1, BinaryRepeatParams());
            PipeBarrier<PIPE_V>();
            // dm = exp(dm)
            Exp<float, false>(dm[dmOffset], dm[dmOffset], MASK_NUL, 1, UnaryRepeatParams());
            PipeBarrier<PIPE_V>();
            ResetMask();
        }

        // gm = hm
        DataCopy(gm[rowOffset], hm[rowOffset], copyParams);
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
        const LocalTensor<float>& dst, const LocalTensor<float>& src, uint32_t rowNumRound, uint32_t elemNum,
        uint32_t elemNumRound)
    {
        if (elemNum >= FLT_PER_REPEAT) {
            BlockReduceSum<float, false>(tv, src, rowNumRound, 0, 1, 1, elemNumRound / DATABLOCK_PER_REPEAT);
            PipeBarrier<PIPE_V>();

            BlockReduceSum<float, false>(dst, tv, CeilDiv(rowNumRound, FLT_PER_DATABLOCK), 0, 1, 1, 8);
            PipeBarrier<PIPE_V>();

            for (uint32_t idx = FLT_PER_REPEAT; idx < RoundDown(elemNum, FLT_PER_REPEAT); idx += FLT_PER_REPEAT) {
                BlockReduceSum<float, false>(tv, src[idx], rowNumRound, 0, 1, 1, elemNumRound / DATABLOCK_PER_REPEAT);
                PipeBarrier<PIPE_V>();

                BlockReduceSum<float, false>(tvb, tv, CeilDiv(rowNumRound, FLT_PER_DATABLOCK), 0, 1, 1, 8);
                PipeBarrier<PIPE_V>();

                SetVecMaskLen(rowNumRound);
                Add<float, false>(dst, dst, tvb, MASK_NUL, 1, BinaryRepeatParams());
                PipeBarrier<PIPE_V>();
                ResetMask();
            }
        }

        if (elemNum % FLT_PER_REPEAT > 0) {
            SetVecMaskLen(elemNum % FLT_PER_REPEAT);
            BlockReduceSum<float, false>(
                tv, src[RoundDown(elemNum, FLT_PER_REPEAT)], rowNumRound, 0, 1, 1, elemNumRound / DATABLOCK_PER_REPEAT);
            PipeBarrier<PIPE_V>();

            SetBlockReduceMaskLen(CeilDiv(elemNum % FLT_PER_REPEAT, DATABLOCK_PER_REPEAT));
            if (elemNum < FLT_PER_REPEAT) {
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
        uint32_t sOffset, uint32_t rowOffset, uint32_t rowNumRound, uint32_t colNum, uint32_t colNumRound)
    {
        if (colNum == 512) {
            RowSum512(ll[rowOffset], ls[sOffset], rowNumRound);
        } else if (colNum == 256) {
            RowSum256(ll[rowOffset], ls[sOffset], rowNumRound);
        } else {
            RowSumTail(ll[rowOffset], ls[sOffset], rowNumRound, colNum, colNumRound);
        }
    }

    CATLASS_DEVICE void UpdateGlobalRowSum(
        uint32_t rowNum, uint32_t rowNumRound, uint32_t dmOffset, uint32_t rowOffset, uint32_t isFirstStackTile)
    {
        if (isFirstStackTile) {
            // gl = ll
            DataCopy(gl[rowOffset], ll[rowOffset], DataCopyParams(1, rowNumRound / DATABLOCK_PER_REPEAT, 0, 0));
            PipeBarrier<PIPE_V>();
        } else {
            SetVecMaskLen(rowNum);
            // gl = dm * gl
            Mul<float, false>(gl[rowOffset], dm[dmOffset], gl[rowOffset], MASK_NUL, 1, BinaryRepeatParams());
            PipeBarrier<PIPE_V>();
            // gl = gl + ll
            Add<float, false>(gl[rowOffset], gl[rowOffset], ll[rowOffset], MASK_NUL, 1, BinaryRepeatParams());
            PipeBarrier<PIPE_V>();
            ResetMask();
        }
    }

    CATLASS_DEVICE void CopyMaskToUb(
        GlobalTensor<ElementMask> gMask, uint32_t colNum, uint32_t colNumRound, uint32_t maskStride,
        uint32_t tokenNumPerHead, uint32_t proTokenIdx, uint32_t proTokenNum, uint32_t integralHeadNum,
        uint32_t epiTokenNum)
    {
        uint32_t blockLen = colNum * sizeof(ElementMask);
        uint32_t srcStride = (maskStride - colNum) * sizeof(ElementMask);
        DataCopyPadExtParams<ElementMask> padParams(false, 0, 0, 0);

        uint32_t innerRowOffset = 0;

        if (proTokenNum != 0) {
            DataCopyPad(
                mask[innerRowOffset], gMask[proTokenIdx * maskStride],
                DataCopyExtParams(proTokenNum /* blockCount */, blockLen, srcStride, 0, 0), padParams);
            innerRowOffset += proTokenNum * colNumRound;
        }

        for (uint32_t i = 0; i < integralHeadNum; i++) {
            DataCopyPad(
                mask[innerRowOffset], gMask,
                DataCopyExtParams(tokenNumPerHead /* blockCount */, blockLen, srcStride, 0, 0), padParams);
            innerRowOffset += tokenNumPerHead * colNumRound;
        }

        if (epiTokenNum != 0) {
            DataCopyPad(
                mask[innerRowOffset], gMask, DataCopyExtParams(epiTokenNum /* blockCount */, blockLen, srcStride, 0, 0),
                padParams);
        }
    }

    CATLASS_DEVICE void ApplyMask(
        const LocalTensor<float>& src, uint32_t rowNum, uint32_t colNumRound, uint32_t maskColumnRound,
        uint32_t addMaskOffset)
    {
        Muls<float, false>(
            mask32, mask32, -3e38f /* -FLT_MAX */, MASK_NUL, CeilDiv(rowNum * maskColumnRound, FLT_PER_REPEAT),
            UnaryRepeatParams());
        PipeBarrier<PIPE_V>();

        if (maskColumnRound == colNumRound) {
            Add<float, false>(
                src, src, mask32, MASK_NUL, CeilDiv(rowNum * maskColumnRound, FLT_PER_REPEAT), BinaryRepeatParams());
        } else {
            for (uint32_t idx = 0; idx < RoundDown(maskColumnRound, FLT_PER_REPEAT); idx += FLT_PER_REPEAT) {
                Add<float, false>(
                    src[addMaskOffset + idx], src[addMaskOffset + idx], mask32[idx], MASK_NUL, rowNum,
                    BinaryRepeatParams(
                        1, 1, 1, colNumRound / DATABLOCK_PER_REPEAT, colNumRound / DATABLOCK_PER_REPEAT,
                        maskColumnRound / DATABLOCK_PER_REPEAT));
            }

            if (maskColumnRound % FLT_PER_REPEAT > 0) {
                SetVecMaskLen(maskColumnRound % FLT_PER_REPEAT);
                Add<float, false>(
                    src[addMaskOffset + RoundDown(maskColumnRound, FLT_PER_REPEAT)],
                    src[addMaskOffset + RoundDown(maskColumnRound, FLT_PER_REPEAT)],
                    mask32[RoundDown(maskColumnRound, FLT_PER_REPEAT)], MASK_NUL, rowNum,
                    BinaryRepeatParams(
                        1, 1, 1, colNumRound / DATABLOCK_PER_REPEAT, colNumRound / DATABLOCK_PER_REPEAT,
                        maskColumnRound / DATABLOCK_PER_REPEAT));
                ResetMask();
            }
        }
        PipeBarrier<PIPE_V>();
    }

    CATLASS_DEVICE void CalcExp(
        const LocalTensor<float>& src, uint32_t rowNum, uint32_t rowNumRound, uint32_t colNum, uint32_t colNumRound,
        uint32_t rowOffset)
    {
        // hm_block = expand_to_block(hm), 存放于 tv
        Brcb(tv, hm[rowOffset], rowNumRound / DATABLOCK_PER_REPEAT, BrcbRepeatParams());
        PipeBarrier<PIPE_V>();

        // ls = ls - hm_block
        for (uint32_t idx = 0; idx < RoundDown(colNum, FLT_PER_REPEAT); idx += FLT_PER_REPEAT) {
            Sub<float, false>(
                src[idx], src[idx], tv, MASK_NUL, rowNum,
                BinaryRepeatParams(1, 1, 0, colNumRound / DATABLOCK_PER_REPEAT, colNumRound / DATABLOCK_PER_REPEAT, 1));
        }

        if (colNum % FLT_PER_REPEAT > 0) {
            SetVecMaskLen(colNum % FLT_PER_REPEAT);
            Sub<float, false>(
                src[RoundDown(colNum, FLT_PER_REPEAT)], src[RoundDown(colNum, FLT_PER_REPEAT)], tv, MASK_NUL, rowNum,
                BinaryRepeatParams(1, 1, 0, colNumRound / DATABLOCK_PER_REPEAT, colNumRound / DATABLOCK_PER_REPEAT, 1));
            ResetMask();
        }
        PipeBarrier<PIPE_V>();

        // ls = exp(ls)
        Exp<float, false>(src, src, MASK_NUL, CeilDiv(rowNum * colNumRound, FLT_PER_REPEAT), UnaryRepeatParams());
        PipeBarrier<PIPE_V>();
    }

    CATLASS_DEVICE void ApplySoftcap(const LocalTensor<float>& src, uint32_t rowNum, uint32_t colNumRound)
    {
        uint32_t repeatTimes = CeilDiv(rowNum * colNumRound, FLT_PER_REPEAT);

        // Clip the input to avoid overflow
        Maxs<float, false>(src, src, -8.8f, MASK_NUL, repeatTimes, UnaryRepeatParams());
        PipeBarrier<PIPE_V>();

        // (2 * softcap) / (1 + exp(-2x)) - softcap
        Muls<float, false>(src, src, -2.0f, MASK_NUL, repeatTimes, UnaryRepeatParams());
        PipeBarrier<PIPE_V>();
        Exp<float, false>(src, src, MASK_NUL, repeatTimes, UnaryRepeatParams());
        PipeBarrier<PIPE_V>();
        Adds<float, false>(src, src, 1.0f, MASK_NUL, repeatTimes, UnaryRepeatParams());

        Duplicate<float, false>(sc, 2.0f * softcapVal, MASK_NUL, 1, 1, 8);
        PipeBarrier<PIPE_V>();
        Div<float, false>(src, sc, src, MASK_NUL, repeatTimes, BinaryRepeatParams(1, 1, 1, 8, 0, 8));
        PipeBarrier<PIPE_V>();
        Adds<float, false>(src, src, -softcapVal, MASK_NUL, repeatTimes, UnaryRepeatParams());
        PipeBarrier<PIPE_V>();
    }

    template <bool doTriUMask>
    CATLASS_DEVICE void SubCoreCompute(
        GlobalTensor<ElementOutput> gOutput, const LayoutOutput& layoutOutput, uint32_t rowOffset,
        uint32_t isFirstStackTile, uint32_t isLastNoMaskStackTile, uint32_t isFirstRowLoop, uint32_t isLastRowLoop,
        uint32_t colNumRound, uint32_t sPingPong, uint32_t curStackTileMod)
    {
        uint32_t rowNumCurLoop = layoutOutput.shape(0);
        uint32_t rowNumCurLoopRound = RoundUp(rowNumCurLoop, DATABLOCK_PER_REPEAT);
        uint32_t colNum = layoutOutput.shape(1);
        uint32_t colNumPad = layoutOutput.stride(0);
        uint32_t sOffset = sPingPong * MAX_UB_S_ELEM_NUM;
        uint32_t dmOffsetCurCycle = curStackTileMod * MAX_ROW_NUM_SUB_CORE + rowOffset;

        if constexpr (LSE_MODE_ == LseModeT::OUT_ONLY) {
            // In lse out-only mode, tv is used in the last stack tile to transport lse
            if (isFirstStackTile && isFirstRowLoop) {
                WaitFlag<HardEvent::MTE3_V>(EVENT_ID4);
            }
        }

        CalcLocalRowMax(sOffset, rowOffset, rowNumCurLoopRound, colNum, colNumRound);
        UpdateGlobalRowMax(rowNumCurLoop, rowNumCurLoopRound, dmOffsetCurCycle, rowOffset, isFirstStackTile);

        CalcExp(ls[sOffset], rowNumCurLoop, rowNumCurLoopRound, colNum, colNumRound, rowOffset);

        if constexpr (!doTriUMask) {
            WaitFlag<HardEvent::MTE3_V>(sPingPong);
        }

        // lp = castfp32to16(ls)
        RoundMode roundMode =
            std::is_same<ElementOutput, bfloat16_t>::value ? RoundMode::CAST_RINT : RoundMode::CAST_NONE;
        Cast<ElementOutput, float, false>(
            lp[sOffset], ls[sOffset], roundMode, MASK_NUL, CeilDiv(rowNumCurLoop * colNumRound, FLT_PER_REPEAT),
            UnaryRepeatParams(1, 1, 4, 8));
        SetFlag<HardEvent::V_MTE3>(sPingPong);

        CalcLocalRowSum(sOffset, rowOffset, rowNumCurLoopRound, colNum, colNumRound);
        SetFlag<HardEvent::V_MTE2>(sPingPong);

        WaitFlag<HardEvent::V_MTE3>(sPingPong);
        DataCopy(
            gOutput, lp[sOffset],
            DataCopyParams(rowNumCurLoop, colNumRound / BLOCK_SIZE, 0, (colNumPad - colNumRound) / BLOCK_SIZE));

        if constexpr (!doTriUMask) {
            SetFlag<HardEvent::MTE3_V>(sPingPong);
            if (isLastNoMaskStackTile && isLastRowLoop) {
                WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            }
        } else {
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }

        UpdateGlobalRowSum(rowNumCurLoop, rowNumCurLoopRound, dmOffsetCurCycle, rowOffset, isFirstStackTile);
    }

    CATLASS_DEVICE void operator()(
        GlobalTensor<ElementOutput> gOutput, GlobalTensor<ElementInput> gInput, const LayoutOutput& layoutOutput,
        const LayoutInput& layoutInput, GemmCoord actualBlockShape, uint32_t isFirstStackTile,
        uint32_t isLastNoMaskStackTile, uint32_t qSBlockSize, uint32_t qNBlockSize, uint32_t curStackTileMod)
    {
        uint32_t rowNum = actualBlockShape.m();
        uint32_t colNum = actualBlockShape.n();
        uint32_t colNumRound = RoundUp(colNum, BLOCK_SIZE);
        uint32_t colNumPad = layoutInput.stride(0);

        uint32_t subcoreIdx = GetSubBlockIdx();
        uint32_t subcoreNum = GetSubBlockNum();

        uint32_t qNSplitSubBlock = qNBlockSize / subcoreNum;
        uint32_t qNThisSubBlock = (qNBlockSize == 1)  ? 0
                                  : (subcoreIdx == 1) ? (qNBlockSize - qNSplitSubBlock)
                                                      : qNSplitSubBlock;
        uint32_t rowSplitSubBlock = (qNBlockSize == 1) ? (qSBlockSize / 2) : (qSBlockSize * qNSplitSubBlock);
        uint32_t rowActualThisSubBlock = (subcoreIdx == 1) ? (rowNum - rowSplitSubBlock) : rowSplitSubBlock;
        uint32_t rowOffsetThisSubBlock = subcoreIdx * rowSplitSubBlock;
        uint32_t maxRowNumPerLoop = MAX_UB_S_ELEM_NUM / colNumRound;
        uint32_t rowNumTile = RoundDown(maxRowNumPerLoop, DATABLOCK_PER_REPEAT);
        rowNumTile = Min(rowNumTile, FLT_PER_REPEAT);
        uint32_t rowLoopNum = CeilDiv(rowActualThisSubBlock, rowNumTile);
        uint32_t preLoad = 1;

        for (uint32_t rowLoopIdx = 0; rowLoopIdx < rowLoopNum + preLoad; rowLoopIdx++) {
            if (rowLoopIdx < rowLoopNum) {
                uint32_t sPingPong = rowLoopIdx % 2;
                uint32_t sOffset = sPingPong * MAX_UB_S_ELEM_NUM;
                uint32_t rowOffsetCurLoop = rowLoopIdx * rowNumTile;
                uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                uint32_t rowNumCurLoop =
                    (rowLoopIdx == rowLoopNum - 1) ? (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;

                auto gInputCurLoop = gInput[layoutInput.GetOffset(MatrixCoord(rowOffsetIoGm, 0))];

                WaitFlag<HardEvent::V_MTE2>(sPingPong);
                DataCopy(
                    ls[sOffset], gInputCurLoop,
                    DataCopyParams(
                        rowNumCurLoop, colNumRound / DATABLOCK_PER_REPEAT,
                        (colNumPad - colNumRound) / DATABLOCK_PER_REPEAT, 0));
                SetFlag<HardEvent::MTE2_V>(sPingPong);
            }

            if (rowLoopIdx >= preLoad) {
                uint32_t delayedRowLoopIdx = rowLoopIdx - preLoad;
                uint32_t sPingPong = delayedRowLoopIdx % 2;
                uint32_t sOffset = sPingPong * MAX_UB_S_ELEM_NUM;
                uint32_t rowOffsetCurLoop = delayedRowLoopIdx * rowNumTile;
                uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                uint32_t rowNumCurLoop =
                    (delayedRowLoopIdx == rowLoopNum - 1) ? (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;

                int64_t offsetOutput = layoutOutput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
                auto gOutputCurLoop = gOutput[offsetOutput];
                auto layoutOutputCurLoop = layoutOutput.GetTileLayout(MatrixCoord(rowNumCurLoop, colNum));
                WaitFlag<HardEvent::MTE2_V>(sPingPong);
                Muls<float, false>(
                    ls[sOffset], ls[sOffset], scaleVal, MASK_NUL, CeilDiv(rowNumCurLoop * colNumRound, FLT_PER_REPEAT),
                    UnaryRepeatParams());
                PipeBarrier<PIPE_V>();

                if constexpr (HAS_SOFTCAP_) {
                    ApplySoftcap(ls[sOffset], rowNumCurLoop, colNumRound);
                }

                SubCoreCompute<false>(
                    gOutputCurLoop, layoutOutputCurLoop, rowOffsetCurLoop, isFirstStackTile, isLastNoMaskStackTile,
                    (delayedRowLoopIdx == 0), (delayedRowLoopIdx == rowLoopNum - 1), colNumRound, sPingPong,
                    curStackTileMod);
            }
        }
    }

    CATLASS_DEVICE void operator()(
        GlobalTensor<ElementOutput> gOutput, GlobalTensor<ElementInput> gInput, GlobalTensor<ElementMask> gMask,
        const LayoutOutput& layoutOutput, const LayoutInput& layoutInput, const LayoutInput& layoutMask,
        GemmCoord actualBlockShape, uint32_t isFirstStackTile, uint32_t qSBlockSize, uint32_t qNBlockSize,
        uint32_t curStackTileMod, Arch::CrossCoreFlag qkReady, uint32_t triUp, uint32_t triDown, uint32_t kvSStartIdx,
        uint32_t kvSEndIdx)
    {
        uint32_t rowNum = actualBlockShape.m();
        uint32_t colNum = actualBlockShape.n();
        uint32_t colNumRound = RoundUp(colNum, NUM_32);
        uint32_t colNumPad = layoutInput.stride(0);
        uint32_t maskStride = layoutMask.stride(0);

        uint32_t subcoreIdx = GetSubBlockIdx();
        uint32_t subcoreNum = GetSubBlockNum();

        uint32_t qNSplitSubBlock = qNBlockSize / subcoreNum;
        uint32_t qNThisSubBlock = (qNBlockSize == 1)  ? 0
                                  : (subcoreIdx == 1) ? (qNBlockSize - qNSplitSubBlock)
                                                      : qNSplitSubBlock;
        uint32_t rowSplitSubBlock = (qNBlockSize == 1) ? (qSBlockSize / 2) : (qSBlockSize * qNSplitSubBlock);
        uint32_t rowActualThisSubBlock = (subcoreIdx == 1) ? (rowNum - rowSplitSubBlock) : rowSplitSubBlock;
        uint32_t rowOffsetThisSubBlock = subcoreIdx * rowSplitSubBlock;

        uint32_t tokenNumPerHeadThisSubBlock = Min(qSBlockSize, rowActualThisSubBlock);
        uint32_t maskOffsetThisSubBlock = (qNBlockSize == 1) ? rowOffsetThisSubBlock : 0;

        // calc mask shift in gm
        uint32_t gmOffsetMaskRow;
        uint32_t gmOffsetMaskColumn;
        uint32_t maskColumn;
        uint32_t ubAddMaskOffset;
        if (triUp >= kvSStartIdx) {
            uint32_t triUpRoundDown = RoundDown(triUp, NUM_32);
            gmOffsetMaskRow = triUp - triUpRoundDown;
            gmOffsetMaskColumn = 0;
            maskColumn = kvSEndIdx - triUpRoundDown;
            ubAddMaskOffset = triUpRoundDown - kvSStartIdx;
        } else {
            gmOffsetMaskRow = 0;
            gmOffsetMaskColumn = kvSStartIdx - triUp;
            maskColumn = colNum;
            ubAddMaskOffset = 0;
        }
        uint32_t maskColumnRound = RoundUp(maskColumn, NUM_32);

        auto gMaskThisSubBlock =
            gMask[layoutMask.GetOffset(MatrixCoord(gmOffsetMaskRow + maskOffsetThisSubBlock, gmOffsetMaskColumn))];
        auto layoutMaskThisSubBlock = layoutMask;

        uint32_t maxRowNumPerLoop = MAX_UB_S_ELEM_NUM / colNumRound;
        uint32_t rowNumTile = RoundDown(maxRowNumPerLoop, DATABLOCK_PER_REPEAT);
        rowNumTile = Min(rowNumTile, FLT_PER_REPEAT);
        uint32_t rowLoopNum = CeilDiv(rowActualThisSubBlock, rowNumTile);
        uint32_t preLoad = 1;

        if (rowActualThisSubBlock == 0) {
            Arch::CrossCoreWaitFlag(qkReady);
            return;
        }

        for (uint32_t rowLoopIdx = 0; rowLoopIdx < rowLoopNum + preLoad; rowLoopIdx++) {
            if (rowLoopIdx < rowLoopNum) {
                uint32_t sPingPong = rowLoopIdx % 2;
                uint32_t sOffset = sPingPong * MAX_UB_S_ELEM_NUM;
                uint32_t rowOffsetCurLoop = rowLoopIdx * rowNumTile;
                uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                uint32_t rowNumCurLoop =
                    (rowLoopIdx == rowLoopNum - 1) ? (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;

                // loop 0 mask load before cross core sync
                if (rowLoopIdx == 0) {
                    // the token idx of the start token of the prologue part
                    uint32_t proTokenIdx = rowOffsetCurLoop % tokenNumPerHeadThisSubBlock;
                    // the token num of the prologue part
                    uint32_t proTokenNum =
                        Min(rowNumCurLoop, (tokenNumPerHeadThisSubBlock - proTokenIdx)) % tokenNumPerHeadThisSubBlock;
                    // the token num of the epilogue part
                    uint32_t integralHeadNum = (rowNumCurLoop - proTokenNum) / tokenNumPerHeadThisSubBlock;
                    // the number of integral heads within a cycle
                    uint32_t epiTokenNum = rowNumCurLoop - proTokenNum - integralHeadNum * tokenNumPerHeadThisSubBlock;
                    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                    CopyMaskToUb(
                        gMaskThisSubBlock, maskColumn, maskColumnRound, maskStride, tokenNumPerHeadThisSubBlock,
                        proTokenIdx, proTokenNum, integralHeadNum, epiTokenNum);
                    SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
                    Arch::CrossCoreWaitFlag(qkReady);
                }

                int64_t offsetInput = layoutInput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
                auto gInputCurLoop = gInput[offsetInput];
                WaitFlag<HardEvent::V_MTE2>(sPingPong);
                DataCopy(
                    ls[sOffset], gInputCurLoop,
                    DataCopyParams(
                        rowNumCurLoop, colNumRound / DATABLOCK_PER_REPEAT,
                        (colNumPad - colNumRound) / DATABLOCK_PER_REPEAT, 0));
                SetFlag<HardEvent::MTE2_V>(sPingPong);
            }

            if (rowLoopIdx >= preLoad) {
                uint32_t delayedRowLoopIdx = rowLoopIdx - preLoad;
                uint32_t sPingPong = delayedRowLoopIdx % 2;
                uint32_t sOffset = sPingPong * MAX_UB_S_ELEM_NUM;
                uint32_t rowOffsetCurLoop = delayedRowLoopIdx * rowNumTile;
                uint32_t rowNumCurLoop =
                    (delayedRowLoopIdx == rowLoopNum - 1) ? (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;

                WaitFlag<HardEvent::MTE2_V>(EVENT_ID2);
                Cast<half, ElementMask, false>(
                    mask16, mask, RoundMode::CAST_NONE, MASK_NUL,
                    CeilDiv(rowNumCurLoop * colNumRound, REPEAT_SIZE / sizeof(half)), UnaryRepeatParams(1, 1, 8, 4));
                PipeBarrier<PIPE_V>();

                Cast<float, half, false>(
                    mask32, mask16, RoundMode::CAST_NONE, MASK_NUL,
                    CeilDiv(rowNumCurLoop * colNumRound, REPEAT_SIZE / sizeof(float)), UnaryRepeatParams(1, 1, 8, 4));
                PipeBarrier<PIPE_V>();

                WaitFlag<HardEvent::MTE2_V>(sPingPong);
                Muls<float, false>(
                    ls[sOffset], ls[sOffset], scaleVal, MASK_NUL, CeilDiv(rowNumCurLoop * colNumRound, FLT_PER_REPEAT),
                    UnaryRepeatParams());
                PipeBarrier<PIPE_V>();

                if constexpr (HAS_SOFTCAP_) {
                    ApplySoftcap(ls[sOffset], rowNumCurLoop, colNumRound);
                }

                ApplyMask(ls[sOffset], rowNumCurLoop, colNumRound, maskColumnRound, ubAddMaskOffset);

                // next loop mask load
                if (rowLoopIdx < rowLoopNum) {
                    uint32_t rowOffsetCurLoop = rowLoopIdx * rowNumTile;
                    uint32_t rowNumCurLoop =
                        (rowLoopIdx == rowLoopNum - 1) ? (rowActualThisSubBlock - rowOffsetCurLoop) : rowNumTile;
                    // the token idx of the start token of the prologue part
                    uint32_t proTokenIdx = rowOffsetCurLoop % tokenNumPerHeadThisSubBlock;
                    // the token num of the prologue part
                    uint32_t proTokenNum =
                        Min(rowNumCurLoop, (tokenNumPerHeadThisSubBlock - proTokenIdx)) % tokenNumPerHeadThisSubBlock;
                    // the number of integral heads within a cycle
                    uint32_t integralHeadNum = (rowNumCurLoop - proTokenNum) / tokenNumPerHeadThisSubBlock;
                    // the token num of the epilogue part
                    uint32_t epiTokenNum = rowNumCurLoop - proTokenNum - integralHeadNum * tokenNumPerHeadThisSubBlock;
                    WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
                    CopyMaskToUb(
                        gMaskThisSubBlock, maskColumn, maskColumnRound, maskStride, tokenNumPerHeadThisSubBlock,
                        proTokenIdx, proTokenNum, integralHeadNum, epiTokenNum);
                    SetFlag<HardEvent::MTE2_V>(EVENT_ID2);
                }

                // online softmax vectorized compute
                uint32_t rowOffsetIoGm = rowOffsetCurLoop + rowOffsetThisSubBlock;
                int64_t offsetOutput = layoutOutput.GetOffset(MatrixCoord(rowOffsetIoGm, 0));
                auto gOutputCurLoop = gOutput[offsetOutput];
                auto layoutOutputCurLoop = layoutOutput.GetTileLayout(MatrixCoord(rowNumCurLoop, colNum));
                SubCoreCompute<true>(
                    gOutputCurLoop, layoutOutputCurLoop, rowOffsetCurLoop, isFirstStackTile, 0,
                    (delayedRowLoopIdx == 0), (delayedRowLoopIdx == rowLoopNum - 1), colNumRound, sPingPong,
                    curStackTileMod);
            }
        }
    }

private:
    float scaleVal;
    float softcapVal;

    LocalTensor<float> ls;
    LocalTensor<ElementOutput> lp;
    LocalTensor<ElementMask> mask;
    LocalTensor<half> mask16;
    LocalTensor<float> mask32;
    LocalTensor<float> lm;
    LocalTensor<float> hm;
    LocalTensor<float> gm;
    LocalTensor<float> dm;
    LocalTensor<float> ll;
    LocalTensor<float> tv;
    LocalTensor<float> tvb;
    LocalTensor<float> gl;
    LocalTensor<float> sc;
};
} // namespace Catlass::Epilogue::Block

#endif