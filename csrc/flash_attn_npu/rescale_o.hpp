/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

#ifndef CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_NO_SPLIT_ROW_HPP_T
#define CATLASS_EPILOGUE_BLOCK_BLOCK_EPILOGUE_RESCALE_O_NO_SPLIT_ROW_HPP_T

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
template <class OutputType_, class InputType_, class UpdateType_, class LseType_>
class BlockEpilogue<EpilogueAtlasA2RescaleOT, OutputType_, InputType_, UpdateType_, LseType_> {
public:
    using ArchTag = Arch::AtlasA2;

    using ElementOutput = typename OutputType_::Element;
    using ElementInput = typename InputType_::Element;
    using ElementUpdate = typename UpdateType_::Element;
    using ElementLse = typename LseType_::Element;

    using LayoutOutput = typename OutputType_::Layout;
    using LayoutInput = typename InputType_::Layout;
    using LayoutUpdate = typename UpdateType_::Layout;
    using LayoutLse = typename LseType_::Layout;

    CATLASS_DEVICE void init(const Arch::Resource<ArchTag>& resource)
    {
        constexpr uint32_t LO_OFFSET = 6 * UB_BLK_SIZE;
        constexpr uint32_t GO_OFFSET = 8 * UB_BLK_SIZE;

        constexpr uint32_t GM_OFFSET = 10 * UB_BLK_SIZE;
        constexpr uint32_t GL_OFFSET = 10 * UB_BLK_SIZE + 4 * UB_RPT_SIZE;
        constexpr uint32_t LSE_OFFSET = 10 * UB_BLK_SIZE + 8 * UB_RPT_SIZE;

        constexpr uint32_t DM_OFFSET = 11 * UB_BLK_SIZE + 4 * UB_RPT_SIZE;
        constexpr uint32_t TV_OFFSET = 11 * UB_BLK_SIZE + 12 * UB_VEC_SIZE;

        lo = resource.ubBuf.template GetBufferByByte<float>(LO_OFFSET);
        go16 = resource.ubBuf.template GetBufferByByte<ElementOutput>(GO_OFFSET);
        go32 = resource.ubBuf.template GetBufferByByte<float>(GO_OFFSET);
        gm = resource.ubBuf.template GetBufferByByte<float>(GM_OFFSET);
        gl = resource.ubBuf.template GetBufferByByte<float>(GL_OFFSET);
        lse = resource.ubBuf.template GetBufferByByte<float>(LSE_OFFSET);
        dm = resource.ubBuf.template GetBufferByByte<float>(DM_OFFSET);
        tv = resource.ubBuf.template GetBufferByByte<float>(TV_OFFSET);
    }

    CATLASS_DEVICE void CopyOToGm(
        const GlobalTensor<ElementOutput>& gOutput, uint32_t rowNum, uint32_t rowOffset, uint32_t qSOffset,
        uint32_t qSBlockSize, uint32_t embed, uint32_t embedRound, uint32_t strideO)
    {
        uint32_t qSBlockIdx = (qSOffset + rowOffset) % qSBlockSize;
        uint32_t qNBlockIdx = (qSOffset + rowOffset) / qSBlockSize;
        uint32_t rowsDone = 0;
        while (rowsDone < rowNum) {
            uint32_t rowsCurHead = Min(rowNum - rowsDone, qSBlockSize - qSBlockIdx);
            DataCopyPad(
                gOutput[qSBlockIdx * strideO + qNBlockIdx * embed], go16[rowsDone * embedRound],
                DataCopyExtParams(
                    rowsCurHead, embed * sizeof(ElementOutput), 0, (strideO - embed) * sizeof(ElementOutput), 0));
            rowsDone += rowsCurHead;
            qSBlockIdx = 0;
            qNBlockIdx++;
        }
    }

    CATLASS_DEVICE void CopyLseToGm(
        const GlobalTensor<ElementLse>& gLse, const GlobalTensor<ElementUpdate>& gUpdate, uint32_t rowNum,
        uint32_t headNum, uint32_t qSOffset, uint32_t qSBlockSize, uint32_t strideLse)
    {
        if (headNum == 1) {
            DataCopyPad(gLse, lse, DataCopyExtParams(1, rowNum * sizeof(float), 0, 0, 0));
            return;
        }

        // *** PreToken ***
        uint32_t preTokenNum = 0;
        uint32_t gmOffset = 0;
        uint32_t ubOffset = 0;
        uint32_t gmAlign = 0;
        if (qSOffset > 0) {
            preTokenNum = qSBlockSize - qSOffset;
            DataCopyPad(gLse, lse, DataCopyExtParams(1, preTokenNum * sizeof(float), 0, 0, 0));
            headNum--;

            gmOffset = strideLse - qSOffset;
            ubOffset = RoundDown<FLT_PER_DATABLOCK>(preTokenNum);
            gmAlign = preTokenNum - ubOffset;
        }

        // *** Align ***
        uint32_t nowRowNum = rowNum - preTokenNum;
        bool preTokenNotAlign = (gmAlign > 0);
        bool qSBlockSizeNotAlign = (qSBlockSize % FLT_PER_DATABLOCK > 0);
        if (preTokenNotAlign || qSBlockSizeNotAlign) {
            // ** UB -> GM **
            DataCopy(gUpdate, lse[ubOffset], DataCopyParams(1, CeilDiv(rowNum - ubOffset, FLT_PER_DATABLOCK), 0, 0));
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            // ** GM -> UB **
            if (qSBlockSizeNotAlign) {
                DataCopyPad(
                    lse[ubOffset], gUpdate[gmAlign], DataCopyExtParams(headNum, qSBlockSize * sizeof(float), 0, 0, 0),
                    DataCopyPadExtParams<ElementLse>());
            } else {
                DataCopy(
                    lse[ubOffset], gUpdate[gmAlign], DataCopyParams(1, CeilDiv(nowRowNum, FLT_PER_DATABLOCK), 0, 0));
            }
            SetFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_MTE3>(EVENT_ID0);
        }

        // *** IntegralHead ***
        uint32_t epiTokenNum = nowRowNum % qSBlockSize;
        headNum = epiTokenNum ? (headNum - 1) : headNum;
        DataCopyPad(
            gLse[gmOffset], lse[ubOffset],
            DataCopyExtParams(headNum, qSBlockSize * sizeof(float), 0, (strideLse - qSBlockSize) * sizeof(float), 0));

        // *** EpiToken ***
        if (epiTokenNum) {
            DataCopyPad(
                gLse[gmOffset + headNum * strideLse], lse[ubOffset + headNum * RoundUp<FLT_PER_DATABLOCK>(qSBlockSize)],
                DataCopyExtParams(1, epiTokenNum * sizeof(float), 0, 0, 0));
        }
    }

    CATLASS_DEVICE void SubCoreCompute(
        const GlobalTensor<ElementOutput>& gOutput, const GlobalTensor<ElementInput>& gInput,
        const GlobalTensor<ElementUpdate>& gUpdate, const GlobalTensor<ElementLse>& gLse,
        const LayoutInput& layoutInput, uint32_t coreRow, uint32_t isFirstStackTile, uint32_t isLastStackTile,
        uint32_t curStackTileMod, uint32_t multipleRowLoop, uint32_t isLastRowLoop, uint32_t rowOffset,
        uint32_t qSBlockSize, uint32_t qSOffset, uint32_t coreHeadNum, uint32_t strideLse, uint32_t strideO,
        uint32_t rowBase)
    {
        uint32_t rowNum = layoutInput.shape(0);
        uint32_t embed = layoutInput.shape(1);
        uint32_t embedRound = layoutInput.stride(0);
        uint32_t dmOffset = curStackTileMod * MAX_ROW_NUM_SUB_CORE + rowBase + rowOffset;

        if (isFirstStackTile) {
            // go = lo
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            DataCopy(go32, gInput, DataCopyParams(1, rowNum * embedRound / FLT_PER_DATABLOCK, 0, 0));
            SetFlag<HardEvent::MTE2_V>(EVENT_ID3);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);
        } else {
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
            DataCopy(lo, gInput, DataCopyParams(1, rowNum * embedRound / FLT_PER_DATABLOCK, 0, 0));
            SetFlag<HardEvent::MTE2_V>(EVENT_ID3);

            // dm_block = expand_to_block(dm)
            Brcb(tv, dm[dmOffset], CeilDiv<FLT_PER_DATABLOCK>(rowNum), BrcbRepeatParams());
            PipeBarrier<PIPE_V>();

            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            if (multipleRowLoop) {
                DataCopy(go32, gUpdate, DataCopyParams(1, rowNum * embedRound / FLT_PER_DATABLOCK, 0, 0));
                SetFlag<HardEvent::MTE2_V>(EVENT_ID4);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID4);
            }

            // go = go * dm_block
            for (uint32_t idx = 0; idx < embed / FLT_PER_REPEAT; idx++) {
                Mul<float, false>(
                    go32[idx * FLT_PER_REPEAT], go32[idx * FLT_PER_REPEAT], tv, MASK_NUL, rowNum,
                    BinaryRepeatParams(1, 1, 0, embedRound / FLT_PER_DATABLOCK, embedRound / FLT_PER_DATABLOCK, 1));
            }
            if (embed % FLT_PER_REPEAT > 0) {
                SetVecMaskLen(embed % FLT_PER_REPEAT);
                Mul<float, false>(
                    go32[RoundDown<FLT_PER_REPEAT>(embed)], go32[RoundDown<FLT_PER_REPEAT>(embed)], tv, MASK_NUL,
                    rowNum,
                    BinaryRepeatParams(1, 1, 0, embedRound / FLT_PER_DATABLOCK, embedRound / FLT_PER_DATABLOCK, 1));
                ResetMask();
            }
            PipeBarrier<PIPE_V>();

            // go = lo + go
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID3);
            Add<float, false>(
                go32, go32, lo, MASK_NUL, CeilDiv<FLT_PER_REPEAT>(rowNum * embedRound), BinaryRepeatParams());
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(EVENT_ID2);
        }

        if (isLastStackTile) {
            // gl_block = expand_to_block(gl)
            Brcb(tv, gl[taskOffset][rowOffset], CeilDiv<FLT_PER_DATABLOCK>(rowNum), BrcbRepeatParams());
            PipeBarrier<PIPE_V>();

            // go = go / gl_block
            for (uint32_t idx = 0; idx < embed / FLT_PER_REPEAT; idx++) {
                Div<float, false>(
                    go32[idx * FLT_PER_REPEAT], go32[idx * FLT_PER_REPEAT], tv, MASK_NUL, rowNum,
                    BinaryRepeatParams(1, 1, 0, embedRound / FLT_PER_DATABLOCK, embedRound / FLT_PER_DATABLOCK, 1));
            }
            if (embed % FLT_PER_REPEAT > 0) {
                SetVecMaskLen(embed % FLT_PER_REPEAT);
                Div<float, false>(
                    go32[RoundDown<FLT_PER_REPEAT>(embed)], go32[RoundDown<FLT_PER_REPEAT>(embed)], tv, MASK_NUL,
                    rowNum,
                    BinaryRepeatParams(1, 1, 0, embedRound / FLT_PER_DATABLOCK, embedRound / FLT_PER_DATABLOCK, 1));
                ResetMask();
            }
            PipeBarrier<PIPE_V>();

            // move lse to GM
            if (isLastRowLoop) {
                Ln<float, false>(lse, gl[taskOffset], MASK_NUL, CeilDiv<FLT_PER_REPEAT>(coreRow), UnaryRepeatParams());
                PipeBarrier<PIPE_V>();
                Add<float, false>(
                    lse, lse, gm[taskOffset], MASK_NUL, CeilDiv<FLT_PER_REPEAT>(coreRow), BinaryRepeatParams());
                SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
                CopyLseToGm(gLse, gUpdate, coreRow, coreHeadNum, qSOffset, qSBlockSize, strideLse);
            }

            // go = castfp32to16(go)
            RoundMode roundMode =
                std::is_same_v<ElementOutput, bfloat16_t> ? RoundMode::CAST_RINT : RoundMode::CAST_NONE;
            Cast<ElementOutput, float, false>(
                go16, go32, roundMode, MASK_NUL, CeilDiv<FLT_PER_REPEAT>(rowNum * embedRound),
                UnaryRepeatParams(1, 1, 4, 8));
            SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);

            // move O to GM
            CopyOToGm(gOutput, rowNum, rowOffset, qSOffset, qSBlockSize, embed, embedRound, strideO);

        } else if (multipleRowLoop) {
            SetFlag<HardEvent::V_MTE3>(EVENT_ID2);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID2);
            DataCopy(gUpdate, go32, DataCopyParams(1, rowNum * embedRound / FLT_PER_DATABLOCK, 0, 0));
        }
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
    }

    CATLASS_DEVICE void operator()(
        const GlobalTensor<ElementOutput>& gOutput, const GlobalTensor<ElementInput>& gInput,
        const GlobalTensor<ElementUpdate>& gUpdate, const GlobalTensor<ElementLse>& gLse,
        const LayoutOutput& layoutOutput, const LayoutInput& layoutInput, const LayoutUpdate& layoutUpdate,
        const LayoutLse& layoutLse, GemmCoord actualBlockShape, uint32_t qSBlockSize, uint32_t qNBlockSize,
        uint32_t isFirstStackTile, uint32_t isLastStackTile, uint32_t curStackTileMod, uint32_t curTaskMod,
        Arch::CrossCoreFlag pvReady, uint32_t rowBase = 0, bool isFirstHead = true)
    {
        uint32_t rowNum = actualBlockShape.m();
        uint32_t embed = actualBlockShape.n();
        uint32_t embedRound = layoutInput.stride(0);
        uint32_t strideO = layoutOutput.stride(0);
        uint32_t strideLse = layoutLse.stride(0);

        uint32_t core1 = GetSubBlockIdx();
        uint32_t rowSplit = rowNum / NUM_2;
        uint32_t coreRow = core1 ? (rowNum - rowSplit) : rowSplit;

        if (coreRow == 0) {
            if (isFirstHead) {
                Arch::CrossCoreWaitFlag(pvReady);
            }
            return;
        }

        // 行在 S/O 里是 head-major 连续排布（行 r -> head r/qSBlockSize、q 偏移 r%qSBlockSize），
        // 因此子核的 head 起点/head 内偏移由起始行直接换算；head 数为奇数时两子核会在
        // head 中间切开，CopyOToGm/CopyLseToGm 已支持跨 head/半个 head 的连续写。
        uint32_t coreRowOffset = core1 ? rowSplit : 0;
        uint32_t qSOffset = coreRowOffset % qSBlockSize;
        uint32_t coreQNOffset = coreRowOffset / qSBlockSize;
        uint32_t coreHeadNum = CeilDiv(coreRow, qSBlockSize);

        uint32_t rowNumTile = RoundDown<FLT_PER_DATABLOCK>(MAX_UB_ELEM_NUM / embedRound);
        uint32_t rowLoop = CeilDiv(coreRow, rowNumTile);

        taskOffset = curTaskMod * MAX_ROW_NUM_SUB_CORE + rowBase;
        if (isFirstHead) {
            Arch::CrossCoreWaitFlag(pvReady);
        }

        for (uint32_t rowIdx = 0; rowIdx < rowLoop; rowIdx++) {
            uint32_t rowOffset = rowIdx * rowNumTile;
            uint32_t rowOffsetActual = coreRowOffset + rowOffset;
            uint32_t rowNumCurLoop = (rowIdx == rowLoop - 1) ? (coreRow - rowOffset) : rowNumTile;

            SubCoreCompute(
                gOutput[layoutOutput.GetOffset(MatrixCoord(0, coreQNOffset * embed))],
                gInput[layoutInput.GetOffset(MatrixCoord(rowOffsetActual, 0))],
                gUpdate[layoutUpdate.GetOffset(MatrixCoord(rowOffsetActual, 0))],
                gLse[layoutLse.GetOffset(MatrixCoord(coreQNOffset, qSOffset))],
                layoutInput.GetTileLayout(MatrixCoord(rowNumCurLoop, embed)), coreRow, isFirstStackTile,
                isLastStackTile, curStackTileMod, (rowLoop > 1), (rowIdx == rowLoop - 1), rowOffset, qSBlockSize,
                qSOffset, coreHeadNum, strideLse, strideO, rowBase);
        }
    }

private:
    uint32_t taskOffset;

    LocalTensor<float> lo;
    LocalTensor<ElementOutput> go16;
    LocalTensor<float> go32;
    LocalTensor<float> gm;
    LocalTensor<float> gl;
    LocalTensor<float> lse;
    LocalTensor<float> dm;
    LocalTensor<float> tv;
};
} // namespace Catlass::Epilogue::Block

#endif
