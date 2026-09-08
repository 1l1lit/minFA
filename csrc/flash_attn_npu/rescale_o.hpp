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

    CATLASS_DEVICE BlockEpilogue() = default;

    CATLASS_DEVICE ~BlockEpilogue() = default;

    CATLASS_DEVICE void init(Arch::Resource<ArchTag>& resource)
    {
        constexpr uint32_t LO_OFFSET = 6 * UB_BLK_SIZE;
        constexpr uint32_t GO_OFFSET = 8 * UB_BLK_SIZE;

        constexpr uint32_t GM_OFFSET = 10 * UB_BLK_SIZE;
        constexpr uint32_t GL_OFFSET = 10 * UB_BLK_SIZE + UB_RPT_SIZE;

        constexpr uint32_t DM_OFFSET = 11 * UB_BLK_SIZE + 2 * UB_RPT_SIZE;
        constexpr uint32_t TV_OFFSET = 11 * UB_BLK_SIZE + 12 * UB_VEC_SIZE;

        lo = resource.ubBuf.template GetBufferByByte<float>(LO_OFFSET);
        go16 = resource.ubBuf.template GetBufferByByte<ElementOutput>(GO_OFFSET);
        go32 = resource.ubBuf.template GetBufferByByte<float>(GO_OFFSET);
        gm = resource.ubBuf.template GetBufferByByte<float>(GM_OFFSET);
        gl = resource.ubBuf.template GetBufferByByte<float>(GL_OFFSET);
        lse = resource.ubBuf.template GetBufferByByte<float>(GL_OFFSET);
        dm = resource.ubBuf.template GetBufferByByte<float>(DM_OFFSET);
        tv = resource.ubBuf.template GetBufferByByte<float>(TV_OFFSET);
    }

    CATLASS_DEVICE void set_curTaskMod(uint32_t curTaskMod)
    {
        taskEvent = curTaskMod + 2;
        taskOffset = curTaskMod * (2 * UB_RPT_SIZE);
    }

    CATLASS_DEVICE void CopyOToGm(
        const GlobalTensor<ElementOutput>& gOutput, uint32_t proTokenIdx, uint32_t proTokenNum, uint32_t epiTokenNum,
        uint32_t integralHeadNum, uint32_t qSThisSubBlock, uint32_t embed, uint32_t strideO)
    {
        uint32_t blockLen = embed * sizeof(ElementOutput);
        uint32_t dstStride = (strideO - embed) * sizeof(ElementOutput);

        uint32_t gOutOffset = 0;
        uint32_t go16Offset = 0;

        if (proTokenNum != 0) {
            DataCopyPad(
                gOutput[gOutOffset + proTokenIdx * strideO], go16[go16Offset],
                DataCopyExtParams(proTokenNum, blockLen, 0, dstStride, 0));
            gOutOffset += embed;
            go16Offset += proTokenNum * embed;
        }

        for (uint32_t i = 0; i < integralHeadNum; i++) {
            DataCopyPad(
                gOutput[gOutOffset], go16[go16Offset], DataCopyExtParams(qSThisSubBlock, blockLen, 0, dstStride, 0));
            gOutOffset += embed;
            go16Offset += qSThisSubBlock * embed;
        }

        if (epiTokenNum != 0) {
            DataCopyPad(
                gOutput[gOutOffset], go16[go16Offset], DataCopyExtParams(epiTokenNum, blockLen, 0, dstStride, 0));
        }
    }

    CATLASS_DEVICE void SubCoreCompute(
        const GlobalTensor<ElementOutput>& gOutput, const GlobalTensor<ElementInput>& gInput,
        const GlobalTensor<ElementUpdate>& gUpdate, const GlobalTensor<ElementLse>& gLse,
        const LayoutOutput& layoutOutput, const LayoutInput& layoutInput, const LayoutUpdate& layoutUpdate,
        const LayoutLse& layoutLse, uint32_t qSThisSubBlock, uint32_t inRowActual, uint32_t isFirstStackTile,
        uint32_t isLastStackTile, uint32_t curStackTileMod, uint32_t multipleRowLoop, uint32_t isLastRowLoop,
        uint32_t rowOffset, uint32_t proTokenIdx, uint32_t proTokenNum, uint32_t epiTokenNum, uint32_t integralHeadNum)
    {
        uint32_t rowNum = layoutInput.shape(0);
        uint32_t rowNumRound = RoundUp(rowNum, FLT_PER_DATABLOCK);
        uint32_t embed = layoutInput.shape(1);
        uint32_t embedRound = layoutInput.stride(0);
        uint32_t strideO = layoutOutput.shape(1);
        uint32_t dmOffset = curStackTileMod * MAX_ROW_NUM_SUB_CORE + rowOffset;

        if (isFirstStackTile) {
            // go = lo
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            DataCopy(go32, gInput, DataCopyParams(1, rowNum * embedRound / FLT_PER_DATABLOCK, 0, 0));
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        } else {
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID2);
            DataCopy(lo, gInput, DataCopyParams(1, rowNum * embedRound / FLT_PER_DATABLOCK, 0, 0));
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);

            // dm_block = expand_to_block(dm)
            Brcb(tv, dm[dmOffset], rowNumRound / FLT_PER_DATABLOCK, BrcbRepeatParams());
            PipeBarrier<PIPE_V>();

            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
            if (multipleRowLoop) {
                DataCopy(go32, gUpdate, DataCopyParams(1, rowNum * embedRound / FLT_PER_DATABLOCK, 0, 0));
                SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
            }

            // go = go * dm_block
            for (uint32_t vmul_idx = 0; vmul_idx < embed / FLT_PER_REPEAT; vmul_idx++) {
                Mul<float, false>(
                    go32[vmul_idx * FLT_PER_REPEAT], go32[vmul_idx * FLT_PER_REPEAT], tv, MASK_NUL, rowNum,
                    BinaryRepeatParams(1, 1, 0, embedRound / FLT_PER_DATABLOCK, embedRound / FLT_PER_DATABLOCK, 1));
            }
            if (embed % FLT_PER_REPEAT > 0) {
                SetVecMaskLen(embed % FLT_PER_REPEAT);
                Mul<float, false>(
                    go32[RoundDown(embed, FLT_PER_REPEAT)], go32[RoundDown(embed, FLT_PER_REPEAT)], tv, MASK_NUL,
                    rowNum,
                    BinaryRepeatParams(1, 1, 0, embedRound / FLT_PER_DATABLOCK, embedRound / FLT_PER_DATABLOCK, 1));
                ResetMask();
            }
            PipeBarrier<PIPE_V>();

            // go = lo + go
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Add<float, false>(
                go32, go32, lo, MASK_NUL, CeilDiv(rowNum * embedRound, FLT_PER_REPEAT), BinaryRepeatParams());
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(EVENT_ID2);
        }

        if (isLastStackTile) {
            // gl_block = expand_to_block(gl)
            Brcb(tv, gl[taskOffset][rowOffset], rowNumRound / FLT_PER_DATABLOCK, BrcbRepeatParams());
            PipeBarrier<PIPE_V>();

            // go = go / gl_block
            for (uint32_t vdiv_idx = 0; vdiv_idx < embed / FLT_PER_REPEAT; vdiv_idx++) {
                Div<float, false>(
                    go32[vdiv_idx * FLT_PER_REPEAT], go32[vdiv_idx * FLT_PER_REPEAT], tv, MASK_NUL, rowNum,
                    BinaryRepeatParams(1, 1, 0, embedRound / FLT_PER_DATABLOCK, embedRound / FLT_PER_DATABLOCK, 1));
            }
            if (embed % FLT_PER_REPEAT > 0) {
                SetVecMaskLen(embed % FLT_PER_REPEAT);
                Div<float, false>(
                    go32[RoundDown(embed, FLT_PER_REPEAT)], go32[RoundDown(embed, FLT_PER_REPEAT)], tv, MASK_NUL,
                    rowNum,
                    BinaryRepeatParams(1, 1, 0, embedRound / FLT_PER_DATABLOCK, embedRound / FLT_PER_DATABLOCK, 1));
                ResetMask();
            }
            PipeBarrier<PIPE_V>();

            // go = castfp32to16(go)
            RoundMode roundMode =
                std::is_same_v<ElementOutput, bfloat16_t> ? RoundMode::CAST_RINT : RoundMode::CAST_NONE;
            Cast<ElementOutput, float, false>(
                go16, go32, roundMode, MASK_NUL, CeilDiv(rowNum * embedRound, FLT_PER_REPEAT),
                UnaryRepeatParams(1, 1, 4, 8));
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

            // move O to GM
            CopyOToGm(gOutput, proTokenIdx, proTokenNum, epiTokenNum, integralHeadNum, qSThisSubBlock, embed, strideO);

            // move lse to GM
            if (isLastRowLoop) {
                Ln<float, false>(
                    lse[taskOffset], gl[taskOffset], MASK_NUL, CeilDiv(inRowActual, FLT_PER_REPEAT),
                    UnaryRepeatParams());
                PipeBarrier<PIPE_V>();
                Add<float, false>(
                    lse[taskOffset], lse[taskOffset], gm[taskOffset], MASK_NUL, CeilDiv(inRowActual, FLT_PER_REPEAT),
                    BinaryRepeatParams());
                SetFlag<HardEvent::V_MTE3>(EVENT_ID4);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID4);
                DataCopyPad(gLse, lse[taskOffset], DataCopyExtParams(1, inRowActual * sizeof(float), 0, 0, 0));
                SetFlag<HardEvent::MTE3_V>(taskEvent); // 通知当前 task 的 RS 完成
            }
        } else if (multipleRowLoop) {
            SetFlag<HardEvent::V_MTE3>(EVENT_ID5);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID5);
            DataCopy(gUpdate, go32, DataCopyParams(1, rowNum * embedRound / FLT_PER_DATABLOCK, 0, 0));
        }
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID1);
    }

    CATLASS_DEVICE void operator()(
        const GlobalTensor<ElementOutput>& gOutput, const GlobalTensor<ElementInput>& gInput,
        const GlobalTensor<ElementUpdate>& gUpdate, const GlobalTensor<ElementLse>& gLse,
        const LayoutOutput& layoutOutput, const LayoutInput& layoutInput, const LayoutUpdate& layoutUpdate,
        const LayoutLse& layoutLse, GemmCoord actualBlockShape, uint32_t qSBlockSize, uint32_t qNBlockSize,
        uint32_t isFirstStackTile, uint32_t isLastStackTile, uint32_t curStackTileMod)
    {
        uint32_t rowNum = actualBlockShape.m();
        uint32_t embed = actualBlockShape.n();
        uint32_t rowNumTile = RoundDown(MAX_UB_ELEM_NUM / embed, FLT_PER_DATABLOCK);

        uint32_t coreIdx = GetSubBlockIdx();
        uint32_t inRowSplit = (qNBlockSize == 1) ? (qSBlockSize / 2) : (qSBlockSize * (qNBlockSize / 2));
        uint32_t inRowActual = (coreIdx == 1) ? (rowNum - inRowSplit) : inRowSplit;
        uint32_t inRowOffset = coreIdx * inRowSplit;

        if (inRowActual == 0) {
            return;
        }

        uint32_t rowLoop = CeilDiv(inRowActual, rowNumTile);
        uint32_t qSThisSubBlock = (qNBlockSize == 1) ? inRowActual : qSBlockSize;

        uint32_t outRowOffset = (qNBlockSize == 1) ? inRowOffset : 0;
        uint32_t outColOffset = (qNBlockSize == 1) ? 0 : coreIdx * (qNBlockSize / 2) * embed;
        uint32_t lseRowOffset = (qNBlockSize == 1) ? 0 : coreIdx * (qNBlockSize / 2);
        uint32_t lseColOffset = (qNBlockSize == 1) ? inRowOffset : 0;

        for (uint32_t rowIdx = 0; rowIdx < rowLoop; rowIdx++) {
            uint32_t rowOffset = rowIdx * rowNumTile;
            uint32_t gmRowOffset = inRowOffset + rowOffset;
            uint32_t rowNumCurLoop = (rowIdx == rowLoop - 1) ? (inRowActual - rowOffset) : rowNumTile;

            uint32_t proTokenIdx = rowOffset % qSThisSubBlock;
            uint32_t proTokenNum = Min(rowNumCurLoop, (qSThisSubBlock - proTokenIdx)) % qSThisSubBlock;
            uint32_t integralHeadNum = (rowNumCurLoop - proTokenNum) / qSThisSubBlock;
            uint32_t epiTokenNum = rowNumCurLoop - proTokenNum - integralHeadNum * qSThisSubBlock;

            SubCoreCompute(
                gOutput
                    [layoutOutput.GetOffset(MatrixCoord(outRowOffset, outColOffset)) +
                     static_cast<int64_t>(rowOffset / qSThisSubBlock * embed)],
                gInput[layoutInput.GetOffset(MatrixCoord(gmRowOffset, 0))],
                gUpdate[layoutUpdate.GetOffset(MatrixCoord(gmRowOffset, 0))],
                gLse[layoutLse.GetOffset(MatrixCoord(lseRowOffset, lseColOffset))], layoutOutput,
                layoutInput.GetTileLayout(MatrixCoord(rowNumCurLoop, embed)),
                layoutUpdate.GetTileLayout(MatrixCoord(rowNumCurLoop, embed)), layoutLse, qSThisSubBlock, inRowActual,
                isFirstStackTile, isLastStackTile, curStackTileMod, (rowLoop > 1), (rowIdx == rowLoop - 1), rowOffset,
                proTokenIdx, proTokenNum, epiTokenNum, integralHeadNum);
        }
    }

private:
    uint32_t taskEvent;
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
