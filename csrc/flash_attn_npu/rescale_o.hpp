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

    static constexpr uint32_t MAX_UB_O_ELEM_NUM = 8192;
    static constexpr uint32_t MAX_ROW_NUM_SUB_CORE = 256;

    CATLASS_DEVICE BlockEpilogue() = default;

    CATLASS_DEVICE ~BlockEpilogue() = default;

    CATLASS_DEVICE void init(Arch::Resource<ArchTag>& resource)
    {
        constexpr uint32_t LO_OFFSET = 6 * UB_BLK_SIZE;
        constexpr uint32_t GO_OFFSET = 8 * UB_BLK_SIZE;

        constexpr uint32_t GM_OFFSET = 10 * UB_BLK_SIZE;
        constexpr uint32_t GL_OFFSET = 10 * UB_BLK_SIZE + UB_VEC_SIZE;
        constexpr uint32_t DM_OFFSET = 10 * UB_BLK_SIZE + 2 * UB_VEC_SIZE;

        constexpr uint32_t TV_OFFSET = 11 * UB_BLK_SIZE + 8 * UB_VEC_SIZE;

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
        taskOffset = curTaskMod * ((4 * UB_VEC_SIZE) / sizeof(float));
    }

    CATLASS_DEVICE void CopyOToGm(
        const GlobalTensor<ElementOutput>& gOutput, uint32_t proTokenIdx, uint32_t proTokenNum, uint32_t epiTokenNum,
        uint32_t integralHeadNum, uint32_t qSThisSubBlock, uint32_t embed, uint32_t oHiddenSize)
    {
        uint32_t blockLen = embed * sizeof(half);
        uint32_t dstStride = (oHiddenSize - embed) * sizeof(half);

        uint32_t innerOGmOffset = 0;
        uint32_t innerGOUbOffset = 0;

        if (proTokenNum != 0) {
            DataCopyPad(
                gOutput[innerOGmOffset + proTokenIdx * oHiddenSize], go16[innerGOUbOffset],
                DataCopyExtParams(proTokenNum, blockLen, 0, dstStride, 0));
            innerOGmOffset += embed;
            innerGOUbOffset += proTokenNum * embed;
        }

        for (uint32_t i = 0; i < integralHeadNum; i++) {
            DataCopyPad(
                gOutput[innerOGmOffset], go16[innerGOUbOffset],
                DataCopyExtParams(qSThisSubBlock, blockLen, 0, dstStride, 0));
            innerOGmOffset += embed;
            innerGOUbOffset += qSThisSubBlock * embed;
        }

        if (epiTokenNum != 0) {
            DataCopyPad(
                gOutput[innerOGmOffset], go16[innerGOUbOffset],
                DataCopyExtParams(epiTokenNum, blockLen, 0, dstStride, 0));
        }
    }

    CATLASS_DEVICE void SubCoreCompute(
        const GlobalTensor<ElementOutput>& gOutput, const GlobalTensor<ElementInput>& gInput,
        const GlobalTensor<ElementUpdate>& gUpdate, const GlobalTensor<ElementLse>& gLse,
        const LayoutOutput& layoutOutput, const LayoutInput& layoutInput, const LayoutUpdate& layoutUpdate,
        const LayoutLse& layoutLse, uint32_t qNThisSubBlock, uint32_t qSThisSubBlock, uint32_t totalRowNum,
        uint32_t isFirstStackTile, uint32_t isLastStackTile, uint32_t curStackTileMod, uint32_t needRowLoop,
        uint32_t isLastRowLoop, uint32_t rowOffset, uint32_t proTokenIdx, uint32_t proTokenNum, uint32_t epiTokenNum,
        uint32_t integralHeadNum)
    {
        uint32_t curRowNum = layoutInput.shape(0);
        uint32_t embed = layoutInput.shape(1);
        uint32_t embedRound = layoutInput.stride(0);
        uint32_t curRowNumRound = RoundUp(curRowNum, DATABLOCK_PER_REPEAT);
        uint32_t qSBlockSize = layoutOutput.shape(0);
        uint32_t oHiddenSize = layoutOutput.shape(1);
        uint32_t qHeads = layoutLse.shape(1);
        uint32_t dmOffsetCurCycle = curStackTileMod * MAX_ROW_NUM_SUB_CORE + rowOffset;

        if (!isFirstStackTile) {
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID3);
            DataCopy(lo, gInput, DataCopyParams(1, curRowNum * embedRound / FLT_PER_DATABLOCK, 0, 0));
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        }

        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID6);
        if (!isFirstStackTile) {
            Brcb(tv, dm[taskOffset][dmOffsetCurCycle], curRowNumRound / DATABLOCK_PER_REPEAT, BrcbRepeatParams());
            PipeBarrier<PIPE_V>();

            if (needRowLoop) {
                DataCopy(go32, gUpdate, DataCopyParams(1, curRowNum * embedRound / FLT_PER_DATABLOCK, 0, 0));
                SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
            }

            // go = go * dm_block
            for (uint32_t vmul_idx = 0; vmul_idx < embed / FLT_PER_REPEAT; ++vmul_idx) {
                Mul<float, false>(
                    go32[vmul_idx * FLT_PER_REPEAT], go32[vmul_idx * FLT_PER_REPEAT], tv, MASK_NUL, curRowNum,
                    BinaryRepeatParams(
                        1, 1, 0, embedRound / DATABLOCK_PER_REPEAT, embedRound / DATABLOCK_PER_REPEAT, 1));
            }

            if (embed % FLT_PER_REPEAT > 0) {
                SetVecMaskLen(embed % FLT_PER_REPEAT);
                Mul<float, false>(
                    go32[RoundDown(embed, FLT_PER_REPEAT)], go32[RoundDown(embed, FLT_PER_REPEAT)], tv, MASK_NUL,
                    curRowNum,
                    BinaryRepeatParams(
                        1, 1, 0, embedRound / DATABLOCK_PER_REPEAT, embedRound / DATABLOCK_PER_REPEAT, 1));
                ResetMask();
            }
            PipeBarrier<PIPE_V>();

            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            // go = lo + go
            Add<float, false>(
                go32, go32, lo, MASK_NUL, (curRowNum * embedRound + FLT_PER_REPEAT - 1) / FLT_PER_REPEAT,
                BinaryRepeatParams());
            PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(EVENT_ID3);
        } else {
            // go = lo
            DataCopy(go32, gInput, DataCopyParams(1, curRowNum * embedRound / DATABLOCK_PER_REPEAT, 0, 0));
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        }

        if (isLastStackTile) {
            // gl_block = expand_to_block(gl)
            Brcb(tv, gl[taskOffset][rowOffset], curRowNumRound / DATABLOCK_PER_REPEAT, BrcbRepeatParams());
            PipeBarrier<PIPE_V>();

            // go = go / gl_block
            for (uint32_t vdiv_idx = 0; vdiv_idx < embed / FLT_PER_REPEAT; ++vdiv_idx) {
                Div<float, false>(
                    go32[vdiv_idx * FLT_PER_REPEAT], go32[vdiv_idx * FLT_PER_REPEAT], tv, MASK_NUL, curRowNum,
                    BinaryRepeatParams(
                        1, 1, 0, embedRound / DATABLOCK_PER_REPEAT, embedRound / DATABLOCK_PER_REPEAT, 1));
            }

            if (embed % FLT_PER_REPEAT > 0) {
                SetVecMaskLen(embed % FLT_PER_REPEAT);
                Div<float, false>(
                    go32[RoundDown(embed, FLT_PER_REPEAT)], go32[RoundDown(embed, FLT_PER_REPEAT)], tv, MASK_NUL,
                    curRowNum,
                    BinaryRepeatParams(
                        1, 1, 0, embedRound / DATABLOCK_PER_REPEAT, embedRound / DATABLOCK_PER_REPEAT, 1));
                ResetMask();
            }
            PipeBarrier<PIPE_V>();

            // go = castfp32to16(go)
            RoundMode roundMode =
                std::is_same_v<ElementOutput, bfloat16_t> ? RoundMode::CAST_RINT : RoundMode::CAST_NONE;
            Cast<ElementOutput, float, false>(
                go16, go32, roundMode, MASK_NUL, CeilDiv(curRowNum * embedRound, FLT_PER_REPEAT),
                UnaryRepeatParams(1, 1, 4, 8));
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

            // move O to GM
            CopyOToGm(
                gOutput, proTokenIdx, proTokenNum, epiTokenNum, integralHeadNum, qSThisSubBlock, embed, oHiddenSize);

            // move lse to GM
            if (isLastRowLoop) {
                Ln<float, false>(
                    lse[taskOffset], gl[taskOffset], MASK_NUL, CeilDiv(totalRowNum, FLT_PER_REPEAT),
                    UnaryRepeatParams());
                PipeBarrier<PIPE_V>();
                Add<float, false>(
                    lse[taskOffset], lse[taskOffset], gm[taskOffset], MASK_NUL, CeilDiv(totalRowNum, FLT_PER_REPEAT),
                    BinaryRepeatParams());
                SetFlag<HardEvent::V_MTE3>(EVENT_ID4);
                WaitFlag<HardEvent::V_MTE3>(EVENT_ID4);

                if (qNThisSubBlock == 0) {
                    DataCopyPad(gLse, lse[taskOffset], DataCopyExtParams(1, totalRowNum * sizeof(float), 0, 0, 0));
                } else {
                    DataCopyPad(
                        gLse, lse[taskOffset],
                        DataCopyExtParams(
                            qNThisSubBlock, qSBlockSize * sizeof(float), 0,
                            (layoutLse.stride(0) - qSBlockSize) * sizeof(float), 0));
                }
                SetFlag<HardEvent::MTE3_V>(taskEvent); // 通知当前 task 的 RS 完成
            }
        } else if (needRowLoop) {
            SetFlag<HardEvent::V_MTE3>(EVENT_ID5);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID5);
            DataCopy(gUpdate, go32, DataCopyParams(1, curRowNum * embedRound / DATABLOCK_PER_REPEAT, 0, 0));
        }
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID6);
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
        uint32_t maxRowNumPerLoop = MAX_UB_O_ELEM_NUM / embed;
        uint32_t rowNumTile = RoundDown(maxRowNumPerLoop, DATABLOCK_PER_REPEAT);

        uint32_t coreIdx = GetSubBlockIdx();
        uint32_t inRowSplit = (qNBlockSize == 1) ? (qSBlockSize / 2) : (qSBlockSize * (qNBlockSize / 2));
        uint32_t inRowActual = (coreIdx == 1) ? (rowNum - inRowSplit) : inRowSplit;
        uint32_t inRowOffset = coreIdx * inRowSplit;

        uint32_t outRowOffset = (qNBlockSize == 1) ? inRowOffset : 0;
        uint32_t outColOffset = (qNBlockSize == 1) ? 0 : coreIdx * (qNBlockSize / 2) * embed;

        uint32_t qSThisSubBlock = (qNBlockSize == 1) ? inRowActual : qSBlockSize;
        uint32_t qNThisSubBlock = (qNBlockSize == 1) ? 0
                                  : (coreIdx == 1)   ? (qNBlockSize - qNBlockSize / 2)
                                                     : (qNBlockSize / 2);

        uint32_t lseRowOffset = (qNBlockSize == 1) ? 0 : coreIdx * (qNBlockSize / 2);
        uint32_t lseColOffset = (qNBlockSize == 1) ? inRowOffset : 0;
        auto gLseThisSubBlock = gLse[layoutLse.GetOffset(MatrixCoord(lseRowOffset, lseColOffset))];

        if (inRowActual > 0) {
            uint32_t rowLoop = CeilDiv(inRowActual, rowNumTile);
            uint32_t needRowLoop = (rowLoop > 1) ? 1 : 0;

            // The rows of each cycle consist of multiple heads with several tokens.
            // There are several integral heads, one prologue head, one epilogue head.
            uint32_t proTokenIdx = 0;     // the token idx of the start token of the prologue part
            uint32_t proTokenIdxPre = 0;  // the token idx of the start token of the pre prologue part
            uint32_t proTokenNum = 0;     // the token num of the prologue part
            uint32_t epiTokenNum = 0;     // the token num of the epilogue part
            uint32_t integralHeadNum = 0; // the number of integral heads within a cycle

            for (uint32_t rowIdx = 0; rowIdx < rowLoop; rowIdx++) {
                uint32_t rowOffset = rowIdx * rowNumTile;
                uint32_t gRowOffset = inRowOffset + rowOffset;
                uint32_t rowNumLoop = (rowIdx == rowLoop - 1) ? (inRowActual - rowOffset) : rowNumTile;

                int64_t offsetOutput = static_cast<int64_t>(rowOffset / qSThisSubBlock * embed) +
                                       layoutOutput.GetOffset(MatrixCoord(outRowOffset, outColOffset));
                auto gOutputCurLoop = gOutput[offsetOutput];

                auto gInputCurLoop = gInput[layoutInput.GetOffset(MatrixCoord(gRowOffset, 0))];
                auto layoutInputCurLoop = layoutInput.GetTileLayout(MatrixCoord(rowNumLoop, embed));

                auto gUpdateCurLoop = gUpdate[layoutUpdate.GetOffset(MatrixCoord(gRowOffset, 0))];
                auto layoutUpdateCurLoop = layoutUpdate.GetTileLayout(MatrixCoord(rowNumLoop, embed));

                proTokenIdx = rowOffset % qSThisSubBlock;
                proTokenNum = Min(rowNumLoop, (qSThisSubBlock - proTokenIdx)) % qSThisSubBlock;
                integralHeadNum = (rowNumLoop - proTokenNum) / qSThisSubBlock;
                epiTokenNum = rowNumLoop - proTokenNum - integralHeadNum * qSThisSubBlock;

                SubCoreCompute(
                    gOutputCurLoop, gInputCurLoop, gUpdateCurLoop, gLseThisSubBlock, layoutOutput, layoutInputCurLoop,
                    layoutUpdateCurLoop, layoutLse, qNThisSubBlock, qSThisSubBlock, inRowActual, isFirstStackTile,
                    isLastStackTile, curStackTileMod, needRowLoop, (rowIdx == rowLoop - 1), rowOffset, proTokenIdx,
                    proTokenNum, epiTokenNum, integralHeadNum);
            }
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
