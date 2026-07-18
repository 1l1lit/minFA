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
template <class OutputType_, class InputType_, class UpdateType_, class LseType_, LseModeT LSE_MODE_>
class BlockEpilogue<EpilogueAtlasA2RescaleOT<LSE_MODE_>, OutputType_, InputType_, UpdateType_, LseType_> {
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

        constexpr uint32_t TV_OFFSET = 10 * UB_BLK_SIZE;
        constexpr uint32_t HM_OFFSET = 10 * UB_BLK_SIZE + 9 * UB_VEC_SIZE;
        constexpr uint32_t GM_OFFSET = 10 * UB_BLK_SIZE + 10 * UB_VEC_SIZE;
        constexpr uint32_t GL_OFFSET = 10 * UB_BLK_SIZE + 12 * UB_VEC_SIZE;
        constexpr uint32_t LSE_OFFSET = 10 * UB_BLK_SIZE + 12 * UB_VEC_SIZE;
        constexpr uint32_t DM_OFFSET = 10 * UB_BLK_SIZE + 13 * UB_VEC_SIZE;

        lo = resource.ubBuf.template GetBufferByByte<float>(LO_OFFSET);
        go16 = resource.ubBuf.template GetBufferByByte<ElementOutput>(GO_OFFSET);
        go32 = resource.ubBuf.template GetBufferByByte<float>(GO_OFFSET);
        tv = resource.ubBuf.template GetBufferByByte<float>(TV_OFFSET);
        hm = resource.ubBuf.template GetBufferByByte<float>(HM_OFFSET);
        gm = resource.ubBuf.template GetBufferByByte<float>(GM_OFFSET);
        gl = resource.ubBuf.template GetBufferByByte<float>(GL_OFFSET);
        lse = resource.ubBuf.template GetBufferByByte<float>(LSE_OFFSET);
        dm = resource.ubBuf.template GetBufferByByte<float>(DM_OFFSET);
    }

    CATLASS_DEVICE void CopyOToGm(
        GlobalTensor<ElementOutput> gOutput, uint32_t proTokenIdx, uint32_t proTokenNum, uint32_t epiTokenNum,
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
        GlobalTensor<ElementOutput> gOutput, GlobalTensor<ElementInput> gInput, GlobalTensor<ElementUpdate> gUpdate,
        GlobalTensor<ElementLse> gLse, const LayoutOutput& layoutOutput, const LayoutInput& layoutInput,
        const LayoutUpdate& layoutUpdate, const LayoutLse& layoutLse, uint32_t qNThisSubBlock, uint32_t qSThisSubBlock,
        uint32_t totalRowNum, uint32_t isFirstStackTile, uint32_t isLastStackTile, uint32_t curStackTileMod,
        uint32_t needRowLoop, uint32_t isLastRowLoop, uint32_t rowOffsetLoop, uint32_t proTokenIdx,
        uint32_t proTokenNum, uint32_t epiTokenNum, uint32_t integralHeadNum)
    {
        uint32_t curRowNum = layoutInput.shape(0);
        uint32_t embed = layoutInput.shape(1);
        uint32_t embedRound = layoutInput.stride(0);
        uint32_t curRowNumRound = RoundUp(curRowNum, DATABLOCK_PER_REPEAT);
        uint32_t qSBlockSize = layoutOutput.shape(0);
        uint32_t oHiddenSize = layoutOutput.shape(1);
        uint32_t qHeads = layoutLse.shape(1);
        uint32_t dmUbOffsetCurStackTile = curStackTileMod * MAX_ROW_NUM_SUB_CORE + rowOffsetLoop;

        if (!isFirstStackTile) {
            WaitFlag<HardEvent::V_MTE2>(EVENT_ID3);
            DataCopy(lo, gInput, DataCopyParams(1, curRowNum * embedRound / DATABLOCK_PER_REPEAT, 0, 0));
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        }

        WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID6);
        if (!isFirstStackTile) {
            Brcb(tv, dm[dmUbOffsetCurStackTile], curRowNumRound / DATABLOCK_PER_REPEAT, BrcbRepeatParams());
            PipeBarrier<PIPE_V>();

            if (needRowLoop) {
                DataCopy(go32, gUpdate, DataCopyParams(1, curRowNum * embedRound / DATABLOCK_PER_REPEAT, 0, 0));
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
            Brcb(tv, gl[rowOffsetLoop], curRowNumRound / DATABLOCK_PER_REPEAT, BrcbRepeatParams());
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
                std::is_same<ElementOutput, bfloat16_t>::value ? RoundMode::CAST_RINT : RoundMode::CAST_NONE;
            Cast<ElementOutput, float, false>(
                go16, go32, roundMode, MASK_NUL, (curRowNum * embedRound + FLT_PER_REPEAT - 1) / FLT_PER_REPEAT,
                UnaryRepeatParams(1, 1, 4, 8));
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

            // move O to GM
            CopyOToGm(
                gOutput, proTokenIdx, proTokenNum, epiTokenNum, integralHeadNum, qSThisSubBlock, embed, oHiddenSize);

            if constexpr (LSE_MODE_ == LseModeT::OUT_ONLY) {
                if (isLastRowLoop) {
                    Ln<float, false>(lse, gl, MASK_NUL, CeilDiv(totalRowNum, FLT_PER_REPEAT), UnaryRepeatParams());
                    PipeBarrier<PIPE_V>();

                    Add<float, false>(
                        lse, lse, gm, MASK_NUL, CeilDiv(totalRowNum, FLT_PER_REPEAT), BinaryRepeatParams());
                    PipeBarrier<PIPE_V>();

                    // lse_block = expand_to_block(lse)
                    Brcb(tv, lse, CeilDiv(totalRowNum, DATABLOCK_PER_REPEAT), BrcbRepeatParams());
                    PipeBarrier<PIPE_V>();
                    SetFlag<HardEvent::V_MTE3>(EVENT_ID4);
                    WaitFlag<HardEvent::V_MTE3>(EVENT_ID4);

                    if (qNThisSubBlock == 0) {
                        DataCopyPad(
                            gLse, tv,
                            DataCopyExtParams(totalRowNum, sizeof(float), 0, (qHeads - 1) * sizeof(float), 0));
                    } else {
                        for (uint32_t qNIdx = 0; qNIdx < qNThisSubBlock; qNIdx++) {
                            DataCopyPad(
                                gLse[qNIdx], tv[qNIdx * qSBlockSize * DATABLOCK_PER_REPEAT],
                                DataCopyExtParams(qSBlockSize, sizeof(float), 0, (qHeads - 1) * sizeof(float), 0));
                        }
                    }
                    SetFlag<HardEvent::MTE3_V>(EVENT_ID4);
                }
            }
        } else if (needRowLoop) {
            SetFlag<HardEvent::V_MTE3>(EVENT_ID5);
            WaitFlag<HardEvent::V_MTE3>(EVENT_ID5);
            DataCopy(gUpdate, go32, DataCopyParams(1, curRowNum * embedRound / DATABLOCK_PER_REPEAT, 0, 0));
        }
        SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID6);
    }

    CATLASS_DEVICE void operator()(
        GlobalTensor<ElementOutput> gOutput, GlobalTensor<ElementInput> gInput, GlobalTensor<ElementUpdate> gUpdate,
        GlobalTensor<ElementLse> gLse, const LayoutOutput& layoutOutput, const LayoutInput& layoutInput,
        const LayoutUpdate& layoutUpdate, const LayoutLse& layoutLse, GemmCoord actualBlockShape, uint32_t qSBlockSize,
        uint32_t qNBlockSize, uint32_t isFirstStackTile, uint32_t isLastStackTile, uint32_t curStackTileMod)
    {
        uint32_t rowNum = actualBlockShape.m();
        uint32_t embed = actualBlockShape.n();
        uint32_t maxRowNumPerLoop = MAX_UB_O_ELEM_NUM / embed;
        uint32_t rowNumTile = RoundDown(maxRowNumPerLoop, DATABLOCK_PER_REPEAT);

        uint32_t subcoreIdx = GetSubBlockIdx();
        uint32_t subcoreNum = GetSubBlockNum();

        uint32_t qNSplitSubBlock = qNBlockSize / subcoreNum;
        uint32_t qNThisSubBlock = (qNBlockSize == 1)  ? 0
                                  : (subcoreIdx == 1) ? (qNBlockSize - qNSplitSubBlock)
                                                      : qNSplitSubBlock;
        uint32_t inRowSplitSubBlock = (qNBlockSize == 1) ? (qSBlockSize / subcoreNum) : (qSBlockSize * qNSplitSubBlock);
        uint32_t inRowActualThisSubBlock = (subcoreIdx == 1) ? (rowNum - inRowSplitSubBlock) : inRowSplitSubBlock;
        uint32_t inRowOffsetThisSubBlock = subcoreIdx * inRowSplitSubBlock;
        uint32_t outRowOffsetThisSubBlock = (qNBlockSize == 1) ? inRowOffsetThisSubBlock : 0;
        uint32_t outColOffsetThisSubBlock = (qNBlockSize == 1) ? 0 : subcoreIdx * qNSplitSubBlock * embed;
        uint32_t qSThisSubBlock = (qNBlockSize == 1) ? inRowActualThisSubBlock : qSBlockSize;
        int64_t outOffsetSubBlock =
            layoutOutput.GetOffset(MatrixCoord(outRowOffsetThisSubBlock, outColOffsetThisSubBlock));

        uint32_t outLseRowOffsetThisSubBlock = (qNBlockSize == 1) ? inRowOffsetThisSubBlock : 0;
        uint32_t outLseColOffsetThisSubBlock = (qNBlockSize == 1) ? 0 : subcoreIdx * qNSplitSubBlock;
        int64_t offsetLse = layoutLse.GetOffset(MatrixCoord(outLseRowOffsetThisSubBlock, outLseColOffsetThisSubBlock));
        auto gLseThisSubBlock = gLse[offsetLse];
        auto layoutOutLseThisSubBlock = layoutLse;

        if (inRowActualThisSubBlock > 0) {
            uint32_t rowLoop = CeilDiv(inRowActualThisSubBlock, rowNumTile);
            uint32_t needRowLoop = (rowLoop > 1) ? 1 : 0;

            // The rows of each cycle consist of multiple heads with several tokens.
            // There are several integral heads, one prologue head, one epilogue head.
            uint32_t proTokenIdx = 0;     // the token idx of the start token of the prologue part
            uint32_t proTokenIdxPre = 0;  // the token idx of the start token of the pre prologue part
            uint32_t proTokenNum = 0;     // the token num of the prologue part
            uint32_t epiTokenNum = 0;     // the token num of the epilogue part
            uint32_t integralHeadNum = 0; // the number of integral heads within a cycle
            uint32_t qSRemian = qSThisSubBlock;
            for (uint32_t rowLoopIdx = 0; rowLoopIdx < rowLoop; rowLoopIdx++) {
                uint32_t rowOffsetLoop = rowLoopIdx * rowNumTile;
                uint32_t rowOffsetCurLoop = inRowOffsetThisSubBlock + rowOffsetLoop;
                uint32_t rowActualCurLoop =
                    (rowLoopIdx == (rowLoop - 1)) ? inRowActualThisSubBlock - rowLoopIdx * rowNumTile : rowNumTile;

                int64_t offsetOutput =
                    static_cast<int64_t>(rowLoopIdx * rowNumTile / qSThisSubBlock * embed) + outOffsetSubBlock;
                auto gOutputCurLoop = gOutput[offsetOutput];
                auto layoutOutputCurLoop = layoutOutput;
                int64_t offsetInput = layoutInput.GetOffset(MatrixCoord(rowOffsetCurLoop, 0));
                auto gInputCurLoop = gInput[offsetInput];
                auto layoutInputCurLoop = layoutInput.GetTileLayout(MatrixCoord(rowActualCurLoop, embed));

                int64_t offsetUpdate = layoutUpdate.GetOffset(MatrixCoord(rowOffsetCurLoop, 0));
                auto gUpdateCurLoop = gUpdate[offsetUpdate];
                auto layoutUpdateCurLoop = layoutUpdate.GetTileLayout(MatrixCoord(rowActualCurLoop, embed));

                proTokenIdx = rowOffsetLoop % qSThisSubBlock;
                proTokenNum = Min(rowActualCurLoop, (qSThisSubBlock - proTokenIdx)) % qSThisSubBlock;
                integralHeadNum = (rowActualCurLoop - proTokenNum) / qSThisSubBlock;
                epiTokenNum = rowActualCurLoop - proTokenNum - integralHeadNum * qSThisSubBlock;

                SubCoreCompute(
                    gOutputCurLoop, gInputCurLoop, gUpdateCurLoop, gLseThisSubBlock, layoutOutputCurLoop,
                    layoutInputCurLoop, layoutUpdateCurLoop, layoutOutLseThisSubBlock, qNThisSubBlock, qSThisSubBlock,
                    inRowActualThisSubBlock, isFirstStackTile, isLastStackTile, curStackTileMod, needRowLoop,
                    (rowLoopIdx == rowLoop - 1), rowOffsetLoop, proTokenIdx, proTokenNum, epiTokenNum, integralHeadNum);
            }
        }
    }

private:
    LocalTensor<float> lo;
    LocalTensor<float> dm;
    LocalTensor<float> hm;
    LocalTensor<float> gl;
    LocalTensor<float> tv;
    LocalTensor<ElementOutput> go16;
    LocalTensor<float> go32;
    LocalTensor<float> gm;
    LocalTensor<float> lse;
};
} // namespace Catlass::Epilogue::Block

#endif
