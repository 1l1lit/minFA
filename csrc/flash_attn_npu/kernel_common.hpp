/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

#ifndef KERNEL_COMMON
#define KERNEL_COMMON

#include "catlass/catlass.hpp"

#define CATLASS_NORMAL __aicore__
#define CATLASS_INLINE __forceinline__ __aicore__

namespace KernelCommon {
constexpr uint32_t QK_READY_ID = 1;
constexpr uint32_t SOFTMAX_READY_ID = 2;
constexpr uint32_t PV_READY_ID = 3;
constexpr uint32_t Q_TILE_CEIL = 256;      // Q 分块的大小上限
constexpr uint32_t MAX_KV_STACK_LEN = 512; // KV 分块的大小上限
constexpr uint32_t WORKSPACE_BLOCK_SIZE_DB = Q_TILE_CEIL * MAX_KV_STACK_LEN;
constexpr uint32_t MAX_ROW_NUM_SUB_CORE = Q_TILE_CEIL / 2;
constexpr uint32_t MAX_UB_ELEM_NUM = 8192;
constexpr uint32_t PRE_LAUNCH = 1;
constexpr uint32_t DOUBLE_BUFFER = 2;
constexpr uint32_t SYNC_MODE = 0x2;
constexpr uint32_t NUM_2 = 2;
constexpr uint32_t NUM_16 = 16;
constexpr uint32_t NUM_32 = 32;
constexpr uint32_t NUM_128 = 128;

constexpr uint64_t MASK_ONE = 1ull;
constexpr uint64_t MASK_NUL = 0ull;
constexpr uint64_t MASK_ALL = ~0ull;

constexpr uint32_t UB_RPT_SIZE = 256;   // 256 B
constexpr uint32_t UB_VEC_SIZE = 1024;  // 1 KB
constexpr uint32_t UB_BLK_SIZE = 16384; // 16 KB

constexpr uint32_t FLT_PER_DATABLOCK = 8;
constexpr uint32_t HALF_PER_DATABLOCK = 16;
constexpr uint32_t FLT_PER_REPEAT = 64;
constexpr uint32_t HALF_PER_REPEAT = 128;

enum class MaskModeT : uint32_t { NO_MASK = 0, MASK_CAUSAL = 1 };

enum class InputLayoutT : uint32_t { BSND = 0, TND = 1 };

struct FAIKernelParams {
    GM_ADDR q;
    GM_ADDR k;
    GM_ADDR v;
    GM_ADDR mask;
    GM_ADDR blockTable;
    GM_ADDR actualQseqlen;
    GM_ADDR actualKvseqlen;
    GM_ADDR o;
    GM_ADDR lse;
    GM_ADDR workspace;
    GM_ADDR tiling;
};

// 获取 Q 的 N 维分块大小（N 维以 head 为单位分块，一个 block 内可包含多个 head）
// 一个 task 内所有 KV 段共享 gl/gm/dm 行状态 buffer，行状态按紧凑行布局（段间无 pad），
// 这样 AIV 侧可以把所有段的行拼成一块，对 softmax/rescale 只调用一次。
// 容量约束：AIV 侧把 task 的 rowNum 行按 rowNum/2 平分给两个 sub-core，每个子核只用
// MAX_ROW_NUM_SUB_CORE 行行状态，所以 task 总行数上限是 2*MAX_ROW_NUM_SUB_CORE
// （= Q_TILE_CEIL = 256）。这里若按 MAX_ROW_NUM_SUB_CORE(128) 卡，qS=64/128 的 tile
// 会被腰斩（4→2、2→1），task 数翻倍，GQA 段内共享的 K/V 被重复搬运
CATLASS_INLINE uint32_t GetQNBlockTile(uint32_t qSeqlen, uint32_t groupSize, uint32_t nCap, uint32_t qHeads)
{
    // Q 在 L1 中按 16 行对齐存放，整个 task 的 Q 占用 = segs * RoundUp<16>(qS) 行，
    // 因此 head 数上限应按对齐后的行数折算（原按 qS 折算过于宽松，会切出过大的 tile）
    uint32_t qSAlign = RoundUp<NUM_16>(qSeqlen);
    uint32_t qNBlockTile = Min(nCap, Q_TILE_CEIL / qSAlign); // L1 Q 空间 + head 数上限
    qNBlockTile = Max(qNBlockTile, 1u);
    // MHA 的 tail 拆分策略（是否把 tail tile 压到 2 个 head）依赖单 batch 的
    // task 总数，在 CalcNormTailTask 里统一决策，这里只返回 fuse 后的 tile。
    // 只覆盖完整的 KV 段：tile >= groupSize 时按 groupSize 对齐（段数 > 1），
    // 否则取 groupSize 的因子（段数 = 1，head 起点仍在段边界）
    if (qNBlockTile >= groupSize) {
        qNBlockTile = (qNBlockTile / groupSize) * groupSize;
    } else {
        while (qNBlockTile > 1 && (groupSize % qNBlockTile) != 0) {
            --qNBlockTile;
        }
    }
    while (qNBlockTile > 1) {
        uint32_t segN = Min(qNBlockTile, groupSize);
        uint32_t segs = CeilDiv(qNBlockTile, segN);
        uint32_t rows = qSeqlen * segN;
        // 段间紧凑排列：行状态总行数 = 总 head 数 * qSeqlen，无对齐 pad
        uint32_t used = segs * rows;
        if (used <= MAX_ROW_NUM_SUB_CORE * NUM_2) { // 两个 sub-core 各拿 rowNum/2 行
            break;
        }
        if (qNBlockTile > groupSize) {
            qNBlockTile -= groupSize; // 保持段对齐
        } else {
            do {
                --qNBlockTile;
            } while (qNBlockTile > 1 && (groupSize % qNBlockTile) != 0);
        }
    }
    // head 数在 task 间均分：task 按 taskIdx += coreNum 轮转，coreNum 为偶数时同一 core
    // 始终拿到同一类 task（taskIdx 奇偶固定），因此 CeilDiv 切出的"大 tile + 小尾巴"
    // 会直接变成核间耗时差（实测 16 head 切成 14+2 时 AIC 524us vs 141us，整核耗时由
    // 慢核决定）。这里把 tile 调成 CeilDiv(qHeads, tileNum) 的均分大小，task 数不变，
    // 只把各 task 的 head 数拉平；bal < tile 保证行状态容量不会变大。
    if (qNBlockTile >= groupSize && groupSize > 0 && (qNBlockTile % groupSize) == 0) {
        uint32_t kvHeads = qHeads / groupSize;
        uint32_t mCap = qNBlockTile / groupSize; // 每 task 覆盖的 KV 段（group）数上限
        if (kvHeads > mCap) {
            uint32_t tileNum = CeilDiv(kvHeads, mCap);
            uint32_t mBal = CeilDiv(kvHeads, tileNum) * groupSize;
            if (mBal < qNBlockTile) {
                qNBlockTile = mBal;
            }
        }
    }
    return qNBlockTile;
}

// 获取 Q 的 S 维分块大小
CATLASS_DEVICE uint32_t GetQSBlockTile() { return Q_TILE_CEIL; }

// 设置向量掩码
CATLASS_DEVICE constexpr void SetVecMaskLen(uint32_t len) { AscendC::SetVectorMask<float>(len); }

// 设置归约掩码
CATLASS_DEVICE constexpr void SetBlockReduceMaskLen(uint32_t len)
{
    uint64_t subMask = (MASK_ONE << len) - 1;
    uint64_t mask = 0;
    for (uint32_t i = 0; i < FLT_PER_REPEAT; i += FLT_PER_DATABLOCK) {
        mask |= (subMask << i);
    }
    AscendC::SetVectorMask<float>(0, mask);
}
} // namespace KernelCommon

#endif