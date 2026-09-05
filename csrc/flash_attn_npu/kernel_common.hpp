/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

#ifndef KERNEL_COMMON
#define KERNEL_COMMON

#include "catlass/catlass.hpp"

namespace KernelCommon {
constexpr uint32_t QK_READY_ID = 1;
constexpr uint32_t SOFTMAX_READY_ID = 2;
constexpr uint32_t PV_READY_ID = 3;
constexpr uint32_t MAX_KV_STACK_LEN = 512; // KV 分块的大小上限
constexpr uint32_t Q_TILE_CEIL = 128;      // Q 分块的大小上限
constexpr uint32_t WORKSPACE_BLOCK_SIZE_DB = Q_TILE_CEIL * MAX_KV_STACK_LEN;
constexpr uint32_t MASK_LEN = 2048;
constexpr uint32_t MAX_UB_ELEM_NUM = 8192;
constexpr uint32_t MAX_ROW_NUM_SUB_CORE = 256;
constexpr uint32_t PRE_LAUNCH = 1;
constexpr uint32_t DOUBLE_BUFFER = 2;
constexpr uint32_t SYNC_MODE = 0x2;
constexpr uint32_t NUM_16 = 16;
constexpr uint32_t NUM_32 = 32;
constexpr uint32_t NUM_128 = 128;

constexpr uint64_t MASK_ONE = 1ull;
constexpr uint64_t MASK_NUL = 0ull;
constexpr uint64_t MASK_ALL = ~0ull;

constexpr uint32_t UB_VEC_SIZE = 1024;  // 1 KB
constexpr uint32_t UB_BLK_SIZE = 16384; // 16 KB

constexpr uint32_t DATABLOCK_PER_REPEAT = 8;
constexpr uint32_t FLT_PER_DATABLOCK = 8;
constexpr uint32_t HALF_PER_DATABLOCK = 16;
constexpr uint32_t FLT_PER_REPEAT = 64;
constexpr uint32_t HALF_PER_REPEAT = 128;
constexpr uint32_t REPEAT_SIZE = 256;

enum class MaskType : uint32_t { NO_MASK = 0, MASK_CAUSAL = 1 };

enum class InputLayout : uint32_t { BSND = 0, TND = 1 };

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

// 获取 Q 的 N 维分块大小
CATLASS_DEVICE uint32_t GetQNBlockTile(uint32_t qSeqlen, uint32_t groupSize)
{
    uint32_t qNBlockTile = Q_TILE_CEIL / qSeqlen;
    qNBlockTile = Min(qNBlockTile, groupSize); // qNBlock 不能跨 group
    qNBlockTile = Max(qNBlockTile, 1u);        // qNBlock 大小至少为 1
    return qNBlockTile;
}

// 获取 Q 的 S 维分块大小
CATLASS_DEVICE uint32_t GetQSBlockTile() { return Q_TILE_CEIL; }

// 设置向量掩码
CATLASS_DEVICE void SetVecMaskLen(uint32_t len)
{
    if (len >= 128) {
        AscendC::SetVectorMask<float>(MASK_ALL, MASK_ALL);
    } else if (len >= 64) {
        AscendC::SetVectorMask<float>((MASK_ONE << (len - 64)) - 1, MASK_ALL);
    } else {
        AscendC::SetVectorMask<float>(0, (MASK_ONE << len) - 1);
    }
}

// 设置归约掩码
CATLASS_DEVICE void SetBlockReduceMaskLen(uint32_t len)
{
    if (len > 8) {
        AscendC::SetVectorMask<float>(MASK_ALL, MASK_ALL);
    } else {
        uint64_t subMask = (MASK_ONE << len) - 1;
        uint64_t mask = 0;
        for (uint32_t i = 0; i < 64; i += 8) {
            mask |= (subMask << i);
        }
        AscendC::SetVectorMask<float>(mask, mask);
    }
}
} // namespace KernelCommon

#endif