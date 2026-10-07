/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

#ifndef FAI_BLOCK_HPP
#define FAI_BLOCK_HPP

#include "catlass/catlass.hpp"
#include "catlass/gemm/dispatch_policy.hpp"

#include "kernel_common.hpp"

namespace Catlass::Gemm {
template <bool PAGED_CACHE_FLAG_ = false>
struct MmadAtlasA2FAIPVT : public Gemm::MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool PAGED_CACHE_FLAG = PAGED_CACHE_FLAG_;
};

template <bool PAGED_CACHE_FLAG_ = false>
struct MmadAtlasA2FAIQKT : public Gemm::MmadAtlasA2 {
    static constexpr uint32_t STAGES = 2;
    static constexpr bool PAGED_CACHE_FLAG = PAGED_CACHE_FLAG_;
};
} // namespace Catlass::Gemm

namespace Catlass::Epilogue {
template <KernelCommon::MaskModeT MASK_MODE_>
struct EpilogueAtlasA2OnlineSoftmaxT {
    static constexpr KernelCommon::MaskModeT MASK_MODE = MASK_MODE_;
};

struct EpilogueAtlasA2RescaleOT {};
} // namespace Catlass::Epilogue

#endif // FAI_BLOCK_HPP