/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

#ifndef FAI_BLOCK_HPP
#define FAI_BLOCK_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/dispatch_policy.hpp"

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
template <bool RETURN_SOFTMAX_ = false, bool HAS_DROPOUT_ = false>
struct EpilogueAtlasA2OnlineSoftmaxT {
    static constexpr bool RETURN_SOFTMAX = RETURN_SOFTMAX_;
    static constexpr bool HAS_DROPOUT = HAS_DROPOUT_;
};

template <bool HAS_DROPOUT_ = false>
struct EpilogueAtlasA2RescaleOT {
    static constexpr bool HAS_DROPOUT = HAS_DROPOUT_;
};
} // namespace Catlass::Epilogue

#endif // FAI_BLOCK_HPP