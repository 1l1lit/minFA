#include <torch/extension.h>

#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "third_party/op-plugin/op_plugin/include/ops.h"
#include "acl/acl.h"
#include "runtime/rt_ffts.h"
#include "kernel_operator.h"
#include "tiling/platform/platform_ascendc.h"

#include "kernel_common.hpp"
#include "mha_fwd_kvcache.hpp"
#include "tiling_data.hpp"

#define FA_INFER(T, PAGED, MASK, LAYOUT)                                                                          \
    FAInfer<T, PAGED, MASK, LAYOUT><<<coreNum, nullptr, aclStream>>>(                                             \
        fftsAddr, qDevice, kDevice, vDevice, maskDevice, blockTableDevice, oDevice, softmaxLseDevice, qSeqDevice, \
        kvSeqDevice, workspaceDevice, tilingDevice)

#define CASE_FA(T, PAGED, CAUSAL, VARLEN)                                   \
    case ((PAGED << 2) | (CAUSAL << 1) | (VARLEN << 0)):                    \
        FA_INFER(                                                           \
            T, PAGED, (CAUSAL ? MaskType::MASK_CAUSAL : MaskType::NO_MASK), \
            (VARLEN ? InputLayout::TND : InputLayout::BSND));               \
        break

std::vector<at::Tensor> mha_fwd_kvcache(
    at::Tensor& q,            // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor& kcache, // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x
                              // num_heads_k x head_size if there's a block_table.
    const at::Tensor& vcache, // batch_size_c x seqlen_k x num_heads_k x head_size or num_blocks x page_block_size x
                              // num_heads_k x head_size if there's a block_table.
    std::optional<const at::Tensor>& k_,               // batch_size x seqlen_knew x num_heads_k x head_size
    std::optional<const at::Tensor>& v_,               // batch_size x seqlen_knew x num_heads_k x head_size
    std::optional<const at::Tensor>& seqlens_k_,       // batch_size
    std::optional<const at::Tensor>& rotary_cos_,      // seqlen_ro x (rotary_dim / 2)
    std::optional<const at::Tensor>& rotary_sin_,      // seqlen_ro x (rotary_dim / 2)
    std::optional<const at::Tensor>& cache_batch_idx_, // indices to index into the KV cache
    std::optional<const at::Tensor>& leftpad_k_,       // batch_size
    std::optional<at::Tensor>& block_table_,           // batch_size x max_num_blocks_per_seq
    std::optional<at::Tensor>& alibi_slopes_,          // num_heads or batch_size x num_heads
    std::optional<at::Tensor>& out_,                   // batch_size x seqlen_q x num_heads x head_size
    const float softmax_scale, bool is_causal, int window_size_left, int window_size_right, const float softcap,
    bool is_rotary_interleaved, // if true, rotary combines indices 0 & 1, else indices 0 & rotary_dim / 2
    int num_splits)
{
    const c10::OptionalDeviceGuard device_guard(device_of(q));
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);

    auto q_dtype = q.dtype();
    bool is_bf16 = q_dtype == torch::kBFloat16;
    bool is_fp16 = q_dtype == torch::kFloat16;

    TORCH_CHECK(is_bf16 || is_fp16, "FlashAttention only supports FP16 and BF16 data types");
    TORCH_CHECK(kcache.dtype() == q_dtype, "query and key_cache must have the same dtype");
    TORCH_CHECK(vcache.dtype() == q_dtype, "query and value_cache must have the same dtype");

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor q must have contiguous last dimension");
    TORCH_CHECK(kcache.stride(-1) == 1, "Input tensor kcache must have contiguous last dimension");
    TORCH_CHECK(vcache.stride(-1) == 1, "Input tensor vcache must have contiguous last dimension");

    at::Tensor tiling_cpu_tensor = at::empty({1024}, at::device(c10::kCPU).dtype(at::kByte));
    auto tiling_cpu_ptr = reinterpret_cast<FAInferTilingData*>(tiling_cpu_tensor.data_ptr<uint8_t>());
    uint32_t coreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    at::Tensor seqlens_k, block_table, out;
    at::Tensor k, v, rotary_cos, rotary_sin, cache_batch_idx, alibi_slopes;
    const bool paged_KV = block_table_.has_value();
    if (paged_KV) {
        block_table = block_table_.value();
        TORCH_CHECK(block_table.dtype() == torch::kInt32, "block_table must have dtype int32");
        TORCH_CHECK(block_table.stride(-1) == 1, "block_table must have contiguous last dimension");
    }

    if (seqlens_k_.has_value()) {
        seqlens_k = seqlens_k_.value();
        TORCH_CHECK(seqlens_k.dtype() == torch::kInt32, "seqlens_k must have dtype int32");
    }

    TORCH_CHECK(!alibi_slopes_.has_value(), "NPU FlashAttention does not support alibi_slopes");
    TORCH_CHECK(!leftpad_k_.has_value(), "NPU FlashAttention does not support leftpad_k");
    TORCH_CHECK(!rotary_cos_.has_value(), "NPU FlashAttention does not support rotary embedding");
    TORCH_CHECK(!rotary_sin_.has_value(), "NPU FlashAttention does not support rotary embedding");
    TORCH_CHECK(softcap == 0.0f, "NPU FlashAttention does not support softcap");
    TORCH_CHECK(window_size_left == -1, "NPU FlashAttention does not support window_size_left");
    TORCH_CHECK(window_size_right == -1, "NPU FlashAttention does not support window_size_right");
    TORCH_CHECK(num_splits == 1 || num_splits == 0, "NPU FlashAttention only supports num_splits=1 or num_splits=0");

    if (k_.has_value()) {
        k = k_.value();
    }
    if (v_.has_value()) {
        v = v_.value();
    }
    if (rotary_cos_.has_value()) {
        rotary_cos = rotary_cos_.value();
    }
    if (rotary_sin_.has_value()) {
        rotary_sin = rotary_sin_.value();
    }
    if (cache_batch_idx_.has_value()) {
        cache_batch_idx = cache_batch_idx_.value();
    }
    if (alibi_slopes_.has_value()) {
        alibi_slopes = alibi_slopes_.value();
    }
    if (out_.has_value()) {
        out = out_.value();
    } else {
        out = torch::empty_like(q);
    }
    const auto sizes = q.sizes();
    const int batch_size = sizes[0];
    int seqlen_q = sizes[1];
    int num_heads = sizes[2];
    const int head_size = sizes[3];
    const int max_num_blocks_per_seq = !paged_KV ? 0 : block_table.size(1);
    const int num_blocks = !paged_KV ? 0 : kcache.size(0);
    const int page_block_size = !paged_KV ? 128 : kcache.size(1);
    const int num_heads_k = kcache.size(2);

    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size <= 256, "FlashAttention only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    tiling_cpu_ptr->set_batch(static_cast<uint32_t>(batch_size));
    tiling_cpu_ptr->set_numHeads(static_cast<uint32_t>(num_heads));
    tiling_cpu_ptr->set_kvHeads(static_cast<uint32_t>(num_heads_k));
    tiling_cpu_ptr->set_embeddingSize(static_cast<uint32_t>(head_size));
    tiling_cpu_ptr->set_numBlocks(static_cast<uint32_t>(num_blocks));
    tiling_cpu_ptr->set_blockSize(static_cast<uint32_t>(page_block_size));
    tiling_cpu_ptr->set_maxNumBlocksPerBatch(static_cast<uint32_t>(max_num_blocks_per_seq));
    tiling_cpu_ptr->set_scaleValue(softmax_scale);
    tiling_cpu_ptr->set_maxQSeqlen(seqlen_q);

    uint64_t WORKSPACE_BLOCK_SIZE_DB = KernelCommon::WORKSPACE_BLOCK_SIZE_DB;
    uint64_t DOUBLE_BUFFER = KernelCommon::DOUBLE_BUFFER;
    uint64_t mm1OutSize = static_cast<uint64_t>(coreNum) * WORKSPACE_BLOCK_SIZE_DB * 4 * DOUBLE_BUFFER;
    uint64_t smOnlineOutSize = static_cast<uint64_t>(coreNum) * WORKSPACE_BLOCK_SIZE_DB * 2 * DOUBLE_BUFFER;
    uint64_t mm2OutSize = static_cast<uint64_t>(coreNum) * WORKSPACE_BLOCK_SIZE_DB * 4 * DOUBLE_BUFFER;
    uint64_t UpdateSize = static_cast<uint64_t>(coreNum) * WORKSPACE_BLOCK_SIZE_DB * 4;
    int64_t workspaceSize = mm1OutSize + smOnlineOutSize + mm2OutSize + UpdateSize;
    at::Tensor workspace_tensor = at::empty({workspaceSize}, at::device(at::kPrivateUse1).dtype(at::kByte));

    tiling_cpu_ptr->set_mm1OutSize(mm1OutSize);
    tiling_cpu_ptr->set_smOnlineOutSize(smOnlineOutSize);
    tiling_cpu_ptr->set_mm2OutSize(mm2OutSize);
    tiling_cpu_ptr->set_UpdateSize(UpdateSize);

    at::Tensor tiling_npu_tensor = tiling_cpu_tensor.to(at::Device(at::kPrivateUse1));
    at::Tensor softmax_lse =
        at::empty({batch_size, num_heads, seqlen_q}, at::device(at::kPrivateUse1).dtype(at::kFloat));

    uint8_t* maskDevice = nullptr;
    at::Tensor mask_npu_tensor;
    if (is_causal) {
        uint32_t mask_row = KernelCommon::MAX_KV_STACK_LEN;
        uint32_t mask_col = 2 * KernelCommon::MAX_KV_STACK_LEN;
        at::Tensor mask_cpu_tensor = at::ones({mask_row, mask_col}, at::device(c10::kCPU).dtype(at::kByte));
        mask_cpu_tensor = at::triu(mask_cpu_tensor, 1);
        mask_npu_tensor = mask_cpu_tensor.to(at::Device(at::kPrivateUse1));
        maskDevice = static_cast<uint8_t*>(mask_npu_tensor.data_ptr());
    }

    uint64_t fftsAddr = 0;
    uint32_t fftsLen = 0;
    rtError_t error = rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);

    auto qDevice = static_cast<uint8_t*>(q.data_ptr());
    auto kDevice = static_cast<uint8_t*>(kcache.data_ptr());
    auto vDevice = static_cast<uint8_t*>(vcache.data_ptr());
    uint8_t* qSeqDevice = nullptr;
    uint8_t* blockTableDevice = nullptr;
    if (paged_KV) {
        blockTableDevice = static_cast<uint8_t*>(block_table.data_ptr());
    }
    auto oDevice = static_cast<uint8_t*>(out.data_ptr());
    auto kvSeqDevice = static_cast<uint8_t*>(seqlens_k.data_ptr());
    auto workspaceDevice = static_cast<uint8_t*>(workspace_tensor.data_ptr());
    auto tilingDevice = static_cast<uint8_t*>(tiling_npu_tensor.data_ptr());
    auto softmaxLseDevice = static_cast<uint8_t*>(softmax_lse.data_ptr());

    int dispatch_code = (static_cast<int>(paged_KV) << 2) | (static_cast<int>(is_causal) << 1);
    if (is_bf16) {
        switch (dispatch_code) {
            CASE_FA(bfloat16_t, 0, 0, 0);
            CASE_FA(bfloat16_t, 0, 1, 0);
            CASE_FA(bfloat16_t, 1, 0, 0);
            CASE_FA(bfloat16_t, 1, 1, 0);
        }
    } else {
        switch (dispatch_code) {
            CASE_FA(half, 0, 0, 0);
            CASE_FA(half, 0, 1, 0);
            CASE_FA(half, 1, 0, 0);
            CASE_FA(half, 1, 1, 0);
        }
    }

    return {out, softmax_lse};
}

std::vector<at::Tensor> mha_fwd(
    at::Tensor& q,                            // batch_size x seqlen_q x num_heads x head_size
    const at::Tensor& k,                      // batch_size x seqlen_k x num_heads_k x head_size
    const at::Tensor& v,                      // batch_size x seqlen_k x num_heads_k x head_size
    std::optional<at::Tensor>& out_,          // batch_size x seqlen_q x num_heads x head_size
    std::optional<at::Tensor>& alibi_slopes_, // num_heads or batch_size x num_heads
    const float p_dropout, const float softmax_scale, bool is_causal, int window_size_left, int window_size_right,
    const float softcap, const bool return_softmax, std::optional<at::Generator> gen_)
{
    const c10::OptionalDeviceGuard device_guard(device_of(q));
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    auto q_dtype = q.dtype();
    bool is_bf16 = q.dtype() == torch::kBFloat16;
    // parameters checking
    TORCH_CHECK(
        q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
        "FlashAttention only support fp16 and bf16 data type");
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");

    // block unsupported params
    TORCH_CHECK(!alibi_slopes_.has_value(), "NPU FlashAttention does not support alibi_slopes.");
    TORCH_CHECK(p_dropout == 0.0, "NPU FlashAttention does not support dropout.");
    TORCH_CHECK(softcap == 0.0, "NPU FlashAttention does not support softcap.");
    TORCH_CHECK(window_size_left == -1, "NPU FlashAttention does not support window_size_left.");
    TORCH_CHECK(window_size_right == -1, "NPU FlashAttention does not support window_size_right.");

    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    const auto sizes = q.sizes();
    const int batch_size = sizes[0];
    int seqlen_q = sizes[1];
    int num_heads = sizes[2];
    const int head_size = sizes[3];
    const int seqlen_k = k.size(1);
    const int num_heads_k = k.size(2);
    TORCH_CHECK(batch_size > 0, "batch size must be positive");
    TORCH_CHECK(head_size <= 256, "FlashAttention only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    // init output tensors
    at::Tensor out = (out_.has_value()) ? out_.value() : torch::empty_like(q);
    auto opts = q.options().device(at::kPrivateUse1);
    auto p = torch::empty({0}, opts);
    auto rng_state = torch::empty({2}, at::device(c10::kCPU).dtype(torch::kInt64));

    // init mask
    uint8_t* maskDevice = nullptr;
    at::Tensor mask_npu_tensor;
    if (is_causal) {
        uint32_t mask_row = KernelCommon::MAX_KV_STACK_LEN;
        uint32_t mask_col = 2 * KernelCommon::MAX_KV_STACK_LEN;
        at::Tensor mask_cpu_tensor = at::ones({mask_row, mask_col}, at::device(c10::kCPU).dtype(at::kByte));
        mask_cpu_tensor = at::triu(mask_cpu_tensor, 1);
        mask_npu_tensor = mask_cpu_tensor.to(at::Device(at::kPrivateUse1));
        maskDevice = static_cast<uint8_t*>(const_cast<void*>(mask_npu_tensor.data_ptr()));
    }

    // init softmax lse
    at::Tensor softmax_lse =
        at::empty({batch_size, num_heads, seqlen_q}, at::device(at::kPrivateUse1).dtype(at::kFloat));
    auto softmaxLseDevice = static_cast<uint8_t*>(const_cast<void*>(softmax_lse.data_ptr()));

    // ffts related
    uint64_t fftsAddr = 0;
    uint32_t fftsLen = 0;
    rtError_t error = rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);

    // set worksapce
    uint32_t coreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    uint64_t WORKSPACE_BLOCK_SIZE_DB = KernelCommon::WORKSPACE_BLOCK_SIZE_DB;
    uint64_t DOUBLE_BUFFER = KernelCommon::DOUBLE_BUFFER;
    uint64_t mm1OutSize = static_cast<uint64_t>(coreNum) * WORKSPACE_BLOCK_SIZE_DB * 4 * DOUBLE_BUFFER;
    uint64_t smOnlineOutSize = static_cast<uint64_t>(coreNum) * WORKSPACE_BLOCK_SIZE_DB * 2 * DOUBLE_BUFFER;
    uint64_t mm2OutSize = static_cast<uint64_t>(coreNum) * WORKSPACE_BLOCK_SIZE_DB * 4 * DOUBLE_BUFFER;
    uint64_t UpdateSize = static_cast<uint64_t>(coreNum) * WORKSPACE_BLOCK_SIZE_DB * 4;
    int64_t workspaceSize = mm1OutSize + smOnlineOutSize + mm2OutSize + UpdateSize;
    at::Tensor workspace_tensor = at::empty({workspaceSize}, at::device(at::kPrivateUse1).dtype(at::kByte));

    // tiling
    at::Tensor tiling_cpu_tensor = at::empty({1024}, at::device(c10::kCPU).dtype(at::kByte));
    auto tiling_cpu_ptr = reinterpret_cast<FAInferTilingData*>(tiling_cpu_tensor.data_ptr<uint8_t>());
    tiling_cpu_ptr->set_batch(static_cast<uint32_t>(batch_size));
    tiling_cpu_ptr->set_numHeads(static_cast<uint32_t>(num_heads));
    tiling_cpu_ptr->set_kvHeads(static_cast<uint32_t>(num_heads_k));
    tiling_cpu_ptr->set_embeddingSize(static_cast<uint32_t>(head_size));
    tiling_cpu_ptr->set_numBlocks(static_cast<uint32_t>(0));
    tiling_cpu_ptr->set_blockSize(static_cast<uint32_t>(128));
    tiling_cpu_ptr->set_maxNumBlocksPerBatch(static_cast<uint32_t>(0));
    tiling_cpu_ptr->set_scaleValue(softmax_scale);
    tiling_cpu_ptr->set_maxQSeqlen(seqlen_q);
    tiling_cpu_ptr->set_mm1OutSize(mm1OutSize);
    tiling_cpu_ptr->set_smOnlineOutSize(smOnlineOutSize);
    tiling_cpu_ptr->set_mm2OutSize(mm2OutSize);
    tiling_cpu_ptr->set_UpdateSize(UpdateSize);

    // device ptrs
    auto qDevice = static_cast<uint8_t*>(const_cast<void*>(q.data_ptr()));
    auto kDevice = static_cast<uint8_t*>(const_cast<void*>(k.data_ptr()));
    auto vDevice = static_cast<uint8_t*>(const_cast<void*>(v.data_ptr()));
    at::Tensor seqlenk_npu_tensor = at::full({batch_size}, seqlen_k, at::device(at::kPrivateUse1).dtype(at::kInt));
    at::Tensor tiling_npu_tensor = tiling_cpu_tensor.to(at::Device(at::kPrivateUse1));
    auto kvSeqDevice = static_cast<uint8_t*>(const_cast<void*>(seqlenk_npu_tensor.data_ptr()));
    auto workspaceDevice = static_cast<uint8_t*>(const_cast<void*>(workspace_tensor.data_ptr()));
    auto tilingDevice = static_cast<uint8_t*>(const_cast<void*>(tiling_npu_tensor.data_ptr()));
    auto oDevice = static_cast<uint8_t*>(const_cast<void*>(out.data_ptr()));
    uint8_t* qSeqDevice = nullptr;
    uint8_t* blockTableDevice = nullptr; // will not be used in non-kvcahce fwd api

    // run kernel
    int dispatch_code = (static_cast<int>(is_causal) << 1);
    if (is_bf16) {
        switch (dispatch_code) {
            CASE_FA(bfloat16_t, 0, 0, 0);
            CASE_FA(bfloat16_t, 0, 1, 0);
        }
    } else {
        switch (dispatch_code) {
            CASE_FA(half, 0, 0, 0);
            CASE_FA(half, 0, 1, 0);
        }
    }

    return {out, softmax_lse, p, rng_state};
}

std::vector<at::Tensor> mha_varlen_fwd(
    at::Tensor& q,                   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    const at::Tensor& k,             // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x
                                     // page_block_size x num_heads_k x head_size if there's a block_table.
    const at::Tensor& v,             // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x
                                     // page_block_size x num_heads_k x head_size if there's a block_table.
    std::optional<at::Tensor>& out_, // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    const at::Tensor& cu_seqlens_q,  // b+1
    const at::Tensor& cu_seqlens_k,  // b+1
    std::optional<at::Tensor>&
        seqused_k_, // b. If given, only this many elements of each batch element's keys are used.
    std::optional<const at::Tensor>& leftpad_k_, // batch_size
    std::optional<at::Tensor>& block_table_,     // batch_size x max_num_blocks_per_seq
    std::optional<at::Tensor>& alibi_slopes_,    // num_heads or b x num_heads
    int max_seqlen_q, const int max_seqlen_k, const float p_dropout, const float softmax_scale, const bool zero_tensors,
    bool is_causal, int window_size_left, int window_size_right, const float softcap, const bool return_softmax,
    std::optional<at::Generator> gen_)
{
    const c10::OptionalDeviceGuard device_guard(device_of(q));
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    at::Tensor tiling_cpu_tensor = at::empty({1024}, at::device(c10::kCPU).dtype(at::kByte));
    auto tiling_cpu_ptr = reinterpret_cast<FAInferTilingData*>(tiling_cpu_tensor.data_ptr<uint8_t>());
    uint32_t coreNum = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();

    bool is_bf16 = q.dtype() == torch::kBFloat16;
    bool is_fp16 = q.dtype() == torch::kFloat16;
    const bool paged_KV = block_table_.has_value();

    TORCH_CHECK(is_bf16 || is_fp16, "NPU FlashAttention only supports Float16 or BFloat16.");
    TORCH_CHECK(!block_table_.has_value(), "NPU FlashAttention does not support paged KV cache with block table.");
    TORCH_CHECK(!seqused_k_.has_value(), "NPU FlashAttention does not support seqused_k.");
    TORCH_CHECK(!leftpad_k_.has_value(), "NPU FlashAttention does not support leftpad_k.");
    TORCH_CHECK(!alibi_slopes_.has_value(), "NPU FlashAttention does not support alibi_slopes.");
    TORCH_CHECK(p_dropout == 0.0, "NPU FlashAttention does not support dropout.");
    TORCH_CHECK(!zero_tensors, "NPU FlashAttention does not support zero_tensors.");
    TORCH_CHECK(softcap == 0.0, "NPU FlashAttention does not support softcap.");
    TORCH_CHECK(window_size_left == -1, "NPU FlashAttention does not support window_size_left.");
    TORCH_CHECK(window_size_right == -1, "NPU FlashAttention does not support window_size_right.");
    TORCH_CHECK(k.dtype() == q.dtype(), "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q.dtype(), "query and value must have the same dtype");
    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");

    at::Tensor seqlens_k, leftpad_k, alibi_slopes, block_table;
    if (seqused_k_.has_value()) {
        seqlens_k = seqused_k_.value();
    }
    if (leftpad_k_.has_value()) {
        leftpad_k = leftpad_k_.value();
    }
    if (alibi_slopes_.has_value()) {
        alibi_slopes = alibi_slopes_.value();
    }
    if (paged_KV) {
        block_table = block_table_.value();
    }

    const auto sizes = q.sizes();
    int total_q = sizes[0];
    int num_heads = sizes[1];
    const int head_size = sizes[2];
    const int batch_size = cu_seqlens_q.numel() - 1;

    const int max_num_blocks_per_seq = !paged_KV ? 0 : block_table.size(1);
    const int num_blocks = !paged_KV ? 0 : k.size(0);
    const int page_block_size = !paged_KV ? 128 : k.size(1);
    const int num_heads_k = k.size(1);

    TORCH_CHECK(head_size <= 256, "FlashAttention only supports head dimension at most 256");
    TORCH_CHECK(num_heads % num_heads_k == 0, "Number of heads in key/value must divide number of heads in query");

    at::Tensor out = (out_.has_value()) ? out_.value() : torch::empty_like(q);
    auto opts = q.options().device(at::kPrivateUse1);
    auto p = torch::empty({0}, opts);
    auto rng_state = torch::empty({2}, at::device(c10::kCPU).dtype(torch::kInt64));

    tiling_cpu_ptr->set_batch(static_cast<uint32_t>(batch_size));
    tiling_cpu_ptr->set_numHeads(static_cast<uint32_t>(num_heads));
    tiling_cpu_ptr->set_kvHeads(static_cast<uint32_t>(num_heads_k));
    tiling_cpu_ptr->set_embeddingSize(static_cast<uint32_t>(head_size));
    tiling_cpu_ptr->set_numBlocks(static_cast<uint32_t>(num_blocks));
    tiling_cpu_ptr->set_blockSize(static_cast<uint32_t>(page_block_size));
    tiling_cpu_ptr->set_maxNumBlocksPerBatch(static_cast<uint32_t>(max_num_blocks_per_seq));
    tiling_cpu_ptr->set_scaleValue(softmax_scale);
    tiling_cpu_ptr->set_maxQSeqlen(max_seqlen_q);

    uint64_t WORKSPACE_BLOCK_SIZE_DB = KernelCommon::WORKSPACE_BLOCK_SIZE_DB;
    uint64_t DOUBLE_BUFFER = KernelCommon::DOUBLE_BUFFER;
    uint64_t mm1OutSize = static_cast<uint64_t>(coreNum) * WORKSPACE_BLOCK_SIZE_DB * 4 * DOUBLE_BUFFER;
    uint64_t smOnlineOutSize = static_cast<uint64_t>(coreNum) * WORKSPACE_BLOCK_SIZE_DB * 2 * DOUBLE_BUFFER;
    uint64_t mm2OutSize = static_cast<uint64_t>(coreNum) * WORKSPACE_BLOCK_SIZE_DB * 4 * DOUBLE_BUFFER;
    uint64_t UpdateSize = static_cast<uint64_t>(coreNum) * WORKSPACE_BLOCK_SIZE_DB * 4;
    int64_t workspaceSize = mm1OutSize + smOnlineOutSize + mm2OutSize + UpdateSize;
    at::Tensor workspace_tensor = at::empty({workspaceSize}, at::device(at::kPrivateUse1).dtype(at::kByte));

    tiling_cpu_ptr->set_mm1OutSize(mm1OutSize);
    tiling_cpu_ptr->set_smOnlineOutSize(smOnlineOutSize);
    tiling_cpu_ptr->set_mm2OutSize(mm2OutSize);
    tiling_cpu_ptr->set_UpdateSize(UpdateSize);

    at::Tensor tiling_npu_tensor = tiling_cpu_tensor.to(at::Device(at::kPrivateUse1));
    at::Tensor softmax_lse = at::empty({num_heads, total_q}, at::device(at::kPrivateUse1).dtype(at::kFloat));

    // attention mask
    uint8_t* maskDevice = nullptr;
    at::Tensor mask_npu_tensor;
    if (is_causal) {
        uint32_t mask_row = KernelCommon::MAX_KV_STACK_LEN;
        uint32_t mask_col = 2 * KernelCommon::MAX_KV_STACK_LEN;
        at::Tensor mask_cpu_tensor = at::ones({mask_row, mask_col}, at::device(c10::kCPU).dtype(at::kByte));
        mask_cpu_tensor = at::triu(mask_cpu_tensor, 1);
        mask_npu_tensor = mask_cpu_tensor.to(at::Device(at::kPrivateUse1));
        maskDevice = static_cast<uint8_t*>(const_cast<void*>(mask_npu_tensor.data_ptr()));
    }

    uint64_t fftsAddr = 0;
    uint32_t fftsLen = 0;
    rtError_t error = rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);

    auto qDevice = static_cast<uint8_t*>(const_cast<void*>(q.data_ptr()));
    auto kDevice = static_cast<uint8_t*>(const_cast<void*>(k.data_ptr()));
    auto vDevice = static_cast<uint8_t*>(const_cast<void*>(v.data_ptr()));
    uint8_t* blockTableDevice = nullptr;
    if (paged_KV) {
        blockTableDevice = static_cast<uint8_t*>(const_cast<void*>(block_table.data_ptr()));
    }
    auto oDevice = static_cast<uint8_t*>(const_cast<void*>(out.data_ptr()));
    auto qSeqDevice = static_cast<uint8_t*>(const_cast<void*>(cu_seqlens_q.data_ptr()));
    auto kvSeqDevice = static_cast<uint8_t*>(const_cast<void*>(cu_seqlens_k.data_ptr()));
    auto workspaceDevice = static_cast<uint8_t*>(const_cast<void*>(workspace_tensor.data_ptr()));
    auto tilingDevice = static_cast<uint8_t*>(const_cast<void*>(tiling_npu_tensor.data_ptr()));
    auto softmaxLseDevice = static_cast<uint8_t*>(const_cast<void*>(softmax_lse.data_ptr()));

    int dispatch_code = (static_cast<int>(paged_KV) << 2) | (static_cast<int>(is_causal) << 1) | (1 << 0);
    if (is_bf16) {
        switch (dispatch_code) {
            CASE_FA(bfloat16_t, 0, 0, 1);
            CASE_FA(bfloat16_t, 0, 1, 1);
            CASE_FA(bfloat16_t, 1, 0, 1);
            CASE_FA(bfloat16_t, 1, 1, 1);
        }
    } else {
        switch (dispatch_code) {
            CASE_FA(half, 0, 0, 1);
            CASE_FA(half, 0, 1, 1);
            CASE_FA(half, 1, 0, 1);
            CASE_FA(half, 1, 1, 1);
        }
    }

    return {out, softmax_lse, p, rng_state};
}

PYBIND11_MODULE(flash_attn, m)
{
    m.doc() = "FlashAttention";
    m.def("fwd_kvcache", &mha_fwd_kvcache, "Forward pass, with KV-cache");
    m.def("fwd", &mha_fwd, "Forward pass");
    m.def("varlen_fwd", &mha_varlen_fwd, "Forward pass (variable length)");
}