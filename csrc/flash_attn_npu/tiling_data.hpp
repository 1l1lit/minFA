/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Modified by Minghua Shen, 2026
 */

#ifndef FLASH_ATTENTION_REGULAR_H
#define FLASH_ATTENTION_REGULAR_H

struct FAInferTilingData {
    uint32_t numHeads;
    uint32_t embeddingSize;
    uint32_t numBlocks;
    uint32_t blockSize;
    uint32_t maxQSeqlen;
    uint32_t maxKvSeqlen;
    uint32_t kvHeads;
    uint32_t batch;
    uint32_t maxNumBlocksPerBatch;
    uint64_t mm1OutSize;
    uint64_t smOnlineOutSize;
    uint64_t mm2OutSize;
    uint64_t UpdateSize;
    uint64_t workspaceSize;
    float scaleValue;

    uint32_t get_numHeads() const { return numHeads; }
    uint32_t get_embeddingSize() const { return embeddingSize; }
    uint32_t get_numBlocks() const { return numBlocks; }
    uint32_t get_blockSize() const { return blockSize; }
    uint32_t get_maxQSeqlen() const { return maxQSeqlen; }
    uint32_t get_maxKvSeqlen() const { return maxKvSeqlen; }
    uint32_t get_kvHeads() const { return kvHeads; }
    uint32_t get_batch() const { return batch; }
    uint32_t get_maxNumBlocksPerBatch() const { return maxNumBlocksPerBatch; }
    uint64_t get_mm1OutSize() const { return mm1OutSize; }
    uint64_t get_smOnlineOutSize() const { return smOnlineOutSize; }
    uint64_t get_mm2OutSize() const { return mm2OutSize; }
    uint64_t get_UpdateSize() const { return UpdateSize; }
    uint64_t get_workspaceSize() const { return workspaceSize; }
    float get_scaleValue() const { return scaleValue; }

    void set_numHeads(uint32_t value) { numHeads = value; }
    void set_embeddingSize(uint32_t value) { embeddingSize = value; }
    void set_numBlocks(uint32_t value) { numBlocks = value; }
    void set_blockSize(uint32_t value) { blockSize = value; }
    void set_maxQSeqlen(uint32_t value) { maxQSeqlen = value; }
    void set_maxKvSeqlen(uint32_t value) { maxKvSeqlen = value; }
    void set_kvHeads(uint32_t value) { kvHeads = value; }
    void set_batch(uint32_t value) { batch = value; }
    void set_maxNumBlocksPerBatch(uint32_t value) { maxNumBlocksPerBatch = value; }
    void set_mm1OutSize(uint64_t value) { mm1OutSize = value; }
    void set_smOnlineOutSize(uint64_t value) { smOnlineOutSize = value; }
    void set_mm2OutSize(uint64_t value) { mm2OutSize = value; }
    void set_UpdateSize(uint64_t value) { UpdateSize = value; }
    void set_workspaceSize(uint64_t value) { workspaceSize = value; }
    void set_scaleValue(float value) { scaleValue = value; }
};

#endif