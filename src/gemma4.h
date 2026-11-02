#pragma once

#include "backend.h"
#include "model_file.h"
#include "tokenizer.h"

#include <cstdint>
#include <string>
#include <vector>

// Gemma 4's language model (the E2B / E4B layout: per-layer embeddings,
// sliding-window and global attention, the last layers sharing the caches of
// earlier ones) with one conversation's K/V cache.
//
// decode() runs tokens or embeddings at a position, which drops everything
// from there on first, so taking back the end of the conversation is
// decoding from an earlier position. The sliding-window layers only keep the
// recent part of the conversation; a rollback further back than a few
// hundred positions needs a guard at that position, set before the
// conversation moved on (guard()).
class Gemma4 {
public:
    Gemma4() = default;
    ~Gemma4();
    Gemma4(const Gemma4 &) = delete;
    Gemma4 & operator=(const Gemma4 &) = delete;

    bool load(const std::string & path, const Backend & backend, int n_ctx);

    // The most tokens one decode() takes.
    static constexpr int MAX_BATCH = 512;

    // Decodes `n` tokens, or `n` embeddings of n_embd() floats, at positions
    // pos .. pos + n - 1, after the first `pos` of the conversation. With
    // `layers`, also fills `states` with the output of those layers for the
    // last of them (n_layer() rows of n_embd(), the others zero).
    bool decode(const int32_t * tokens, const float * embd, int n, int pos,
                const std::vector<int> * layers = nullptr, std::vector<float> * states = nullptr);

    // Logits for the last position decoded.
    const std::vector<float> & logits() const { return logits_; }

    // Keeps what a rollback to `pos` needs, until the slot's next guard or a
    // decode before `pos`; 0 clears it. Two slots, e.g. the end of the system
    // prompt and the start of the current turn.
    static constexpr int GUARDS = 2;
    void guard(int slot, int pos);

    // Whether the file has the model's audio encoder in it too (a combined
    // file, training/turn/combine.py), to load with AudioEncoder.
    bool has_audio_encoder() const { return file_.find_key("clip.audio.projector_type") >= 0; }

    const Tokenizer & tokenizer() const { return tokenizer_; }
    int n_layer() const { return (int) layers_.size(); }
    int n_embd() const { return n_embd_; }
    int n_vocab() const { return n_vocab_; }
    int n_ctx() const { return n_ctx_; }

private:
    struct Layer {
        bool swa = false;
        int  kv  = 0;  // the cache it reads (its own, or an earlier layer's)
        int  n_head = 0, n_head_kv = 0, head_dim = 0;
        ggml_tensor * attn_norm = nullptr;
        ggml_tensor * wq = nullptr, * wk = nullptr, * wv = nullptr, * wo = nullptr;
        ggml_tensor * q_norm = nullptr, * k_norm = nullptr;
        ggml_tensor * attn_post_norm = nullptr;
        ggml_tensor * ffn_norm = nullptr, * ffn_gate = nullptr, * ffn_up = nullptr, * ffn_down = nullptr;
        ggml_tensor * ffn_post_norm = nullptr;
        ggml_tensor * ple_gate = nullptr, * ple_proj = nullptr, * ple_post_norm = nullptr;
        ggml_tensor * out_scale = nullptr;
    };

    struct Cache {
        ggml_tensor * k = nullptr;  // [head_dim * n_head_kv, cells], f16
        ggml_tensor * v = nullptr;
    };

    // The window below a rollback point, saved once the sliding-window
    // caches are about to drop it.
    struct Guard {
        int  pos   = -1;
        bool saved = false;
        std::vector<ggml_tensor *> k, v;  // per cache, [head_dim * n_head_kv, n_swa]
    };

    bool place(int pos, int n);
    // Copies `n` cells of each sliding-window cache, from cell `src` to `dst`;
    // with a guard, from or into it.
    bool copy_cells(int src, int dst, int n, Guard * from = nullptr, Guard * to = nullptr);
    ggml_tensor * norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w) const;

    ModelFile file_;
    Tokenizer tokenizer_;
    ggml_backend_sched_t sched_ = nullptr;
    bool flash_attn_ = false;

    int   n_embd_ = 0, n_vocab_ = 0, n_ple_ = 0, n_swa_ = 0, n_ctx_ = 0;
    float eps_ = 1e-6f, softcap_ = 0.f, rope_base_ = 0.f, rope_base_swa_ = 0.f;
    int   n_ctx_train_ = 0;

    ggml_tensor *       tok_embd_ = nullptr;
    const ggml_tensor * ple_embd_ = nullptr;  // read on the host, a row per token
    ggml_tensor *       ple_proj_ = nullptr, * ple_proj_norm_ = nullptr;
    ggml_tensor *       rope_freqs_ = nullptr;
    ggml_tensor *       output_norm_ = nullptr;
    std::vector<Layer>  layers_;

    // The caches. Global layers hold every position at its own cell; the
    // sliding-window layers hold positions swa_base_ .. at cells 0 ..
    ggml_context *        cache_ctx_ = nullptr;
    ggml_backend_buffer_t cache_buf_ = nullptr;
    std::vector<Cache>    caches_;
    int swa_cells_ = 0;
    int swa_base_  = 0;
    Guard guards_[GUARDS];

    std::vector<float> logits_;
    std::vector<float> ple_rows_;
    std::vector<ggml_fp16_t> mask_, mask_swa_;
};
