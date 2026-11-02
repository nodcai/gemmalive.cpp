#pragma once

#include "backend.h"
#include "mel.h"
#include "model_file.h"

#include <string>
#include <vector>

// Gemma 4's audio encoder: a conformer over log-mel features, four frames
// (40 ms) per position, and its projection into the language model's
// embeddings, one per position.
//
// Its attention sees the 12 positions before each chunk of 12 and nothing
// after it, and its convolutions only look back, so a chunk (480 ms) is
// final once its audio was heard. It is encoded on its own, with the
// state the previous chunk left.
class AudioEncoder {
public:
    static constexpr int CHUNK = 12;

    // What a chunk leaves the next one: per layer, the keys and values of
    // its positions and the last inputs of the depthwise convolution.
    struct State {
        int n_chunks = 0;
        std::vector<std::vector<float>> k, v, conv;
    };

    AudioEncoder() = default;
    ~AudioEncoder();
    AudioEncoder(const AudioEncoder &) = delete;
    AudioEncoder & operator=(const AudioEncoder &) = delete;

    bool load(const std::string & path, const Backend & backend);

    int n_embd() const { return n_out_; }
    State new_state() const;

    // Encodes the next chunk from `mel`, which holds at least its frames:
    // CHUNK positions, or with `last` what is left of them. Appends
    // n_embd() floats per position to `out`.
    bool encode(const Mel & mel, State & st, bool last, std::vector<float> & out);

private:
    struct Layer {
        ggml_tensor * ffn_norm, * ffn_up, * ffn_down, * ffn_post_norm;
        ggml_tensor * attn_norm, * q, * k, * v, * o, * k_rel, * per_dim_scale, * attn_post_norm;
        ggml_tensor * conv_norm, * conv_pw1, * conv_dw, * conv_dw_norm, * conv_pw2;
        ggml_tensor * ffn_norm_1, * ffn_up_1, * ffn_down_1, * ffn_post_norm_1;
        ggml_tensor * out_norm;
    };
    struct Clamp {
        float in_min, in_max, out_min, out_max;
    };

    ggml_tensor * mm(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x) const;
    ggml_tensor * norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w) const;
    ggml_tensor * ffn(ggml_context * ctx, ggml_tensor * x, ggml_tensor * n, ggml_tensor * up, ggml_tensor * down,
                      ggml_tensor * post) const;

    ModelFile file_;
    ggml_backend_sched_t sched_ = nullptr;

    int n_embd_ = 0, n_head_ = 0, head_dim_ = 0, n_conv_ = 0, n_out_ = 0;
    ggml_tensor * stem_conv_[2] = {}, * stem_norm_[2] = {};
    ggml_tensor * stem_proj_ = nullptr;
    ggml_tensor * out_proj_ = nullptr, * out_proj_b_ = nullptr;
    ggml_tensor * embd_norm_ = nullptr, * embd_proj_ = nullptr;
    std::vector<Layer> layers_;
    std::vector<std::pair<const ggml_tensor *, Clamp>> clamps_;
    std::vector<float> pos_emb_;
};

// Audio on its way into embeddings: its features and the encoder's state.
// A copy goes on independently, e.g. to finish() it while this one goes on.
class AudioStream {
public:
    explicit AudioStream(AudioEncoder & encoder);

    // At most 30 s.
    void push(const float * pcm, size_t n);

    // The embeddings of chunks that are complete, n_embd() floats each,
    // appended to `out`; how many in `n`.
    bool pull(std::vector<float> & out, int & n);
    // The rest, up to the end of the audio. Nothing can follow.
    bool finish(std::vector<float> & out, int & n);

    size_t n_samples() const { return mel_.n_samples(); }

private:
    AudioEncoder *       encoder_;
    Mel                  mel_;
    AudioEncoder::State  state_;
    bool                 finished_ = false;
};
