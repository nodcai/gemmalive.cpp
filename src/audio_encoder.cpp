#include "audio_encoder.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>

static constexpr size_t GRAPH_SIZE = 4096;
static constexpr float  EPS        = 1e-6f;
static constexpr int    MAX_PAST   = 12;  // positions before a chunk its attention sees
static constexpr int    CONV_PAST  = 4;   // the depthwise convolution's kernel, less one
static constexpr int    FRAMES     = 4;   // mel frames per position
static constexpr float  SOFTCAP    = 50.f;

// Positions of `n` frames after the stem's two stride-2 convolutions.
static int n_positions(int n) {
    for (int i = 0; i < 2; i++) {
        n = (n - 1) / 2 + 1;
    }
    return n;
}

AudioEncoder::~AudioEncoder() {
    if (sched_) ggml_backend_sched_free(sched_);
}

bool AudioEncoder::load(const std::string & path, const Backend & backend) {
    if (!file_.open(path)) {
        return false;
    }
    if (file_.str("clip.audio.projector_type") != "gemma4a") {
        fprintf(stderr, "audio: %s has no Gemma 4 audio encoder\n", path.c_str());
        return false;
    }
    const auto audio = [](const std::string & name) { return name.rfind("a.", 0) == 0 || name.rfind("mm.a.", 0) == 0; };
    if (!file_.load(backend.device(), audio)) {
        return false;
    }

    n_embd_   = (int) file_.u32("clip.audio.embedding_length");
    n_head_   = (int) file_.u32("clip.audio.attention.head_count");
    head_dim_ = n_embd_ / n_head_;
    const int n_layer = (int) file_.u32("clip.audio.block_count");

    bool ok = true;
    auto get = [&](const std::string & name, bool required = true) {
        ggml_tensor * t = file_.get(name);
        if (!t && required) {
            fprintf(stderr, "audio: %s is missing\n", name.c_str());
            ok = false;
        }
        if (t && ggml_n_dims(t) == 2) {
            // Some projections are clamped on the way in and out.
            const std::string base = name.substr(0, name.size() - 7);  // less ".weight"
            const Clamp c = { file_.scalar(base + ".input_min", -FLT_MAX), file_.scalar(base + ".input_max", FLT_MAX),
                              file_.scalar(base + ".output_min", -FLT_MAX), file_.scalar(base + ".output_max", FLT_MAX) };
            if (c.in_min > -FLT_MAX || c.in_max < FLT_MAX || c.out_min > -FLT_MAX || c.out_max < FLT_MAX) {
                clamps_.push_back({ t, c });
            }
        }
        return t;
    };
    for (int i = 0; i < 2; i++) {
        stem_conv_[i] = get("a.conv1d." + std::to_string(i) + ".weight");
        stem_norm_[i] = get("a.conv1d." + std::to_string(i) + ".norm.weight");
    }
    stem_proj_  = get("a.input_projection.weight");
    out_proj_   = get("a.pre_encode.out.weight");
    out_proj_b_ = get("a.pre_encode.out.bias", false);
    embd_norm_  = get("mm.a.soft_emb_norm.weight", false);
    embd_proj_  = get("mm.a.input_projection.weight");

    layers_.resize(n_layer);
    for (int il = 0; il < n_layer; il++) {
        Layer & l = layers_[il];
        auto w = [&](const char * t, bool required = true) {
            return get("a.blk." + std::to_string(il) + "." + t + ".weight", required);
        };
        l.ffn_norm        = w("ffn_norm");
        l.ffn_up          = w("ffn_up");
        l.ffn_down        = w("ffn_down");
        l.ffn_post_norm   = w("ffn_post_norm", false);
        l.attn_norm       = w("attn_pre_norm");
        l.q               = w("attn_q");
        l.k               = w("attn_k");
        l.v               = w("attn_v");
        l.o               = w("attn_out");
        l.k_rel           = w("attn_k_rel");
        l.per_dim_scale   = w("per_dim_scale", false);
        l.attn_post_norm  = w("attn_post_norm", false);
        l.conv_norm       = w("conv_norm");
        l.conv_pw1        = w("conv_pw1");
        l.conv_dw         = w("conv_dw");
        l.conv_dw_norm    = w("norm_conv", false);
        l.conv_pw2        = w("conv_pw2");
        l.ffn_norm_1      = w("ffn_norm_1");
        l.ffn_up_1        = w("ffn_up_1");
        l.ffn_down_1      = w("ffn_down_1");
        l.ffn_post_norm_1 = w("ffn_post_norm_1", false);
        l.out_norm        = w("ln2", false);
    }
    if (!ok) {
        return false;
    }
    n_conv_ = (int) layers_[0].conv_pw1->ne[1] / 2;
    n_out_  = (int) embd_proj_->ne[1];

    // Sinusoidal embeddings of the relative distances MAX_PAST .. 0.
    pos_emb_.assign((size_t) n_embd_ * (MAX_PAST + 1), 0.f);
    const int   n_ts = n_embd_ / 2;
    const float step = std::log(10000.f) / std::max(n_ts - 1, 1);
    for (int p = 0; p <= MAX_PAST; p++) {
        for (int i = 0; i < n_ts; i++) {
            const float x = (float) (MAX_PAST - p) * std::exp(-(float) i * step);
            pos_emb_[(size_t) p * n_embd_ + i]        = std::sin(x);
            pos_emb_[(size_t) p * n_embd_ + i + n_ts] = std::cos(x);
        }
    }

    sched_ = backend.new_sched(GRAPH_SIZE);
    return sched_ != nullptr;
}

AudioEncoder::State AudioEncoder::new_state() const {
    State st;
    st.k.assign(layers_.size(), std::vector<float>((size_t) n_embd_ * MAX_PAST, 0.f));
    st.v.assign(layers_.size(), std::vector<float>((size_t) n_embd_ * MAX_PAST, 0.f));
    st.conv.assign(layers_.size(), std::vector<float>((size_t) n_conv_ * CONV_PAST, 0.f));
    return st;
}

ggml_tensor * AudioEncoder::mm(ggml_context * ctx, ggml_tensor * w, ggml_tensor * x) const {
    for (const auto & [t, c] : clamps_) {
        if (t == w) {
            x = ggml_clamp(ctx, x, c.in_min, c.in_max);
            return ggml_clamp(ctx, ggml_mul_mat(ctx, w, x), c.out_min, c.out_max);
        }
    }
    return ggml_mul_mat(ctx, w, x);
}

ggml_tensor * AudioEncoder::norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w) const {
    x = ggml_rms_norm(ctx, x, EPS);
    return w ? ggml_mul(ctx, x, w) : x;
}

// A half-step feed-forward block's output, before the residual.
ggml_tensor * AudioEncoder::ffn(ggml_context * ctx, ggml_tensor * x, ggml_tensor * n, ggml_tensor * up,
                                ggml_tensor * down, ggml_tensor * post) const {
    x = norm(ctx, x, n);
    x = mm(ctx, down, ggml_silu(ctx, mm(ctx, up, x)));
    return post ? norm(ctx, x, post) : x;
}

bool AudioEncoder::encode(const Mel & mel, State & st, bool last, std::vector<float> & out) {
    const int K   = st.n_chunks;
    const int f0  = std::max(0, K * CHUNK * FRAMES - 2 * FRAMES);
    const int f1  = last ? mel.n_frames() : (K + 1) * CHUNK * FRAMES;
    const int n   = last ? n_positions(mel.n_frames()) - K * CHUNK : CHUNK;
    const int S   = MAX_PAST + n;  // keys: the past, then the chunk
    const int R   = MAX_PAST + 1;  // relative distances
    const int T   = f1 - f0;
    if (n <= 0) {
        return true;
    }
    if (n > CHUNK || f1 > mel.n_frames()) {
        fprintf(stderr, "audio: chunk %d is not complete\n", K);
        return false;
    }

    ggml_init_params params = { ggml_tensor_overhead() * GRAPH_SIZE + ggml_graph_overhead_custom(GRAPH_SIZE, false), nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_cgraph *  gf  = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);

    // The stem over the chunk's frames and the 8 before them, which the
    // first two of its positions need; its last `n` positions are the chunk's.
    ggml_tensor * inp = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, T, Mel::N_MEL);
    ggml_set_input(inp);
    ggml_tensor * x = ggml_cont(ctx, ggml_transpose(ctx, inp));
    for (int i = 0; i < 2; i++) {
        x = ggml_conv_2d(ctx, stem_conv_[i], x, 2, 2, 1, 1, 1, 1);
        x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3));
        x = ggml_mul(ctx, ggml_norm(ctx, x, EPS), stem_norm_[i]);
        x = ggml_cont(ctx, ggml_permute(ctx, x, 2, 0, 1, 3));
        x = ggml_relu(ctx, x);
    }
    x = ggml_cont(ctx, ggml_permute(ctx, x, 1, 2, 0, 3));
    x = ggml_reshape_2d(ctx, x, x->ne[0] * x->ne[1], x->ne[2]);
    x = mm(ctx, stem_proj_, x);
    x = ggml_cont(ctx, ggml_view_2d(ctx, x, x->ne[0], n, x->nb[1], (size_t) (x->ne[1] - n) * x->nb[1]));

    ggml_tensor * pos_emb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd_, R);
    ggml_set_input(pos_emb);
    ggml_tensor * mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, S, n, 1);
    ggml_set_input(mask);

    const int D = head_dim_, H = n_head_;
    const float q_scale = 1.f / std::sqrt((float) D) / std::log(2.f);
    const float k_scale = std::log(1.f + std::exp(1.f)) / std::log(2.f);

    const int n_layer = (int) layers_.size();
    std::vector<ggml_tensor *> past_k(n_layer), past_v(n_layer), past_conv(n_layer);
    std::vector<ggml_tensor *> new_k(n_layer, nullptr), new_v(n_layer, nullptr), new_conv(n_layer, nullptr);

    for (int il = 0; il < n_layer; il++) {
        const Layer & l = layers_[il];

        x = ggml_add(ctx, x, ggml_scale(ctx, ffn(ctx, x, l.ffn_norm, l.ffn_up, l.ffn_down, l.ffn_post_norm), 0.5f));

        // Attention over the past positions and the chunk, with relative
        // positions and soft-capped scores.
        {
            ggml_tensor * cur = norm(ctx, x, l.attn_norm);
            ggml_tensor * q = ggml_reshape_3d(ctx, mm(ctx, l.q, cur), D, H, n);
            ggml_tensor * k = ggml_reshape_3d(ctx, mm(ctx, l.k, cur), D, H, n);
            ggml_tensor * v = ggml_reshape_3d(ctx, mm(ctx, l.v, cur), D, H, n);
            q = ggml_scale(ctx, q, q_scale);
            if (l.per_dim_scale) {
                q = ggml_mul(ctx, q, ggml_reshape_3d(ctx, l.per_dim_scale, D, 1, 1));
            }
            k = ggml_scale(ctx, k, k_scale);

            past_k[il] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, H, MAX_PAST);
            past_v[il] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D, H, MAX_PAST);
            ggml_set_input(past_k[il]);
            ggml_set_input(past_v[il]);
            if (!last) {
                new_k[il] = ggml_cont(ctx, k);
                new_v[il] = ggml_cont(ctx, v);
                ggml_set_output(new_k[il]);
                ggml_set_output(new_v[il]);
            }
            ggml_tensor * kw = ggml_concat(ctx, past_k[il], k, 2);  // [D, H, S]
            ggml_tensor * vw = ggml_concat(ctx, past_v[il], v, 2);

            ggml_tensor * q4 = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, q, D, H, n, 1), 0, 3, 1, 2));    // [D, n, 1, H]
            ggml_tensor * k4 = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, kw, D, H, S, 1), 0, 3, 1, 2));   // [D, S, 1, H]
            ggml_tensor * v4 = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_4d(ctx, vw, D, H, S, 1), 1, 3, 0, 2));   // [S, D, 1, H]

            ggml_tensor * scores = ggml_mul_mat(ctx, k4, q4);  // [S, n, 1, H]

            // Relative positions: scores against the R distances, shifted
            // so that each query's distance 0 lines up with its own key.
            ggml_tensor * p = ggml_mul_mat(ctx, l.k_rel, pos_emb);
            p = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, p, D, H, R), 0, 2, 1, 3));  // [D, R, H]
            ggml_tensor * bd = ggml_mul_mat(ctx, p, ggml_reshape_3d(ctx, q4, D, n, H));        // [R, n, H]
            bd = ggml_pad(ctx, ggml_reshape_4d(ctx, bd, R, n, 1, H), S + 1 - R, 0, 0, 0);
            bd = ggml_reshape_3d(ctx, bd, (int64_t) (S + 1) * n, 1, H);
            bd = ggml_cont(ctx, ggml_view_3d(ctx, bd, (int64_t) n * S, 1, H, bd->nb[1], bd->nb[2], 0));
            bd = ggml_reshape_4d(ctx, bd, S, n, 1, H);
            scores = ggml_add(ctx, scores, bd);

            scores = ggml_scale(ctx, ggml_tanh(ctx, ggml_scale(ctx, scores, 1.f / SOFTCAP)), SOFTCAP);
            scores = ggml_soft_max(ctx, ggml_add(ctx, scores, mask));

            ggml_tensor * a = ggml_mul_mat(ctx, v4, scores);  // [D, n, 1, H]
            a = ggml_cont_2d(ctx, ggml_permute(ctx, a, 0, 2, 3, 1), (int64_t) D * H, n);
            a = mm(ctx, l.o, a);
            if (l.attn_post_norm) {
                a = norm(ctx, a, l.attn_post_norm);
            }
            x = ggml_add(ctx, x, a);
        }

        // Convolution: gated, then depthwise over time with the inputs the
        // previous chunk left.
        {
            ggml_tensor * c = mm(ctx, l.conv_pw1, norm(ctx, x, l.conv_norm));
            const int64_t d = c->ne[0] / 2;
            ggml_tensor * gate = ggml_sigmoid(ctx, ggml_cont(ctx, ggml_view_2d(ctx, c, d, n, c->nb[1], d * c->nb[0])));
            c = ggml_mul(ctx, ggml_view_2d(ctx, c, d, n, c->nb[1], 0), gate);
            c = ggml_cont(ctx, ggml_transpose(ctx, c));  // [n, n_conv]

            past_conv[il] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, CONV_PAST, n_conv_);
            ggml_set_input(past_conv[il]);
            if (!last) {
                new_conv[il] = ggml_cont(ctx, ggml_view_2d(ctx, c, CONV_PAST, n_conv_, c->nb[1], (size_t) (n - CONV_PAST) * c->nb[0]));
                ggml_set_output(new_conv[il]);
            }
            c = ggml_ssm_conv(ctx, ggml_concat(ctx, past_conv[il], c, 0), l.conv_dw);
            if (l.conv_dw_norm) {
                c = norm(ctx, c, l.conv_dw_norm);
            }
            c = mm(ctx, l.conv_pw2, ggml_silu(ctx, c));
            x = ggml_add(ctx, x, c);
        }

        x = ggml_add(ctx, x, ggml_scale(ctx, ffn(ctx, x, l.ffn_norm_1, l.ffn_up_1, l.ffn_down_1, l.ffn_post_norm_1), 0.5f));
        if (l.out_norm) {
            x = norm(ctx, x, l.out_norm);
        }
    }

    x = mm(ctx, out_proj_, x);
    if (out_proj_b_) {
        x = ggml_add(ctx, x, out_proj_b_);
    }
    x = mm(ctx, embd_proj_, norm(ctx, x, embd_norm_));
    ggml_set_output(x);
    ggml_build_forward_expand(gf, x);
    for (int il = 0; il < n_layer && !last; il++) {
        ggml_build_forward_expand(gf, new_k[il]);
        ggml_build_forward_expand(gf, new_v[il]);
        ggml_build_forward_expand(gf, new_conv[il]);
    }

    ggml_backend_sched_reset(sched_);
    if (!ggml_backend_sched_alloc_graph(sched_, gf)) {
        fprintf(stderr, "audio: cannot allocate the graph\n");
        ggml_free(ctx);
        return false;
    }
    {
        std::vector<float> frames((size_t) T * Mel::N_MEL);
        for (int t = 0; t < T; t++) {
            const float * f = mel.frame(f0 + t);
            for (int m = 0; m < Mel::N_MEL; m++) {
                frames[(size_t) m * T + t] = f[m];
            }
        }
        ggml_backend_tensor_set(inp, frames.data(), 0, frames.size() * sizeof(float));
    }
    ggml_backend_tensor_set(pos_emb, pos_emb_.data(), 0, pos_emb_.size() * sizeof(float));
    {
        // A query sees the keys up to MAX_PAST - 1 positions back; before
        // the first chunk there are none.
        std::vector<float> m((size_t) S * n, -INFINITY);
        for (int qi = 0; qi < n; qi++) {
            const int gq = K * CHUNK + qi;
            for (int ki = 0; ki < S; ki++) {
                const int gk = K * CHUNK - MAX_PAST + ki;
                if (gk >= 0 && gk <= gq && gq - gk < MAX_PAST) {
                    m[(size_t) qi * S + ki] = 0.f;
                }
            }
        }
        ggml_backend_tensor_set(mask, m.data(), 0, m.size() * sizeof(float));
    }
    for (int il = 0; il < n_layer; il++) {
        ggml_backend_tensor_set(past_k[il], st.k[il].data(), 0, st.k[il].size() * sizeof(float));
        ggml_backend_tensor_set(past_v[il], st.v[il].data(), 0, st.v[il].size() * sizeof(float));
        ggml_backend_tensor_set(past_conv[il], st.conv[il].data(), 0, st.conv[il].size() * sizeof(float));
    }

    const bool ok = ggml_backend_sched_graph_compute(sched_, gf) == GGML_STATUS_SUCCESS;
    if (ok) {
        const size_t at = out.size();
        out.resize(at + (size_t) n * n_out_);
        ggml_backend_tensor_get(x, out.data() + at, 0, (size_t) n * n_out_ * sizeof(float));
        for (int il = 0; il < n_layer && !last; il++) {
            ggml_backend_tensor_get(new_k[il], st.k[il].data(), 0, st.k[il].size() * sizeof(float));
            ggml_backend_tensor_get(new_v[il], st.v[il].data(), 0, st.v[il].size() * sizeof(float));
            ggml_backend_tensor_get(new_conv[il], st.conv[il].data(), 0, st.conv[il].size() * sizeof(float));
        }
        st.n_chunks++;
    } else {
        fprintf(stderr, "audio: encoding failed\n");
    }
    ggml_free(ctx);
    return ok;
}

AudioStream::AudioStream(AudioEncoder & encoder)
    : encoder_(&encoder), state_(encoder.new_state()) {}

void AudioStream::push(const float * pcm, size_t n) {
    if (!finished_) {
        mel_.push(pcm, n);
    }
}

bool AudioStream::pull(std::vector<float> & out, int & n) {
    const size_t at = out.size();
    while (!finished_ && mel_.n_frames() >= (state_.n_chunks + 1) * AudioEncoder::CHUNK * FRAMES) {
        if (!encoder_->encode(mel_, state_, false, out)) {
            return false;
        }
    }
    n = (int) ((out.size() - at) / encoder_->n_embd());
    return true;
}

bool AudioStream::finish(std::vector<float> & out, int & n) {
    if (finished_) {
        n = 0;
        return true;
    }
    mel_.finish();
    const size_t at = out.size();
    int n_full = 0;
    if (!pull(out, n_full)) {
        return false;
    }
    finished_ = true;
    if (!encoder_->encode(mel_, state_, true, out)) {
        return false;
    }
    n = (int) ((out.size() - at) / encoder_->n_embd());
    return true;
}
