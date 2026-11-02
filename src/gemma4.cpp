#include "gemma4.h"

#include "ggml-alloc.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

static constexpr size_t GRAPH_SIZE = 8192;

// Positions the sliding-window caches keep below a batch, besides the
// window: room for taking back the end of a reply or turn without a guard.
static constexpr int SWA_SLACK = 256;

// A per-layer value of the model's metadata: an array, or one for all.
static std::vector<uint32_t> per_layer(const ModelFile & f, const char * key, int n_layer) {
    const int64_t k = f.find_key(key);
    std::vector<uint32_t> out(n_layer, 0);
    if (k < 0) {
        return out;
    }
    if (gguf_get_kv_type(f.gguf(), k) == GGUF_TYPE_ARRAY) {
        const auto * d = (const uint32_t *) gguf_get_arr_data(f.gguf(), k);
        for (int i = 0; i < n_layer && i < (int) gguf_get_arr_n(f.gguf(), k); i++) {
            out[i] = d[i];
        }
    } else {
        std::fill(out.begin(), out.end(), f.u32(key));
    }
    return out;
}

Gemma4::~Gemma4() {
    if (sched_)     ggml_backend_sched_free(sched_);
    if (cache_buf_) ggml_backend_buffer_free(cache_buf_);
    if (cache_ctx_) ggml_free(cache_ctx_);
}

bool Gemma4::load(const std::string & path, const Backend & backend, int n_ctx) {
    if (!file_.open(path)) {
        return false;
    }
    if (file_.str("general.architecture") != "gemma4") {
        fprintf(stderr, "gemma4: %s is not a Gemma 4 model\n", path.c_str());
        return false;
    }
    if (!tokenizer_.load(file_.gguf())) {
        return false;
    }

    const int n_layer = (int) file_.u32("gemma4.block_count");
    n_embd_        = (int) file_.u32("gemma4.embedding_length");
    n_ple_         = (int) file_.u32("gemma4.embedding_length_per_layer_input");
    n_swa_         = (int) file_.u32("gemma4.attention.sliding_window");
    n_ctx_train_   = (int) file_.u32("gemma4.context_length");
    eps_           = file_.f32("gemma4.attention.layer_norm_rms_epsilon", 1e-6f);
    softcap_       = file_.f32("gemma4.final_logit_softcapping", 0.f);
    rope_base_     = file_.f32("gemma4.rope.freq_base", 10000.f);
    rope_base_swa_ = file_.f32("gemma4.rope.freq_base_swa", 10000.f);
    n_ctx_         = std::min(n_ctx, n_ctx_train_);

    const auto n_head    = per_layer(file_, "gemma4.attention.head_count", n_layer);
    const auto n_head_kv = per_layer(file_, "gemma4.attention.head_count_kv", n_layer);
    const int  head_dim  = (int) file_.u32("gemma4.attention.key_length");
    const int  head_dim_swa = (int) file_.u32("gemma4.attention.key_length_swa");
    const int  n_shared  = (int) file_.u32("gemma4.attention.shared_kv_layers");
    const int  n_layer_kv = n_layer - n_shared;

    std::vector<bool> swa(n_layer, false);
    {
        const int64_t k = file_.find_key("gemma4.attention.sliding_window_pattern");
        if (k < 0 || (int) gguf_get_arr_n(file_.gguf(), k) < n_layer) {
            fprintf(stderr, "gemma4: no sliding window pattern\n");
            return false;
        }
        const auto * d = (const int8_t *) gguf_get_arr_data(file_.gguf(), k);
        for (int i = 0; i < n_layer; i++) {
            swa[i] = d[i] != 0;
        }
    }

    // Everything on the device but the per-layer token embeddings, which are
    // read a row per token.
    // A combined file has the audio encoder's and the heads' tensors too.
    const auto own = [](const std::string & name) {
        for (const char * n : { "token_embd.weight", "per_layer_model_proj.weight", "per_layer_proj_norm.weight",
                                "rope_freqs.weight", "output_norm.weight" }) {
            if (name == n) {
                return true;
            }
        }
        return name.rfind("blk.", 0) == 0;
    };
    if (!file_.load(backend.device(), own)) {
        return false;
    }
    auto get = [&](const std::string & name, bool required = true) {
        ggml_tensor * t = file_.get(name);
        if (!t && required) {
            fprintf(stderr, "gemma4: %s is missing\n", name.c_str());
        }
        return t;
    };
    tok_embd_      = get("token_embd.weight");
    ple_embd_      = file_.info("per_layer_token_embd.weight");
    ple_proj_      = get("per_layer_model_proj.weight");
    ple_proj_norm_ = get("per_layer_proj_norm.weight");
    rope_freqs_    = get("rope_freqs.weight", false);
    output_norm_   = get("output_norm.weight");
    if (!tok_embd_ || !ple_embd_ || !ple_proj_ || !ple_proj_norm_ || !output_norm_) {
        return false;
    }
    n_vocab_ = (int) tok_embd_->ne[1];

    layers_.resize(n_layer);
    for (int il = 0; il < n_layer; il++) {
        Layer & l = layers_[il];
        auto name = [&](const char * t) { return "blk." + std::to_string(il) + "." + t + ".weight"; };
        l.swa       = swa[il];
        l.n_head    = (int) n_head[il];
        l.n_head_kv = (int) n_head_kv[il];
        l.head_dim  = l.swa ? head_dim_swa : head_dim;
        l.kv        = il < n_layer_kv ? il : n_layer_kv - (l.swa ? 2 : 1);
        l.attn_norm      = get(name("attn_norm"));
        l.wq             = get(name("attn_q"));
        l.q_norm         = get(name("attn_q_norm"));
        l.wo             = get(name("attn_output"));
        l.attn_post_norm = get(name("post_attention_norm"));
        if (il < n_layer_kv) {
            l.wk     = get(name("attn_k"));
            l.wv     = get(name("attn_v"), false);
            l.k_norm = get(name("attn_k_norm"));
        }
        l.ffn_norm      = get(name("ffn_norm"));
        l.ffn_gate      = get(name("ffn_gate"));
        l.ffn_up        = get(name("ffn_up"));
        l.ffn_down      = get(name("ffn_down"));
        l.ffn_post_norm = get(name("post_ffw_norm"));
        l.ple_gate      = get(name("inp_gate"));
        l.ple_proj      = get(name("proj"));
        l.ple_post_norm = get(name("post_norm"));
        l.out_scale     = get(name("layer_output_scale"), false);
        if (!l.attn_norm || !l.wq || !l.q_norm || !l.wo || !l.attn_post_norm || !l.ffn_norm || !l.ffn_gate ||
            !l.ffn_up || !l.ffn_down || !l.ffn_post_norm || !l.ple_gate || !l.ple_proj || !l.ple_post_norm ||
            (il < n_layer_kv && (!l.wk || !l.k_norm))) {
            return false;
        }
    }
    if (n_layer_kv < 2 || layers_[n_layer_kv - 1].swa || !layers_[n_layer_kv - 2].swa) {
        fprintf(stderr, "gemma4: unexpected layout of the shared caches\n");
        return false;
    }

    // The caches, and per guard a copy of a window of each sliding-window one.
    swa_cells_ = std::min(n_ctx_, 4 * n_swa_);
    {
        ggml_init_params params = { ggml_tensor_overhead() * (size_t) n_layer_kv * (2 + 2 * GUARDS), nullptr, true };
        cache_ctx_ = ggml_init(params);
        caches_.resize(n_layer_kv);
        for (int il = 0; il < n_layer_kv; il++) {
            const Layer & l = layers_[il];
            const int64_t row   = (int64_t) l.head_dim * l.n_head_kv;
            const int64_t cells = l.swa ? swa_cells_ : n_ctx_;
            caches_[il].k = ggml_new_tensor_2d(cache_ctx_, GGML_TYPE_F16, row, cells);
            caches_[il].v = ggml_new_tensor_2d(cache_ctx_, GGML_TYPE_F16, row, cells);
            for (Guard & g : guards_) {
                g.k.push_back(l.swa ? ggml_new_tensor_2d(cache_ctx_, GGML_TYPE_F16, row, n_swa_) : nullptr);
                g.v.push_back(l.swa ? ggml_new_tensor_2d(cache_ctx_, GGML_TYPE_F16, row, n_swa_) : nullptr);
            }
        }
        cache_buf_ = ggml_backend_alloc_ctx_tensors(cache_ctx_, backend.main());
        if (!cache_buf_) {
            fprintf(stderr, "gemma4: out of memory for the cache\n");
            return false;
        }
        ggml_backend_buffer_clear(cache_buf_, 0);
    }

    // Flash attention where the device has it for both head sizes.
    {
        ggml_init_params params = { ggml_tensor_overhead() * 16, nullptr, true };
        ggml_context * ctx = ggml_init(params);
        flash_attn_ = true;
        for (const int d : { head_dim, head_dim_swa }) {
            ggml_tensor * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, 1, 8);
            ggml_tensor * k = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, d, 256, 2);
            ggml_tensor * v = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, d, 256, 2);
            ggml_tensor * m = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, 256, 1);
            flash_attn_ = flash_attn_ && ggml_backend_supports_op(backend.main(), ggml_flash_attn_ext(ctx, q, k, v, m, 1.f, 0.f, 0.f));
        }
        ggml_free(ctx);
    }

    sched_ = backend.new_sched(GRAPH_SIZE);
    logits_.assign(n_vocab_, 0.f);
    return sched_ != nullptr;
}

ggml_tensor * Gemma4::norm(ggml_context * ctx, ggml_tensor * x, ggml_tensor * w) const {
    x = ggml_rms_norm(ctx, x, eps_);
    return w ? ggml_mul(ctx, x, w) : x;
}

void Gemma4::guard(int slot, int pos) {
    Guard & g = guards_[slot];
    g.pos   = pos > 0 ? pos : -1;
    g.saved = false;
}

bool Gemma4::copy_cells(int src, int dst, int n, Guard * from, Guard * to) {
    if (n <= 0) {
        return true;
    }
    ggml_init_params params = { ggml_tensor_overhead() * 512 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_cgraph *  gf  = ggml_new_graph(ctx);
    for (size_t il = 0; il < caches_.size(); il++) {
        if (!layers_[il].swa) {
            continue;
        }
        for (int kv = 0; kv < 2; kv++) {
            ggml_tensor * cache = kv == 0 ? caches_[il].k : caches_[il].v;
            ggml_tensor * s = from ? (kv == 0 ? from->k[il] : from->v[il]) : cache;
            ggml_tensor * d = to ? (kv == 0 ? to->k[il] : to->v[il]) : cache;
            ggml_build_forward_expand(gf, ggml_cpy(ctx,
                ggml_view_2d(ctx, s, s->ne[0], n, s->nb[1], (size_t) src * s->nb[1]),
                ggml_view_2d(ctx, d, d->ne[0], n, d->nb[1], (size_t) dst * d->nb[1])));
        }
    }
    ggml_backend_sched_reset(sched_);
    const bool ok = ggml_backend_sched_alloc_graph(sched_, gf) &&
                    ggml_backend_sched_graph_compute(sched_, gf) == GGML_STATUS_SUCCESS;
    ggml_free(ctx);
    return ok;
}

// Makes room in the sliding-window caches for positions pos .. pos + n - 1,
// with the window below them.
bool Gemma4::place(int pos, int n) {
    for (Guard & g : guards_) {
        if (pos < g.pos) {
            g.pos = -1;  // what it guards is being replaced
        }
    }
    const int lo = std::max(0, pos - n_swa_ + 1);
    if (pos == 0) {
        // Starting over: there is nothing below position 0 to bring back.
        swa_base_ = 0;
    } else if (lo < swa_base_) {
        Guard * g = nullptr;
        for (Guard & c : guards_) {
            if (c.pos == pos && c.saved) {
                g = &c;
            }
        }
        if (!g) {
            fprintf(stderr, "gemma4: cannot go back to position %d\n", pos);
            return false;
        }
        swa_base_ = std::max(0, pos - n_swa_);
        if (!copy_cells(0, 0, pos - swa_base_, g)) {
            return false;
        }
        for (Guard & c : guards_) {
            if (!c.saved && c.pos >= 0 && std::max(0, c.pos - n_swa_) < swa_base_) {
                c.pos = -1;
            }
        }
    }
    if (pos + n - swa_base_ > swa_cells_) {
        // Move the recent cells to the front. The window of a guard that
        // would be dropped is saved first.
        const int base = pos - n_swa_ - SWA_SLACK;
        for (Guard & g : guards_) {
            const int g0 = std::max(0, g.pos - n_swa_);
            if (g.pos < 0 || g.saved || g0 >= base) {
                continue;
            }
            if (g0 < swa_base_ || !copy_cells(g0 - swa_base_, 0, g.pos - g0, nullptr, &g)) {
                g.pos = -1;
                continue;
            }
            g.saved = true;
        }
        if (!copy_cells(base - swa_base_, 0, pos - base)) {
            return false;
        }
        swa_base_ = base;
    }
    return true;
}

bool Gemma4::decode(const int32_t * tokens, const float * embd, int n, int pos,
                    const std::vector<int> * capture, std::vector<float> * states) {
    if (n <= 0 || n > MAX_BATCH || pos < 0 || pos + n > n_ctx_) {
        fprintf(stderr, "gemma4: cannot decode %d at %d\n", n, pos);
        return false;
    }
    if (!place(pos, n)) {
        return false;
    }
    const int n_layer = (int) layers_.size();

    // The global caches: every position up to the batch's last. The
    // sliding-window ones: from the first query's window on.
    const int n_kv      = pos + n;
    const int swa_first = std::max(0, pos - n_swa_ + 1) - swa_base_;
    const int n_kv_swa  = pos + n - swa_base_ - swa_first;

    ggml_init_params params = { ggml_tensor_overhead() * GRAPH_SIZE + ggml_graph_overhead_custom(GRAPH_SIZE, false), nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_cgraph *  gf  = ggml_new_graph_custom(ctx, GRAPH_SIZE, false);

    ggml_tensor * inp_tokens = nullptr;
    ggml_tensor * inp_embd   = nullptr;
    ggml_tensor * x;
    if (tokens) {
        inp_tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
        ggml_set_input(inp_tokens);
        x = ggml_get_rows(ctx, tok_embd_, inp_tokens);
        x = ggml_scale(ctx, x, std::sqrt((float) n_embd_));
    } else {
        inp_embd = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_embd_, n);
        ggml_set_input(inp_embd);
        x = inp_embd;
    }

    ggml_tensor * inp_pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
    ggml_set_input(inp_pos);
    ggml_tensor * mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv, n);
    ggml_set_input(mask);
    ggml_tensor * mask_swa = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, n_kv_swa, n);
    ggml_set_input(mask_swa);

    // Per-layer inputs: the token's own embedding per layer (for embeddings,
    // the padding token's) plus a projection of the input embedding.
    const int n_ple_rows = tokens ? n : 1;
    ggml_tensor * inp_ple = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, (int64_t) n_ple_ * n_layer, n_ple_rows);
    ggml_set_input(inp_ple);
    ggml_tensor * ple;
    {
        ggml_tensor * own = ggml_reshape_3d(ctx, inp_ple, n_ple_, n_layer, n_ple_rows);
        own = ggml_scale(ctx, own, std::sqrt((float) n_ple_));
        ple = ggml_mul_mat(ctx, ple_proj_, x);
        ple = ggml_scale(ctx, ple, 1.f / std::sqrt((float) n_embd_));
        ple = ggml_reshape_3d(ctx, ple, n_ple_, n_layer, n);
        ple = norm(ctx, ple, ple_proj_norm_);
        ple = ggml_add(ctx, ple, own);
        ple = ggml_scale(ctx, ple, 1.f / std::sqrt(2.f));
        ple = ggml_cont(ctx, ggml_permute(ctx, ple, 0, 2, 1, 3));  // [n_ple, n, n_layer]
    }

    std::vector<ggml_tensor *> outs(n_layer, nullptr);
    int n_rows = n;
    for (int il = 0; il < n_layer; il++) {
        const Layer & l = layers_[il];
        const Cache & c = caches_[l.kv];
        const int d = l.head_dim;

        ggml_tensor * cur = norm(ctx, x, l.attn_norm);

        ggml_tensor * q = ggml_mul_mat(ctx, l.wq, cur);
        q = ggml_reshape_3d(ctx, q, d, l.n_head, n);
        q = norm(ctx, q, l.q_norm);
        q = ggml_rope_ext(ctx, q, inp_pos, l.swa ? nullptr : rope_freqs_, d, GGML_ROPE_TYPE_NEOX, n_ctx_train_,
                          l.swa ? rope_base_swa_ : rope_base_, 1.f, 0.f, 1.f, 32.f, 1.f);

        const int64_t row = (int64_t) d * l.n_head_kv;
        const int first_cell = l.swa ? pos - swa_base_ : pos;
        if (l.kv == il) {
            ggml_tensor * k = ggml_mul_mat(ctx, l.wk, cur);
            ggml_tensor * v = l.wv ? ggml_mul_mat(ctx, l.wv, cur) : k;
            k = ggml_reshape_3d(ctx, k, d, l.n_head_kv, n);
            v = ggml_reshape_3d(ctx, v, d, l.n_head_kv, n);
            k = norm(ctx, k, l.k_norm);
            v = ggml_rms_norm(ctx, v, eps_);
            k = ggml_rope_ext(ctx, k, inp_pos, l.swa ? nullptr : rope_freqs_, d, GGML_ROPE_TYPE_NEOX, n_ctx_train_,
                              l.swa ? rope_base_swa_ : rope_base_, 1.f, 0.f, 1.f, 32.f, 1.f);
            // The writes come before the attention that reads them.
            ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_reshape_2d(ctx, k, row, n),
                ggml_view_2d(ctx, c.k, row, n, c.k->nb[1], (size_t) first_cell * c.k->nb[1])));
            ggml_build_forward_expand(gf, ggml_cpy(ctx, ggml_reshape_2d(ctx, v, row, n),
                ggml_view_2d(ctx, c.v, row, n, c.v->nb[1], (size_t) first_cell * c.v->nb[1])));
        }

        const int kv0  = l.swa ? swa_first : 0;
        const int n_kv_l = l.swa ? n_kv_swa : n_kv;
        ggml_tensor * k = ggml_view_3d(ctx, c.k, d, n_kv_l, l.n_head_kv, c.k->nb[1], ggml_row_size(c.k->type, d),
                                       (size_t) kv0 * c.k->nb[1]);
        ggml_tensor * v = ggml_view_3d(ctx, c.v, d, n_kv_l, l.n_head_kv, c.v->nb[1], ggml_row_size(c.v->type, d),
                                       (size_t) kv0 * c.v->nb[1]);
        ggml_tensor * m = l.swa ? mask_swa : mask;
        q = ggml_permute(ctx, q, 0, 2, 1, 3);  // [d, n, n_head]
        if (flash_attn_) {
            cur = ggml_flash_attn_ext(ctx, q, k, v, m, 1.f, 0.f, 0.f);
            ggml_prec_set_acc(cur, GGML_PREC_F32);
            cur = ggml_reshape_2d(ctx, cur, (int64_t) d * l.n_head, n);
        } else {
            ggml_tensor * kq = ggml_mul_mat(ctx, k, q);
            ggml_prec_set_acc(kq, GGML_PREC_F32);
            kq = ggml_soft_max_ext(ctx, kq, m, 1.f, 0.f);
            ggml_tensor * vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));
            cur = ggml_mul_mat(ctx, vt, kq);  // [d, n, n_head]
            cur = ggml_cont_2d(ctx, ggml_permute(ctx, cur, 0, 2, 1, 3), (int64_t) d * l.n_head, n);
        }
        cur = ggml_mul_mat(ctx, l.wo, cur);

        // The last layer only matters for the last position.
        ggml_tensor * ple_l = ggml_view_2d(ctx, ple, n_ple_, n, ple->nb[1], (size_t) il * ple->nb[2]);
        if (il == n_layer - 1 && n > 1) {
            cur   = ggml_view_2d(ctx, cur, cur->ne[0], 1, cur->nb[1], (size_t) (n - 1) * cur->nb[1]);
            x     = ggml_view_2d(ctx, x, x->ne[0], 1, x->nb[1], (size_t) (n - 1) * x->nb[1]);
            ple_l = ggml_view_2d(ctx, ple_l, n_ple_, 1, ple_l->nb[1], (size_t) (n - 1) * ple_l->nb[1]);
            n_rows = 1;
        }
        cur = norm(ctx, cur, l.attn_post_norm);
        ggml_tensor * attn_out = ggml_add(ctx, cur, x);

        cur = norm(ctx, attn_out, l.ffn_norm);
        cur = ggml_geglu_split(ctx, ggml_mul_mat(ctx, l.ffn_gate, cur), ggml_mul_mat(ctx, l.ffn_up, cur));
        cur = ggml_mul_mat(ctx, l.ffn_down, cur);
        cur = norm(ctx, cur, l.ffn_post_norm);
        cur = ggml_add(ctx, cur, attn_out);

        ggml_tensor * pe = ggml_gelu(ctx, ggml_mul_mat(ctx, l.ple_gate, cur));
        pe = ggml_mul(ctx, pe, ple_l);
        pe = ggml_mul_mat(ctx, l.ple_proj, pe);
        pe = norm(ctx, pe, l.ple_post_norm);
        cur = ggml_add(ctx, cur, pe);

        if (l.out_scale) {
            cur = ggml_mul(ctx, cur, l.out_scale);
        }
        x = cur;

        if (capture && std::find(capture->begin(), capture->end(), il) != capture->end()) {
            outs[il] = ggml_cont(ctx, ggml_view_1d(ctx, x, n_embd_, (size_t) (n_rows - 1) * x->nb[1]));
            ggml_set_output(outs[il]);
            ggml_build_forward_expand(gf, outs[il]);
        }
    }

    x = ggml_view_2d(ctx, x, n_embd_, 1, x->nb[1], (size_t) (n_rows - 1) * x->nb[1]);
    x = norm(ctx, x, output_norm_);
    ggml_tensor * logits = ggml_mul_mat(ctx, tok_embd_, x);
    if (softcap_ > 0.f) {
        logits = ggml_scale(ctx, ggml_tanh(ctx, ggml_scale(ctx, logits, 1.f / softcap_)), softcap_);
    }
    ggml_set_output(logits);
    ggml_build_forward_expand(gf, logits);

    ggml_backend_sched_reset(sched_);
    if (!ggml_backend_sched_alloc_graph(sched_, gf)) {
        fprintf(stderr, "gemma4: cannot allocate the graph\n");
        ggml_free(ctx);
        return false;
    }

    if (tokens) {
        ggml_backend_tensor_set(inp_tokens, tokens, 0, (size_t) n * sizeof(int32_t));
    } else {
        ggml_backend_tensor_set(inp_embd, embd, 0, (size_t) n * n_embd_ * sizeof(float));
    }
    {
        std::vector<int32_t> positions(n);
        for (int i = 0; i < n; i++) {
            positions[i] = pos + i;
        }
        ggml_backend_tensor_set(inp_pos, positions.data(), 0, positions.size() * sizeof(int32_t));
    }
    {
        const ggml_fp16_t zero = ggml_fp32_to_fp16(0.f);
        const ggml_fp16_t inf  = ggml_fp32_to_fp16(-INFINITY);
        mask_.assign((size_t) n_kv * n, inf);
        mask_swa_.assign((size_t) n_kv_swa * n, inf);
        for (int i = 0; i < n; i++) {
            const int q = pos + i;
            std::fill(mask_.begin() + (size_t) i * n_kv, mask_.begin() + (size_t) i * n_kv + q + 1, zero);
            for (int j = 0; j < n_kv_swa; j++) {
                const int p = swa_base_ + swa_first + j;
                if (p <= q && q - p < n_swa_) {
                    mask_swa_[(size_t) i * n_kv_swa + j] = zero;
                }
            }
        }
        ggml_backend_tensor_set(mask, mask_.data(), 0, mask_.size() * sizeof(ggml_fp16_t));
        ggml_backend_tensor_set(mask_swa, mask_swa_.data(), 0, mask_swa_.size() * sizeof(ggml_fp16_t));
    }
    {
        std::vector<int32_t> rows(n_ple_rows, 0);
        if (tokens) {
            rows.assign(tokens, tokens + n);
        }
        file_.prefetch(ple_embd_, rows.data(), n_ple_rows);
        ple_rows_.resize((size_t) n_ple_rows * ple_embd_->ne[0]);
        for (int i = 0; i < n_ple_rows; i++) {
            file_.row(ple_embd_, rows[i], ple_rows_.data() + (size_t) i * ple_embd_->ne[0]);
        }
        ggml_backend_tensor_set(inp_ple, ple_rows_.data(), 0, ple_rows_.size() * sizeof(float));
    }

    const bool ok = ggml_backend_sched_graph_compute(sched_, gf) == GGML_STATUS_SUCCESS;
    if (ok) {
        ggml_backend_tensor_get(logits, logits_.data(), 0, logits_.size() * sizeof(float));
        if (states) {
            states->assign((size_t) n_layer * n_embd_, 0.f);
            for (int il = 0; il < n_layer; il++) {
                if (outs[il]) {
                    ggml_backend_tensor_get(outs[il], states->data() + (size_t) il * n_embd_, 0, n_embd_ * sizeof(float));
                }
            }
        }
    } else {
        fprintf(stderr, "gemma4: decode failed\n");
    }
    ggml_free(ctx);
    return ok;
}
