#include "turn_head.h"

#include "ggml.h"
#include "gguf.h"
#include "model_file.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>

static constexpr const char * ARCH = "gemmalive-heads";

// SHA-256 (FIPS 180-4), for GGUF header fingerprints.
static std::string sha256_hex(const unsigned char * data, size_t size) {
    static const uint32_t k[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    };
    uint32_t h[8] = { 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };
    auto rotr = [](uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };

    // Message plus padding: 0x80, zeros, 64-bit big-endian bit length.
    const size_t n_blocks = (size_t) (size + 9 + 63) / 64;
    std::array<unsigned char, 64> block;
    for (size_t b = 0; b < n_blocks; b++) {
        for (size_t i = 0; i < 64; i++) {
            const size_t pos = b * 64 + i;
            if (pos < size) {
                block[i] = data[pos];
            } else if (pos == size) {
                block[i] = 0x80;
            } else if (pos >= n_blocks * 64 - 8) {
                block[i] = (unsigned char) ((uint64_t) size * 8 >> (8 * (n_blocks * 64 - 1 - pos)));
            } else {
                block[i] = 0;
            }
        }
        uint32_t w[64];
        for (int i = 0; i < 16; i++) {
            w[i] = (uint32_t) block[4 * i] << 24 | (uint32_t) block[4 * i + 1] << 16 |
                   (uint32_t) block[4 * i + 2] << 8 | (uint32_t) block[4 * i + 3];
        }
        for (int i = 16; i < 64; i++) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], bb = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; i++) {
            const uint32_t t1 = hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
            const uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & bb) ^ (a & c) ^ (bb & c));
            hh = g; g = f; f = e; e = d + t1; d = c; c = bb; bb = a; a = t1 + t2;
        }
        h[0] += a; h[1] += bb; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }

    char hex[65];
    for (int i = 0; i < 8; i++) {
        snprintf(hex + 8 * i, 9, "%08x", h[i]);
    }
    return std::string(hex, 64);
}

// SHA-256 of a GGUF file's header: all metadata and the tensor layout, i.e.
// everything before the tensor data. Cheap, and differs for any other model,
// quantization or conversion.
static std::string gguf_fingerprint(const std::string & path) {
    gguf_init_params params = { /*no_alloc=*/true, /*ctx=*/nullptr };
    gguf_context * ctx = gguf_init_from_file(path.c_str(), params);
    if (!ctx) {
        return {};
    }
    const size_t offset = gguf_get_data_offset(ctx);
    gguf_free(ctx);

    std::vector<unsigned char> header(offset);
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        return {};
    }
    const bool ok = fread(header.data(), 1, offset, f) == offset;
    fclose(f);
    return ok ? sha256_hex(header.data(), header.size()) : std::string();
}

static bool get_floats(const ModelFile & file, const std::string & name, std::vector<float> & out, size_t n) {
    const ggml_tensor * t = file.info(name);
    if (!t || t->type != GGML_TYPE_F32 || ggml_n_dims(t) > 2 || (size_t) ggml_nelements(t) != n) {
        fprintf(stderr, "heads: tensor %s missing or malformed\n", name.c_str());
        return false;
    }
    out.resize(n);
    for (int64_t i = 0; i < t->ne[1]; i++) {
        file.row(t, i, out.data() + i * t->ne[0]);
    }
    return true;
}

std::string TurnHead::path_for(const std::string & model_path) {
    if (std::filesystem::exists(model_path)) {
        gguf_init_params params = { /*no_alloc=*/true, /*ctx=*/nullptr };
        gguf_context * ctx = gguf_init_from_file(model_path.c_str(), params);
        const bool combined = ctx && gguf_find_key(ctx, "heads.tasks") >= 0;
        gguf_free(ctx);
        if (combined) {
            return model_path;
        }
    }
    const size_t ext = model_path.size() >= 5 && model_path.compare(model_path.size() - 5, 5, ".gguf") == 0
                           ? model_path.size() - 5 : model_path.size();
    return model_path.substr(0, ext) + "-heads.gguf";
}

bool TurnHead::load(const std::string & path, const std::string & task, const Expect & expect) {
    ModelFile file;
    if (!file.open(path)) {
        return false;
    }
    const gguf_context * ctx = file.gguf();

    // Heads in the model's own file were put there for it.
    const bool combined = path == expect.model_path;
    bool ok = combined ? file.find_key("heads.tasks") >= 0 : file.str("general.architecture") == ARCH;
    if (!ok) {
        fprintf(stderr, "heads: %s is not a heads file\n", path.c_str());
    }

    // Trained on these files and this prompt; anything else changes the
    // hidden states the heads read. Heads trained on many prompts take any.
    const int64_t any_id     = gguf_find_key(ctx, "heads.any_system_prompt");
    const bool    any_prompt = any_id >= 0 && gguf_get_kv_type(ctx, any_id) == GGUF_TYPE_BOOL &&
                               gguf_get_val_bool(ctx, any_id);
    const auto fingerprint = [&](const std::string & of) { return combined ? std::string() : gguf_fingerprint(of); };
    const struct { const char * what, * key; std::string have; bool needed; } checks[] = {
        { "model",         "heads.model.fingerprint",  fingerprint(expect.model_path), !combined },
        { "audio projector", "heads.mmproj.fingerprint", fingerprint(expect.mmproj_path), !combined },
        { "system prompt", "heads.system_prompt",      expect.system_prompt, !any_prompt },
    };
    for (const auto & c : checks) {
        if (ok && c.needed && file.str(c.key) != c.have) {
            fprintf(stderr, "heads: %s was trained for a different %s\n", path.c_str(), c.what);
            ok = false;
        }
    }

    // The task's keys and tensors: heads.<task>.layers, <task>.mix, ...
    const std::string key = "heads." + task + ".";
    const std::string tensor = task + ".";
    if (ok) {
        const int64_t layers_id = gguf_find_key(ctx, (key + "layers").c_str());
        const int64_t embd_id   = gguf_find_key(ctx, (key + "n_embd").c_str());
        const int64_t hidden_id = gguf_find_key(ctx, (key + "hidden").c_str());
        const int64_t thr_id    = gguf_find_key(ctx, (key + "threshold").c_str());
        ok = layers_id >= 0 && embd_id >= 0 && hidden_id >= 0 && thr_id >= 0 &&
             gguf_get_arr_type(ctx, layers_id) == GGUF_TYPE_INT32;
        if (ok) {
            const int32_t * l = (const int32_t *) gguf_get_arr_data(ctx, layers_id);
            layers_.assign(l, l + gguf_get_arr_n(ctx, layers_id));
            n_embd_    = (int) gguf_get_val_u32(ctx, embd_id);
            hidden_    = (int) gguf_get_val_u32(ctx, hidden_id);
            threshold_ = gguf_get_val_f32(ctx, thr_id);
        } else {
            fprintf(stderr, "heads: %s has no %s head\n", path.c_str(), task.c_str());
        }
    }

    if (ok) {
        const size_t n_layers = layers_.size();
        const size_t n_embd   = (size_t) n_embd_;
        const size_t hidden   = (size_t) hidden_;
        std::vector<float> b2;
        ok = get_floats(file, tensor + "mix", mix_, n_layers) &&
             get_floats(file, tensor + "mean", mean_, n_embd) &&
             get_floats(file, tensor + "std", std_, n_embd) &&
             (hidden == 0 || (get_floats(file, tensor + "fc1.weight", w1_, hidden * n_embd) &&
                              get_floats(file, tensor + "fc1.bias", b1_, hidden))) &&
             get_floats(file, tensor + "fc2.weight", w2_, hidden ? hidden : n_embd) &&
             get_floats(file, tensor + "fc2.bias", b2, 1);
        b2_ = ok ? b2[0] : 0.f;
    }

    if (!ok) {
        n_embd_ = 0;
    }
    return ok;
}

float TurnHead::predict(const std::vector<float> & layer_states) const {
    std::vector<float> h(n_embd_, 0.f);
    for (size_t i = 0; i < layers_.size(); i++) {
        const float * row = layer_states.data() + (size_t) layers_[i] * n_embd_;
        for (int e = 0; e < n_embd_; e++) {
            h[e] += mix_[i] * row[e];
        }
    }
    for (int e = 0; e < n_embd_; e++) {
        h[e] = (h[e] - mean_[e]) / std_[e];
    }

    float logit = b2_;
    if (hidden_ == 0) {
        for (int e = 0; e < n_embd_; e++) {
            logit += w2_[e] * h[e];
        }
    } else {
        for (int j = 0; j < hidden_; j++) {
            const float * w = w1_.data() + (size_t) j * n_embd_;
            float a = b1_[j];
            for (int e = 0; e < n_embd_; e++) {
                a += w[e] * h[e];
            }
            const float gelu = 0.5f * a * (1.f + std::erf(a / std::sqrt(2.f)));
            logit += w2_[j] * gelu;
        }
    }
    return 1.f / (1.f + std::exp(-logit));
}
