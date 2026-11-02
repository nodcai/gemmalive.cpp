#include "vad.h"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>

// Silero VAD v5 per 512-sample window, as whisper.cpp computes it:
//   reflect pad 64 | STFT (256-tap basis, hop 128, 4 frames) -> 129 magnitudes
//   conv k3 129->128 s1 | 128->64 s2 | 64->64 s2 | 64->128 s1, each + ReLU
//   LSTM cell (128) | ReLU | dot with 128 weights + bias | sigmoid

static constexpr int PAD    = 64;
static constexpr int N_FFT  = 256;
static constexpr int HOP    = 128;
static constexpr int N_BINS = N_FFT / 2 + 1;
static constexpr int HIDDEN = 128;

namespace {

struct Reader {
    const std::vector<uint8_t> & d;
    size_t off = 0;
    bool ok = true;

    int32_t i32() {
        int32_t v = 0;
        if (off + 4 > d.size()) {
            ok = false;
            return 0;
        }
        std::memcpy(&v, d.data() + off, 4);
        off += 4;
        return v;
    }
};

// y[o][t] = b[o] + sum_c sum_k w[o][c][k] * x[c][t * stride + k - 1], zero padded.
void conv3(const std::vector<float> & w, const std::vector<float> & b, int n_in, int n_out,
           const float * x, int len, int stride, float * y, int & out_len) {
    out_len = (len + 2 - 3) / stride + 1;
    for (int o = 0; o < n_out; o++) {
        for (int t = 0; t < out_len; t++) {
            float acc = b[o];
            for (int c = 0; c < n_in; c++) {
                const float * wk = w.data() + ((size_t) o * n_in + c) * 3;
                const float * xc = x + (size_t) c * len;
                for (int k = 0; k < 3; k++) {
                    const int i = t * stride + k - 1;
                    if (i >= 0 && i < len) {
                        acc += wk[k] * xc[i];
                    }
                }
            }
            y[(size_t) o * out_len + t] = std::max(acc, 0.f);
        }
    }
}

float dot(const float * a, const float * b, int n) {
    float acc = 0.f;
    for (int i = 0; i < n; i++) {
        acc += a[i] * b[i];
    }
    return acc;
}

float sigmoid(float x) {
    return 1.f / (1.f + std::exp(-x));
}

}  // namespace

bool Vad::load(const std::string & path) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "vad: cannot open %s\n", path.c_str());
        return false;
    }
    std::vector<uint8_t> data;
    uint8_t buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
        data.insert(data.end(), buf, buf + n);
    }
    fclose(f);

    Reader r{ data };
    if ((uint32_t) r.i32() != 0x67676d6c) {  // "ggml"
        fprintf(stderr, "vad: %s is not a ggml file\n", path.c_str());
        return false;
    }
    const int32_t type_len = r.i32();
    const std::string type = r.ok && r.off + type_len <= data.size()
        ? std::string((const char *) data.data() + r.off, type_len) : std::string();
    r.off += type_len;
    const int major = r.i32();
    r.i32();  // minor
    r.i32();  // patch
    const int n_window = r.i32();
    r.i32();  // context
    const int n_layers = r.i32();
    if (!r.ok || type != "silero-16k" || major != 5 || n_window != WINDOW || n_layers != 4) {
        fprintf(stderr, "vad: %s is not a Silero VAD v5 model\n", path.c_str());
        return false;
    }
    for (int i = 0; i < 4; i++) {
        enc_in_[i]  = r.i32();
        enc_out_[i] = r.i32();
        if (r.i32() != 3) {
            r.ok = false;
        }
    }
    const int lstm_in = r.i32(), lstm_hidden = r.i32();
    r.i32();  // final conv in
    r.i32();  // final conv out
    if (!r.ok || enc_in_[0] != N_BINS || lstm_in != HIDDEN || lstm_hidden != HIDDEN || enc_out_[3] != HIDDEN) {
        fprintf(stderr, "vad: unexpected layout in %s\n", path.c_str());
        return false;
    }

    std::map<std::string, std::vector<float> *> want = {
        { "_model.stft.forward_basis_buffer", &stft_ },
        { "_model.decoder.rnn.weight_ih",     &w_ih_ },
        { "_model.decoder.rnn.weight_hh",     &w_hh_ },
        { "_model.decoder.rnn.bias_ih",       &b_ih_ },
        { "_model.decoder.rnn.bias_hh",       &b_hh_ },
        { "_model.decoder.decoder.2.weight",  &out_w_ },
    };
    std::vector<float> out_b;
    want["_model.decoder.decoder.2.bias"] = &out_b;
    for (int i = 0; i < 4; i++) {
        want["_model.encoder." + std::to_string(i) + ".reparam_conv.weight"] = &enc_w_[i];
        want["_model.encoder." + std::to_string(i) + ".reparam_conv.bias"]   = &enc_b_[i];
    }

    int n_loaded = 0;
    while (r.off < data.size()) {
        const int n_dims = r.i32(), name_len = r.i32(), ttype = r.i32();
        size_t count = 1;
        for (int i = 0; i < n_dims && r.ok; i++) {
            count *= (size_t) r.i32();
        }
        if (!r.ok || n_dims < 0 || n_dims > 4 || (ttype != 0 && ttype != 1) || r.off + name_len > data.size()) {
            r.ok = false;
            break;
        }
        const std::string name((const char *) data.data() + r.off, name_len);
        r.off += name_len;
        const size_t bytes = count * (ttype == 0 ? 4 : 2);
        if (r.off + bytes > data.size()) {
            r.ok = false;
            break;
        }
        auto it = want.find(name);
        if (it != want.end()) {
            std::vector<float> & dst = *it->second;
            dst.resize(count);
            const uint8_t * src = data.data() + r.off;
            for (size_t i = 0; i < count; i++) {
                if (ttype == 0) {
                    std::memcpy(&dst[i], src + 4 * i, 4);
                } else {
                    ggml_fp16_t h;
                    std::memcpy(&h, src + 2 * i, 2);
                    dst[i] = ggml_fp16_to_fp32(h);
                }
            }
            n_loaded++;
        }
        r.off += bytes;
    }
    if (!r.ok || n_loaded != (int) want.size() || stft_.size() != (size_t) 2 * N_BINS * N_FFT ||
        w_ih_.size() != (size_t) 4 * HIDDEN * HIDDEN || out_w_.size() != HIDDEN || out_b.size() != 1) {
        fprintf(stderr, "vad: missing or malformed tensors in %s\n", path.c_str());
        return false;
    }
    out_b_  = out_b[0];
    loaded_ = true;
    return true;
}

void Vad::reset(State & st) const {
    st.h.assign(HIDDEN, 0.f);
    st.c.assign(HIDDEN, 0.f);
    st.pending.clear();
    st.prob = 0.f;
}

float Vad::push(State & st, const float * pcm, size_t n_samples) const {
    if (st.h.empty()) {
        reset(st);
    }
    st.pending.insert(st.pending.end(), pcm, pcm + n_samples);
    size_t off = 0;
    for (; off + WINDOW <= st.pending.size(); off += WINDOW) {
        st.prob = window(st, st.pending.data() + off);
    }
    st.pending.erase(st.pending.begin(), st.pending.begin() + off);
    return st.prob;
}

float Vad::window(State & st, const float * x) const {
    // Reflect padding, as torch's "reflect" mode.
    float xp[WINDOW + 2 * PAD];
    for (int i = 0; i < PAD; i++) {
        xp[i]                  = x[PAD - i];
        xp[PAD + WINDOW + i]   = x[WINDOW - 2 - i];
    }
    std::memcpy(xp + PAD, x, sizeof(float) * WINDOW);

    // STFT magnitudes, [bin][frame].
    constexpr int N_FRAMES = (WINDOW + 2 * PAD - N_FFT) / HOP + 1;
    float mag[N_BINS * N_FRAMES];
    for (int t = 0; t < N_FRAMES; t++) {
        const float * seg = xp + t * HOP;
        for (int b = 0; b < N_BINS; b++) {
            const float re = dot(stft_.data() + (size_t) b * N_FFT, seg, N_FFT);
            const float im = dot(stft_.data() + (size_t) (b + N_BINS) * N_FFT, seg, N_FFT);
            mag[b * N_FRAMES + t] = std::sqrt(re * re + im * im);
        }
    }

    float a[HIDDEN * N_FRAMES], b[HIDDEN * N_FRAMES];
    int len = N_FRAMES;
    static constexpr int STRIDE[4] = { 1, 2, 2, 1 };
    const float * in = mag;
    float * out = a;
    for (int l = 0; l < 4; l++) {
        int out_len = 0;
        conv3(enc_w_[l], enc_b_[l], enc_in_[l], enc_out_[l], in, len, STRIDE[l], out, out_len);
        len = out_len;
        in  = out;
        out = out == a ? b : a;
    }
    // `in` holds [128][1]: the encoder's single remaining step.

    float gates[4 * HIDDEN];
    for (int j = 0; j < 4 * HIDDEN; j++) {
        gates[j] = b_ih_[j] + b_hh_[j] + dot(w_ih_.data() + (size_t) j * HIDDEN, in, HIDDEN) +
                   dot(w_hh_.data() + (size_t) j * HIDDEN, st.h.data(), HIDDEN);
    }
    float out_acc = out_b_;
    for (int i = 0; i < HIDDEN; i++) {
        const float ig = sigmoid(gates[i]);
        const float fg = sigmoid(gates[HIDDEN + i]);
        const float gg = std::tanh(gates[2 * HIDDEN + i]);
        const float og = sigmoid(gates[3 * HIDDEN + i]);
        st.c[i] = fg * st.c[i] + ig * gg;
        st.h[i] = og * std::tanh(st.c[i]);
        out_acc += out_w_[i] * std::max(st.h[i], 0.f);
    }
    return sigmoid(out_acc);
}
