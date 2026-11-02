#pragma once

#include <cstddef>
#include <string>
#include <vector>

// Silero VAD v5 (snakers4/silero-vad): the probability that a 32 ms window
// of 16 kHz audio is speech. Weights in whisper.cpp's ggml format
// (ggml-org/whisper-vad); the small network runs on the CPU.
class Vad {
public:
    static constexpr int SAMPLE_RATE = 16000;
    static constexpr int WINDOW      = 512;

    // Per-stream state: the LSTM and the samples of an incomplete window.
    struct State {
        std::vector<float> h, c, pending;
        float prob = 0.f;  // of the latest complete window
    };

    bool load(const std::string & path);
    bool loaded() const { return loaded_; }

    // Feeds audio; returns the probability of the latest complete window.
    float push(State & st, const float * pcm, size_t n_samples) const;
    void  reset(State & st) const;

private:
    float window(State & st, const float * x) const;

    bool loaded_ = false;

    // [out][in][k] like PyTorch, all f32.
    std::vector<float> stft_;          // 258 x 256
    std::vector<float> enc_w_[4], enc_b_[4];
    int                enc_in_[4]  = {}, enc_out_[4] = {};
    std::vector<float> w_ih_, w_hh_, b_ih_, b_hh_;  // 512 x 128, 512
    std::vector<float> out_w_;                      // 128
    float              out_b_ = 0.f;
};
