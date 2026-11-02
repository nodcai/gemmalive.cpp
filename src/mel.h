#pragma once

#include <cstddef>
#include <vector>

// Gemma 4's audio features, computed as audio arrives: 128 log-mel bins
// every 10 ms over 20 ms Hann windows (zero-padded to 512 samples) of 16 kHz
// audio, the first window centred on the first sample.
//
// A frame is final once all of its window was heard; finish() adds the
// last ones, over zeros past the end.
class Mel {
public:
    static constexpr int SAMPLE_RATE = 16000;
    static constexpr int N_MEL       = 128;
    static constexpr int HOP         = 160;

    Mel();

    void push(const float * pcm, size_t n);
    void finish();

    // Frames so far, each N_MEL values.
    int n_frames() const { return (int) (frames_.size() / N_MEL); }
    const float * frame(int i) const { return frames_.data() + (size_t) i * N_MEL; }
    size_t n_samples() const { return pcm_.size() - PAD; }

private:
    static constexpr int N_FFT  = 512;
    static constexpr int WINDOW = 320;
    static constexpr int PAD    = WINDOW / 2;

    void compute(const float * samples);
    void fft(std::vector<float> & re, std::vector<float> & im) const;

    std::vector<float> pcm_;     // PAD zeros, then the audio
    std::vector<float> frames_;  // time-major
    std::vector<float> window_;
    std::vector<float> filters_;  // [N_MEL][N_FFT / 2 + 1]
    std::vector<float> cos_, sin_;
    std::vector<float> re_, im_;
};
