#include "mel.h"

#include <algorithm>
#include <cmath>

static constexpr double PI = 3.14159265358979323846;

Mel::Mel() : pcm_(PAD, 0.f) {
    window_.assign(N_FFT, 0.f);
    for (int i = 0; i < WINDOW; i++) {
        window_[i] = 0.5f - 0.5f * std::cos(2.f * (float) PI * i / WINDOW);
    }

    // Triangular filters, evenly spaced on the HTK mel scale up to Nyquist.
    const int n_bins = N_FFT / 2 + 1;
    const auto hz_to_mel = [](double hz) { return 2595.0 * std::log10(1.0 + hz / 700.0); };
    const auto mel_to_hz = [](double mel) { return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0); };
    const double top = hz_to_mel(SAMPLE_RATE / 2.0);
    std::vector<double> edges(N_MEL + 2);
    for (int i = 0; i < N_MEL + 2; i++) {
        edges[i] = mel_to_hz(top * i / (N_MEL + 1));
    }
    filters_.assign((size_t) N_MEL * n_bins, 0.f);
    for (int m = 0; m < N_MEL; m++) {
        const double left = edges[m], centre = edges[m + 1], right = edges[m + 2];
        for (int k = 0; k < n_bins; k++) {
            const double f = k * (double) SAMPLE_RATE / N_FFT;
            double w = 0.0;
            if (f >= left && f <= centre) {
                w = (f - left) / std::max(1e-30, centre - left);
            } else if (f > centre && f <= right) {
                w = (right - f) / std::max(1e-30, right - centre);
            }
            filters_[(size_t) m * n_bins + k] = (float) w;
        }
    }

    cos_.resize(N_FFT / 2);
    sin_.resize(N_FFT / 2);
    for (int i = 0; i < N_FFT / 2; i++) {
        cos_[i] = (float) std::cos(2.0 * PI * i / N_FFT);
        sin_[i] = (float) std::sin(2.0 * PI * i / N_FFT);
    }
    re_.resize(N_FFT);
    im_.resize(N_FFT);
}

void Mel::push(const float * pcm, size_t n) {
    pcm_.insert(pcm_.end(), pcm, pcm + n);
    while (pcm_.size() >= (size_t) n_frames() * HOP + N_FFT) {
        compute(pcm_.data() + (size_t) n_frames() * HOP);
    }
}

void Mel::finish() {
    // As many frames as windows of WINDOW + 1 samples fit, the last ones
    // reaching into zeros.
    const int n_real = (int) n_samples();
    const int total  = std::max(0, (n_real + PAD - (WINDOW + 1)) / HOP + 1);
    std::vector<float> tail(N_FFT, 0.f);
    while (n_frames() < total) {
        const size_t start = (size_t) n_frames() * HOP;
        std::fill(tail.begin(), tail.end(), 0.f);
        std::copy(pcm_.begin() + std::min(start, pcm_.size()), pcm_.begin() + std::min(start + N_FFT, pcm_.size()), tail.begin());
        compute(tail.data());
    }
}

void Mel::fft(std::vector<float> & re, std::vector<float> & im) const {
    const int n = N_FFT;
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) {
            j ^= bit;
        }
        j ^= bit;
        if (i < j) {
            std::swap(re[i], re[j]);
            std::swap(im[i], im[j]);
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        const int step = n / len;
        for (int i = 0; i < n; i += len) {
            for (int k = 0; k < len / 2; k++) {
                const float wr = cos_[k * step], wi = -sin_[k * step];
                const float ur = re[i + k], ui = im[i + k];
                const float xr = re[i + k + len / 2], xi = im[i + k + len / 2];
                const float vr = xr * wr - xi * wi, vi = xr * wi + xi * wr;
                re[i + k] = ur + vr;
                im[i + k] = ui + vi;
                re[i + k + len / 2] = ur - vr;
                im[i + k + len / 2] = ui - vi;
            }
        }
    }
}

void Mel::compute(const float * samples) {
    for (int i = 0; i < N_FFT; i++) {
        re_[i] = window_[i] * samples[i];
        im_[i] = 0.f;
    }
    fft(re_, im_);
    const int n_bins = N_FFT / 2 + 1;
    for (int k = 0; k < n_bins; k++) {
        re_[k] = std::sqrt(re_[k] * re_[k] + im_[k] * im_[k]);
    }
    for (int m = 0; m < N_MEL; m++) {
        const float * f = filters_.data() + (size_t) m * n_bins;
        double sum = 0.0;
        for (int k = 0; k < n_bins; k++) {
            sum += re_[k] * f[k];
        }
        frames_.push_back((float) std::log(std::max(sum, 0.001)));
    }
}
