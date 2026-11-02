#include "sampler.h"

#include <algorithm>
#include <cmath>

void Sampler::init(int top_k, float top_p, float temperature, uint32_t seed) {
    top_k_       = top_k;
    top_p_       = top_p;
    temperature_ = temperature;
    seed_        = seed;
    reset();
}

void Sampler::reset() {
    rng_.seed(seed_ ? seed_ : std::random_device{}());
}

int32_t Sampler::sample(const std::vector<float> & logits) {
    const int n = (int) logits.size();
    const int k = top_k_ > 0 ? std::min(top_k_, n) : n;
    candidates_.resize(n);
    for (int i = 0; i < n; i++) {
        candidates_[i] = { logits[i], i };
    }
    std::partial_sort(candidates_.begin(), candidates_.begin() + k, candidates_.end(),
                      [](const auto & a, const auto & b) { return a.first > b.first; });
    candidates_.resize(k);
    if (temperature_ <= 0.f) {
        return candidates_[0].second;
    }

    const auto softmax = [&](float scale) {
        std::vector<double> p(candidates_.size());
        double sum = 0.0;
        for (size_t i = 0; i < p.size(); i++) {
            p[i] = std::exp((double) (candidates_[i].first - candidates_[0].first) * scale);
            sum += p[i];
        }
        for (double & x : p) {
            x /= sum;
        }
        return p;
    };

    if (top_p_ < 1.f) {
        const std::vector<double> p = softmax(1.0);
        double cum = 0.0;
        size_t keep = p.size();
        for (size_t i = 0; i < p.size(); i++) {
            cum += p[i];
            if (cum >= top_p_) {
                keep = i + 1;
                break;
            }
        }
        candidates_.resize(keep);
    }

    const std::vector<double> p = softmax(1.0 / temperature_);
    double r = std::uniform_real_distribution<double>(0.0, 1.0)(rng_);
    for (size_t i = 0; i < p.size(); i++) {
        r -= p[i];
        if (r <= 0.0) {
            return candidates_[i].second;
        }
    }
    return candidates_.back().second;
}
