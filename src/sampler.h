#pragma once

#include <cstdint>
#include <random>
#include <vector>

// Picks the next token from logits: the top_k most likely, of those the
// fewest that make up top_p of the probability, at a temperature.
class Sampler {
public:
    void init(int top_k, float top_p, float temperature, uint32_t seed);
    // Starts the random sequence over, from the seed (0: a new random one).
    void reset();
    int32_t sample(const std::vector<float> & logits);

private:
    int   top_k_ = 64;
    float top_p_ = 0.95f;
    float temperature_ = 1.f;
    uint32_t seed_ = 0;
    std::mt19937 rng_;
    std::vector<std::pair<float, int32_t>> candidates_;
};
