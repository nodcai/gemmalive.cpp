#pragma once

#include "llm.h"
#include "logging.h"

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

// For the tests with a conversation: the model from GEMMALIVE_MODEL (with
// GEMMALIVE_MMPROJ, if its audio encoder is in a file of its own), and
// sound to put into a turn.

static bool load(Llm & llm, const char * model, Llm::Params params = {}) {
    const char * mmproj = getenv("GEMMALIVE_MMPROJ");
    params.model_path  = model;
    params.mmproj_path = mmproj ? mmproj : "";
    init_logging(false);
    return llm.load(params);
}

// A tone in noise: what the audio is does not matter to the caches.
static std::vector<float> sound(double seconds, int rate) {
    std::vector<float> pcm((size_t) (seconds * rate));
    unsigned s = 1;
    for (size_t i = 0; i < pcm.size(); i++) {
        s = s * 1664525u + 1013904223u;
        pcm[i] = 0.1f * (float) std::sin(2.0 * 3.14159265358979323846 * 220.0 * (double) i / rate) +
                 0.02f * ((float) (s >> 8) / 8388608.f - 1.f);
    }
    return pcm;
}

// Into the open turn in 20 ms pieces, as a microphone brings it.
static bool hear(Llm & llm, const std::vector<float> & pcm) {
    const size_t frame = (size_t) llm.audio_sample_rate() / 50;
    for (size_t at = 0; at < pcm.size(); at += frame) {
        if (!llm.push_audio(pcm.data() + at, std::min(frame, pcm.size() - at))) {
            return false;
        }
    }
    return true;
}
