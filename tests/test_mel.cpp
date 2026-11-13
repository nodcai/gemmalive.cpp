// The audio features: how many frames audio gives, that they do not depend
// on how the audio arrives, and that a tone lands in its mel bin.
#include "check.h"
#include "mel.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <vector>

static constexpr double PI = 3.14159265358979323846;

static std::vector<float> tone(double hz, int n, float level = 0.5f) {
    std::vector<float> pcm(n);
    for (int i = 0; i < n; i++) {
        pcm[i] = level * (float) std::sin(2.0 * PI * hz * i / Mel::SAMPLE_RATE);
    }
    return pcm;
}

// Where a bin's filter is centred: 128 triangles evenly spaced on the HTK
// mel scale up to 8 kHz.
static double centre_of(int bin) {
    const double top = 2595.0 * std::log10(1.0 + (Mel::SAMPLE_RATE / 2.0) / 700.0);
    return 700.0 * (std::pow(10.0, top * (bin + 1) / (Mel::N_MEL + 1) / 2595.0) - 1.0);
}

int main() {
    // A frame every 10 ms, final once its 32 ms of samples were heard (the
    // first is centred on the first sample); finish() adds the last ones.
    {
        Mel m;
        const std::vector<float> pcm = tone(440, Mel::SAMPLE_RATE);
        CHECK_EQ(m.n_frames(), 0);
        m.push(pcm.data(), 351);
        CHECK_EQ(m.n_frames(), 0);
        m.push(pcm.data() + 351, 1);
        CHECK_EQ(m.n_frames(), 1);
        m.push(pcm.data() + 352, pcm.size() - 352);
        CHECK_EQ(m.n_frames(), 98);
        CHECK_EQ(m.n_samples(), pcm.size());
        m.finish();
        CHECK_EQ(m.n_frames(), 99);
    }
    {
        Mel m;
        m.finish();
        CHECK_EQ(m.n_frames(), 0);
    }

    // In 20 ms pieces, in odd pieces, or at once: the same frames.
    {
        const std::vector<float> pcm = tone(300, 3 * Mel::SAMPLE_RATE + 123);
        Mel whole, frames, odd;
        whole.push(pcm.data(), pcm.size());
        for (size_t at = 0; at < pcm.size(); at += 320) {
            frames.push(pcm.data() + at, std::min<size_t>(320, pcm.size() - at));
        }
        for (size_t at = 0, n = 1; at < pcm.size(); at += n, n = n * 7 % 501 + 1) {
            n = std::min(n, pcm.size() - at);
            odd.push(pcm.data() + at, n);
        }
        whole.finish();
        frames.finish();
        odd.finish();
        CHECK_EQ(frames.n_frames(), whole.n_frames());
        CHECK_EQ(odd.n_frames(), whole.n_frames());
        int differ = 0;
        for (int f = 0; f < whole.n_frames() && f < frames.n_frames() && f < odd.n_frames(); f++) {
            for (int b = 0; b < Mel::N_MEL; b++) {
                differ += whole.frame(f)[b] != frames.frame(f)[b] || whole.frame(f)[b] != odd.frame(f)[b];
            }
        }
        CHECK_EQ(differ, 0);
    }

    // A tone is loudest in a bin of its own frequency, in the middle of the
    // audio: within the 100 Hz either side that a 20 ms window smears it over.
    for (const double hz : { 250.0, 1000.0, 3000.0, 6000.0 }) {
        Mel m;
        const std::vector<float> pcm = tone(hz, Mel::SAMPLE_RATE);
        m.push(pcm.data(), pcm.size());
        const float * f = m.frame(50);
        int loudest = 0;
        for (int b = 1; b < Mel::N_MEL; b++) {
            if (f[b] > f[loudest]) {
                loudest = b;
            }
        }
        g_checks++;
        if (std::fabs(centre_of(loudest) - hz) > 100.0) {
            fprintf(stderr, "FAIL: %.0f Hz is loudest in bin %d, at %.0f Hz\n", hz, loudest, centre_of(loudest));
            g_failed++;
        }
        CHECK(f[loudest] > 0.f);
    }

    // Silence sits on the floor, log(0.001).
    {
        Mel m;
        const std::vector<float> pcm(Mel::SAMPLE_RATE / 2, 0.f);
        m.push(pcm.data(), pcm.size());
        CHECK(m.n_frames() > 0);
        bool floor = true;
        for (int f = 0; f < m.n_frames(); f++) {
            for (int b = 0; b < Mel::N_MEL; b++) {
                floor = floor && std::fabs(m.frame(f)[b] - std::log(0.001f)) < 1e-5f;
            }
        }
        CHECK(floor);
    }
    return done();
}
