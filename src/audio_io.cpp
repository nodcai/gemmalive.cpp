#include "audio_io.h"

#define MINIAUDIO_IMPLEMENTATION
#define MA_NO_ENCODING
#define MA_NO_DEVICE_IO
#define MA_NO_RESOURCE_MANAGER
#define MA_NO_NODE_GRAPH
#define MA_NO_ENGINE
#define MA_NO_GENERATION
#include "miniaudio.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>

static bool decode(const std::string & path, ma_decoder_config config, std::vector<float> & out, int & rate, int & channels) {
    ma_decoder decoder;
    if (ma_decoder_init_file(path.c_str(), &config, &decoder) != MA_SUCCESS) {
        return false;
    }
    ma_uint64 n = 0;
    bool ok = ma_decoder_get_length_in_pcm_frames(&decoder, &n) == MA_SUCCESS;
    rate     = (int) decoder.outputSampleRate;
    channels = (int) decoder.outputChannels;
    out.resize((size_t) n * channels);
    ma_uint64 read = 0;
    ok = ok && ma_decoder_read_pcm_frames(&decoder, out.data(), n, &read) == MA_SUCCESS;
    out.resize((size_t) read * channels);
    ma_decoder_uninit(&decoder);
    return ok;
}

bool read_audio(const std::string & path, int sample_rate, std::vector<float> & out) {
    int rate = 0, channels = 0;
    return decode(path, ma_decoder_config_init(ma_format_f32, 1, (ma_uint32) sample_rate), out, rate, channels);
}

bool read_audio_raw(const std::string & path, std::vector<float> & out, int & sample_rate, int & channels) {
    return decode(path, ma_decoder_config_init(ma_format_f32, 0, 0), out, sample_rate, channels);
}

bool write_wav(const std::string & path, const float * pcm, size_t n, int sample_rate) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        return false;
    }
    const auto u32 = [&](uint32_t v) { fwrite(&v, 4, 1, f); };
    const auto u16 = [&](uint16_t v) { fwrite(&v, 2, 1, f); };
    const uint32_t bytes = (uint32_t) (n * 2);
    fwrite("RIFF", 1, 4, f);
    u32(36 + bytes);
    fwrite("WAVEfmt ", 1, 8, f);
    u32(16);
    u16(1);  // PCM
    u16(1);  // mono
    u32((uint32_t) sample_rate);
    u32((uint32_t) sample_rate * 2);
    u16(2);
    u16(16);
    fwrite("data", 1, 4, f);
    u32(bytes);
    std::vector<int16_t> s(n);
    for (size_t i = 0; i < n; i++) {
        s[i] = (int16_t) std::lround(std::clamp(pcm[i], -1.f, 1.f) * 32767.f);
    }
    const bool ok = fwrite(s.data(), 2, n, f) == n;
    return fclose(f) == 0 && ok;
}
