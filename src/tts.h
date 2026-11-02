#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>

struct lv_model;
struct lv_voice;
struct lv_stream;

// Streaming TTS (livevoice) fed with text while it is being generated.
//
// begin() starts a session and push_text() appends text as it arrives. end()
// marks the end without waiting, finish() also waits until all audio was
// delivered, abort() drops the rest. Audio is delivered on the engine's
// worker thread. One session at a time; begin() releases a finished one.
class Tts {
public:
    struct Params {
        std::string model_path;
        std::string voice_path;
        uint32_t    seed              = 0;   // 0 = random
        // With fewer steps the voice fades over a long reply (3: about
        // 4.5 dB a minute).
        int         n_diffusion_steps = 5;
        float       cfg_scale         = 1.5f;
        // Room for a full reply (Llm::Params::max_tokens): about 5 minutes
        // of speech at 7.5 frames per second.
        int         max_speech_frames = 2400;
        int         max_text_tokens   = 2048;
    };

    using AudioCallback = std::function<void(const float * pcm, int n_samples)>;

    Tts() = default;
    ~Tts();
    Tts(const Tts &) = delete;
    Tts & operator=(const Tts &) = delete;

    bool load(const Params & params);

    // Runs a short session so the first real reply does not pay for
    // first-use setup.
    void warmup();

    // Output sample rate (48 kHz with the DFN post-filter, 24 kHz without).
    int sample_rate() const;

    bool begin(AudioCallback on_audio);
    bool push_text(std::string_view text);
    void end();
    void finish();
    void abort();

    // True while a session is still producing audio.
    bool active() const;

private:
    static int on_audio_trampoline(const float * pcm, int n_samples, void * user);

    Params          params_;
    AudioCallback   on_audio_;
    lv_model *      model_  = nullptr;
    lv_voice *      voice_  = nullptr;
    lv_stream *     stream_ = nullptr;
    std::string     held_;      // the end of the text, until what follows shows how to say it
    char            said_ = 0;  // the last character handed to the TTS
};
