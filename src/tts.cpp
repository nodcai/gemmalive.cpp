#include "tts.h"

#include "internal.h"
#include "livevoice.h"

#include <cstdio>
#include <string>

using namespace internal;

// Symbols the model writes but the TTS can't say ("15°C" comes out as
// "15 du c"), in words. A degree sign, or one with its unit, at the end of
// `text` is left in `held` until the next text shows what it is, unless
// `last`. `prev` is the last character said before `text`, and is updated.
std::string internal::spoken(std::string text, std::string & held, char & prev, bool last) {
    static const std::string deg = "\xC2\xB0";  // °
    const auto letter = [](char c) { return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'); };
    held.clear();
    std::string out;
    size_t pos = 0;
    for (size_t at; (at = text.find(deg, pos)) != std::string::npos; pos = at) {
        out.append(text, pos, at - pos);
        size_t      end  = at + deg.size();
        const char  unit = end < text.size() ? text[end] : 0;
        if (!last && (end == text.size() || ((unit == 'C' || unit == 'F') && end + 1 == text.size()))) {
            held = text.substr(at);
            text.resize(at);
            pos = at;
            break;
        }
        const char * words = "degrees";
        if ((unit == 'C' || unit == 'F') && (end + 1 == text.size() || !letter(text[end + 1]))) {
            words = unit == 'C' ? "degrees Celsius" : "degrees Fahrenheit";
            end++;
        }
        const char before = out.empty() ? prev : out.back();
        if (before && before != ' ') {
            out += ' ';
        }
        out += words;
        if (end < text.size() && letter(text[end])) {
            out += ' ';
        }
        at = end;
    }
    if (pos < text.size()) {
        out.append(text, pos);
    }
    if (!out.empty()) {
        prev = out.back();
    }
    return out;
}

Tts::~Tts() {
    abort();
    if (voice_) lv_voice_free(voice_);
    if (model_) lv_model_free(model_);
}

bool Tts::load(const Params & params) {
    params_ = params;
    model_ = lv_model_load(params.model_path.c_str());
    if (!model_) {
        fprintf(stderr, "tts: failed to load %s\n", params.model_path.c_str());
        return false;
    }
    voice_ = lv_voice_load(model_, params.voice_path.c_str());
    if (!voice_) {
        fprintf(stderr, "tts: failed to load %s\n", params.voice_path.c_str());
        return false;
    }
    return true;
}

void Tts::warmup() {
    if (begin([](const float *, int) {})) {
        push_text("Hello.");
        finish();
    }
}

int Tts::sample_rate() const {
    return model_ ? lv_sample_rate(model_, nullptr) : 0;
}

bool Tts::begin(AudioCallback on_audio) {
    if (active()) {
        fprintf(stderr, "tts: session already running\n");
        return false;
    }
    finish();
    on_audio_ = std::move(on_audio);

    lv_params p;
    lv_params_init(&p);
    p.seed            = params_.seed;
    p.n_steps         = params_.n_diffusion_steps;
    p.cfg_scale       = params_.cfg_scale;
    p.max_frames      = params_.max_speech_frames;
    p.max_text_tokens = params_.max_text_tokens;

    held_.clear();
    said_ = 0;
    stream_ = model_ ? lv_stream_begin(model_, voice_, &p, &Tts::on_audio_trampoline, this) : nullptr;
    return stream_ != nullptr;
}

bool Tts::push_text(std::string_view text) {
    if (!stream_) {
        return false;
    }
    const std::string say = spoken(held_ + std::string(text), held_, said_, false);
    return say.empty() || lv_stream_push(stream_, say.c_str()) == LV_OK;
}

void Tts::end() {
    if (stream_) {
        if (!held_.empty()) {
            const std::string say = spoken(held_, held_, said_, true);
            lv_stream_push(stream_, say.c_str());
        }
        lv_stream_end(stream_);
    }
}

bool Tts::active() const {
    return stream_ && !lv_stream_done(stream_);
}

void Tts::finish() {
    if (stream_) {
        lv_stream_free(stream_);
        stream_ = nullptr;
    }
}

void Tts::abort() {
    if (stream_) {
        lv_stream_abort(stream_);
        lv_stream_free(stream_);
        stream_ = nullptr;
    }
}

int Tts::on_audio_trampoline(const float * pcm, int n_samples, void * user) {
    auto * self = static_cast<Tts *>(user);
    if (self->on_audio_) {
        self->on_audio_(pcm, n_samples);
    }
    return 0;
}
