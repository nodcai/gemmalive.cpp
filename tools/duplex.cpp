// Runs recorded user audio through a live session in real time and records
// what the assistant says, on the same timeline: the offline counterpart of
// a conversation in an app, for full-duplex benchmarks such as Full-Duplex-Bench.
//
//   gemmalive-duplex --list jobs.tsv [options]
//
// Each line of the list is "<input audio>\t<output wav>". The input is fed in
// 20 ms frames at real-time pace while the assistant's speech is played out
// at the same pace (dropped when it is interrupted), and after the input
// ends the run continues until the assistant has finished. Every input gets
// a fresh conversation. Next to each output, <output>.events.jsonl lists the
// session's events with their time in ms.

#include "audio_io.h"
#include "llm.h"
#include "logging.h"
#include "session.h"
#include "tts.h"
#include "turn_head.h"
#include "vad.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;

static constexpr int FRAME_MS        = 20;
static constexpr int MIN_TAIL_MS     = 3000;   // after the input, for an open turn to end
static constexpr int QUIET_MS        = 1000;   // assistant silent this long: done
static constexpr int MAX_TAIL_MS     = 30000;

struct Playback {
    std::mutex        mtx;
    std::deque<float> queue;
    bool              paused = false;
};

static bool run(Llm & llm, Tts & tts, const TurnHead * head, const TurnHead * addressee, const Vad * vad,
                const std::string & in_path, const std::string & out_path) {
    std::vector<float> input;
    if (!llm.read_audio_file(in_path, input)) {
        fprintf(stderr, "cannot read %s\n", in_path.c_str());
        return false;
    }
    if (!llm.reset()) {
        return false;
    }

    const int in_rate   = llm.audio_sample_rate();
    const int out_rate  = tts.sample_rate();
    const int in_frame  = in_rate * FRAME_MS / 1000;
    const int out_frame = out_rate * FRAME_MS / 1000;

    Playback play;
    std::vector<float> output;
    std::ofstream events(out_path + ".events.jsonl");
    std::mutex events_mtx;
    const auto t0 = Clock::now();

    Session::Callbacks cb;
    cb.on_audio = [&](const float * pcm, int n) {
        std::lock_guard<std::mutex> lock(play.mtx);
        play.queue.insert(play.queue.end(), pcm, pcm + n);
    };
    cb.on_interrupt = [&] {
        std::lock_guard<std::mutex> lock(play.mtx);
        play.queue.clear();
        play.paused = false;
    };
    cb.on_pause = [&] {
        std::lock_guard<std::mutex> lock(play.mtx);
        play.paused = true;
    };
    cb.on_resume = [&] {
        std::lock_guard<std::mutex> lock(play.mtx);
        play.paused = false;
    };
    cb.queued = [&] {
        std::lock_guard<std::mutex> lock(play.mtx);
        return (double) play.queue.size() / out_rate;
    };
    cb.on_event = [&](const std::string & json) {
        const long ms = (long) std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - t0).count();
        std::lock_guard<std::mutex> lock(events_mtx);
        events << "{\"ms\":" << ms << ",\"event\":" << json << "}\n";
    };

    {
        Session session(llm, tts, head, addressee, vad, cb);
        const std::vector<float> silence(in_frame, 0.f);
        const size_t n_in_frames = (input.size() + in_frame - 1) / in_frame;
        int quiet_ms = 0;

        auto next = Clock::now();
        for (size_t f = 0;; f++) {
            next += std::chrono::milliseconds(FRAME_MS);
            std::this_thread::sleep_until(next);

            // The microphone keeps sending after the recording ends.
            if (f < n_in_frames) {
                const size_t off = f * in_frame;
                session.push_audio(input.data() + off, std::min((size_t) in_frame, input.size() - off));
            } else {
                session.push_audio(silence.data(), silence.size());
            }

            bool speaking;
            {
                std::lock_guard<std::mutex> lock(play.mtx);
                const size_t n = play.paused ? 0 : std::min(play.queue.size(), (size_t) out_frame);
                output.insert(output.end(), play.queue.begin(), play.queue.begin() + n);
                output.resize(output.size() + (out_frame - n), 0.f);
                play.queue.erase(play.queue.begin(), play.queue.begin() + n);
                speaking = n > 0 || !play.queue.empty();
            }
            quiet_ms = speaking || tts.active() ? 0 : quiet_ms + FRAME_MS;

            const long tail_ms = ((long) f - (long) n_in_frames) * FRAME_MS;
            if ((tail_ms >= MIN_TAIL_MS && quiet_ms >= QUIET_MS) || tail_ms >= MAX_TAIL_MS) {
                break;
            }
        }
    }

    if (!write_wav(out_path, output.data(), output.size(), out_rate)) {
        fprintf(stderr, "cannot write %s\n", out_path.c_str());
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    Llm::Params llm_params;
    llm_params.gemma("E2B");

    Tts::Params tts_params;
    tts_params.model_path = "models/livevoice-0.5b-COMPACT.gguf";
    tts_params.voice_path = "models/voices/en-Clarissa_female.gguf";

    std::string list_path;
    std::string heads_path;
    std::string vad_path = "models/ggml-silero-v5.1.2.bin";
    bool        verbose  = false;
    const auto usage = [&] {
        fprintf(stderr, "usage: %s --list jobs.tsv [--heads FILE|none] [--vad FILE|none]\n"
                        "          [--gemma E2B|E4B] [--model FILE] [--mmproj FILE] [--tts-model FILE] [--voice FILE]\n"
                        "          [--system TEXT | --system-file FILE] [-v]\n", argv[0]);
        return 1;
    };
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg == "-v" || arg == "--verbose") {
            verbose = true;
            continue;
        }
        if (i + 1 >= argc) {
            return usage();
        }
        const char * value = argv[++i];
        if      (arg == "--list")      list_path = value;
        else if (arg == "--gemma")     { if (!llm_params.gemma(value)) return usage(); }
        else if (arg == "--model")     llm_params.model_path = value;
        else if (arg == "--mmproj")    llm_params.mmproj_path = value;
        else if (arg == "--tts-model") tts_params.model_path = value;
        else if (arg == "--voice")     tts_params.voice_path = value;
        else if (arg == "--heads")     heads_path = value;
        else if (arg == "--vad")       vad_path = value;
        else if (arg == "--system")    llm_params.system_prompt = value;
        else if (arg == "--system-file") {
            // The system prompt from a file, e.g. to compare prompts.
            std::ifstream f(value);
            if (!f) {
                fprintf(stderr, "cannot read %s\n", value);
                return 1;
            }
            std::stringstream text;
            text << f.rdbuf();
            llm_params.system_prompt = text.str();
            while (!llm_params.system_prompt.empty() && llm_params.system_prompt.back() == '\n') {
                llm_params.system_prompt.pop_back();
            }
        } else {
            return usage();
        }
    }
    if (list_path.empty()) {
        return usage();
    }
    std::ifstream list(list_path);
    if (!list) {
        fprintf(stderr, "cannot read %s\n", list_path.c_str());
        return 1;
    }

    init_logging(verbose);

    Llm llm;
    Tts tts;
    if (!llm.load(llm_params) || !tts.load(tts_params)) {
        return 1;
    }
    tts.warmup();

    if (heads_path.empty()) {
        heads_path = TurnHead::path_for(llm_params.model_path);
    }
    const TurnHead::Expect expect = { llm_params.model_path, llm_params.mmproj_path, llm_params.system_prompt };
    TurnHead head, addressee;
    if (heads_path != "none" && std::filesystem::exists(heads_path) &&
        !(head.load(heads_path, TurnHead::END_OF_TURN, expect) &&
          addressee.load(heads_path, TurnHead::ADDRESSEE, expect))) {
        return 1;
    }
    fprintf(stderr, "end of turn: %s\n", head.loaded() ? heads_path.c_str() : "fixed silence");
    fprintf(stderr, "addressee: %s\n", addressee.loaded() ? heads_path.c_str() : "everyone");

    Vad vad;
    if (vad_path != "none" && std::filesystem::exists(vad_path) && !vad.load(vad_path)) {
        return 1;
    }
    fprintf(stderr, "speech: %s\n", vad.loaded() ? vad_path.c_str() : "energy only");

    int n = 0, failed = 0;
    for (std::string line; std::getline(list, line);) {
        const size_t tab = line.find('\t');
        if (tab == std::string::npos) {
            continue;
        }
        const std::string in = line.substr(0, tab), out = line.substr(tab + 1);
        const auto t = Clock::now();
        const bool ok = run(llm, tts, &head, &addressee, &vad, in, out);
        failed += !ok;
        fprintf(stderr, "%s %s (%.0f s)\n", ok ? "done" : "FAILED", out.c_str(),
                std::chrono::duration<double>(Clock::now() - t).count());
        n++;
    }
    fprintf(stderr, "%d runs, %d failed\n", n, failed);
    return failed ? 1 : 0;
}
