#include "audio_io.h"
#include "device.h"
#include "llm.h"
#include "logging.h"
#include "tts.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

static bool g_verbose = false;

static double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

static bool run_turn(Llm & llm, Tts & tts, const std::string & input, const std::string & output) {
    std::vector<float> pcm;
    if (!llm.read_audio_file(input, pcm)) {
        fprintf(stderr, "failed to read %s\n", input.c_str());
        return false;
    }
    const int in_rate = llm.audio_sample_rate();
    fprintf(stderr, "\n> %s (%.2f s)\n", input.c_str(), (double) pcm.size() / in_rate);

    // Feed the user's speech in 20 ms chunks, as a live source would.
    auto t_user = Clock::now();
    if (!llm.begin_user_turn()) {
        return false;
    }
    const size_t chunk = in_rate / 50;
    for (size_t off = 0; off < pcm.size(); off += chunk) {
        if (!llm.push_audio(pcm.data() + off, std::min(chunk, pcm.size() - off))) {
            return false;
        }
    }
    const double ms_encode = ms_since(t_user);

    // Latency is measured from the end of the user's speech.
    auto t_end = Clock::now();
    if (!llm.end_user_turn()) {
        return false;
    }
    const double ms_finalize = ms_since(t_end);

    std::mutex mtx;
    std::vector<float> reply;
    double ms_first_audio = -1;
    const auto on_audio = [&](const float * samples, int n) {
        std::lock_guard<std::mutex> lock(mtx);
        if (ms_first_audio < 0) {
            ms_first_audio = ms_since(t_end);
        }
        reply.insert(reply.end(), samples, samples + n);
    };
    if (!tts.begin(on_audio)) {
        return false;
    }

    double ms_first_token = -1;
    int n_pieces = 0;
    llm.generate([&](std::string_view piece) {
        if (ms_first_token < 0) {
            ms_first_token = ms_since(t_end);
        }
        n_pieces++;
        fwrite(piece.data(), 1, piece.size(), stdout);
        fflush(stdout);
        return tts.push_text(piece);
    });
    const double ms_text = ms_since(t_end);
    fputc('\n', stdout);

    tts.finish();
    const double ms_total = ms_since(t_end);

    const int out_rate = tts.sample_rate();
    if (!write_wav(output, reply.data(), reply.size(), out_rate)) {
        fprintf(stderr, "failed to write %s\n", output.c_str());
        return false;
    }

    fprintf(stderr,
        "\n"
        "audio encode (streamed)  %7.0f ms\n"
        "finalize + prompt        %7.0f ms\n"
        "first token              %7.0f ms\n"
        "first audio              %7.0f ms\n"
        "text done (%3d pieces)   %7.0f ms\n"
        "audio done               %7.0f ms\n"
        "reply audio              %7.2f s -> %s\n",
        ms_encode, ms_finalize, ms_first_token, ms_first_audio, n_pieces, ms_text, ms_total,
        (double) reply.size() / out_rate, output.c_str());

    return true;
}

static void usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s -i <input audio> [-i <input audio> ...] [options]\n"
        "  -i, --input FILE      user speech (wav/mp3/flac), one turn per -i\n"
        "  -o, --output FILE     reply speech wav (default: reply.wav, numbered\n"
        "                        reply-1.wav, reply-2.wav, ... for several turns)\n"
        "  --gemma E2B|E4B       which Gemma 4 (default: E2B)\n"
        "  --model FILE          Gemma gguf, instead of models/<gemma>\n"
        "  --mmproj FILE         Gemma audio projector gguf, for a model without it\n"
        "  --tts-model FILE      livevoice gguf\n"
        "  --voice FILE          livevoice voice gguf\n"
        "  --system TEXT         system prompt\n"
        "  --locale ID           the user's locale, e.g. de_DE (default: from LANG)\n"
        "  -v, --verbose         show library logs\n",
        argv0);
}

int main(int argc, char ** argv) {
    Llm::Params llm_params;
    llm_params.gemma("E2B");

    Tts::Params tts_params;
    tts_params.model_path = "models/livevoice-0.5b-COMPACT.gguf";
    tts_params.voice_path = "models/voices/en-Clarissa_female.gguf";

    std::vector<std::string> inputs;
    std::string output = "reply.wav";
    Device device;
    device.locale = environment_locale();

    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                usage(argv[0]);
                exit(1);
            }
            return argv[++i];
        };
        if      (arg == "-i" || arg == "--input")  inputs.push_back(next());
        else if (arg == "-o" || arg == "--output") output = next();
        else if (arg == "--gemma")                 { if (!llm_params.gemma(next())) { usage(argv[0]); return 1; } }
        else if (arg == "--model")                 llm_params.model_path = next();
        else if (arg == "--mmproj")                llm_params.mmproj_path = next();
        else if (arg == "--tts-model")             tts_params.model_path = next();
        else if (arg == "--voice")                 tts_params.voice_path = next();
        else if (arg == "--system")                llm_params.system_prompt = next();
        else if (arg == "--locale")                device.locale = next();
        else if (arg == "-v" || arg == "--verbose") g_verbose = true;
        else { usage(argv[0]); return 1; }
    }
    if (inputs.empty()) {
        usage(argv[0]);
        return 1;
    }

    llm_params.context      = [device] { return device_context(device); };
    llm_params.turn_context = [device] { return clock_note(device); };

    init_logging(g_verbose);

    auto t_load = Clock::now();
    Llm llm;
    if (!llm.load(llm_params)) {
        return 1;
    }
    Tts tts;
    if (!tts.load(tts_params)) {
        return 1;
    }
    fprintf(stderr, "loaded in %.0f ms\n", ms_since(t_load));

    for (size_t t = 0; t < inputs.size(); t++) {
        std::string out_path = output;
        if (inputs.size() > 1) {
            const size_t dot = output.rfind('.');
            const std::string n = "-" + std::to_string(t + 1);
            out_path = dot == std::string::npos ? output + n : output.substr(0, dot) + n + output.substr(dot);
        }
        if (!run_turn(llm, tts, inputs[t], out_path)) {
            return 1;
        }
    }
    return 0;
}
