// Answers spoken prompts for benchmarks: each record is one user turn in a
// fresh conversation, run through the same path as the live session.
//
// Input: records of [uint32 n_samples][float32 x n_samples], mono 16 kHz,
// from a file or stdin ("-"). With --text, records of [uint32 n][n bytes of
// UTF-8] that are answered as typed turns.
// Output: one JSON object per record on stdout:
//   {"response": "...", "audio_s": 3.4, "first_token_ms": 52, "total_ms": 410}

#include "llm.h"
#include "logging.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using Clock = std::chrono::steady_clock;

static std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (const char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if ((unsigned char) c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

static double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

int main(int argc, char ** argv) {
    Llm::Params params;
    params.gemma("E2B");

    std::string in_path = "-";
    bool text_mode = false;
    bool oneshot   = false;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--text") {
            text_mode = true;
            continue;
        }
        if (std::string(argv[i]) == "--oneshot") {
            oneshot = true;
            continue;
        }
        if (i + 1 >= argc) {
            fprintf(stderr, "missing value for %s\n", argv[i]);
            return 1;
        }
        const std::string arg = argv[i++];
        const char * value = argv[i];
        if      (arg == "--gemma")      { if (!params.gemma(value)) { fprintf(stderr, "--gemma: E2B or E4B\n"); return 1; } }
        else if (arg == "--model")      params.model_path = value;
        else if (arg == "--mmproj")     params.mmproj_path = value;
        else if (arg == "--in")         in_path = value;
        else if (arg == "--system")     params.system_prompt = value;
        else if (arg == "--temp")       params.temperature = (float) std::atof(value);
        else if (arg == "--max-tokens") params.max_tokens = std::atoi(value);
        else if (arg == "--seed")       params.seed = (unsigned) std::atoi(value);
        else {
            fprintf(stderr,
                "usage: %s [--in clips.pcm|-] [--text|--oneshot] [--system TEXT] [--temp T] [--max-tokens N]\n"
                "          [--seed N] [--gemma E2B|E4B] [--model FILE] [--mmproj FILE]\n",
                argv[0]);
            return 1;
        }
    }

    init_logging();

    Llm llm;
    if (!llm.load(params)) {
        return 1;
    }
    const int    prefix      = llm.n_past();
    const int    rate        = llm.audio_sample_rate();
    const size_t max_samples = llm.max_turn_samples();

    FILE * in = in_path == "-" ? stdin : fopen(in_path.c_str(), "rb");
    if (!in) {
        fprintf(stderr, "failed to open %s\n", in_path.c_str());
        return 1;
    }

    std::vector<float> pcm;
    std::string        text;
    int n = 0;
    uint32_t n_items = 0;
    while (fread(&n_items, sizeof(n_items), 1, in) == 1) {
        size_t skip = 0;
        if (text_mode) {
            text.resize(n_items);
            if (fread(text.data(), 1, n_items, in) != n_items) {
                fprintf(stderr, "truncated input at record %d\n", n);
                return 1;
            }
        } else {
            pcm.resize(n_items);
            if (fread(pcm.data(), sizeof(float), n_items, in) != n_items) {
                fprintf(stderr, "truncated input at record %d\n", n);
                return 1;
            }
            // An audio segment holds < 30 s; keep the end, where the question is.
            skip = pcm.size() > max_samples ? pcm.size() - max_samples : 0;
        }

        const bool streamed = !text_mode && !oneshot;
        if (!llm.truncate(prefix) ||
            (streamed && (!llm.begin_user_turn() || !llm.push_audio(pcm.data() + skip, pcm.size() - skip)))) {
            fprintf(stderr, "failed at record %d\n", n);
            return 1;
        }

        // Timed from the end of the user's speech, as in the live session.
        const auto t_end = Clock::now();
        const bool ok = text_mode ? llm.user_text(text)
                      : oneshot   ? llm.user_audio_oneshot(pcm.data() + skip, pcm.size() - skip)
                                  : llm.end_user_turn();
        if (!ok) {
            fprintf(stderr, "failed at record %d\n", n);
            return 1;
        }
        std::string response;
        double first_token_ms = -1;
        llm.generate([&](std::string_view piece) {
            if (first_token_ms < 0) {
                first_token_ms = ms_since(t_end);
            }
            response += piece;
            return true;
        });

        printf("{\"response\": \"%s\", \"audio_s\": %.2f, \"first_token_ms\": %.0f, \"total_ms\": %.0f}\n",
               json_escape(response).c_str(), (double) (pcm.size() - skip) / rate, first_token_ms, ms_since(t_end));
        fflush(stdout);
        n++;
    }

    if (in != stdin) {
        fclose(in);
    }
    fprintf(stderr, "done: %d prompts\n", n);
    return 0;
}
