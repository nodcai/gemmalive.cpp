// Extracts Gemma hidden states at the end of committed audio, for training
// the end-of-turn and addressee heads. Runs the same path as the live
// session: system prompt, conversation, user turn, streamed audio segment,
// commit.
//
// Input: records of [uint32 n_samples][float32 x n_samples], mono 16 kHz,
// from a file or stdin ("-").
// Output: header [int32 n_layer][int32 n_embd], then per record the output
// of every layer at the segment's closing token as float16.
//
// With --prompts, each record gets a system prompt drawn from a file (one per
// line, "-" for none), so that a head learns to ignore it; --prompt-log
// writes the line number drawn for each record.
//
// With --history, records come in blocks (--block) that each follow a
// conversation of their own, so that a head learns to read past one: up to
// three earlier user turns, drawn from a file of records like the input,
// each answered by the model, the last answer sometimes cut short as when
// the user speaks over it. The prompt is drawn per block, and --prompt-log
// adds the turns drawn (record numbers, or -) and how many pieces of the
// last answer were kept (-1: all).
//
// With --depth N, each conversation goes on with drawn turns until it holds
// N tokens, none cut short: to check the heads late in long conversations.

#include "llm.h"
#include "logging.h"

#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <random>
#include <string>
#include <vector>

// Earlier turns of the conversation: 0 to 3 of `history`, each answered,
// the last answer cut short in 3 of 10 conversations; or with `depth`, as
// many as fill that many tokens, none cut. `drawn` gets the turns and the
// pieces kept of the last answer, or -1.
static bool converse(Llm & llm, const std::vector<std::vector<float>> & history, int depth, std::mt19937 & rng,
                     std::string & drawn) {
    const int n_turns = depth > 0 ? INT_MAX : std::discrete_distribution<int>({ 20, 35, 25, 20 })(rng);
    const int cut     = depth == 0 && n_turns > 0 && rng() % 10 < 3 ? 1 + (int) (rng() % 30) : -1;
    std::string turns;
    int kept = -1;
    for (int t = 0; t < n_turns && (depth == 0 || llm.n_past() < depth); t++) {
        const size_t h = rng() % history.size();
        turns += (turns.empty() ? "" : ",") + std::to_string(h);
        const std::vector<float> & pcm = history[h];
        const size_t n = std::min(pcm.size(), llm.max_turn_samples());
        if (!llm.begin_user_turn() || !llm.push_audio(pcm.data(), n) || !llm.end_user_turn() || !llm.begin_reply()) {
            return false;
        }
        const bool last = t == n_turns - 1;
        int pieces = 0;
        std::string piece;
        bool going = true;
        while (going && !(last && cut >= 0 && pieces == cut)) {
            going = llm.next_piece(piece);
            pieces += going;
        }
        if (going) {
            kept = pieces;
            if (!llm.hold_reply() || !llm.end_reply()) {
                return false;
            }
        } else if (llm.reply_failed()) {
            return false;
        }
    }
    drawn = (turns.empty() ? "-" : turns) + "\t" + std::to_string(kept);
    return true;
}

int main(int argc, char ** argv) {
    Llm::Params params;
    params.gemma("E2B");
    params.n_ctx       = 2048;
    params.max_tokens  = 0;

    std::string in_path;
    std::string out_path;
    std::string prompts_path;
    std::string prompt_log_path;
    std::string history_path;
    int         block = 32;
    int         depth = 0;
    unsigned    seed = 1;
    bool bad = argc % 2 == 0;  // every option takes a value
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string arg = argv[i];
        if      (arg == "--gemma")  bad |= !params.gemma(argv[i + 1]);
        else if (arg == "--model")  params.model_path = argv[i + 1];
        else if (arg == "--mmproj") params.mmproj_path = argv[i + 1];
        else if (arg == "--in")     in_path = argv[i + 1];
        else if (arg == "--out")    out_path = argv[i + 1];
        else if (arg == "--prompts")    prompts_path = argv[i + 1];
        else if (arg == "--prompt-log") prompt_log_path = argv[i + 1];
        else if (arg == "--seed")       seed = (unsigned) std::strtoul(argv[i + 1], nullptr, 10);
        else if (arg == "--history")    history_path = argv[i + 1];
        else if (arg == "--block")      block = std::atoi(argv[i + 1]);
        else if (arg == "--depth")      depth = std::atoi(argv[i + 1]);
        else if (arg == "--threads")    params.n_threads = std::atoi(argv[i + 1]);
        else bad = true;
    }
    if (bad || in_path.empty() || out_path.empty() || block < 1 || depth < 0) {
        fprintf(stderr, "usage: %s --in clips.pcm|- --out features.f16 [--gemma E2B|E4B] [--model FILE] [--mmproj FILE]\n"
                        "       [--prompts FILE] [--history clips.pcm [--block N] [--depth N]]\n"
                        "       [--prompt-log FILE] [--seed N] [--threads N]\n",
                argv[0]);
        return 1;
    }

    std::vector<std::string> prompts;
    if (!prompts_path.empty()) {
        std::ifstream f(prompts_path);
        for (std::string line; std::getline(f, line);) {
            if (!line.empty()) {
                prompts.push_back(line == "-" ? std::string() : line);
            }
        }
        if (prompts.empty()) {
            fprintf(stderr, "no prompts in %s\n", prompts_path.c_str());
            return 1;
        }
    }
    std::vector<std::vector<float>> history;
    if (!history_path.empty()) {
        FILE * f = fopen(history_path.c_str(), "rb");
        uint32_t n = 0;
        while (f && fread(&n, sizeof(n), 1, f) == 1) {
            history.emplace_back(n);
            if (fread(history.back().data(), sizeof(float), n, f) != n) {
                history.clear();
                break;
            }
        }
        if (f) {
            fclose(f);
        }
        if (history.empty()) {
            fprintf(stderr, "no turns in %s\n", history_path.c_str());
            return 1;
        }
        // Answers about as long as most first replies; a longer one is cut
        // as if the user spoke over it.
        params.max_tokens = 96;
        params.n_ctx      = 4096;
        params.seed       = seed;
    }
    std::mt19937 rng(seed);
    if (depth > 0) {
        // The conversation, its last turn's answer and the clip.
        params.n_ctx = std::max(params.n_ctx, depth + 4096);
    }
    FILE * prompt_log = prompt_log_path.empty() ? nullptr : fopen(prompt_log_path.c_str(), "w");
    if (!prompt_log_path.empty() && !prompt_log) {
        fprintf(stderr, "cannot write %s\n", prompt_log_path.c_str());
        return 1;
    }

    init_logging();

    Llm llm;
    if (!llm.load(params)) {
        return 1;
    }
    int prefix = llm.n_past();
    int prompt = -1;
    std::string drawn;  // the block's turns and cut, for the log
    const int32_t n_layer = llm.n_layer();
    const int32_t n_embd  = llm.n_embd();
    const size_t  max_samples = llm.max_turn_samples();

    FILE * in  = in_path == "-" ? stdin : fopen(in_path.c_str(), "rb");
    FILE * out = fopen(out_path.c_str(), "wb");
    if (!in || !out) {
        fprintf(stderr, "failed to open %s or %s\n", in_path.c_str(), out_path.c_str());
        return 1;
    }
    fwrite(&n_layer, sizeof(n_layer), 1, out);
    fwrite(&n_embd, sizeof(n_embd), 1, out);

    std::vector<float>       pcm;
    std::vector<float>       states;
    std::vector<ggml_fp16_t> half((size_t) n_layer * n_embd);

    const auto t0 = std::chrono::steady_clock::now();
    double audio_s = 0;
    double t_begin = 0, t_push = 0, t_commit = 0;
    int n = 0;
    uint32_t n_samples = 0;
    while (fread(&n_samples, sizeof(n_samples), 1, in) == 1) {
        pcm.resize(n_samples);
        if (fread(pcm.data(), sizeof(float), n_samples, in) != n_samples) {
            fprintf(stderr, "truncated input at record %d\n", n);
            return 1;
        }
        // A turn holds < 30 s; keep the end, where the turn decision is.
        const size_t skip = pcm.size() > max_samples ? pcm.size() - max_samples : 0;

        const auto t_a = std::chrono::steady_clock::now();
        bool ok = true;
        if (history.empty()) {
            if (!prompts.empty()) {
                const int p = (int) (rng() % prompts.size());
                if (p != prompt) {
                    ok = llm.reset(prompts[p]);
                    prefix = llm.n_past();
                    prompt = p;
                }
                if (prompt_log) {
                    fprintf(prompt_log, "%d\n", p);
                }
            }
        } else {
            if (n % block == 0) {
                if (!prompts.empty()) {
                    prompt = (int) (rng() % prompts.size());
                    ok = llm.reset(prompts[prompt]);
                } else {
                    ok = llm.reset();
                }
                ok = ok && converse(llm, history, depth, rng, drawn);
                prefix = llm.n_past();
            }
            if (prompt_log) {
                fprintf(prompt_log, "%d\t%s\n", prompt, drawn.c_str());
            }
        }
        ok = ok && llm.truncate(prefix) && llm.begin_user_turn();
        const auto t_b = std::chrono::steady_clock::now();
        ok = ok && llm.push_audio(pcm.data() + skip, pcm.size() - skip);
        const auto t_c = std::chrono::steady_clock::now();
        ok = ok && llm.commit_audio(&states);
        const auto t_d = std::chrono::steady_clock::now();
        if (!ok) {
            fprintf(stderr, "failed at record %d\n", n);
            return 1;
        }
        t_begin  += std::chrono::duration<double>(t_b - t_a).count();
        t_push   += std::chrono::duration<double>(t_c - t_b).count();
        t_commit += std::chrono::duration<double>(t_d - t_c).count();
        ggml_fp32_to_fp16_row(states.data(), half.data(), (int64_t) half.size());
        fwrite(half.data(), sizeof(ggml_fp16_t), half.size(), out);

        n++;
        audio_s += (double) (pcm.size() - skip) / llm.audio_sample_rate();
        if (n % 500 == 0) {
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            fprintf(stderr, "%d clips, %.1f clips/s, %.0fx realtime\n", n, n / s, audio_s / s);
        }
    }

    fclose(in);
    fclose(out);
    if (prompt_log) {
        fclose(prompt_log);
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr, "done: %d clips in %.0f s (begin %.1f s, audio %.1f s, commit %.1f s)\n", n, s, t_begin, t_push,
            t_commit);
    return 0;
}
