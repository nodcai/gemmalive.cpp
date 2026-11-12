#include "session.h"

#include "llm.h"
#include "tts.h"
#include "turn_head.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

// Turn-taking, in 20 ms frames.
static constexpr int PREROLL_FRAMES       = 25;   // speech before the onset that is kept
static constexpr int ONSET_FRAMES         = 4;    // speech needed to open a turn
static constexpr int ONSET_FRAMES_PLAYING = 10;   // ... while the assistant is talking
static constexpr int END_SILENCE_FRAMES   = 35;   // silence that ends a turn (no turn head)
static constexpr int COMMIT_FRAMES        = 10;   // pause after which the turn head decides
static constexpr int MAX_PAUSE_FRAMES     = 40;   // pause that ends the turn regardless
static constexpr int REOPEN_FRAMES        = 75;   // speech this soon after a turn closed continues it
static constexpr int MAX_TURN_FRAMES      = 1400; // a turn holds < 30 s of audio, pauses included
static constexpr int ADDRESS_CHECK_FRAMES = 75;   // speech over the assistant checked by then at the latest
static constexpr int IGNORE_END_FRAMES    = 10;   // silence that ends speech we ignore
static constexpr int DOUBT_END_FRAMES     = 40;   // silence after which speech we doubt is ignored

// Assistant speech synthesized ahead of playback.
static constexpr double MAX_LEAD_SECONDS = 1.5;

// How long a reply waits for a tool's answer.
static constexpr std::chrono::seconds TOOL_TIMEOUT{8};

// VAD: speech probability that counts as speech, and how long it keeps
// running after the last sound. In quiet it rests, and keeps its state: from
// a fresh state it takes the onset of tones and chimes for speech.
static constexpr float VAD_THRESHOLD   = 0.7f;
static constexpr int   VAD_HOLD_FRAMES = 25;

// Distance: speech this far below the user's level is someone else's. The
// level is the 90th percentile of speech frames: the last ones for speech
// that starts, a turn's for the user, who is followed per answered turn and,
// slower, per ignored one, so that a user who moved away gets heard again.
static constexpr float DISTANT_DB         = 10.f;
static constexpr int   LEVEL_FRAMES       = 50;
static constexpr int   UTTERANCE_GAP      = 10;   // silence that ends an utterance's level
static constexpr float USER_LEVEL_RATE    = 0.3f;
static constexpr float IGNORED_LEVEL_RATE = 0.1f;

// Energy detector: speech is this far above the tracked noise floor, and
// above an absolute minimum.
static constexpr float SPEECH_OVER_NOISE_DB = 12.f;
static constexpr float SPEECH_MIN_DB        = -50.f;

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

Session::Session(Llm & llm, Tts & tts, const TurnHead * turn_head, const TurnHead * addressee_head,
                 const Vad * vad, Callbacks callbacks)
    : llm_(llm), tts_(tts), turn_head_(turn_head && turn_head->loaded() ? turn_head : nullptr),
      // Addressing is decided at the turn head's commits.
      addressee_head_(turn_head_ && addressee_head && addressee_head->loaded() ? addressee_head : nullptr),
      vad_(vad && vad->loaded() ? vad : nullptr),
      cb_(std::move(callbacks)), frame_samples_(llm.audio_sample_rate() / 50) {
    for (const TurnHead * head : { turn_head_, addressee_head_ }) {
        if (head) {
            head_layers_.insert(head_layers_.end(), head->layers().begin(), head->layers().end());
        }
    }
    if (const char * dir = getenv("GEMMALIVE_DUMP")) {
        dump_dir_ = dir;
    }
    answers_ = std::make_shared<Answers>();
    answers_->session = this;
    worker_ = std::thread([this] { run(); });
}

Session::~Session() {
    {
        std::lock_guard<std::mutex> lock(answers_->mtx);
        answers_->session = nullptr;
    }
    stop_ = true;
    cv_.notify_all();
    worker_.join();
}

void Session::push_audio(const float * pcm, size_t n_samples) {
    {
        std::lock_guard<std::mutex> lock(mtx_);
        input_.insert(input_.end(), pcm, pcm + n_samples);
        largest_push_ = std::max(largest_push_, n_samples);
        last_push_    = Clock::now();
    }
    cv_.notify_one();
}

void Session::run() {
    emit("listening");

    using namespace std::chrono;

    std::vector<float> pending;
    std::vector<float> frame(frame_samples_);
    std::vector<std::pair<int, std::string>> results;
    auto last_input = steady_clock::now();
    while (!stop_) {
        bool got_input;
        size_t largest_push;
        Clock::time_point arrived;  // when the end of `pending` came in
        {
            // While the reply grows, only take what has arrived.
            std::unique_lock<std::mutex> lock(mtx_);
            const bool growing = replying_ && !llm_.reply_held() && !llm_.awaiting_tool() && !speech_ended_;
            cv_.wait_for(lock, milliseconds(growing ? 0 : 50), [this] {
                return stop_ || (int) input_.size() >= frame_samples_ || !tool_results_.empty();
            });
            got_input = !input_.empty();
            pending.insert(pending.end(), input_.begin(), input_.end());
            input_.clear();
            largest_push = largest_push_;
            arrived      = last_push_;
            results.swap(tool_results_);
        }

        // Clients stop sending while their source is silent; count the gap
        // as silence so that an open turn ends and time keeps running for
        // the reopen window. Some send in large chunks (an iPhone's
        // microphone tap every 100 ms): only a gap well beyond those counts.
        const auto now = steady_clock::now();
        const auto gap = milliseconds(std::max<int64_t>(100, (int64_t) (2 * largest_push * 20 / frame_samples_)));
        if (got_input) {
            last_input = now;
        } else if (now - last_input >= gap) {
            const int n_frames = (int) (duration_cast<milliseconds>(now - last_input).count() / 20);
            pending.insert(pending.end(), (size_t) n_frames * frame_samples_, 0.f);
            last_input += milliseconds(20 * n_frames);
            arrived = now;
        }

        size_t off = 0;
        while (!stop_ && pending.size() - off >= (size_t) frame_samples_) {
            frame.assign(pending.begin() + off, pending.begin() + off + frame_samples_);
            process_frame(frame);
            off += frame_samples_;
            if (cb_.on_frame) {
                // The audio after the frame came in with it, or after it.
                const auto after = microseconds((int64_t) (pending.size() - off) * 20000 / frame_samples_);
                cb_.on_frame({ arrived - after, Clock::now(), frame_db_, noise_db_, frame_speech_, frame_voice_,
                               vad_ ? vad_state_.prob : NAN, state_ });
            }
        }
        pending.erase(pending.begin(), pending.begin() + off);

        // Between frames, the tool's answer the reply waits for, or its next
        // piece.
        if (!stop_ && replying_ && llm_.awaiting_tool()) {
            for (const auto & [id, text] : results) {
                if (id == tool_id_ && llm_.awaiting_tool()) {
                    answer_tool(text);
                }
            }
            if (llm_.awaiting_tool() && steady_clock::now() - tool_start_ >= TOOL_TIMEOUT) {
                answer_tool("error: no answer in time");
            }
        } else if (!stop_ && replying_ && !llm_.reply_held()) {
            // After a tool call, the rest of the reply is new speech, once
            // what came before it is synthesized.
            if (speech_ended_ && !tts_.active()) {
                speech_ended_ = false;
                if (!start_speech()) {
                    emit("error", "failed to start speech synthesis");
                    interrupt();
                }
            }
            if (replying_ && !speech_ended_) {
                step_reply();
            }
        }
        results.clear();
    }

    synth_stop_ = true;
    tts_.abort();
    if (replying_) {
        llm_.end_reply();
        replying_ = false;
    }
}

void Session::process_frame(const std::vector<float> & frame) {
    const bool speech = is_speech(frame);
    // What opens a turn while the assistant is silent, and resumes one;
    // within a turn the energy detector alone finds the pauses. Speech over
    // the assistant takes the energy detector too, so that it stops quickly;
    // the addressee head then decides about it.
    bool voice = speech;
    if (vad_) {
        vad_quiet_ = speech ? 0 : vad_quiet_ + 1;
        if (vad_quiet_ < VAD_HOLD_FRAMES) {
            const auto t = Clock::now();
            vad_->push(vad_state_, frame.data(), frame.size());
            span("vad", t, vad_state_.prob);
        }
        voice = speech && vad_state_.prob >= VAD_THRESHOLD;
    }
    frame_speech_ = speech;
    frame_voice_  = voice;
    quiet_frames_ = speech ? 0 : quiet_frames_ + 1;
    if (quiet_frames_ == UTTERANCE_GAP && !distant_run_) {
        recent_db_.clear();
    }
    if (speech) {
        recent_db_.push_back(frame_db_);
        if ((int) recent_db_.size() > LEVEL_FRAMES) {
            recent_db_.pop_front();
        }
        if (state_ != State::Listening) {
            turn_db_.push_back(frame_db_);
        }
    }

    if (state_ == State::Listening) {
        preroll_.push_back(frame);
        if ((int) preroll_.size() > PREROLL_FRAMES) {
            preroll_.pop_front();
        }

        if (ignoring_) {
            // Speech that was not meant for us, until it stops.
            ignore_silence_ = speech ? 0 : ignore_silence_ + 1;
            if (ignore_silence_ < IGNORE_END_FRAMES) {
                return;
            }
            ignoring_ = false;
        }

        const bool playing = replying_ || tts_.active() || (cb_.queued && cb_.queued() > 0);
        speech_run_ = (playing ? speech : voice) ? speech_run_ + 1 : 0;
        if (frames_since_end_ >= 0) {
            since_end_.insert(since_end_.end(), frame.begin(), frame.end());
            if (++frames_since_end_ > REOPEN_FRAMES + ONSET_FRAMES_PLAYING) {
                frames_since_end_ = -1;
                since_end_.clear();
            }
        }

        if (distant_run_ && quiet_frames_ >= UTTERANCE_GAP) {
            distant_run_ = false;
            learn_level(std::vector<float>(recent_db_.begin(), recent_db_.end()), IGNORED_LEVEL_RATE);
            recent_db_.clear();
            emit("distant");
        }
        if (speech_run_ >= (playing ? ONSET_FRAMES_PLAYING : ONSET_FRAMES) && is_distant()) {
            distant_run_ = true;
        } else if (speech_run_ >= (playing ? ONSET_FRAMES_PLAYING : ONSET_FRAMES)) {
            // Where the speech started, counted from the end of the last turn.
            const int onset = frames_since_end_ - speech_run_;
            const bool reopen = frames_since_end_ >= 0 && onset <= REOPEN_FRAMES && llm_.can_reopen_user_turn();
            if (reopen) {
                if (playing) {
                    interrupt();
                }
                reopen_user_turn();
            } else if (playing && addressee_head_) {
                // Hold the reply until we know whether this is meant for us.
                if (cb_.on_pause) {
                    cb_.on_pause();
                }
                emit("pause");
                const auto t = Clock::now();
                if (replying_ && !llm_.hold_reply()) {
                    emit("error", "failed to hold the reply");
                    interrupt();
                } else if (replying_) {
                    span("hold", t);
                }
                if (start_user_turn()) {
                    overlap_   = true;
                    addressed_ = false;
                } else {
                    interrupt();
                }
            } else {
                if (playing) {
                    interrupt();
                }
                // Speech while the assistant is silent is answered once the
                // addressee head takes it for ours, at its first pause: side
                // talk right after a reply is no turn either.
                if (start_user_turn() && !playing && addressee_head_) {
                    addressed_ = false;
                }
            }
        }
        return;
    }

    if (state_ == State::UserPaused) {
        preroll_.push_back(frame);
        if ((int) preroll_.size() > PREROLL_FRAMES) {
            preroll_.pop_front();
        }
        // The pause is part of the turn as the turn head sees it.
        const auto t = Clock::now();
        llm_.push_audio(frame.data(), frame.size());
        span("hear", t, 0.02f);
        paused_pushed_++;
        speech_run_ = voice ? speech_run_ + 1 : 0;
        pause_frames_++;
        if (speech_run_ >= ONSET_FRAMES) {
            resume_user_turn();
        } else if (pause_frames_ >= MAX_PAUSE_FRAMES) {
            end_user_turn();
        }
        return;
    }

    // UserSpeaking
    const auto t = Clock::now();
    llm_.push_audio(frame.data(), frame.size());
    span("hear", t, 0.02f);
    turn_frames_++;
    // Without a turn head, the silence that ends the turn is the VAD's: the
    // energy detector also hears a room's noise.
    silence_run_ = (turn_head_ ? speech : voice) ? 0 : silence_run_ + 1;
    if (llm_.turn_audio().size() >= (size_t) MAX_TURN_FRAMES * frame_samples_) {
        if (addressed_ || check_addressee()) {
            end_user_turn();
            // The turn is full: what follows is a turn of its own.
            frames_since_end_ = -1;
            since_end_.clear();
        }
    } else if (turn_head_ && silence_run_ == COMMIT_FRAMES) {
        commit_pause();
    } else if (overlap_ && !addressed_ && turn_frames_ - checked_frames_ >= ADDRESS_CHECK_FRAMES) {
        // Long speech over the assistant, e.g. a TV: decide without a pause.
        if (check_addressee()) {
            const auto t = Clock::now();
            if (!llm_.continue_user_turn()) {
                end_user_turn();
            } else {
                span("continue", t);
            }
        }
    } else if (doubted_ && silence_run_ >= DOUBT_END_FRAMES) {
        ignore_turn();
    } else if (!turn_head_ && !doubted_ && silence_run_ >= END_SILENCE_FRAMES) {
        end_user_turn();
    }
}

// 90th percentile of the levels, NAN for too few.
static float speech_level(std::vector<float> db) {
    if (db.size() < ONSET_FRAMES) {
        return NAN;
    }
    std::nth_element(db.begin(), db.begin() + (db.size() * 9) / 10, db.end());
    return db[(db.size() * 9) / 10];
}

bool Session::is_distant() const {
    const float level = speech_level(std::vector<float>(recent_db_.begin(), recent_db_.end()));
    return !std::isnan(user_db_) && !std::isnan(level) && level < user_db_ - DISTANT_DB;
}

void Session::learn_level(const std::vector<float> & db, float rate) {
    const float level = speech_level(db);
    if (std::isnan(level)) {
        return;
    }
    user_db_ = std::isnan(user_db_) ? level : user_db_ + rate * (level - user_db_);
}

bool Session::is_speech(const std::vector<float> & frame) {
    double sum = 0.0;
    for (const float x : frame) {
        sum += (double) x * x;
    }
    const float db = 10.f * std::log10((float) (sum / frame.size()) + 1e-10f);
    frame_db_ = db;

    const bool speech = db > std::max(noise_db_ + SPEECH_OVER_NOISE_DB, SPEECH_MIN_DB);
    if (!speech) {
        // Follow the floor down immediately, up slowly.
        noise_db_ = db < noise_db_ ? db : 0.95f * noise_db_ + 0.05f * db;
        noise_db_ = std::clamp(noise_db_, -90.f, -30.f);
    }
    return speech;
}

bool Session::start_user_turn() {
    turn_db_.clear();
    auto t = Clock::now();
    if (!llm_.begin_user_turn()) {
        emit("error", "failed to start user turn");
        speech_run_ = 0;
        return false;
    }
    span("begin turn", t);
    frames_since_end_ = -1;
    since_end_.clear();
    addressed_      = true;
    overlap_        = false;
    doubted_        = false;
    checked_frames_ = 0;
    n_turns_++;
    n_checks_ = 0;
    t = Clock::now();
    for (const auto & f : preroll_) {
        llm_.push_audio(f.data(), f.size());
    }
    span("hear", t, 0.02f * preroll_.size());
    turn_frames_ = (int) preroll_.size();
    preroll_.clear();
    silence_run_ = 0;
    state_ = State::UserSpeaking;
    emit("user_start");
    return true;
}

void Session::commit_pause() {
    if (!addressed_) {
        if (!check_addressee()) {
            return;
        }
    } else {
        const auto t = Clock::now();
        if (!llm_.commit_audio(&layer_states_, &head_layers_)) {
            emit("error", "failed to commit audio");
            end_user_turn();
            return;
        }
        span("commit", t);
    }
    state_ = State::UserPaused;
    const auto t = Clock::now();
    const float p = turn_head_->predict(layer_states_);
    span("turn head", t, p);
    dump_turn(p);
    emit("turn_check", std::to_string(p));
    if (p >= turn_head_->threshold()) {
        end_user_turn();
        return;
    }
    pause_frames_  = silence_run_;
    paused_pushed_ = 0;
    speech_run_    = 0;
    preroll_.clear();
}

// Commits the audio and asks the addressee head about the turn so far. If it
// is for us, a held reply is dropped and the turn goes on; if not, the reply
// (if any) is played on, and the turn is forgotten unless this was the first
// check of the speech (doubted_), which gets another once there is more.
bool Session::check_addressee() {
    auto t = Clock::now();
    if (!llm_.commit_audio(&layer_states_, &head_layers_)) {
        emit("error", "failed to commit audio");
        end_user_turn();
        return false;
    }
    span("commit", t);
    t = Clock::now();
    const float p = addressee_head_->predict(layer_states_);
    span("addressee head", t, p);
    emit("address_check", std::to_string(p));
    checked_frames_ = turn_frames_;
    if (p < addressee_head_->threshold()) {
        t = Clock::now();
        if (doubted_) {
            ignore_turn();
        } else if (!llm_.continue_user_turn()) {
            emit("error", "failed to continue user turn");
            ignore_turn();
        } else {
            span("continue", t);
            doubted_ = true;
            if (cb_.on_resume) {
                cb_.on_resume();
            }
            emit("doubted");
        }
        return false;
    }
    addressed_ = true;
    if (overlap_) {
        overlap_ = false;
        interrupt();
    }
    return true;
}

void Session::ignore_turn() {
    learn_level(turn_db_, IGNORED_LEVEL_RATE);
    auto t = Clock::now();
    llm_.abandon_user_turn();
    span("abandon", t);
    t = Clock::now();
    if (replying_ && llm_.reply_held()) {
        if (llm_.resume_reply()) {
            span("resume", t);
        } else {
            emit("error", "failed to resume the reply");
            interrupt();
        }
    }
    state_          = State::Listening;
    speech_run_     = 0;
    ignoring_       = true;
    ignore_silence_ = 0;
    preroll_.clear();
    if (overlap_) {
        overlap_ = false;
        if (cb_.on_resume) {
            cb_.on_resume();
        }
    }
    emit("ignored");
}

void Session::resume_user_turn() {
    // The speech onset was pushed already, during the pause.
    const int lead_in = std::min((int) preroll_.size(), paused_pushed_);
    const auto t = Clock::now();
    if (!llm_.continue_user_turn()) {
        end_user_turn();
        return;
    }
    span("continue", t);
    turn_frames_ = lead_in;
    preroll_.clear();
    silence_run_ = 0;
    state_ = State::UserSpeaking;
    emit("user_resume");
}

void Session::reopen_user_turn() {
    frames_since_end_ = -1;
    addressed_ = true;  // the user goes on with their own turn
    overlap_   = false;
    const int lead_in = std::min((int) preroll_.size(), (int) (since_end_.size() / frame_samples_));
    const auto t = Clock::now();
    const bool ok = llm_.reopen_user_turn(since_end_.data(), since_end_.size());
    span("reopen", t, (float) since_end_.size() / llm_.audio_sample_rate());
    since_end_.clear();
    if (!ok) {
        start_user_turn();
        return;
    }
    turn_frames_ = lead_in;
    preroll_.clear();
    silence_run_ = 0;
    state_ = State::UserSpeaking;
    emit("turn_reopened");
}

void Session::end_user_turn() {
    if (state_ == State::UserSpeaking) {
        dump_turn(-1.f);  // closed without asking the turn head
    }
    state_ = State::Listening;
    speech_run_ = 0;
    if (replying_) {
        interrupt();
    }

    const auto t = Clock::now();
    if (!llm_.end_user_turn()) {
        emit("error", "failed to end user turn");
        llm_.abandon_user_turn();
        return;
    }
    span("end turn", t);
    emit("user_end");
    // Learned once: if the turn is reopened, only what follows counts again.
    learn_level(turn_db_, USER_LEVEL_RATE);
    turn_db_.clear();
    frames_since_end_ = 0;
    since_end_.clear();
    respond();
}

bool Session::start_speech() {
    synth_stop_  = false;
    speech_text_ = 0;
    const auto t = Clock::now();
    const bool ok = tts_.begin([this, last = Clock::time_point()](const float * pcm, int n_samples) mutable {
        span("synthesize", std::max(last, Clock::time_point(Clock::duration(speech_text_.load()))),
             (float) n_samples / tts_.sample_rate());
        cb_.on_audio(pcm, n_samples);
        // Stay just ahead of playback, so that speech which is cut off, or a
        // reply to a turn that goes on after all, is mostly never synthesized.
        const auto wait = Clock::now();
        bool waited = false;
        while (cb_.queued && cb_.queued() > MAX_LEAD_SECONDS && !synth_stop_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            waited = true;
        }
        if (waited) {
            span("lead", wait);
        }
        last = Clock::now();
    });
    span("begin speech", t);
    return ok;
}

void Session::respond() {
    spoken_       = false;
    speech_ended_ = false;
    if (!start_speech()) {
        emit("error", "failed to start speech synthesis");
        // Close the model turn all the same, so the next user turn follows a
        // complete one.
        if (llm_.begin_reply()) {
            llm_.end_reply();
        }
        return;
    }
    const auto t = Clock::now();
    if (!llm_.begin_reply()) {
        emit("error", "failed to start the reply");
        tts_.end();
        return;
    }
    span("begin reply", t);
    replying_ = true;
}

void Session::step_reply() {
    std::string piece;
    const auto t = Clock::now();
    const bool more = llm_.next_piece(piece);
    span("token", t, NAN, piece);
    if (more) {
        if (!piece.empty()) {
            if (speech_text_ == 0) {
                speech_text_ = Clock::now().time_since_epoch().count();
            }
            tts_.push_text(piece);
            spoken_ = true;
            emit("assistant", piece);
        }
        return;
    }
    if (llm_.awaiting_tool()) {
        if (spoken_) {
            tts_.end();
            spoken_       = false;
            speech_ended_ = true;
        }
        call_tool();
        return;
    }
    replying_ = false;
    tts_.end();
    if (llm_.reply_failed()) {
        emit("error", "failed to generate the reply");
    }
    emit("assistant_end");
}

void Session::call_tool() {
    ToolCall call;
    llm_.tool_call(call);
    // A new id per call: answers to calls taken back, e.g. by holding the
    // reply, are dropped.
    tool_id_++;
    tool_start_ = std::chrono::steady_clock::now();
    emit("tool_call", call.name + " " + call.args);
    if (!cb_.on_tool_call) {
        answer_tool("error: tools are not available");
        return;
    }
    cb_.on_tool_call(call, [answers = answers_, id = tool_id_](std::string result) {
        std::lock_guard<std::mutex> lock(answers->mtx);
        if (Session * s = answers->session) {
            {
                std::lock_guard<std::mutex> inner(s->mtx_);
                s->tool_results_.emplace_back(id, std::move(result));
            }
            s->cv_.notify_one();
        }
    });
}

void Session::answer_tool(const std::string & result) {
    emit("tool_result", result);
    const auto t = Clock::now();
    if (!llm_.tool_response(result)) {
        emit("error", "failed to answer the tool call");
        interrupt();
        return;
    }
    span("tool answer", t);
}

void Session::interrupt() {
    synth_stop_   = true;
    speech_ended_ = false;
    auto t = Clock::now();
    tts_.abort();
    span("stop speech", t);
    if (replying_) {
        t = Clock::now();
        llm_.end_reply();
        span("end reply", t);
        replying_ = false;
    }
    if (cb_.on_interrupt) {
        cb_.on_interrupt();
    }
    emit("interrupt");
}

void Session::dump_turn(float p) {
    if (dump_dir_.empty()) {
        return;
    }
    // The turn as the turn head saw it, in the training tools' record format.
    const std::vector<float> & pcm = llm_.turn_audio();
    if (FILE * f = fopen((dump_dir_ + "/turns.pcm").c_str(), "ab")) {
        const uint32_t n = (uint32_t) pcm.size();
        fwrite(&n, sizeof(n), 1, f);
        fwrite(pcm.data(), sizeof(float), n, f);
        fclose(f);
    }
    if (FILE * f = fopen((dump_dir_ + "/turns.tsv").c_str(), "a")) {
        fprintf(f, "%d\t%d\t%.6f\t%zu\n", n_turns_, n_checks_, p, pcm.size());
        fclose(f);
    }
    n_checks_++;
}

void Session::span(const char * stage, Clock::time_point start, float value, std::string text) const {
    if (cb_.on_span) {
        cb_.on_span({ stage, start, Clock::now(), value, std::move(text) });
    }
}

void Session::emit(const std::string & type, const std::string & text) {
    if (!cb_.on_event) {
        return;
    }
    std::string json = "{\"type\":\"" + type + "\"";
    if (!text.empty()) {
        json += ",\"text\":\"" + json_escape(text) + "\"";
    }
    json += "}";
    cb_.on_event(json);
}
