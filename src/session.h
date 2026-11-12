#pragma once

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "vad.h"

class Llm;
class Tts;
class TurnHead;
struct ToolCall;

// One live conversation: detects when the user speaks, streams their speech
// into the model, speaks the reply and stops it when the user barges in.
//
// With a turn head, a short pause commits the audio and the head decides
// whether the user is done; if not, the turn continues when they go on.
// Without one, a longer fixed silence ends the turn, as the VAD hears it if
// there is one. Either way, if the user goes on right after their turn was
// closed, the reply is taken back and the turn reopened.
//
// With an addressee head as well, speech over the assistant pauses its reply
// and is only answered if it is meant for the assistant: at the first commit,
// or after 1.5 s of speech without one, the head decides whether the reply is
// dropped and the turn goes on, or the reply plays on. The first words are
// often a "by the way" that could go either way, so if the speech goes on,
// the head decides once more on all of it; otherwise the turn is forgotten
// and the reply's text goes on too. Speech while the assistant is silent
// gets the same decision at its first commit, so that side talk right after
// a reply is not answered either; only the user going on with a turn that
// was just closed is taken for theirs without asking.
//
// With a VAD, only speech opens or resumes a turn; on its own, the energy
// detector also takes chimes, clicks and music for speech. Speech well below
// the user's level, e.g. a TV or someone across the room, opens none: once
// the user has spoken, it neither pauses a reply nor gets one.
//
// The reply is generated a token at a time on the same thread, between
// microphone frames, so that speech over it is noticed at once however long
// it is. Speech over a reply holds it; the reply goes on if the speech was
// not meant for the assistant.
//
// A reply that calls a tool waits for the answer, which the app gives from
// any thread (Callbacks::on_tool_call); speech goes on being heard meanwhile.
// Without an answer in time, the model is told the tool failed. What the
// reply said before the call, e.g. "Let me check.", is spoken at once: speech
// synthesis waits for more text otherwise. The rest is synthesized anew.
//
// Microphone audio goes in through push_audio() from any thread; all model
// work happens on the session's worker thread.
class Session {
public:
    using Clock = std::chrono::steady_clock;

    enum class State { Listening, UserSpeaking, UserPaused };

    // A piece of work, for an app that shows where the time goes: the
    // stage that ran (Callbacks::on_span), when, and what it was about.
    struct Span {
        const char *      stage;
        Clock::time_point start;
        Clock::time_point end;
        float             value = NAN;  // seconds of audio, or a head's probability
        std::string       text;         // a reply's piece
    };

    // A 20 ms frame of the microphone as the session took it.
    struct Frame {
        Clock::time_point heard;    // when its audio came in, about
        Clock::time_point handled;  // when the session was done with it
        float             db;       // its level
        float             noise_db; // the noise floor the energy detector follows
        bool              speech;   // the energy detector's verdict
        bool              voice;    // ... and the VAD's, if there is one
        float             vad;      // the VAD's speech probability, NAN without one
        State             state;    // after the frame
    };

    struct Callbacks {
        // Assistant speech at Tts::sample_rate(), from the TTS worker thread.
        std::function<void(const float * pcm, int n_samples)> on_audio;
        // Drop assistant speech that was not played yet.
        std::function<void()> on_interrupt;
        // Hold and continue playback of assistant speech.
        std::function<void()> on_pause;
        std::function<void()> on_resume;
        // Seconds of assistant speech still queued for playback.
        std::function<double()> queued;
        // JSON event for the client.
        std::function<void(const std::string & json)> on_event;
        // A tool call of the reply (Llm::Params::tools), on the worker
        // thread; return at once. The answer goes to `done`, from any thread;
        // an answer the reply no longer waits for is dropped.
        std::function<void(const ToolCall & call, std::function<void(std::string)> done)> on_tool_call;
        // Each piece of work, once done, from the thread that did it. On the
        // worker thread: "vad", "hear" (the turn's audio into Gemma, with its
        // seconds), "begin turn", "commit", "turn head" and "addressee head"
        // (with the probability), "continue", "reopen" (with the seconds
        // heard again), "abandon", "end turn", "begin reply", "token" (with
        // its piece), "tool answer", "hold", "resume", "end reply", "begin
        // speech", "stop speech". On the speech synthesis thread:
        // "synthesize" (from the text or the audio before, with the seconds
        // of speech) and "lead" (waiting for playback to catch up).
        std::function<void(const Span & span)> on_span;
        // Each frame of the microphone, on the worker thread.
        std::function<void(const Frame & frame)> on_frame;
    };

    Session(Llm & llm, Tts & tts, const TurnHead * turn_head, const TurnHead * addressee_head,
            const Vad * vad, Callbacks callbacks);
    ~Session();
    Session(const Session &) = delete;
    Session & operator=(const Session &) = delete;

    // Mono f32 at Llm::audio_sample_rate().
    void push_audio(const float * pcm, size_t n_samples);

private:
    void run();
    void process_frame(const std::vector<float> & frame);
    bool is_speech(const std::vector<float> & frame);
    bool is_distant() const;
    void learn_level(const std::vector<float> & db, float rate);
    bool start_user_turn();
    void commit_pause();
    bool check_addressee();
    void ignore_turn();
    void resume_user_turn();
    void reopen_user_turn();
    void end_user_turn();
    void respond();
    bool start_speech();
    void step_reply();
    void call_tool();
    void answer_tool(const std::string & result);
    void interrupt();
    void emit(const std::string & type, const std::string & text = {});
    // A span of `stage` from `start` to now.
    void span(const char * stage, Clock::time_point start, float value = NAN, std::string text = {}) const;
    void dump_turn(float p);

    Llm &            llm_;
    Tts &            tts_;
    const TurnHead * turn_head_;
    const TurnHead * addressee_head_;
    const Vad *      vad_;
    Callbacks        cb_;

    const int frame_samples_;

    std::mutex              mtx_;
    std::condition_variable cv_;
    std::vector<float>      input_;
    size_t                  largest_push_ = 0;  // samples: how much a client sends at once
    Clock::time_point       last_push_;
    std::atomic<bool>       stop_{false};
    std::atomic<bool>       synth_stop_{false};  // releases speech synthesis waiting for playback
    std::thread             worker_;

    // Tool answers, which may come after the session is gone.
    struct Answers {
        std::mutex                                mtx;
        Session *                                 session = nullptr;
    };
    std::shared_ptr<Answers>                      answers_;
    std::vector<std::pair<int, std::string>>      tool_results_;  // guarded by mtx_
    int                                           tool_id_ = 0;   // of the call the reply waits for
    std::chrono::steady_clock::time_point         tool_start_;

    State                          state_ = State::Listening;
    std::deque<std::vector<float>> preroll_;
    int                            speech_run_  = 0;
    int                            silence_run_ = 0;
    int                            turn_frames_ = 0;
    int                            pause_frames_ = 0;
    int                            paused_pushed_ = 0;      // frames pushed since the pause began
    bool                           addressed_ = true;       // the open turn is meant for the assistant
    bool                           overlap_   = false;      // the open turn paused the assistant's reply
    bool                           doubted_   = false;      // ... and might not be meant for us: the next check decides
    int                            checked_frames_ = 0;     // turn_frames_ at the last addressee check
    bool                           ignoring_  = false;      // speech not meant for us goes on
    int                            ignore_silence_ = 0;
    int                            frames_since_end_ = -1;  // since the last turn closed, -1 = long ago
    std::vector<float>             since_end_;              // audio since then, for reopening
    std::vector<float>             layer_states_;
    std::vector<int>               head_layers_;  // the layers the heads read
    bool                           replying_         = false;  // the reply's text is still being generated
    bool                           spoken_           = false;  // ... and some of it went to speech synthesis
    bool                           speech_ended_     = false;  // ... which was ended for a tool call
    std::atomic<Clock::rep>        speech_text_{0};            // when it got its first text, 0 = not yet

    // GEMMALIVE_DUMP=<dir> records each turn as the turn head saw it and
    // the head's verdict, for replaying live turns offline.
    std::string dump_dir_;
    int         n_turns_  = 0;
    int         n_checks_ = 0;
    float                          noise_db_    = -60.f;
    float                          frame_db_    = -90.f;  // of the latest frame
    bool                           frame_speech_ = false;
    bool                           frame_voice_  = false;
    std::deque<float>              recent_db_;            // of the current utterance's speech frames
    int                            quiet_frames_ = 0;
    std::vector<float>             turn_db_;              // of the open turn's speech frames
    float                          user_db_     = NAN;    // the user's speech level, NAN = not heard yet
    bool                           distant_run_ = false;  // distant speech goes on
    Vad::State                     vad_state_;
    int                            vad_quiet_   = 0;  // frames since the energy detector heard sound
};
