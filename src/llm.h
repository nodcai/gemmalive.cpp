#pragma once

#include "audio_encoder.h"
#include "backend.h"
#include "gemma4.h"
#include "sampler.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

// A function the model can call (Gemma 4 tool calling).
struct Tool {
    struct Param {
        std::string name;
        std::string type;  // STRING, NUMBER, INTEGER or BOOLEAN
        std::string description;
        bool        required = false;
    };
    std::string        name;
    std::string        description;
    std::vector<Param> params;
};

// A call the model made: the tool's name and its arguments as a JSON object.
struct ToolCall {
    std::string name;
    std::string args;
};

// Gemma 4 conversation with streamed audio input and streamed text output.
//
// A user turn is opened with begin_user_turn() and fed with push_audio()
// while the user speaks, and while they pause. The turn is one audio
// segment, pauses included: commit_audio() closes it where the audio is
// now, e.g. when the user pauses, and continue_user_turn() takes the close
// back and goes on when they do. end_user_turn() closes the turn and the
// model's reply follows a token at a time (next_piece()), or all at once
// (generate()). The KV cache keeps the whole conversation, so turns build on
// each other.
//
// The heads read their features in the conversation, at the end of the
// current turn's audio: they are trained on clips after conversations of
// their own.
//
// With tools, the reply may stop for a tool call (tool_call()); the tool's
// answer, given to tool_response(), goes into the reply and it goes on.
class Llm {
public:
    struct Params {
        std::string model_path;
        // The model's audio encoder. Not looked at when the model's file
        // has it (a combined file, training/turn/combine.py).
        std::string mmproj_path;
        // The file of a Gemma 4 variant, "E2B" or "E4B", as
        // scripts/download_models.cmake puts it in models/: the model with
        // its audio encoder and heads. False for another name.
        bool gemma(const std::string & variant);
        std::string system_prompt =
            "You are a friendly, knowledgeable voice assistant. Your replies are spoken aloud. "
            "Match the length to the question: a simple fact gets a sentence or two, a how-to "
            "or an explanation gets a short paragraph with the key points and the reason behind "
            "them, and only a request for detail gets more. Keep greetings and small talk brief. "
            "Speak in plain conversational sentences, without markdown, lists, emojis or special "
            "characters.";
        int         n_ctx       = 8192;
        int         n_threads   = 0;     // 0 = auto
        int         max_tokens  = 1024;  // per reply
        float       temperature = 1.0f;
        int         top_k       = 64;
        float       top_p       = 0.95f;
        unsigned    seed        = 0;     // 0 = random

        // Declared in the system turn.
        std::vector<Tool> tools;
        // A tool call is made whenever the model gives it at least this
        // probability: sampled, Gemma often talks about looking something up
        // instead, or asks what it was told. Spoken, at the start of a reply,
        // E4B gave calls of lookup tools 0.0001 to 0.6 where one was due
        // (most above 0.002 with a reminder that its knowledge is dated in
        // the turn's note), 0.00004 at most where none was.
        // 0 = sampled like any token.
        float             call_threshold = 0.002f;
        // Said, in turn, before a tool call that would open the reply, so
        // that the user hears something while the tool runs.
        std::vector<std::string> call_fillers = { "Let me check.", "One moment." };
        // Goes with each tool's answer. Without it, Gemma takes an answer
        // that does not help, e.g. an article for a quiz question, for one
        // that failed and says it found nothing; told in the system prompt,
        // it takes itself for a lookup service and refuses small talk.
        std::string       response_note = "Answer the question now. Use this where it helps: it is newer than "
                                          "what you know. Where it does not help, answer from what you know, "
                                          "without mentioning the lookup.";
        // Added to the system prompt at the start of each conversation, e.g.
        // the date and where the user is.
        std::function<std::string()> context;
        // Put after each user turn's speech, where the heads do not see it,
        // e.g. the time.
        std::function<std::string()> turn_context;
    };

    // Called per text piece; return false to stop generating.
    using PieceCallback = std::function<bool(std::string_view)>;
    // Answers a tool call.
    using ToolCallback = std::function<std::string(const ToolCall &)>;

    Llm() = default;
    ~Llm();
    Llm(const Llm &) = delete;
    Llm & operator=(const Llm &) = delete;

    bool load(const Params & params);

    // Forget the conversation, keep the system prompt.
    bool reset();
    // Forget the conversation and start over with another system prompt.
    bool reset(const std::string & system_prompt);

    // The most audio a user turn holds: the turn is one audio segment, which
    // holds < 30 s.
    size_t max_turn_samples() const;

    // PCM rate expected by push_audio(), mono f32.
    int audio_sample_rate() const;

    // Decodes a wav/mp3/flac file to mono f32 at audio_sample_rate().
    bool read_audio_file(const std::string & path, std::vector<float> & out) const;

    // A typed user turn instead of speech, e.g. to compare benchmark scores.
    bool user_text(std::string_view text);

    // A spoken user turn encoded all at once rather than as it is heard.
    bool user_audio_oneshot(const float * pcm, size_t n_samples);

    bool begin_user_turn();

    // Audio of the open turn. While the segment is closed it is held until
    // continue_user_turn(). A turn holds < 30 s of audio.
    bool push_audio(const float * pcm, size_t n_samples);

    // Closes the segment where the audio is now. With `layer_states`, also
    // returns the output of the layers at the end of the turn so far,
    // n_layer() rows of n_embd() floats: of every layer, or only of `layers`
    // (the others are left zero). Each layer read makes the GPU wait.
    bool commit_audio(std::vector<float> * layer_states = nullptr, const std::vector<int> * layers = nullptr);

    // Takes back the close and goes on with the audio held since.
    bool continue_user_turn();
    bool end_user_turn();

    // Takes back the reply to the last user turn and reopens that turn, for
    // when the user was not done after all. `since_end` is the audio since
    // the turn ended. Only possible until the next user turn starts, and
    // while the turn still holds the audio; if it fails halfway, the turn is
    // abandoned.
    bool reopen_user_turn(const float * since_end, size_t n_samples);
    bool can_reopen_user_turn() const { return reopen_pos_ >= 0; }

    // Removes the open user turn as if it never happened, e.g. when the
    // speech was not meant for the assistant.
    bool abandon_user_turn();

    // All audio of the current (or last) turn, pauses included.
    const std::vector<float> & turn_audio() const { return turn_pcm_; }

    // The reply to the last user turn, a token at a time, so that other work
    // can go on in between: begin_reply() after end_user_turn(), then
    // next_piece() gives the next piece of text (which may be empty) until it
    // returns false, once the reply is complete and the model turn closed.
    //
    // A reply can be held, e.g. for speech over it: hold_reply() closes the
    // model turn where it is, so that a user turn can start. If that turn is
    // abandoned, resume_reply() takes the close back and the reply goes on;
    // end_reply() leaves it closed where it is, as it does for a reply that
    // is being generated.
    //
    // When next_piece() returns false because the model called a tool,
    // tool_call() gives the call and the reply waits for tool_response() to
    // go on. A call that is cut short, by holding or ending the reply, is
    // taken back: the conversation keeps no call without its answer.
    bool begin_reply();
    bool next_piece(std::string & piece);
    bool hold_reply();
    bool resume_reply();
    bool end_reply();
    bool reply_held() const { return reply_ == Reply::Held; }
    bool awaiting_tool() const { return reply_ == Reply::Tool; }
    bool tool_call(ToolCall & call) const;
    bool tool_response(std::string_view result);
    // Whether the last reply ended because decoding failed.
    bool reply_failed() const { return reply_failed_; }

    // The whole reply at once; tool calls go to `on_tool`, or are answered
    // with an error without one.
    bool generate(const PieceCallback & on_piece, const ToolCallback & on_tool = nullptr);

    int n_past() const { return n_past_; }
    int n_layer() const;
    int n_embd() const;

    // Drops everything after the first `n_past` positions, e.g. to go back
    // to the system prompt without decoding it again.
    bool truncate(int n_past);

private:
    enum class Reply { None, Generating, Held, Tool };

    void close_stream();
    bool feed();
    bool decode_prefix();
    void drop_call();
    int32_t special_token(const char * text) const;
    float probability(int32_t token, float floor) const;
    std::vector<int32_t> tokenize(std::string_view text, bool add_bos, bool parse_special = true) const;
    bool decode_text(std::string_view text, bool add_bos = false, bool parse_special = true);
    void guard_chat(int pos);
    bool decode_token(int32_t token);
    bool decode_embd(const float * embd, int n_tokens);
    bool decode(const int32_t * tokens, const float * embd, int n);

    Params params_;

    Backend      backend_;
    Gemma4       model_;
    AudioEncoder encoder_;
    Sampler      sampler_;

    int  n_past_  = 0;
    bool in_turn_ = false;

    std::unique_ptr<AudioStream> stream_;  // the turn's audio
    size_t fed_             = 0;      // samples of the turn pushed into stream_
    bool   closed_          = false;  // the segment's closing tail is decoded
    int    chat_tail_pos_   = 0;      // where that tail starts

    int  reopen_pos_    = -1;  // >= 0 while the last turn can be reopened
    int  turn_pos_      = 0;   // chat position where the open turn starts

    std::vector<float> turn_pcm_;

    Reply   reply_          = Reply::None;
    int     reply_tokens_   = 0;
    bool    reply_at_start_ = true;   // no text yet: leading newlines are dropped
    bool    reply_failed_   = false;
    int32_t last_token_     = 0;      // the last one in the conversation, of the reply or its opening
    int     held_pos_       = 0;      // where the close of a held reply starts
    bool    reply_spoke_    = false;  // the reply has text: what follows a tool's answer is set apart
    bool    call_next_      = false;  // a filler was said: the call follows
    size_t  fillers_said_   = 0;

    // Tool calls: Gemma's call markup, and the call being made.
    int32_t     call_begin_  = -1;
    int32_t     call_end_    = -1;
    int32_t     response_    = -1;
    bool        in_call_     = false;
    int         call_pos_    = 0;  // where the call starts
    int32_t     before_call_ = 0;  // the token before it
    std::string call_text_;
    ToolCall    call_;
};
