#include "llm.h"

#include "audio_io.h"
#include "internal.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <thread>
#include <vector>

using namespace internal;

// Gemma 4 turn markup, see the model's chat template.
static constexpr const char * TURN_SYSTEM = "<|turn>system\n";
static constexpr const char * TURN_USER   = "<|turn>user\n";
static constexpr const char * TURN_MODEL  = "<|turn>model\n";
static constexpr const char * TURN_END    = "<turn|>\n";
static constexpr const char * AUDIO_BEGIN = "<|audio>";
static constexpr const char * AUDIO_END   = "<audio|>";
static constexpr const char * TOOL_BEGIN  = "<|tool>";
static constexpr const char * TOOL_END    = "<tool|>";
static constexpr const char * CALL_BEGIN  = "<|tool_call>";
static constexpr const char * CALL_END    = "<tool_call|>";
static constexpr const char * RESPONSE_BEGIN = "<|tool_response>";
static constexpr const char * RESPONSE_END   = "<tool_response|>";
static constexpr const char * QUOTE       = "<|\"|>";

// A tool's answer goes into the context; this much of it at most.
static constexpr size_t MAX_RESPONSE_BYTES = 2048;

// The model's guard slots: rollback points further back than its
// sliding-window caches reach.
static constexpr int GUARD_PREFIX = 0;  // the end of the system turn
static constexpr int GUARD_CHAT   = 1;  // the start or end of the current turn

static constexpr int MAX_TURN_SECONDS = 30;

// A tool as the chat template declares it:
// declaration:name{description:<|"|>...<|"|>,parameters:{properties:{...},required:[...],type:<|"|>OBJECT<|"|>}}
std::string internal::declare(const Tool & tool) {
    const auto quote = [](const std::string & s) { return QUOTE + s + QUOTE; };
    std::vector<const Tool::Param *> params;
    for (const auto & p : tool.params) {
        params.push_back(&p);
    }
    std::sort(params.begin(), params.end(), [](auto * a, auto * b) { return a->name < b->name; });

    std::string out = "declaration:" + tool.name + "{description:" + quote(tool.description);
    if (!params.empty()) {
        out += ",parameters:{properties:{";
        std::string required;
        for (const auto * p : params) {
            if (out.back() != '{') {
                out += ",";
            }
            out += p->name + ":{description:" + quote(p->description) + ",type:" + quote(p->type) + "}";
            if (p->required) {
                required += (required.empty() ? "" : ",") + quote(p->name);
            }
        }
        out += "},";
        if (!required.empty()) {
            out += "required:[" + required + "],";
        }
        out += "type:" + quote("OBJECT") + "}";
    }
    return out + "}";
}

std::string internal::json_string(std::string_view s) {
    std::string out = "\"";
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
    return out + "\"";
}

// One value of Gemma's call syntax, as JSON: <|"|>text<|"|>, {key:value,...},
// [value,...], or a bare number, true, false or null.
static bool call_value(std::string_view & s, std::string & json) {
    const std::string_view quote = QUOTE;
    if (s.substr(0, quote.size()) == quote) {
        const size_t end = s.find(quote, quote.size());
        if (end == std::string_view::npos) {
            return false;
        }
        json += json_string(s.substr(quote.size(), end - quote.size()));
        s.remove_prefix(end + quote.size());
        return true;
    }
    if (!s.empty() && (s[0] == '{' || s[0] == '[')) {
        const bool object = s[0] == '{';
        const char close  = object ? '}' : ']';
        json += s[0];
        s.remove_prefix(1);
        while (!s.empty() && s[0] != close) {
            if (json.back() != '{' && json.back() != '[') {
                if (s[0] != ',') {
                    return false;
                }
                json += ',';
                s.remove_prefix(1);
            }
            if (object) {
                const size_t colon = s.find(':');
                if (colon == std::string_view::npos || colon == 0) {
                    return false;
                }
                json += json_string(s.substr(0, colon)) + ":";
                s.remove_prefix(colon + 1);
            }
            if (!call_value(s, json)) {
                return false;
            }
        }
        if (s.empty()) {
            return false;
        }
        json += close;
        s.remove_prefix(1);
        return true;
    }
    const size_t end = s.find_first_of(",}]");
    const std::string_view word = s.substr(0, end);
    if (word.empty()) {
        return false;
    }
    const bool number = word.find_first_not_of("+-.0123456789eE") == std::string_view::npos;
    json += number || word == "true" || word == "false" || word == "null" ? std::string(word) : json_string(word);
    s.remove_prefix(word.size());
    return true;
}

// call:name{key:value,...} -> the name and the arguments as a JSON object.
bool internal::parse_call(std::string_view s, ToolCall & call) {
    const std::string_view prefix = "call:";
    const size_t brace = s.find('{');
    if (s.substr(0, prefix.size()) != prefix || brace == std::string_view::npos || brace == prefix.size()) {
        return false;
    }
    call.name = std::string(s.substr(prefix.size(), brace - prefix.size()));
    call.args.clear();
    s.remove_prefix(brace);
    return call_value(s, call.args) && s.find_first_not_of(" \t\r\n") == std::string_view::npos;
}

// At most `max` bytes of `s`, not cutting a UTF-8 character.
std::string_view internal::cut(std::string_view s, size_t max) {
    if (s.size() <= max) {
        return s;
    }
    size_t n = max;
    while (n > 0 && ((unsigned char) s[n] & 0xc0) == 0x80) {
        n--;
    }
    return s.substr(0, n);
}

bool Llm::Params::gemma(const std::string & variant) {
    if (variant != "E2B" && variant != "E4B") {
        return false;
    }
    model_path = "models/gemma-4-" + variant + "-live-XS.gguf";
    mmproj_path.clear();
    return true;
}

Llm::~Llm() {
    close_stream();
}

bool Llm::load(const Params & params) {
    params_ = params;
    const bool ok = backend_.init(params.n_threads) && model_.load(params.model_path, backend_, params.n_ctx);
    if (ok && model_.has_audio_encoder()) {
        params_.mmproj_path = params.model_path;
    } else if (ok && params.mmproj_path.empty()) {
        fprintf(stderr, "llm: %s has no audio encoder, and no mmproj is given\n", params.model_path.c_str());
        return false;
    }
    if (!ok || !encoder_.load(params_.mmproj_path, backend_)) {
        fprintf(stderr, "llm: failed to load %s with %s\n", params.model_path.c_str(), params_.mmproj_path.c_str());
        return false;
    }
    if (encoder_.n_embd() != model_.n_embd()) {
        fprintf(stderr, "llm: audio embeddings of %d for a model of %d\n", encoder_.n_embd(), model_.n_embd());
        return false;
    }
    sampler_.init(params.top_k, params.top_p, params.temperature, params.seed);

    call_begin_ = special_token(CALL_BEGIN);
    call_end_   = special_token(CALL_END);
    response_   = special_token(RESPONSE_BEGIN);
    if (!params.tools.empty() && (call_begin_ < 0 || call_end_ < 0 || response_ < 0)) {
        fprintf(stderr, "llm: the model has no tool calls\n");
        return false;
    }

    return decode_prefix();
}

bool Llm::reset() {
    close_stream();
    reply_      = Reply::None;
    in_call_    = false;
    call_next_  = false;
    in_turn_    = false;
    reopen_pos_ = -1;
    sampler_.reset();
    n_past_ = 0;
    return decode_prefix();
}

bool Llm::reset(const std::string & system_prompt) {
    params_.system_prompt = system_prompt;
    return reset();
}

size_t Llm::max_turn_samples() const {
    return (size_t) audio_sample_rate() * MAX_TURN_SECONDS - 1;
}

// Keeps what a rollback to `pos` needs.
void Llm::guard_chat(int pos) {
    model_.guard(GUARD_CHAT, pos);
}

bool Llm::truncate(int n_past) {
    if (n_past < 0 || n_past > n_past_) {
        return false;
    }
    close_stream();
    in_turn_    = false;
    reopen_pos_ = -1;
    reply_      = Reply::None;
    in_call_    = false;
    call_next_  = false;
    guard_chat(0);
    n_past_ = n_past;
    return true;
}

int Llm::n_layer() const {
    return model_.n_layer();
}

int Llm::n_embd() const {
    return model_.n_embd();
}

bool Llm::decode_prefix() {
    std::string system = params_.system_prompt;
    const std::string context = params_.context ? params_.context() : "";
    if (!context.empty()) {
        system += (system.empty() ? "" : "\n\n") + context;
    }
    for (const Tool & tool : params_.tools) {
        system += TOOL_BEGIN + declare(tool) + TOOL_END;
    }
    std::string prefix;
    if (!system.empty()) {
        prefix = std::string(TURN_SYSTEM) + system + TURN_END;
    }
    if (!decode_text(prefix, /*add_bos=*/true)) {
        return false;
    }
    model_.guard(GUARD_PREFIX, n_past_);
    return true;
}

int Llm::audio_sample_rate() const {
    return Mel::SAMPLE_RATE;
}

bool Llm::read_audio_file(const std::string & path, std::vector<float> & out) const {
    return read_audio(path, audio_sample_rate(), out);
}

bool Llm::user_text(std::string_view text) {
    if (in_turn_) {
        fprintf(stderr, "llm: user turn already open\n");
        return false;
    }
    reopen_pos_ = -1;
    const std::string note = params_.turn_context ? params_.turn_context() : "";
    return decode_text(std::string(TURN_USER) + std::string(text) + note + TURN_END + TURN_MODEL);
}

bool Llm::user_audio_oneshot(const float * pcm, size_t n_samples) {
    if (in_turn_) {
        fprintf(stderr, "llm: user turn already open\n");
        return false;
    }
    reopen_pos_ = -1;
    if (n_samples > max_turn_samples()) {
        fprintf(stderr, "llm: turn too long\n");
        return false;
    }
    AudioStream stream(encoder_);
    stream.push(pcm, n_samples);
    std::vector<float> embd;
    int n = 0;
    return stream.finish(embd, n) &&
           decode_text(std::string(TURN_USER) + AUDIO_BEGIN) &&
           (n == 0 || decode_embd(embd.data(), n)) &&
           decode_text(std::string(AUDIO_END) + TURN_END + TURN_MODEL);
}

bool Llm::begin_user_turn() {
    if (in_turn_) {
        fprintf(stderr, "llm: user turn already open\n");
        return false;
    }
    if (reply_ == Reply::Generating) {
        fprintf(stderr, "llm: hold or end the reply first\n");
        return false;
    }
    // A turn holds up to 30 s of audio (~750 tokens), plus its markup and the reply, and with tools a few of their answers.
    // Start over rather than run out of context mid-turn.
    const int answers = params_.tools.empty() ? 0 : 3 * (int) MAX_RESPONSE_BYTES / 4;
    if (n_past_ + 750 + 32 + params_.max_tokens + answers > model_.n_ctx()) {
        fprintf(stderr, "llm: context nearly full, starting a new conversation\n");
        if (!reset()) {
            return false;
        }
    }
    close_stream();
    reopen_pos_  = -1;
    turn_pos_    = n_past_;
    turn_pcm_.clear();
    guard_chat(turn_pos_);

    stream_ = std::make_unique<AudioStream>(encoder_);
    if (!decode_text(std::string(TURN_USER) + AUDIO_BEGIN)) {
        return false;
    }
    fed_     = 0;
    closed_  = false;
    in_turn_ = true;
    return true;
}

bool Llm::push_audio(const float * pcm, size_t n_samples) {
    if (!in_turn_) {
        return false;
    }
    turn_pcm_.insert(turn_pcm_.end(), pcm, pcm + n_samples);
    // While the segment is closed, e.g. during a pause, the audio waits for
    // the user to go on, and is dropped if they do not.
    return closed_ || feed();
}

bool Llm::commit_audio(std::vector<float> * layer_states, const std::vector<int> * layers) {
    if (!in_turn_ || closed_) {
        return false;
    }
    // A copy of the stream is flushed, so that the stream itself can go on.
    AudioStream copy = *stream_;
    std::vector<float> embd;
    int n_new = 0;
    chat_tail_pos_ = n_past_;
    if (!copy.finish(embd, n_new) || (n_new > 0 && !decode_embd(embd.data(), n_new))) {
        return false;
    }
    closed_ = true;
    const std::vector<int32_t> tokens = tokenize(AUDIO_END, false);
    if (!layer_states) {
        return decode(tokens.data(), nullptr, (int) tokens.size());
    }
    std::vector<int> wanted;
    for (int il = 0; il < n_layer(); il++) {
        if (!layers || std::find(layers->begin(), layers->end(), il) != layers->end()) {
            wanted.push_back(il);
        }
    }
    if (tokens.size() != 1 || n_past_ + 1 > model_.n_ctx()) {
        return false;
    }
    if (!model_.decode(tokens.data(), nullptr, 1, n_past_, &wanted, layer_states)) {
        return false;
    }
    n_past_++;
    return true;
}

bool Llm::continue_user_turn() {
    if (!in_turn_ || !closed_) {
        return false;
    }
    // Take back the segment's closing tail.
    n_past_ = chat_tail_pos_;
    closed_ = false;
    return feed();
}

bool Llm::end_user_turn() {
    if (!in_turn_) {
        return false;
    }
    // The stream is kept, in case the turn is reopened.
    if (!closed_ && !commit_audio()) {
        return false;
    }
    in_turn_    = false;
    reopen_pos_ = chat_tail_pos_;
    guard_chat(chat_tail_pos_);
    // The note follows the speech, after the heads' features.
    const std::string note = params_.turn_context ? params_.turn_context() : "";
    return decode_text(note + TURN_END + TURN_MODEL);
}

bool Llm::abandon_user_turn() {
    if (!in_turn_) {
        return false;
    }
    close_stream();
    guard_chat(0);
    n_past_     = turn_pos_;
    in_turn_    = false;
    turn_pcm_.clear();
    return true;
}

bool Llm::reopen_user_turn(const float * since_end, size_t n_samples) {
    if (in_turn_ || reopen_pos_ < 0 || turn_pcm_.size() + n_samples > max_turn_samples()) {
        return false;
    }
    // Taking back the segment's tail also drops the turn's closing tokens
    // and the reply.
    reopen_pos_ = -1;
    reply_      = Reply::None;
    in_turn_    = true;
    turn_pcm_.insert(turn_pcm_.end(), since_end, since_end + n_samples);
    if (!continue_user_turn()) {
        abandon_user_turn();
        return false;
    }
    return true;
}

bool Llm::begin_reply() {
    if (in_turn_ || reply_ != Reply::None) {
        fprintf(stderr, "llm: no turn to reply to\n");
        return false;
    }
    const std::vector<int32_t> opening = tokenize(TURN_MODEL, false);
    if (opening.empty()) {
        return false;
    }
    reply_          = Reply::Generating;
    reply_tokens_   = 0;
    reply_at_start_ = true;
    reply_spoke_    = false;
    reply_failed_   = false;
    in_call_        = false;
    call_next_      = false;
    last_token_     = opening.back();
    return true;
}

bool Llm::next_piece(std::string & piece) {
    piece.clear();
    if (reply_ != Reply::Generating) {
        return false;
    }
    const Tokenizer & vocab = model_.tokenizer();
    int32_t tok;
    if (call_next_ || (!params_.tools.empty() && !in_call_ && params_.call_threshold > 0.f &&
                       probability(call_begin_, params_.call_threshold) >= params_.call_threshold)) {
        tok = call_begin_;
    } else {
        tok = sampler_.sample(model_.logits());
    }
    if (vocab.is_eog(tok) || reply_tokens_ >= params_.max_tokens) {
        end_reply();
        return false;
    }

    // A call that would open the reply waits for a filler, said as the
    // model's own words: a call taken back leaves it, as it was heard.
    if (tok == call_begin_ && !call_next_ && !in_call_ && !reply_spoke_ && !params_.call_fillers.empty()) {
        const std::string & filler = params_.call_fillers[fillers_said_++ % params_.call_fillers.size()];
        const std::vector<int32_t> tokens = tokenize(filler, false);
        if (!tokens.empty() && decode_text(filler)) {
            reply_tokens_  += (int) tokens.size();
            last_token_     = tokens.back();
            reply_at_start_ = false;
            reply_spoke_    = true;
            call_next_      = true;
            piece           = filler;
            return true;
        }
    }
    call_next_ = false;

    // A tool call is not text of the reply: it is collected, markup and all,
    // and the reply waits for the answer once it is complete.
    bool called = false;
    if (!params_.tools.empty() && tok == call_begin_ && !in_call_) {
        in_call_     = true;
        call_pos_    = n_past_;
        before_call_ = last_token_;
        call_text_.clear();
    } else if (in_call_ && tok == call_end_) {
        in_call_ = false;
        called   = true;
        if (!parse_call(call_text_, call_)) {
            fprintf(stderr, "llm: malformed tool call: %s\n", call_text_.c_str());
            in_call_ = true;  // taken back with the rest of the call
            end_reply();
            return false;
        }
    }

    const std::string token_text = vocab.piece(tok, /*special=*/in_call_);
    std::string_view text = token_text;
    if (in_call_) {
        if (tok != call_begin_) {
            call_text_ += text;
        }
        text = {};
    } else if (reply_at_start_) {
        // Later turns tend to open with a newline, and so does the text
        // after a tool's answer, which is set apart from what came before.
        const size_t first = text.find_first_not_of(" \t\r\n");
        text = first == std::string_view::npos ? std::string_view() : text.substr(first);
        if (!text.empty()) {
            reply_at_start_ = false;
            if (reply_spoke_) {
                piece = " ";
            }
        }
    }
    if (!decode_token(tok)) {
        reply_failed_ = true;
        reply_        = Reply::None;
        in_call_      = false;
        return false;
    }
    reply_tokens_++;
    last_token_ = tok;
    if (called) {
        // The answer follows the call at once, as the chat template has it.
        if (!decode_token(response_)) {
            reply_failed_ = true;
            reply_        = Reply::None;
            return false;
        }
        last_token_ = response_;
        reply_      = Reply::Tool;
        return false;
    }
    reply_spoke_ = reply_spoke_ || !text.empty();
    piece.append(text);
    return true;
}

bool Llm::tool_call(ToolCall & call) const {
    if (reply_ != Reply::Tool) {
        return false;
    }
    call = call_;
    return true;
}

bool Llm::tool_response(std::string_view result) {
    if (reply_ != Reply::Tool) {
        return false;
    }
    // The answer is data: its text is never read as markup. With a note, the
    // response is a mapping, keys in order as the chat template has them.
    reply_ = Reply::Generating;
    const std::string note = params_.response_note.empty() ? "" : "note:" + (QUOTE + params_.response_note + QUOTE) + ",";
    const bool ok = decode_text(std::string("response:") + call_.name + "{" + note + "value:" + QUOTE) &&
                    decode_text(cut(result, MAX_RESPONSE_BYTES), false, /*parse_special=*/false) &&
                    decode_text(std::string(QUOTE) + "}" + RESPONSE_END);
    if (!ok) {
        reply_failed_ = true;
        reply_        = Reply::None;
        return false;
    }
    reply_at_start_ = true;
    return true;
}

// Takes back the call being made, or the one waiting for its answer.
void Llm::drop_call() {
    if (!in_call_ && reply_ != Reply::Tool) {
        return;
    }
    n_past_     = call_pos_;
    last_token_ = before_call_;
    in_call_    = false;
}

bool Llm::hold_reply() {
    if (reply_ != Reply::Generating && reply_ != Reply::Tool) {
        return false;
    }
    drop_call();
    call_next_ = false;  // on resuming, the model decides again
    held_pos_  = n_past_;
    reply_    = Reply::Held;
    return decode_text(TURN_END);
}

bool Llm::resume_reply() {
    if (reply_ != Reply::Held || in_turn_) {
        return false;
    }
    // Take back the close, and the reply's last token with it: decoding that
    // again gives the logits for the next one.
    n_past_ = held_pos_ - 1;
    reply_  = Reply::Generating;
    if (!decode_token(last_token_)) {
        reply_        = Reply::None;
        reply_failed_ = true;
        return false;
    }
    return true;
}

bool Llm::end_reply() {
    const Reply was = reply_;
    drop_call();
    reply_ = Reply::None;
    // Close the model turn so the next user turn continues the conversation.
    return was == Reply::None || was == Reply::Held || decode_text(TURN_END);
}

bool Llm::generate(const PieceCallback & on_piece, const ToolCallback & on_tool) {
    if (!begin_reply()) {
        return false;
    }
    std::string piece;
    for (;;) {
        while (next_piece(piece)) {
            if (!piece.empty() && !on_piece(piece)) {
                return end_reply();
            }
        }
        ToolCall call;
        if (!tool_call(call)) {
            break;
        }
        if (!tool_response(on_tool ? on_tool(call) : "error: tools are not available")) {
            break;
        }
    }
    return !reply_failed_;
}

bool Llm::decode_token(int32_t token) {
    return decode(&token, nullptr, 1);
}

void Llm::close_stream() {
    stream_.reset();
    closed_ = false;
}

// Feeds the stream the turn's audio it has not seen yet.
bool Llm::feed() {
    if (turn_pcm_.size() > max_turn_samples()) {
        fprintf(stderr, "llm: turn too long\n");
        return false;
    }
    const size_t from = fed_;
    fed_ = turn_pcm_.size();
    stream_->push(turn_pcm_.data() + from, fed_ - from);
    std::vector<float> embd;
    int n_new = 0;
    return stream_->pull(embd, n_new) && (n_new <= 0 || decode_embd(embd.data(), n_new));
}

// The probability the model gives `token` next, before sampling, or 0 if it
// is below `floor`. It is at most e^(logit - the largest logit), which rules
// out most tokens without the softmax over the whole vocabulary.
float Llm::probability(int32_t token, float floor) const {
    const std::vector<float> & logits = model_.logits();
    const int n = (int) logits.size();
    if (token < 0 || token >= n) {
        return 0.f;
    }
    const float max = *std::max_element(logits.begin(), logits.end());
    if (logits[token] - max < std::log(floor)) {
        return 0.f;
    }
    float sum = 0.f;
    for (int i = 0; i < n; i++) {
        sum += std::exp(logits[i] - max);
    }
    return std::exp(logits[token] - max) / sum;
}

int32_t Llm::special_token(const char * text) const {
    const std::vector<int32_t> tokens = tokenize(text, false);
    return tokens.size() == 1 ? tokens[0] : -1;
}

std::vector<int32_t> Llm::tokenize(std::string_view text, bool add_bos, bool parse_special) const {
    return model_.tokenizer().encode(text, add_bos, parse_special);
}

bool Llm::decode_text(std::string_view text, bool add_bos, bool parse_special) {
    const std::vector<int32_t> tokens = tokenize(text, add_bos, parse_special);
    if (tokens.empty()) {
        return text.empty();
    }
    return decode(tokens.data(), nullptr, (int) tokens.size());
}

bool Llm::decode_embd(const float * embd, int n_tokens) {
    return decode(nullptr, embd, n_tokens);
}

// Tokens or embeddings at the end of the conversation, a batch at a time.
bool Llm::decode(const int32_t * tokens, const float * embd, int n) {
    if (n_past_ + n > model_.n_ctx()) {
        fprintf(stderr, "llm: context full\n");
        return false;
    }
    for (int start = 0; start < n; start += Gemma4::MAX_BATCH) {
        const int count = std::min(n - start, Gemma4::MAX_BATCH);
        if (!model_.decode(tokens ? tokens + start : nullptr, embd ? embd + (size_t) start * model_.n_embd() : nullptr,
                           count, n_past_)) {
            return false;
        }
        n_past_ += count;
    }
    return true;
}
