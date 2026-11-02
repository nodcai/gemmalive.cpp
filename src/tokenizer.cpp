#include "tokenizer.h"

#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <list>
#include <queue>

static const std::string SPACE = "\xe2\x96\x81";  // U+2581

static void replace_all(std::string & s, const std::string & from, const std::string & to) {
    std::string out;
    size_t pos = 0;
    for (size_t at; (at = s.find(from, pos)) != std::string::npos; pos = at + from.size()) {
        out.append(s, pos, at - pos);
        out += to;
    }
    out.append(s, pos);
    s = std::move(out);
}

static size_t utf8_len(char c) {
    const uint8_t b = (uint8_t) c;
    return b < 0x80 ? 1 : (b >> 5) == 0x6 ? 2 : (b >> 4) == 0xe ? 3 : (b >> 3) == 0x1e ? 4 : 1;
}

bool Tokenizer::load(gguf_context * gguf) {
    const int64_t k_tokens = gguf_find_key(gguf, "tokenizer.ggml.tokens");
    const int64_t k_types  = gguf_find_key(gguf, "tokenizer.ggml.token_type");
    const int64_t k_merges = gguf_find_key(gguf, "tokenizer.ggml.merges");
    if (k_tokens < 0 || k_types < 0 || k_merges < 0) {
        fprintf(stderr, "tokenizer: no vocabulary in the model\n");
        return false;
    }
    const size_t n = gguf_get_arr_n(gguf, k_tokens);
    const int32_t * types = (const int32_t *) gguf_get_arr_data(gguf, k_types);
    tokens_.resize(n);
    types_.resize(n);
    ids_.reserve(n);
    for (size_t i = 0; i < n; i++) {
        tokens_[i] = gguf_get_arr_str(gguf, k_tokens, i);
        types_[i]  = types[i];
        ids_.emplace(tokens_[i], (int32_t) i);
        if (types_[i] == CONTROL || types_[i] == USER_DEFINED || types_[i] == UNKNOWN) {
            markup_.push_back((int32_t) i);
        }
    }
    std::stable_sort(markup_.begin(), markup_.end(),
                     [&](int32_t a, int32_t b) { return tokens_[a].size() > tokens_[b].size(); });

    const size_t n_merges = gguf_get_arr_n(gguf, k_merges);
    for (size_t i = 0; i < n_merges; i++) {
        const std::string merge = gguf_get_arr_str(gguf, k_merges, i);
        const size_t space = merge.find(' ', 1);
        if (space != std::string::npos) {
            ranks_.emplace(merge.substr(0, space) + '\n' + merge.substr(space + 1), (int) i);
        }
    }

    const int64_t k_bos = gguf_find_key(gguf, "tokenizer.ggml.bos_token_id");
    bos_ = k_bos >= 0 ? (int32_t) gguf_get_val_u32(gguf, k_bos) : find("<bos>");
    for (const char * t : { "<eos>", "<turn|>", "<|tool_response>" }) {
        if (find(t) >= 0) {
            eog_.push_back(find(t));
        }
    }
    const int64_t k_eos = gguf_find_key(gguf, "tokenizer.ggml.eos_token_id");
    if (k_eos >= 0) {
        eog_.push_back((int32_t) gguf_get_val_u32(gguf, k_eos));
    }
    return true;
}

int32_t Tokenizer::find(std::string_view text) const {
    auto it = ids_.find(std::string(text));
    return it == ids_.end() ? -1 : it->second;
}

bool Tokenizer::is_eog(int32_t token) const {
    return std::find(eog_.begin(), eog_.end(), token) != eog_.end();
}

std::vector<int32_t> Tokenizer::encode(std::string_view text, bool add_bos, bool special) const {
    // Split off the markup, longest token first; what is left is text.
    struct Fragment {
        int32_t     token;  // -1 for text
        std::string text;
    };
    std::list<Fragment> fragments = { { -1, std::string(text) } };
    for (const int32_t id : markup_) {
        if (!special && types_[id] != USER_DEFINED) {
            continue;
        }
        const std::string & mark = tokens_[id];
        for (auto it = fragments.begin(); it != fragments.end(); ++it) {
            if (it->token >= 0) {
                continue;
            }
            size_t at = it->text.find(mark);
            while (at != std::string::npos) {
                std::string rest = it->text.substr(at + mark.size());
                it->text.resize(at);
                auto next = std::next(it);
                if (it->text.empty()) {
                    it->token = id;
                } else {
                    it = fragments.insert(next, { id, {} });
                }
                if (rest.empty()) {
                    break;
                }
                it = fragments.insert(std::next(it), { -1, std::move(rest) });
                at = it->text.find(mark);
            }
        }
    }

    std::vector<int32_t> out;
    if (add_bos && bos_ >= 0) {
        out.push_back(bos_);
    }
    for (const Fragment & f : fragments) {
        if (f.token >= 0) {
            out.push_back(f.token);
        } else if (!f.text.empty()) {
            encode_text(f.text, out);
        }
    }
    return out;
}

void Tokenizer::encode_text(const std::string & raw, std::vector<int32_t> & out) const {
    std::string text = raw;
    replace_all(text, " ", SPACE);
    // Words are runs of newlines and runs of everything else.
    size_t start = 0;
    while (start < text.size()) {
        const bool newlines = text[start] == '\n';
        size_t end = start;
        while (end < text.size() && (text[end] == '\n') == newlines) {
            end++;
        }
        encode_word(text.substr(start, end - start), out);
        start = end;
    }
}

void Tokenizer::encode_word(const std::string & word, std::vector<int32_t> & out) const {
    if (word[0] == '\n') {
        const int32_t id = find(word);
        if (id >= 0) {
            out.push_back(id);
            return;
        }
    }

    struct Symbol {
        int    prev, next;
        size_t start, n;
    };
    std::vector<Symbol> symbols;
    for (size_t at = 0; at < word.size();) {
        const size_t n = std::min(word.size() - at, utf8_len(word[at]));
        symbols.push_back({ (int) symbols.size() - 1, -1, at, n });
        at += n;
    }
    for (size_t i = 0; i + 1 < symbols.size(); i++) {
        symbols[i].next = (int) i + 1;
    }

    struct Bigram {
        int         left, right, rank;
        std::string text;
    };
    auto later = [](const Bigram & a, const Bigram & b) {
        return a.rank > b.rank || (a.rank == b.rank && a.left > b.left);
    };
    std::priority_queue<Bigram, std::vector<Bigram>, decltype(later)> queue(later);
    auto text_of = [&](const Symbol & s) { return word.substr(s.start, s.n); };
    auto add = [&](int left, int right) {
        if (left < 0 || right < 0) {
            return;
        }
        const std::string l = text_of(symbols[left]);
        const std::string r = text_of(symbols[right]);
        auto it = ranks_.find(l + '\n' + r);
        if (it != ranks_.end()) {
            queue.push({ left, right, it->second, l + r });
        }
    };
    for (size_t i = 1; i < symbols.size(); i++) {
        add((int) i - 1, (int) i);
    }
    while (!queue.empty()) {
        const Bigram b = queue.top();
        queue.pop();
        Symbol & l = symbols[b.left];
        Symbol & r = symbols[b.right];
        if (l.n == 0 || r.n == 0 || text_of(l) + text_of(r) != b.text) {
            continue;
        }
        l.n += r.n;
        r.n = 0;
        l.next = r.next;
        if (r.next >= 0) {
            symbols[r.next].prev = b.left;
        }
        add(l.prev, b.left);
        add(b.left, l.next);
    }

    for (const Symbol & s : symbols) {
        if (s.n == 0) {
            continue;
        }
        const std::string text = text_of(s);
        const int32_t id = find(text);
        if (id >= 0) {
            out.push_back(id);
            continue;
        }
        for (const char c : text) {
            char hex[8];
            snprintf(hex, sizeof(hex), "<0x%02X>", (uint8_t) c);
            const int32_t b = find(hex);
            if (b >= 0) {
                out.push_back(b);
            }
        }
    }
}

std::string Tokenizer::piece(int32_t token, bool special) const {
    if (token < 0 || token >= (int32_t) tokens_.size()) {
        return {};
    }
    const std::string & text = tokens_[token];
    switch (types_[token]) {
        case NORMAL: {
            std::string out = text;
            replace_all(out, SPACE, " ");
            return out;
        }
        case BYTE:
            return std::string(1, (char) std::strtol(text.c_str() + 3, nullptr, 16));
        case USER_DEFINED:
            return text;
        default:
            return special ? text : std::string();
    }
}
