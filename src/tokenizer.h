#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

struct gguf_context;

// Gemma 4's tokenizer: BPE over UTF-8 with spaces written as U+2581, byte
// tokens for what the vocabulary lacks, and its markup as single tokens.
class Tokenizer {
public:
    bool load(gguf_context * gguf);

    // `special`: markup in the text becomes its token; without, only text.
    std::vector<int32_t> encode(std::string_view text, bool add_bos, bool special = true) const;

    // The text of a token. Markup is only shown with `special`.
    std::string piece(int32_t token, bool special = false) const;

    int32_t find(std::string_view text) const;  // -1 if none
    int32_t bos() const { return bos_; }
    bool    is_eog(int32_t token) const;
    int     n_tokens() const { return (int) tokens_.size(); }

private:
    enum Type { NORMAL = 1, UNKNOWN = 2, CONTROL = 3, USER_DEFINED = 4, UNUSED = 5, BYTE = 6 };

    void encode_text(const std::string & text, std::vector<int32_t> & out) const;
    void encode_word(const std::string & word, std::vector<int32_t> & out) const;

    std::vector<std::string> tokens_;
    std::vector<int>         types_;
    std::unordered_map<std::string, int32_t> ids_;
    std::unordered_map<std::string, int> ranks_;  // "left\nright" -> rank
    std::vector<int32_t> markup_;  // longest first
    std::vector<int32_t> eog_;
    int32_t bos_ = -1;
};
