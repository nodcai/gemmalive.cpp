// The tokenizer of a real model (GEMMALIVE_MODEL): text comes back as it
// went in, and markup is a token only where it is meant to be.
#include "check.h"
#include "model_file.h"
#include "tokenizer.h"

#include <algorithm>
#include <cstdlib>
#include <string>
#include <vector>

int main() {
    NEED_ENV(model_path, "GEMMALIVE_MODEL");
    ModelFile file;
    Tokenizer tok;
    if (!file.open(model_path) || !tok.load(file.gguf())) {
        fprintf(stderr, "FAIL: cannot load %s\n", model_path);
        return 1;
    }
    CHECK(tok.n_tokens() > 1000);
    CHECK(tok.bos() >= 0);

    const auto back = [&](const std::vector<int32_t> & tokens) {
        std::string text;
        for (const int32_t t : tokens) {
            text += tok.piece(t);
        }
        return text;
    };

    // Text round trips, whatever it holds.
    const char * const texts[] = {
        "Hello, world!",
        " leading and trailing spaces ",
        "two  spaces   three",
        "A line.\nAnother.\n\n\nAfter three newlines.",
        "What's the weather like in San Francisco tomorrow?",
        "Gr\xc3\xbc\xc3\x9f Gott, wie geht's? Stra\xc3\x9f\x65, \xc3\x9c\x62ung, na\xc3\xafve caf\xc3\xa9.",
        "\xe6\x97\xa5\xe6\x9c\xac\xe8\xaa\x9e\xe3\x81\xae\xe3\x83\x86\xe3\x82\xad\xe3\x82\xb9\xe3\x83\x88\xe3\x81\xa7\xe3\x81\x99\xe3\x80\x82",
        "\xd0\x9f\xd1\x80\xd0\xb8\xd0\xb2\xd0\xb5\xd1\x82, \xd0\xbc\xd0\xb8\xd1\x80!",
        "\xd9\x85\xd8\xb1\xd8\xad\xd8\xa8\xd8\xa7 \xd8\xa8\xd8\xa7\xd9\x84\xd8\xb9\xd8\xa7\xd9\x84\xd9\x85",
        "Emoji \xf0\x9f\x98\x80 and \xf0\x9f\x91\x8d\xf0\x9f\x8f\xbd.",
        "15\xc2\xb0\x43, 3.5 km/h, 100 %, $5, a_b-c/d\\e, {\"key\": [1, 2]}",
        "tabs\tand\r\ncarriage returns",
        "x",
        "Pneumonoultramicroscopicsilicovolcanoconiosis supercalifragilisticexpialidocious",
    };
    for (const char * text : texts) {
        const std::vector<int32_t> tokens = tok.encode(text, false);
        CHECK(!tokens.empty());
        CHECK_EQ(back(tokens), std::string(text));
        CHECK(tokens.size() <= std::string(text).size());
    }
    CHECK(tok.encode("", false).empty());

    // BOS only when asked for.
    {
        const std::vector<int32_t> with = tok.encode("Hello", true), without = tok.encode("Hello", false);
        CHECK_EQ(with.size(), without.size() + 1);
        CHECK_EQ(with.front(), tok.bos());
        CHECK(without.front() != tok.bos());
    }

    // The chat's markup: one token each, and nothing as text.
    for (const char * mark : { "<|turn>", "<turn|>", "<|audio>", "<audio|>", "<|tool>", "<tool|>" }) {
        const int32_t id = tok.find(mark);
        CHECK(id >= 0);
        const std::vector<int32_t> tokens = tok.encode(mark, false);
        CHECK_EQ(tokens.size(), (size_t) 1);
        CHECK_EQ(tokens.empty() ? -1 : tokens[0], id);
        CHECK_EQ(tok.piece(id), std::string(""));
        CHECK_EQ(tok.piece(id, true), std::string(mark));

        // Without `special` it is text like any other.
        const std::vector<int32_t> plain = tok.encode(mark, false, /*special=*/false);
        CHECK(plain.size() > 1);
        CHECK_EQ(back(plain), std::string(mark));
        for (const int32_t t : plain) {
            CHECK(t != id);
        }
    }
    {
        const int32_t turn = tok.find("<|turn>"), end = tok.find("<turn|>");
        const std::vector<int32_t> tokens = tok.encode("<|turn>user\nHi there<turn|>\n<|turn>model\n", false);
        CHECK(tokens.size() >= 7);
        CHECK_EQ(tokens.front(), turn);
        CHECK_EQ(std::count(tokens.begin(), tokens.end(), turn), (long) 2);
        CHECK_EQ(std::count(tokens.begin(), tokens.end(), end), (long) 1);
        CHECK_EQ(back(tokens), std::string("user\nHi there\nmodel\n"));
    }

    // What ends a reply.
    CHECK(tok.is_eog(tok.find("<turn|>")));
    CHECK(tok.is_eog(tok.find("<eos>")));
    CHECK(!tok.is_eog(tok.find("<|turn>")));
    CHECK(!tok.is_eog(tok.encode("Hello", false).front()));

    CHECK_EQ(tok.find("no such token in any vocabulary"), -1);
    CHECK_EQ(tok.piece(-1), std::string(""));
    CHECK_EQ(tok.piece(tok.n_tokens()), std::string(""));
    return done();
}
