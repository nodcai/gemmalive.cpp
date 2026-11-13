#pragma once

#include "llm.h"

#include <cstddef>
#include <string>
#include <string_view>

// Parts of the library that the tests check on their own (tests/). Not for
// apps: they change with the code around them.
namespace internal {

// llm.cpp: Gemma's tool markup.
std::string      declare(const Tool & tool);
std::string      json_string(std::string_view s);
bool             parse_call(std::string_view s, ToolCall & call);
std::string_view cut(std::string_view s, size_t max);

// tts.cpp: the reply's text as it is spoken.
std::string spoken(std::string text, std::string & held, char & prev, bool last);

// turn_head.cpp
std::string sha256_hex(const unsigned char * data, size_t size);

}  // namespace internal
