#pragma once

#include <optional>
#include <string>

// What the assistant knows of the user's device, for the system prompt
// (Llm::Params::context) and each turn (Llm::Params::turn_context). The date
// and time come from the clock, in the device's time zone if it is known,
// else in this machine's.
struct Device {
    std::string        locale;     // e.g. "de_DE" or "en-US"; "" = unknown
    std::string        time_zone;  // e.g. "Europe/Berlin"; "" = unknown
    std::optional<int> utc_offset; // minutes ahead of UTC; none = this machine's
};

// The date, the time zone and the user's locale.
std::string device_context(const Device & device);

// Minutes the user's clock is ahead of UTC: the device's offset if known,
// else this machine's.
int local_utc_offset(const Device & device);

// The time, put after each user turn, with `aside` if any:
// "\n(14:05; aside)".
std::string clock_note(const Device & device, const std::string & aside = "");

// Whether the user measures in Fahrenheit and miles: their locale's region
// does, and their time zone, if known, is in it. A locale may stand for a
// language alone, "en-US" for most English speakers anywhere.
bool imperial_units(const Device & device);

// The locale of the environment (LC_ALL, LC_MESSAGES, LANG), e.g. "de_DE",
// or "".
std::string environment_locale();
