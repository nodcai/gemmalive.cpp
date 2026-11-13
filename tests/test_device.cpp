// What the assistant is told about the user's device.
#include "check.h"
#include "device.h"

#include <regex>
#include <string>

static bool has(const std::string & text, const char * part) {
    return text.find(part) != std::string::npos;
}

static Device device(const char * locale, const char * time_zone) {
    Device d;
    d.locale    = locale;
    d.time_zone = time_zone;
    return d;
}

int main() {
    // Fahrenheit and miles: the locale's region uses them and the time zone,
    // if known, is in it.
    CHECK(imperial_units(device("en_US", "")));
    CHECK(imperial_units(device("en-US", "America/New_York")));
    CHECK(imperial_units(device("en_US.UTF-8", "Pacific/Honolulu")));
    CHECK(!imperial_units(device("en_US", "Europe/Berlin")));
    CHECK(!imperial_units(device("de_DE", "America/New_York")));
    CHECK(!imperial_units(device("en_GB", "")));
    CHECK(imperial_units(device("en_LR", "Africa/Monrovia")));
    CHECK(!imperial_units(device("en", "")));
    CHECK(!imperial_units(device("", "")));

    {
        Device d = device("de_DE", "Europe/Berlin");
        d.utc_offset = 120;
        const std::string c = device_context(d);
        CHECK(std::regex_search(c, std::regex("^Today is [A-Z][a-z]+day, [1-9][0-9]? [A-Z][a-z]+ 20[0-9][0-9]\\.")));
        CHECK(has(c, "The user's time zone is Europe/Berlin, UTC+02:00."));
        CHECK(has(c, "The user's locale is de_DE; use Celsius and metric units."));
        CHECK_EQ(local_utc_offset(d), 120);
    }
    {
        Device d = device("en_US", "");
        d.utc_offset = -330;
        const std::string c = device_context(d);
        CHECK(has(c, "The user's time zone is UTC-05:30."));
        CHECK(has(c, "use Fahrenheit and miles."));
        CHECK_EQ(local_utc_offset(d), -330);
    }
    {
        // This machine's time zone, whatever it is.
        const std::string c = device_context(Device{});
        CHECK(has(c, "Today is "));
        CHECK(!has(c, "locale"));
        const int offset = local_utc_offset(Device{});
        CHECK(offset > -13 * 60 && offset < 15 * 60);
    }

    Device d;
    d.utc_offset = 0;
    CHECK(std::regex_match(clock_note(d), std::regex("\n\\([0-2][0-9]:[0-5][0-9]\\)")));
    CHECK(std::regex_match(clock_note(d, "an aside"), std::regex("\n\\([0-2][0-9]:[0-5][0-9]; an aside\\)")));
    return done();
}
