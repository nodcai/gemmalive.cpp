#include "device.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <initializer_list>

// The time where the user is: at their UTC offset if known, else this
// machine's local time.
static std::tm local_now(const Device & device) {
    std::time_t t = std::time(nullptr);
    std::tm tm{};
    if (device.utc_offset) {
        t += (std::time_t) *device.utc_offset * 60;
#ifdef _WIN32
        gmtime_s(&tm, &t);
#else
        gmtime_r(&t, &tm);
#endif
        return tm;
    }
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    return tm;
}

static std::string format(const char * fmt, const std::tm & tm) {
    char buf[128];
    const size_t n = std::strftime(buf, sizeof(buf), fmt, &tm);
    return std::string(buf, n);
}

// "en_US.UTF-8", "en-US" -> "US".
static std::string region(const std::string & locale) {
    const size_t sep = locale.find_first_of("_-");
    std::string out;
    for (size_t i = sep + 1; sep != std::string::npos && i < locale.size() && std::isalpha((unsigned char) locale[i]); i++) {
        out += (char) std::toupper((unsigned char) locale[i]);
    }
    return out;
}

bool imperial_units(const Device & device) {
    const std::string   r  = region(device.locale);
    const std::string & tz = device.time_zone;
    const auto in = [&](std::initializer_list<const char *> prefixes) {
        for (const char * p : prefixes) {
            if (tz.rfind(p, 0) == 0) {
                return true;
            }
        }
        return tz.empty();
    };
    if (r == "US") {
        return in({ "America/", "US/", "Pacific/Honolulu", "Pacific/Guam", "Pacific/Pago_Pago", "Pacific/Saipan" });
    }
    if (r == "LR") {
        return in({ "Africa/Monrovia" });
    }
    if (r == "MM") {
        return in({ "Asia/Yangon", "Asia/Rangoon" });
    }
    return false;
}

std::string device_context(const Device & device) {
    const std::tm tm = local_now(device);
    std::string date = format("%A, %d %B %Y", tm);
    const size_t comma = date.find(", 0");
    if (comma != std::string::npos) {
        date.erase(comma + 2, 1);  // "Friday, 03 ..." -> "Friday, 3 ..."
    }
    // "+0200" -> "UTC+02:00"
    std::string offset;
    if (device.utc_offset) {
        const int m = std::abs(*device.utc_offset);
        char buf[16];
        snprintf(buf, sizeof(buf), "UTC%c%02d:%02d", *device.utc_offset < 0 ? '-' : '+', m / 60, m % 60);
        offset = buf;
    } else {
        offset = format("%z", tm);
        if (offset.size() == 5) {
            offset = "UTC" + offset.substr(0, 3) + ":" + offset.substr(3);
        }
    }
    const std::string zone = !device.time_zone.empty() ? device.time_zone
                           : device.utc_offset         ? ""
                                                       : format("%Z", tm);

    std::string out = "Today is " + date + ".";
    if (!zone.empty() || !offset.empty()) {
        out += " The user's time zone is " + zone + (zone.empty() || offset.empty() ? "" : ", ") + offset + ".";
    }
    if (!device.locale.empty()) {
        out += " The user's locale is " + device.locale + "; use " +
               (imperial_units(device) ? "Fahrenheit and miles." : "Celsius and metric units.");
    }
    return out;
}

int local_utc_offset(const Device & device) {
    if (device.utc_offset) {
        return *device.utc_offset;
    }
    // "+0200" -> 120
    const std::string z = format("%z", local_now(device));
    int hh = 0, mm = 0;
    if (z.size() != 5 || sscanf(z.c_str() + 1, "%2d%2d", &hh, &mm) != 2) {
        return 0;
    }
    return (z[0] == '-' ? -1 : 1) * (hh * 60 + mm);
}

std::string clock_note(const Device & device, const std::string & aside) {
    return "\n(" + format("%H:%M", local_now(device)) + (aside.empty() ? "" : "; " + aside) + ")";
}

std::string environment_locale() {
    for (const char * name : { "LC_ALL", "LC_MESSAGES", "LANG" }) {
        const char * value = std::getenv(name);
        if (!value || !*value) {
            continue;
        }
        std::string locale = value;
        locale = locale.substr(0, locale.find_first_of(".@"));
        return locale == "C" || locale == "POSIX" ? "" : locale;
    }
    return "";
}
