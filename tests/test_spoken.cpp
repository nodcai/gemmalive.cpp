// The reply's text on its way to speech: degree signs in words, also when
// the sign and its unit come in different pieces.
#include "check.h"
#include "internal.h"

#include <initializer_list>
#include <string>

using namespace internal;

#define DEG "\xC2\xB0"

// The pieces as a reply brings them, then its end.
static std::string say(std::initializer_list<const char *> pieces) {
    std::string held, out;
    char prev = 0;
    for (const char * piece : pieces) {
        out += spoken(held + piece, held, prev, false);
    }
    if (!held.empty()) {
        out += spoken(held, held, prev, true);
    }
    return out;
}

int main() {
    CHECK_EQ(say({ "Hello there." }), std::string("Hello there."));
    CHECK_EQ(say({ "Hello", " there", "." }), std::string("Hello there."));
    CHECK_EQ(say({ "It is 15" DEG "C today." }), std::string("It is 15 degrees Celsius today."));
    CHECK_EQ(say({ "It is 59" DEG "F." }), std::string("It is 59 degrees Fahrenheit."));
    CHECK_EQ(say({ "Turn 90" DEG " left." }), std::string("Turn 90 degrees left."));
    CHECK_EQ(say({ "minus 5 " DEG "C." }), std::string("minus 5 degrees Celsius."));
    CHECK_EQ(say({ "20" DEG "Celsius" }), std::string("20 degrees Celsius"));
    CHECK_EQ(say({ "15" DEG "C to 20" DEG "C" }), std::string("15 degrees Celsius to 20 degrees Celsius"));

    // The sign at the end of a piece waits for what follows.
    CHECK_EQ(say({ "It is 15", DEG, "C in Paris." }), std::string("It is 15 degrees Celsius in Paris."));
    CHECK_EQ(say({ "It is 15" DEG, "C in Paris." }), std::string("It is 15 degrees Celsius in Paris."));
    CHECK_EQ(say({ "It is 15" DEG "C", " in Paris." }), std::string("It is 15 degrees Celsius in Paris."));
    CHECK_EQ(say({ "It is 15" DEG "C", "elsius." }), std::string("It is 15 degrees Celsius."));
    CHECK_EQ(say({ "A right angle has 90", DEG }), std::string("A right angle has 90 degrees"));
    CHECK_EQ(say({ "It is 15", DEG "C" }), std::string("It is 15 degrees Celsius"));

    // Held text is not said twice, and nothing is held at the end.
    {
        std::string held;
        char prev = 0;
        CHECK_EQ(spoken("15" DEG, held, prev, false), std::string("15"));
        CHECK_EQ(held, std::string(DEG));
        CHECK_EQ(spoken(held, held, prev, true), std::string(" degrees"));
        CHECK_EQ(held, std::string(""));
    }
    return done();
}
