// Gemma's tool markup: declarations as the chat template writes them, and
// the model's calls as JSON.
#include "check.h"
#include "internal.h"

#include <string>

using namespace internal;

#define Q "<|\"|>"

static std::string args(const std::string & call) {
    ToolCall c;
    return parse_call(call, c) ? c.args : "FAILED";
}

int main() {
    // Calls.
    {
        ToolCall c;
        CHECK(parse_call("call:get_weather{location:" Q "Paris, France" Q "}", c));
        CHECK_EQ(c.name, std::string("get_weather"));
        CHECK_EQ(c.args, std::string("{\"location\":\"Paris, France\"}"));
    }
    CHECK_EQ(args("call:get_weather{}"), std::string("{}"));
    CHECK_EQ(args("call:f{a:{b:1,c:[1,2," Q "x" Q "]},d:true,e:null,f:-1.5e3}"),
             std::string("{\"a\":{\"b\":1,\"c\":[1,2,\"x\"]},\"d\":true,\"e\":null,\"f\":-1.5e3}"));
    CHECK_EQ(args("call:f{unit:celsius}"), std::string("{\"unit\":\"celsius\"}"));
    CHECK_EQ(args("call:f{text:" Q "say \"hi\"\n\tC:\\" Q "}"), std::string("{\"text\":\"say \\\"hi\\\"\\n\\tC:\\\\\"}"));
    CHECK_EQ(args("call:f{a:" Q Q ",b:[]}"), std::string("{\"a\":\"\",\"b\":[]}"));
    CHECK_EQ(args("call:f{a:1} \n"), std::string("{\"a\":1}"));
    // Only JSON's numbers are numbers: the rest of what is made of their
    // characters is text, e.g. a date or a range.
    CHECK_EQ(args("call:f{a:0,b:-12,c:3.25,d:1e9,e:2E-3}"), std::string("{\"a\":0,\"b\":-12,\"c\":3.25,\"d\":1e9,\"e\":2E-3}"));
    CHECK_EQ(args("call:f{date:2026-11-13}"), std::string("{\"date\":\"2026-11-13\"}"));
    CHECK_EQ(args("call:f{a:e,b:5-10,c:+1,d:.5,e:1.,f:007,g:-}"),
             std::string("{\"a\":\"e\",\"b\":\"5-10\",\"c\":\"+1\",\"d\":\".5\",\"e\":\"1.\",\"f\":\"007\",\"g\":\"-\"}"));

    // What is no call.
    for (const char * bad : { "", "get_weather{}", "call:{}", "call:f", "call:f{a:1", "call:f{a:" Q "x}", "call:f{a:1}x",
                              "call:f{a}", "call:f{a:}", "call:f{a:1,}x", "call:f{a:[1,2}" }) {
        ToolCall c;
        if (parse_call(bad, c)) {
            fprintf(stderr, "FAIL: \"%s\" was taken for a call: %s\n", bad, c.args.c_str());
            g_failed++;
        }
        g_checks++;
    }

    // Declarations: parameters by name, the required ones listed.
    CHECK_EQ(declare({ "f", "Does it.", {} }), std::string("declaration:f{description:" Q "Does it." Q "}"));
    CHECK_EQ(declare({ "get_weather", "The weather.", { { "location", "STRING", "The city.", false } } }),
             std::string("declaration:get_weather{description:" Q "The weather." Q ",parameters:{properties:{"
                         "location:{description:" Q "The city." Q ",type:" Q "STRING" Q "}},type:" Q "OBJECT" Q "}}"));
    CHECK_EQ(declare({ "f", "d", { { "b", "NUMBER", "B.", true }, { "a", "STRING", "A.", false }, { "c", "BOOLEAN", "C.", true } } }),
             std::string("declaration:f{description:" Q "d" Q ",parameters:{properties:{"
                         "a:{description:" Q "A." Q ",type:" Q "STRING" Q "},"
                         "b:{description:" Q "B." Q ",type:" Q "NUMBER" Q "},"
                         "c:{description:" Q "C." Q ",type:" Q "BOOLEAN" Q "}},"
                         "required:[" Q "b" Q "," Q "c" Q "],type:" Q "OBJECT" Q "}}"));

    // JSON strings.
    CHECK_EQ(json_string("a\"b\\c\n\r\t\x01z"), std::string("\"a\\\"b\\\\c\\n\\r\\t\\u0001z\""));
    CHECK_EQ(json_string("h\xc3\xa9llo"), std::string("\"h\xc3\xa9llo\""));

    // A cut never splits a character.
    CHECK_EQ(std::string(cut("abc", 5)), std::string("abc"));
    CHECK_EQ(std::string(cut("abc", 3)), std::string("abc"));
    CHECK_EQ(std::string(cut("abc", 0)), std::string(""));
    CHECK_EQ(std::string(cut("h\xc3\xa9llo", 2)), std::string("h"));
    CHECK_EQ(std::string(cut("h\xc3\xa9llo", 3)), std::string("h\xc3\xa9"));
    CHECK_EQ(std::string(cut("\xe6\x97\xa5\xe6\x9c\xac", 3)), std::string("\xe6\x97\xa5"));
    CHECK_EQ(std::string(cut("\xe6\x97\xa5\xe6\x9c\xac", 5)), std::string("\xe6\x97\xa5"));
    CHECK_EQ(std::string(cut("\xf0\x9f\x98\x80", 3)), std::string(""));
    return done();
}
