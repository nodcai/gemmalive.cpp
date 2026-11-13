// A reply that is held for speech over it goes on as if nothing had
// happened once that speech is abandoned: with the same choices (the
// likeliest token each time), the same reply as without the hold.
#include "check.h"
#include "conversation.h"

static std::string reply(Llm & llm, int hold_after, double seconds) {
    std::string text, piece;
    if (!llm.user_text("Name the seven days of the week and say one short sentence about each.") || !llm.begin_reply()) {
        return "FAILED to start";
    }
    for (int n = 0; llm.next_piece(piece); n++) {
        text += piece;
        if (n + 1 == hold_after) {
            // Speech over the reply: held, heard, not meant for us, resumed.
            const int at = llm.n_past();
            if (!llm.hold_reply() || !llm.reply_held() || !llm.begin_user_turn() ||
                !hear(llm, sound(seconds, llm.audio_sample_rate())) || !llm.commit_audio() ||
                !llm.abandon_user_turn() || !llm.resume_reply() || llm.reply_held() || llm.n_past() != at) {
                return "FAILED to hold and resume";
            }
        }
    }
    return llm.reply_failed() ? "FAILED to finish" : text;
}

int main() {
    NEED_ENV(model, "GEMMALIVE_MODEL");
    Llm::Params params;
    params.temperature = 0.f;
    params.max_tokens  = 120;
    Llm llm;
    if (!load(llm, model, params)) {
        fprintf(stderr, "FAIL: cannot load %s\n", model);
        return 1;
    }
    const int prefix = llm.n_past();

    const std::string plain = reply(llm, 0, 0);
    printf("%s\n", plain.c_str());
    CHECK(plain.size() > 100);
    CHECK(plain.find("Monday") != std::string::npos);

    // Held early and late, for speech of 1 s and of 3 s.
    for (const int after : { 1, 10, 60 }) {
        for (const double seconds : { 1.0, 3.0 }) {
            CHECK(llm.truncate(prefix));
            CHECK_EQ(reply(llm, after, seconds), plain);
        }
    }

    // A reply that is held cannot be resumed while the turn over it is open.
    CHECK(llm.truncate(prefix));
    std::string piece;
    CHECK(llm.user_text("Say hello.") && llm.begin_reply() && llm.next_piece(piece));
    CHECK(llm.hold_reply());
    CHECK(llm.begin_user_turn());
    CHECK(!llm.resume_reply());
    CHECK(llm.abandon_user_turn());
    CHECK(llm.resume_reply());
    CHECK(llm.end_reply());
    CHECK(!llm.resume_reply());
    return done();
}
