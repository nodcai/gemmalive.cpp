// A turn that is abandoned leaves the conversation as it was, wherever in
// it the turn was and however long: speech that was not meant for the
// assistant is abandoned, and the next turn starts where it started.
//
// The sliding-window caches only keep the recent positions and move on
// every so often; a long turn that crosses such a move is taken back from
// what was saved for it. So the turns here are long, 25 s, and they start
// all through the first few thousand positions.
#include "check.h"
#include "conversation.h"

int main() {
    NEED_ENV(model, "GEMMALIVE_MODEL");
    Llm::Params params;
    params.seed       = 1;
    params.max_tokens = 16;
    Llm llm;
    if (!load(llm, model, params)) {
        fprintf(stderr, "FAIL: cannot load %s\n", model);
        return 1;
    }
    const std::vector<float> pcm = sound(25, llm.audio_sample_rate());

    // A typed turn of some 250 tokens moves the conversation on between the
    // spoken ones: less than the stretch of positions a long turn can fail at.
    std::string filler;
    for (int i = 0; i < 24; i++) {
        filler += "The tide comes in twice a day and goes out twice a day. ";
    }

    bool stuck = false;
    while (llm.n_past() < 4200 && !stuck) {
        const int at = llm.n_past();
        bool ok = llm.begin_user_turn() && hear(llm, pcm);
        const int grown = llm.n_past() - at;
        ok = ok && llm.commit_audio() && llm.abandon_user_turn();
        if (!ok || llm.n_past() != at || grown < 500) {
            fprintf(stderr, "FAIL: the turn at %d was not heard and abandoned (%d positions)\n", at, grown);
            g_failed++;
            break;
        }
        // The next turn, as after speech that was ignored. Abandoned again,
        // it must leave no trace either.
        g_checks++;
        if (!llm.begin_user_turn()) {
            fprintf(stderr, "FAIL: no turn can start at %d, after a turn there grew to %d and was abandoned\n", at, at + grown);
            g_failed++;
            stuck = !llm.begin_user_turn();
            fprintf(stderr, "      a second try %s\n", stuck ? "fails too: the conversation is stuck" : "works");
        }
        if (stuck) {
            break;
        }
        CHECK(hear(llm, sound(1, llm.audio_sample_rate())));
        CHECK(llm.abandon_user_turn());
        CHECK_EQ(llm.n_past(), at);

        // A reply to a typed turn still works, and moves on.
        std::string reply;
        CHECK(llm.user_text(filler + "Say OK.") && llm.generate([&](std::string_view piece) {
            reply += piece;
            return true;
        }));
        CHECK(!reply.empty());
        CHECK(llm.n_past() > at + 200);
    }
    CHECK(stuck || llm.n_past() >= 4200);
    return done();
}
