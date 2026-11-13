// Sampling: the same seed gives the same tokens, and top-k, top-p and the
// temperature keep it to the tokens they should.
#include "check.h"
#include "sampler.h"

#include <cmath>
#include <set>
#include <vector>

static std::vector<int32_t> draw(Sampler & s, const std::vector<float> & logits, int n) {
    std::vector<int32_t> out;
    for (int i = 0; i < n; i++) {
        out.push_back(s.sample(logits));
    }
    return out;
}

int main() {
    // Many tokens of similar weight, the largest at 700.
    std::vector<float> logits(1000);
    for (size_t i = 0; i < logits.size(); i++) {
        logits[i] = std::sin((float) i * 0.37f);
    }
    logits[700] = 1.5f;

    Sampler a, b, c;
    a.init(64, 0.95f, 1.f, 7);
    b.init(64, 0.95f, 1.f, 7);
    c.init(64, 0.95f, 1.f, 8);
    const std::vector<int32_t> first = draw(a, logits, 100);
    CHECK(first == draw(b, logits, 100));
    CHECK(first != draw(c, logits, 100));
    CHECK(std::set<int32_t>(first.begin(), first.end()).size() > 10);

    // reset() starts the seed's sequence over.
    a.reset();
    CHECK(first == draw(a, logits, 100));

    // Without a temperature, and with one candidate, the largest logit.
    Sampler greedy;
    greedy.init(64, 0.95f, 0.f, 1);
    Sampler top1;
    top1.init(1, 1.f, 1.f, 1);
    for (int i = 0; i < 20; i++) {
        CHECK_EQ(greedy.sample(logits), 700);
        CHECK_EQ(top1.sample(logits), 700);
    }

    // top_k: only the k largest.
    {
        Sampler s;
        s.init(3, 1.f, 5.f, 3);
        const std::vector<float> l = { 0.f, 2.f, 1.9f, -1.f, 1.8f, 1.f };
        std::set<int32_t> seen;
        for (const int32_t t : draw(s, l, 300)) {
            seen.insert(t);
        }
        CHECK(seen == std::set<int32_t>({ 1, 2, 4 }));
    }
    // top_p: a token with more than top_p of the probability is the only one,
    // however hot the temperature; otherwise the fewest that make it up.
    {
        Sampler s;
        s.init(64, 0.95f, 10.f, 3);
        const std::vector<float> sure = { 5.f, 0.f, 0.f, 0.f };  // 0.98
        for (const int32_t t : draw(s, sure, 100)) {
            CHECK_EQ(t, 0);
        }
        Sampler half;
        half.init(64, 0.5f, 10.f, 3);
        const std::vector<float> two = { 1.f, 1.1f, 0.f, 0.f, 0.f, 0.f };  // 0.26 and 0.29 of 1
        std::set<int32_t> seen;
        for (const int32_t t : draw(half, two, 300)) {
            seen.insert(t);
        }
        CHECK(seen == std::set<int32_t>({ 0, 1 }));
    }
    // Every token can come when nothing is cut.
    {
        Sampler s;
        s.init(0, 1.f, 1.f, 5);
        const std::vector<float> flat(8, 0.f);
        std::set<int32_t> seen;
        for (const int32_t t : draw(s, flat, 500)) {
            seen.insert(t);
        }
        CHECK_EQ(seen.size(), (size_t) 8);
    }
    return done();
}
