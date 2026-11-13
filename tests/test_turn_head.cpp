// The heads: their file's fingerprint check, and predict() against the
// formula, on small files written here.
#include "check.h"
#include "internal.h"
#include "turn_head.h"

#include "ggml.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace internal;

namespace fs = std::filesystem;

// A stand-in for a model: any GGUF file has a header to fingerprint.
static void write_model(const std::string & path, const char * name) {
    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.name", name);
    gguf_write_to_file(g, path.c_str(), /*only_meta=*/false);
    gguf_free(g);
}

// What TurnHead checks a model by: the SHA-256 of its header.
static std::string fingerprint(const std::string & path) {
    gguf_init_params params = { /*no_alloc=*/true, /*ctx=*/nullptr };
    gguf_context * g = gguf_init_from_file(path.c_str(), params);
    const size_t n = gguf_get_data_offset(g);
    gguf_free(g);
    std::vector<unsigned char> header(n);
    std::ifstream(path, std::ios::binary).read((char *) header.data(), (std::streamsize) n);
    return sha256_hex(header.data(), n);
}

struct Head {
    std::string        task;
    std::vector<int>   layers;
    std::vector<float> mix, mean, std, w1, b1, w2;
    float              b2, threshold;
};

static void write_heads(const std::string & path, const std::string & model, const std::string & mmproj,
                        const std::string & prompt, bool any_prompt, const std::vector<Head> & heads,
                        const char * arch = "gemmalive-heads") {
    ggml_init_params params = { 1 << 20, nullptr, false };
    ggml_context * ctx = ggml_init(params);
    gguf_context * g = gguf_init_empty();
    gguf_set_val_str(g, "general.architecture", arch);
    gguf_set_val_str(g, "heads.model.fingerprint", fingerprint(model).c_str());
    gguf_set_val_str(g, "heads.mmproj.fingerprint", fingerprint(mmproj).c_str());
    gguf_set_val_str(g, "heads.system_prompt", prompt.c_str());
    gguf_set_val_bool(g, "heads.any_system_prompt", any_prompt);
    std::vector<const char *> tasks;
    for (const Head & h : heads) {
        tasks.push_back(h.task.c_str());
    }
    gguf_set_arr_str(g, "heads.tasks", tasks.data(), tasks.size());
    for (const Head & h : heads) {
        const std::string key = "heads." + h.task + ".";
        const std::vector<int32_t> layers(h.layers.begin(), h.layers.end());
        gguf_set_arr_data(g, (key + "layers").c_str(), GGUF_TYPE_INT32, layers.data(), layers.size());
        gguf_set_val_u32(g, (key + "n_embd").c_str(), (uint32_t) h.mean.size());
        gguf_set_val_u32(g, (key + "hidden").c_str(), (uint32_t) h.b1.size());
        gguf_set_val_f32(g, (key + "threshold").c_str(), h.threshold);
        const auto add = [&](const char * name, const std::vector<float> & values) {
            if (values.empty()) {
                return;
            }
            ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, (int64_t) values.size());
            ggml_set_name(t, (h.task + "." + name).c_str());
            std::copy(values.begin(), values.end(), (float *) t->data);
            gguf_add_tensor(g, t);
        };
        add("mix", h.mix);
        add("mean", h.mean);
        add("std", h.std);
        add("fc1.weight", h.w1);
        add("fc1.bias", h.b1);
        add("fc2.weight", h.w2);
        add("fc2.bias", { h.b2 });
    }
    gguf_write_to_file(g, path.c_str(), /*only_meta=*/false);
    gguf_free(g);
    ggml_free(ctx);
}

// predict() as the training script's head computes it.
static float expect(const Head & h, const std::vector<float> & states) {
    const size_t n_embd = h.mean.size();
    std::vector<double> x(n_embd, 0.0);
    for (size_t i = 0; i < h.layers.size(); i++) {
        for (size_t e = 0; e < n_embd; e++) {
            x[e] += (double) h.mix[i] * states[(size_t) h.layers[i] * n_embd + e];
        }
    }
    for (size_t e = 0; e < n_embd; e++) {
        x[e] = (x[e] - h.mean[e]) / h.std[e];
    }
    double logit = h.b2;
    if (h.b1.empty()) {
        for (size_t e = 0; e < n_embd; e++) {
            logit += h.w2[e] * x[e];
        }
    } else {
        for (size_t j = 0; j < h.b1.size(); j++) {
            double a = h.b1[j];
            for (size_t e = 0; e < n_embd; e++) {
                a += h.w1[j * n_embd + e] * x[e];
            }
            logit += h.w2[j] * 0.5 * a * (1.0 + std::erf(a / std::sqrt(2.0)));
        }
    }
    return (float) (1.0 / (1.0 + std::exp(-logit)));
}

int main() {
    // SHA-256 (FIPS 180-4 examples, and lengths around a block's padding).
    const auto sha = [](const std::string & s) { return sha256_hex((const unsigned char *) s.data(), s.size()); };
    CHECK_EQ(sha(""), std::string("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"));
    CHECK_EQ(sha("abc"), std::string("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"));
    CHECK_EQ(sha("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
             std::string("248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"));
    CHECK_EQ(sha(std::string(55, 'a')), std::string("9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318"));
    CHECK_EQ(sha(std::string(56, 'a')), std::string("b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a"));
    CHECK_EQ(sha(std::string(64, 'a')), std::string("ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb"));
    CHECK_EQ(sha(std::string(1000000, 'a')), std::string("cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"));

    CHECK_EQ(TurnHead::path_for("models/gemma.gguf"), std::string("models/gemma-heads.gguf"));
    CHECK_EQ(TurnHead::path_for("gemma"), std::string("gemma-heads.gguf"));

    const fs::path dir = fs::temp_directory_path() / ("gemmalive-test-heads-" + std::to_string((long long) std::time(nullptr)));
    fs::create_directories(dir);
    const std::string model = (dir / "model.gguf").string(), mmproj = (dir / "mmproj.gguf").string();
    const std::string other = (dir / "other.gguf").string(), heads = (dir / "model-heads.gguf").string();
    write_model(model, "a model");
    write_model(mmproj, "its audio projector");
    write_model(other, "another model");

    // Five layers of four values; a linear head on two of them and a small
    // MLP on three.
    const Head turn = { TurnHead::END_OF_TURN, { 1, 3 }, { 0.25f, 0.75f }, { 0.1f, -0.2f, 0.3f, 0.f }, { 1.f, 2.f, 0.5f, 1.5f },
                        {}, {}, { 0.7f, -1.1f, 0.4f, 0.9f }, -0.3f, 0.45f };
    const Head addressee = { TurnHead::ADDRESSEE, { 0, 2, 4 }, { 0.5f, 0.3f, 0.2f }, { 0.f, 0.5f, -0.5f, 1.f }, { 2.f, 1.f, 1.f, 0.5f },
                             { 0.2f, -0.4f, 0.6f, 0.1f, -0.3f, 0.5f, 0.7f, -0.2f, 0.9f, 0.3f, 0.3f, -0.8f }, { 0.1f, -0.1f, 0.2f },
                             { 1.2f, -0.7f, 0.5f }, 0.2f, 0.3f };
    std::vector<float> states(5 * 4);
    for (size_t i = 0; i < states.size(); i++) {
        states[i] = std::sin((float) i * 1.3f) * 2.f;
    }

    write_heads(heads, model, mmproj, "Be brief.", false, { turn, addressee });
    const TurnHead::Expect ours = { model, mmproj, "Be brief." };
    {
        TurnHead h;
        CHECK(!h.loaded());
        CHECK(h.load(heads, TurnHead::END_OF_TURN, ours));
        CHECK(h.loaded());
        CHECK(h.layers() == turn.layers);
        CHECK_EQ(h.threshold(), 0.45f);
        CHECK(std::fabs(h.predict(states) - expect(turn, states)) < 1e-5f);
    }
    {
        TurnHead h;
        CHECK(h.load(heads, TurnHead::ADDRESSEE, ours));
        CHECK(h.layers() == addressee.layers);
        CHECK_EQ(h.threshold(), 0.3f);
        CHECK(std::fabs(h.predict(states) - expect(addressee, states)) < 1e-5f);
        CHECK(std::fabs(h.predict(states) - expect(turn, states)) > 1e-3f);
    }

    // Heads are for the files and the prompt they were trained on.
    fprintf(stderr, "(the refusals that follow are expected)\n");
    {
        TurnHead h;
        CHECK(!h.load(heads, TurnHead::END_OF_TURN, { other, mmproj, "Be brief." }));
        CHECK(!h.loaded());
        CHECK(!h.load(heads, TurnHead::END_OF_TURN, { model, other, "Be brief." }));
        CHECK(!h.load(heads, TurnHead::END_OF_TURN, { model, mmproj, "Be long." }));
        CHECK(!h.load(heads, "no_such_task", ours));
        CHECK(!h.load((dir / "missing.gguf").string(), TurnHead::END_OF_TURN, ours));
        CHECK(!h.load(model, TurnHead::END_OF_TURN, ours));
        CHECK(!h.loaded());
    }
    // ... unless they were trained on many prompts.
    {
        const std::string any = (dir / "any-heads.gguf").string();
        write_heads(any, model, mmproj, "", true, { turn });
        TurnHead h;
        CHECK(h.load(any, TurnHead::END_OF_TURN, { model, mmproj, "Be long." }));
        CHECK(!h.load(any, TurnHead::ADDRESSEE, ours));
        CHECK(!h.load(any, TurnHead::END_OF_TURN, { other, mmproj, "Be long." }));
    }
    {
        const std::string wrong = (dir / "wrong-heads.gguf").string();
        write_heads(wrong, model, mmproj, "Be brief.", false, { turn }, "something-else");
        TurnHead h;
        CHECK(!h.load(wrong, TurnHead::END_OF_TURN, ours));
    }

    // A model's file may have its heads in it: those are taken as they are.
    {
        const std::string combined = (dir / "combined.gguf").string();
        write_heads(combined, other, other, "", true, { turn, addressee }, "a-model");
        CHECK_EQ(TurnHead::path_for(combined), combined);
        CHECK_EQ(TurnHead::path_for(model), heads);
        TurnHead h;
        CHECK(h.load(combined, TurnHead::ADDRESSEE, { combined, "", "Be long." }));
        CHECK(std::fabs(h.predict(states) - expect(addressee, states)) < 1e-5f);
        CHECK(!h.load(combined, TurnHead::ADDRESSEE, ours));
    }

    fs::remove_all(dir);
    return done();
}
