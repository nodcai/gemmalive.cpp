#pragma once

#include <string>
#include <vector>

// A classifier on Gemma's hidden states at the end of committed audio (see
// training/turn): whether the user finished their turn, or whether the
// speech is meant for the assistant. Mixes a few layers, normalises and
// applies a linear or small MLP head.
//
// Both heads for a model are in one GGUF file next to it, <model>-heads.gguf,
// or in the model's own (a combined file, training/turn/combine.py); see
// path_for(). A heads file records what the heads were trained on; load()
// refuses it if the model, audio projector or system prompt in use differ.
class TurnHead {
public:
    static constexpr const char * END_OF_TURN = "end_of_turn";
    static constexpr const char * ADDRESSEE   = "addressee";

    struct Expect {
        std::string model_path;
        std::string mmproj_path;
        std::string system_prompt;
    };

    // The file with a model's heads: models/x.gguf -> models/x-heads.gguf,
    // or the model's own when it has them.
    static std::string path_for(const std::string & model_path);

    // Loads the head for `task` (END_OF_TURN or ADDRESSEE) from a heads file,
    // or from the model's (`path` is expect.model_path).
    bool load(const std::string & path, const std::string & task, const Expect & expect);
    bool loaded() const { return n_embd_ > 0; }

    // The head's probability (the user finished their turn, or the speech is
    // meant for the assistant). `layer_states` is the n_layer x n_embd output
    // of Llm::commit_audio().
    float predict(const std::vector<float> & layer_states) const;

    float threshold() const { return threshold_; }

    // The layers predict() reads.
    const std::vector<int> & layers() const { return layers_; }

private:
    int                n_embd_ = 0;
    int                hidden_ = 0;
    std::vector<int>   layers_;
    std::vector<float> mix_;
    std::vector<float> mean_;
    std::vector<float> std_;
    std::vector<float> w1_, b1_, w2_;
    float              b2_ = 0.f;
    float              threshold_ = 0.5f;
};
