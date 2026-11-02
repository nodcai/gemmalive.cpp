#pragma once

#include "ggml-backend.h"

#include <vector>

// The device the models run on, and a CPU backend for what it can't run.
//
// GEMMALIVE_BACKEND=cpu (or the name of a device, e.g. "metal") picks one;
// otherwise the first GPU, else the CPU.
class Backend {
public:
    Backend() = default;
    ~Backend();
    Backend(const Backend &) = delete;
    Backend & operator=(const Backend &) = delete;

    bool init(int n_threads);

    ggml_backend_t     main() const { return main_; }
    ggml_backend_t     cpu() const { return cpu_; }
    ggml_backend_dev_t device() const { return ggml_backend_get_device(main_); }
    bool               is_cpu() const { return main_ == cpu_; }
    const char *       name() const;

    // A scheduler over the device and the CPU, for graphs of up to
    // `graph_size` nodes.
    ggml_backend_sched_t new_sched(size_t graph_size) const;

private:
    ggml_backend_t main_ = nullptr;
    ggml_backend_t cpu_  = nullptr;
};
