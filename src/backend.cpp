#include "backend.h"

#include "ggml-cpu.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

static std::string lower(const char * s) {
    std::string r = s ? s : "";
    for (char & c : r) {
        c = (char) std::tolower((unsigned char) c);
    }
    return r;
}

Backend::~Backend() {
    if (main_ && main_ != cpu_) {
        ggml_backend_free(main_);
    }
    if (cpu_) {
        ggml_backend_free(cpu_);
    }
}

bool Backend::init(int n_threads) {
#if defined(__APPLE__) && !TARGET_OS_OSX
    // On iOS, Metal can't wrap large host allocations without copying them,
    // and fails at load; private buffers cost nothing on unified memory.
    setenv("GGML_METAL_SHARED_BUFFERS_DISABLE", "1", 0);
#endif
    ggml_backend_load_all();

    cpu_ = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!cpu_) {
        fprintf(stderr, "backend: no CPU backend\n");
        return false;
    }
    if (n_threads <= 0) {
        n_threads = (int) std::max(1u, std::thread::hardware_concurrency() / 2);
    }
    ggml_backend_cpu_set_n_threads(cpu_, n_threads);

    const std::string want = lower(std::getenv("GEMMALIVE_BACKEND"));
    if (want != "cpu") {
        for (size_t i = 0; i < ggml_backend_dev_count() && !main_; i++) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            const bool match = want.empty()
                ? ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU
                : lower(ggml_backend_dev_name(dev)).find(want) != std::string::npos;
            if (match) {
                main_ = ggml_backend_dev_init(dev, nullptr);
            }
        }
        if (!main_ && !want.empty()) {
            fprintf(stderr, "backend: no device matches GEMMALIVE_BACKEND=%s, using the CPU\n", want.c_str());
        }
    }
    if (!main_) {
        main_ = cpu_;
    }
    return true;
}

const char * Backend::name() const {
    return main_ ? ggml_backend_name(main_) : "none";
}

ggml_backend_sched_t Backend::new_sched(size_t graph_size) const {
    std::vector<ggml_backend_t> backends = { main_ };
    if (main_ != cpu_) {
        backends.push_back(cpu_);
    }
    return ggml_backend_sched_new(backends.data(), nullptr, (int) backends.size(), graph_size,
                                  /*parallel=*/false, /*op_offload=*/true);
}
