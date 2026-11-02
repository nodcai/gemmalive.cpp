#pragma once

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

// A GGUF file, memory-mapped. Its tensors stay in the file's pages: a device
// that can address host memory runs on them where they are, others get a
// copy. Tensors that are not loaded can still be read on the host, a row at
// a time (row()), which only brings the pages of those rows in.
class ModelFile {
public:
    ModelFile() = default;
    ~ModelFile();
    ModelFile(const ModelFile &) = delete;
    ModelFile & operator=(const ModelFile &) = delete;

    bool open(const std::string & path);

    // Puts the tensors `want` picks on `dev`.
    bool load(ggml_backend_dev_t dev, const std::function<bool(const std::string &)> & want);

    // A loaded tensor, or nullptr.
    ggml_tensor * get(const std::string & name) const;
    // A tensor of the file, loaded or not.
    const ggml_tensor * info(const std::string & name) const;

    // Row `i` of a 2D tensor of the file, as floats.
    void row(const ggml_tensor * t, int64_t i, float * out) const;
    // Asks the OS to bring these rows in, all at once.
    void prefetch(const ggml_tensor * t, const int32_t * rows, int n) const;

    // A scalar tensor's value, or `def` without it.
    float scalar(const std::string & name, float def) const;

    gguf_context * gguf() const { return gguf_; }
    int64_t     find_key(const char * key) const;
    uint32_t    u32(const char * key, uint32_t def = 0) const;
    float       f32(const char * key, float def = 0.f) const;
    std::string str(const char * key) const;

private:
    const uint8_t * data(const ggml_tensor * t) const;

    std::string    path_;
    gguf_context * gguf_ = nullptr;
    ggml_context * meta_ = nullptr;  // the file's tensors, without data
    ggml_context * ctx_  = nullptr;  // the loaded ones
    void *         map_  = nullptr;
    size_t         size_ = 0;
    size_t         data_offset_ = 0;
    std::vector<ggml_backend_buffer_t> buffers_;
    std::unordered_map<std::string, ggml_tensor *> loaded_;
};
