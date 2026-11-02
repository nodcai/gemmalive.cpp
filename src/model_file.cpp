#include "model_file.h"

#include <algorithm>
#include <cfloat>
#include <cstdio>
#include <cstring>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

ModelFile::~ModelFile() {
    for (ggml_backend_buffer_t b : buffers_) {
        ggml_backend_buffer_free(b);
    }
    if (ctx_)  ggml_free(ctx_);
    if (meta_) ggml_free(meta_);
    if (gguf_) gguf_free(gguf_);
#ifdef _WIN32
    if (map_) UnmapViewOfFile(map_);
#else
    if (map_) munmap(map_, size_);
#endif
}

bool ModelFile::open(const std::string & path) {
    path_ = path;
    gguf_init_params params = { /*no_alloc=*/true, /*ctx=*/&meta_ };
    gguf_ = gguf_init_from_file(path.c_str(), params);
    if (!gguf_) {
        fprintf(stderr, "model: cannot read %s\n", path.c_str());
        return false;
    }
    data_offset_ = gguf_get_data_offset(gguf_);

#ifdef _WIN32
    HANDLE file = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        return false;
    }
    LARGE_INTEGER size;
    GetFileSizeEx(file, &size);
    size_ = (size_t) size.QuadPart;
    HANDLE mapping = CreateFileMappingA(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    CloseHandle(file);
    if (!mapping) {
        return false;
    }
    map_ = MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(mapping);
#else
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        fprintf(stderr, "model: cannot open %s\n", path.c_str());
        return false;
    }
    struct stat st;
    fstat(fd, &st);
    size_ = (size_t) st.st_size;
    map_  = mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);
    if (map_ == MAP_FAILED) {
        map_ = nullptr;
    }
#endif
    if (!map_) {
        fprintf(stderr, "model: cannot map %s\n", path.c_str());
        return false;
    }
    return true;
}

const uint8_t * ModelFile::data(const ggml_tensor * t) const {
    const int64_t id = gguf_find_tensor(gguf_, t->name);
    return (const uint8_t *) map_ + data_offset_ + gguf_get_tensor_offset(gguf_, id);
}

bool ModelFile::load(ggml_backend_dev_t dev, const std::function<bool(const std::string &)> & want) {
    std::vector<const ggml_tensor *> picked;
    for (ggml_tensor * t = ggml_get_first_tensor(meta_); t; t = ggml_get_next_tensor(meta_, t)) {
        if (want(t->name)) {
            picked.push_back(t);
        }
    }
    std::sort(picked.begin(), picked.end(), [&](auto * a, auto * b) { return data(a) < data(b); });

    ggml_init_params params = { ggml_tensor_overhead() * (picked.size() + 1), nullptr, /*no_alloc=*/true };
    ctx_ = ggml_init(params);
    std::vector<ggml_tensor *> tensors;
    for (const ggml_tensor * t : picked) {
        ggml_tensor * copy = ggml_new_tensor(ctx_, t->type, GGML_MAX_DIMS, t->ne);
        ggml_set_name(copy, t->name);
        tensors.push_back(copy);
        loaded_[t->name] = copy;
    }
    if (tensors.empty()) {
        return true;
    }

    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(dev, &props);
    if (!props.caps.buffer_from_host_ptr) {
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx_, ggml_backend_dev_buffer_type(dev));
        if (!buf) {
            fprintf(stderr, "model: out of device memory for %s\n", path_.c_str());
            return false;
        }
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        buffers_.push_back(buf);
        for (size_t i = 0; i < tensors.size(); i++) {
            ggml_backend_tensor_set(tensors[i], data(picked[i]), 0, ggml_nbytes(tensors[i]));
        }
        return true;
    }

    // One buffer per run of picked tensors, so that the ones in between stay
    // out of the device's working set.
    const size_t page = 16384;
    size_t i = 0;
    while (i < picked.size()) {
        size_t j = i + 1;
        size_t max_tensor = ggml_nbytes(picked[i]);
        while (j < picked.size() && data(picked[j]) - (data(picked[j - 1]) + ggml_nbytes(picked[j - 1])) < (ptrdiff_t) page) {
            max_tensor = std::max(max_tensor, ggml_nbytes(picked[j]));
            j++;
        }
        const uint8_t * base  = (const uint8_t *) map_;
        const size_t    first = (size_t) (data(picked[i]) - base) / page * page;
        const size_t    last  = (size_t) (data(picked[j - 1]) - base) + ggml_nbytes(picked[j - 1]);
        ggml_backend_buffer_t buf = ggml_backend_dev_buffer_from_host_ptr(dev, (void *) (base + first), last - first, max_tensor);
        if (!buf) {
            fprintf(stderr, "model: cannot map %s for the device\n", path_.c_str());
            return false;
        }
        ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        buffers_.push_back(buf);
        for (size_t k = i; k < j; k++) {
            if (ggml_backend_tensor_alloc(buf, tensors[k], (void *) data(picked[k])) != GGML_STATUS_SUCCESS) {
                fprintf(stderr, "model: cannot place %s\n", picked[k]->name);
                return false;
            }
        }
        i = j;
    }
    return true;
}

ggml_tensor * ModelFile::get(const std::string & name) const {
    auto it = loaded_.find(name);
    return it == loaded_.end() ? nullptr : it->second;
}

const ggml_tensor * ModelFile::info(const std::string & name) const {
    return ggml_get_tensor(meta_, name.c_str());
}

void ModelFile::row(const ggml_tensor * t, int64_t i, float * out) const {
    const uint8_t * src = data(t) + i * t->nb[1];
    if (t->type == GGML_TYPE_F32) {
        memcpy(out, src, t->ne[0] * sizeof(float));
    } else {
        ggml_get_type_traits(t->type)->to_float(src, out, t->ne[0]);
    }
}

void ModelFile::prefetch(const ggml_tensor * t, const int32_t * rows, int n) const {
#ifndef _WIN32
    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    for (int k = 0; k < n; k++) {
        const uintptr_t start = (uintptr_t) (data(t) + (size_t) rows[k] * t->nb[1]);
        const uintptr_t first = start / page * page;
        madvise((void *) first, start + t->nb[1] - first, MADV_WILLNEED);
    }
#else
    (void) t; (void) rows; (void) n;
#endif
}

float ModelFile::scalar(const std::string & name, float def) const {
    const ggml_tensor * t = info(name);
    if (!t || t->type != GGML_TYPE_F32 || ggml_nelements(t) != 1) {
        return def;
    }
    float v;
    memcpy(&v, data(t), sizeof(v));
    return v;
}

int64_t ModelFile::find_key(const char * key) const {
    return gguf_find_key(gguf_, key);
}

uint32_t ModelFile::u32(const char * key, uint32_t def) const {
    const int64_t k = find_key(key);
    if (k < 0) {
        return def;
    }
    switch (gguf_get_kv_type(gguf_, k)) {
        case GGUF_TYPE_UINT32: return gguf_get_val_u32(gguf_, k);
        case GGUF_TYPE_INT32:  return (uint32_t) gguf_get_val_i32(gguf_, k);
        default:               return def;
    }
}

float ModelFile::f32(const char * key, float def) const {
    const int64_t k = find_key(key);
    return k >= 0 && gguf_get_kv_type(gguf_, k) == GGUF_TYPE_FLOAT32 ? gguf_get_val_f32(gguf_, k) : def;
}

std::string ModelFile::str(const char * key) const {
    const int64_t k = find_key(key);
    return k >= 0 && gguf_get_kv_type(gguf_, k) == GGUF_TYPE_STRING ? gguf_get_val_str(gguf_, k) : "";
}
