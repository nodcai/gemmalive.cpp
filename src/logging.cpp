#include "logging.h"

#include "livevoice.h"

#include <cstdio>

namespace {

bool g_verbose = false;

void log(lv_log_level level, const char * msg, void *) {
    if (g_verbose || level <= LV_LOG_WARN) {
        fputs(msg, stderr);
    }
}

}  // namespace

void init_logging(bool verbose) {
    g_verbose = verbose;
    // livevoice routes ggml's log through its own, which is shared by every
    // model here.
    lv_set_log_level(verbose ? LV_LOG_DEBUG : LV_LOG_WARN);
    lv_set_log_callback(log, nullptr);
}
