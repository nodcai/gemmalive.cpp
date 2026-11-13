#pragma once

#include <cstdio>
#include <sstream>
#include <string>

// CHECK(condition) and CHECK_EQ(got, want) report a failure with its line
// and go on; a test's main() returns done().

static int g_checks = 0;
static int g_failed = 0;

#define CHECK(condition)                                                              \
    do {                                                                              \
        g_checks++;                                                                   \
        if (!(condition)) {                                                           \
            g_failed++;                                                               \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition);      \
        }                                                                             \
    } while (0)

#define CHECK_EQ(got, want)                                                           \
    do {                                                                              \
        g_checks++;                                                                   \
        const auto & got_ = (got);                                                    \
        const auto & want_ = (want);                                                  \
        if (!(got_ == want_)) {                                                       \
            g_failed++;                                                               \
            std::ostringstream os_;                                                   \
            os_ << "got  " << got_ << "\n     want " << want_;                        \
            fprintf(stderr, "FAIL %s:%d: %s\n     %s\n", __FILE__, __LINE__, #got, os_.str().c_str()); \
        }                                                                             \
    } while (0)

static int done() {
    printf("%d checks, %d failed\n", g_checks, g_failed);
    return g_failed ? 1 : 0;
}

// What a test with a model needs, or exit code 77: skipped.
#define NEED_ENV(variable, name)                                                      \
    const char * variable = getenv(name);                                             \
    if (!variable || !*variable) {                                                    \
        fprintf(stderr, "skip: set %s\n", name);                                      \
        return 77;                                                                    \
    }
