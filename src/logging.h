#pragma once

// Sends the logs of ggml and livevoice to stderr: warnings and errors, or
// with `verbose` everything.
void init_logging(bool verbose = false);
