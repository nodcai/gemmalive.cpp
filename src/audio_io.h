#pragma once

#include <string>
#include <vector>

// Decodes a WAV, MP3 or FLAC file to mono floats at `sample_rate`.
bool read_audio(const std::string & path, int sample_rate, std::vector<float> & out);

// Decodes a file as it is: its own rate, all channels interleaved.
bool read_audio_raw(const std::string & path, std::vector<float> & out, int & sample_rate, int & channels);

// Writes mono floats as a 16-bit WAV.
bool write_wav(const std::string & path, const float * pcm, size_t n, int sample_rate);
