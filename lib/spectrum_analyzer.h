#ifndef _spectrum_analyzer_included_
#define _spectrum_analyzer_included_

#include <algorithm>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <numbers>
#include <vector>
#include <fstream>
#include <iostream>

constexpr size_t fft_length = 4096; // s*Hz; better to be a power of 2
constexpr double fft_sample_interval = 0.001; // The spectrum scan interval, in seconds

extern std::array<std::array<float,fft_length>,4> fft_4ch_muxed_real(const float* muxed);
// float (* fft_4ch_muxed_real(
//     const float* muxed, std::size_t samples_per_channel))[fft_length];
extern std::vector<double> fftfreq(int n, double fs);
// extern void write_magnitude_file(const std::string& filename, float magnitude[4][fft_length], std::vector<double> freqs, double timestamp);
extern void write_magnitude_file(const std::string& filename, std::array<std::array<float,fft_length>,4>  magnitude, std::vector<double> freqs, double timestamp);


#endif