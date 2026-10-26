#include "spectrum_analyzer.h"

using cfloat = std::complex<float>;

# define M_2PI		6.28318530717958647692	/* 2pi */
# define M_2PI_f	6.28318530717958647692f	/* 2pi */


inline bool is_power_of_two(std::size_t n)
{
    return n != 0 && ((n & (n - 1)) == 0);
}

// In-place radix-2 complex FFT.
inline void fft_radix2_inplace(cfloat* x, std::size_t n, bool inverse = false)
{
    assert(x != nullptr);
    std::cout<<"n=" << n << std::endl;
    assert(is_power_of_two(n));

    // Reorder input into bit-reversed order.
    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;

        while (j & bit) {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;

        if (i < j) {
            std::swap(x[i], x[j]);
        }
    }

    const float sign = inverse ? 1.0f : -1.0f;

    // Iterative decimation-in-time radix-2 FFT.
    for (std::size_t len = 2; len <= n; len <<= 1) {
        const std::size_t half = len >> 1;
        const float angle =
            sign * M_2PI_f / static_cast<float>(len);

        const cfloat wlen(std::cos(angle), std::sin(angle));

        for (std::size_t base = 0; base < n; base += len) {
            cfloat w(1.0f, 0.0f);

            for (std::size_t j = 0; j < half; ++j) {
                const cfloat u = x[base + j];
                const cfloat v = x[base + j + half] * w;

                x[base + j]        = u + v;
                x[base + j + half] = u - v;

                w *= wlen;
            }
        }
    }

    // Normalize only the inverse, so fft -> ifft restores x.
    if (inverse) {
        const float inv_n = 1.0f / static_cast<float>(n);

        for (std::size_t i = 0; i < n; ++i) {
            x[i] *= inv_n;
        }
    }
}

inline void fft_radix2_inplace(std::vector<cfloat>& x, bool inverse = false)
{
    fft_radix2_inplace(x.data(), x.size(), inverse);
}


std::array<std::array<float,fft_length>,4> fft_4ch_muxed_real(const float* muxed)
{
    // Deinterleave and promote real samples to complex samples.
    cfloat spectra[4][fft_length];
    std::array<std::array<float,fft_length>,4>  power_spectra{};
    std::cout<<"Creating FFT containers" << std::endl;
    for (std::size_t n = 0; n < fft_length; ++n) {
        const std::size_t src = 4 * n;
        // std::cout<<"muxed = " << muxed[src + 0] << std::endl;
        spectra[0][n] = cfloat(muxed[src + 0], 0.0f);
        spectra[1][n] = cfloat(muxed[src + 1], 0.0f);
        spectra[2][n] = cfloat(muxed[src + 2], 0.0f);
        spectra[3][n] = cfloat(muxed[src + 3], 0.0f);
    }

     for (std::size_t ch = 0; ch < 4; ++ch) {
        std::cout<<"FFT Start"<< std::endl;
        fft_radix2_inplace(spectra[ch], fft_length,false);
        for (std::size_t n = 0; n < fft_length; ++n) {
            power_spectra[ch][n] = std::abs(spectra[ch][n]);
            // std::cout<< "power_spectra[" << ch << "][" << n << "] = " << power_spectra[ch][n] << ", spectra real[" << ch << "][" << n << "] = " << std::real(spectra[ch][n]) << std::endl;
        }
    }
    return power_spectra;
}


void write_magnitude_file(const std::string& filename, std::array<std::array<float,fft_length>,4>  magnitude, std::vector<double> freqs, double timestamp) 
{

    std::cout<< "Saving spectra to file: " << filename << std::endl;
    std::ofstream out(filename.c_str(),
                        std::ios::binary | std::ios::app);

    if (!out) {
        throw std::runtime_error("Cannot open output file: " + filename);
    }
    out.write(
            reinterpret_cast<const char*>(&fft_length),
            static_cast<std::streamsize>(sizeof(size_t))
        );
    out.write(
            reinterpret_cast<const char*>(&timestamp),
            static_cast<std::streamsize>(sizeof(double))
        );
    
    out.write(
            reinterpret_cast<const char*>(freqs.data()),
            static_cast<std::streamsize>(fft_length * sizeof(double))
        );

    for (std::size_t ch = 0; ch < 4; ++ch) {
        out.write(
            reinterpret_cast<const char*>(magnitude[ch].data()),
            static_cast<std::streamsize>(fft_length * sizeof(float))
        );
    }

    if (!out) {
        throw std::runtime_error("Failed while writing: " + filename);
    }
}



std::vector<double> fftfreq(int n, double fs) {
    std::vector<double> freq(n);
    double val = fs / n ;
    
    int N = (n - 1) / 2 + 1;
    for (int i = 0; i < N; ++i) {
        freq[i] = i * val;
    }
    
    int M = n / 2;
    for (int i = 0; i < M; ++i) {
        freq[n - M + i] = -(M - i) * val;
    }
    std::cout<<"Max freq: "<<freq[n/2]<<std::endl;
    return freq;
}
