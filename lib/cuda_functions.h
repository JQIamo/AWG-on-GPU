#ifndef _cudaFunctions_included_
#define _cudaFunctions_included_

# define M_2PI		6.28318530717958647692	/* 2pi */
# define M_2PI_f	6.28318530717958647692f	/* 2pi */

    // ----- CUDA includes -----
#include "server.h"
#include "../spcm_header/spcm_cuda_common.h"
#include <math.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "../spcm_header/dlltyp.h"
#include "../spcm_header/regs.h"
#include "../spcm_header/spcerr.h"
#include "../spcm_header/spcm_drv.h"
#include "amp_map.h"
// ----- standard c include files -----
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream> 

// Call a CUDA API function and check for errors
#define CUDA_CHECK(call)                                      \
do {                                                          \
    cudaError_t err = call;                                   \
    if (__builtin_expect(err != cudaSuccess,0)) {             \
        std::cerr << "CUDA Error: "                           \
        << cudaGetErrorString(err)                            \
        << " at " << __FILE__ << ":" << __LINE__              \
        << std::endl;                                         \
        throw std::runtime_error(cudaGetErrorString(err));    \
    }                                                         \
} while (0)

extern drv_handle hCard;

extern void tone_counter(int dynamic);
extern int staticBufferInit();
extern int static_length;
extern size_t lBytesPerChannelInNotifySize;
extern size_t int_temp;
extern double double_temp;
extern float amp_map_static[4*MAX_TWEEZERS_PER_CHANNEL];
extern float * amp_map_static_cuda;
extern float total_divider[4];
extern float * total_divider_cuda;
extern void StaticAmpMapper(float * __restrict__ frequency, const float * __restrict__ amp_map, float * amp_map_static, float *total_divider);
// extern __device__ double cos2pi_fast(double x);
// extern __device__ double sin2pi_fast(double x);
extern __device__ double modfeq(double x);
extern __device__ float modfeqf(float x);

__device__ __forceinline__ double cos2pi_fast(double x){
    return cospi(2. * modfeq(x));
}

__device__ __forceinline__ double sin2pi_fast(double x){
    return sinpi(2. * modfeq(x));
}

__device__ __forceinline__ float cos2pi_fastf(float x){
    return cospif(2.f * modfeqf(x));
}

__device__ __forceinline__ float sin2pi_fastf(float x){
    return sinpif(2.f * modfeqf(x));
}

extern __global__ void StaticWaveGeneration (float* __restrict__ frequency, double* pnOut,double** sumOut);
extern __global__ void StaticWaveGeneration_amp (float* __restrict__ frequency, float* __restrict__ amp, double* pnOut,double** sumOut);
extern __global__ void StaticWaveGeneration_amp_mapped (float* __restrict__ frequency, float* __restrict__ amp_map_static,float* __restrict__ total_divider, double* pnOut,double** sumOut,float*phase_list);
extern __global__ void StaticWaveGeneration (float* __restrict__ frequency, double* pnOut,double** sumOut,float*phase_list);
extern __global__ void StaticWaveGeneration_amp (float* __restrict__ frequency, float* __restrict__ amp, double* pnOut,double** sumOut,float*phase_list);
extern __global__ void StaticWaveGeneration_single (float* __restrict__ frequency, double* pnOut,double** sumOut);
extern __global__ void StaticWaveGeneration_update (float* __restrict__ frequency, double* pnOut,double** sumOut,float* __restrict__ phase_list);
extern __global__ void ShakeWaveGeneration (float* __restrict__ frequency, double shake_amp, double shake_freq, double* pnOut,double** sumOut,float*phase_list);
extern __global__ void ShakeWaveGeneration_amp (float* __restrict__ frequency, float* __restrict__ amp, double shake_amp, double shake_freq, double* pnOut,double** sumOut,float*phase_list);
extern __global__ void SingleStaticRecombiner(int ch, int *__restrict__ single_static_map_cuda, double* __restrict__ pnOut,double** sumOut);
extern __global__ void printer (double* __restrict__ list,int count);
extern __global__ void printer (short* __restrict__ list,int count);

extern __global__ void StaticWaveGeneration_update_amp (float* __restrict__ frequency, float* __restrict__ amp, double* pnOut,double** sumOut,float* __restrict__ phase_list);

extern __global__ void phase_reorder_update(int* __restrict__ indexmap,float*__restrict__ newphaselist,float*phaselist,int length);
extern __global__ void StaticMux ( double** __restrict__ buffer,short* pnOut);

extern __device__ __constant__ unsigned int static_num_cuda[4];
extern __device__ __constant__ int    channel_num_cuda;
extern __device__ __constant__ double illSamplerate_cuda;
extern __device__ __constant__ double idynamic_bufferlength_cuda;
extern __device__ __constant__ float map_interval_cuda;
extern __device__ __constant__ unsigned int static_bufferlength_cuda;
extern __device__ __constant__ int    dynamic_num_cuda[4];
extern __device__ __constant__ unsigned int dynamic_bufferlength_cuda;
extern __device__ __constant__ float ipower_normalizer_cuda[4];
extern __device__ __constant__ float imaximal_tweezer_number_cuda[4];
extern __device__ __constant__ int dynamic_loopcount_cuda;
extern cudaError_t eCudaErr;
extern __device__ double istatic_num_cuda[4];
extern double* summed_buffer[4];
extern double* saved_buffer[4];
extern short* dynamic_saved_buffer[4];
extern double** summed_buffer_cuda;
extern double** saved_buffer_cuda;
extern double* static_buffer_cuda;
extern float * real_static_freq_cuda;
extern float real_static_freq[4*MAX_TWEEZERS_PER_CHANNEL];
extern unsigned int dynamic_total;
extern unsigned int static_total;
extern float * amp_map_cuda;
extern __device__ __constant__ unsigned int tone_count_cuda[5];
extern __device__ __constant__ unsigned int dynamic_tone_count_cuda[5];
// ------Dynamics----------------------------
extern float real_destination_freq[4*MAX_TWEEZERS_PER_CHANNEL];
extern float* real_destination_freq_cuda;
extern int * dynamic_list_cuda;
extern int * static_list_cuda;
extern float * amp_list_cuda;
extern float * final_amp_list_cuda;
extern float * phase_list_cuda;
extern float * new_phase_list_cuda;
extern int * update_index_map_cuda;
extern int * single_static_map_cuda;
extern int * single_dynamic_map_cuda;
extern float * rearrange_freq_list_cuda;

extern int dynamic_loopcount;
extern bool not_arrived;
#endif