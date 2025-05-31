#ifndef _cudaFunctions_included_
#define _cudaFunctions_included_

# define M_2PI		6.28318530717958647692	/* pi */
# define M_2PI_f	6.28318530717958647692f	/* pi */

    // ----- CUDA includes -----
#include "server.h"
#include "../spcm_header/spcm_cuda_common.h"
#include <math.h>
#include <cuda_fp16.h>
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

extern drv_handle hCard;

extern void tone_counter(int dynamic);
extern int staticBufferInit();
extern size_t static_length;
extern size_t lBytesPerChannelInNotifySize;
extern size_t int_temp;
extern double double_temp;
extern float amp_map_static[16384];
extern float * amp_map_static_cuda;
extern float total_divider[4];
extern float * total_divider_cuda;
extern void StaticAmpMapper(double * __restrict__ frequency, const float * __restrict__ amp_map, float * amp_map_static, float *total_divider);
extern __global__ void StaticWaveGeneration (double* __restrict__ frequency, double* pnOut,double** sumOut);
extern __global__ void StaticWaveGeneration_amp (double* __restrict__ frequency, float* __restrict__ amp, double* pnOut,double** sumOut);
extern __global__ void StaticWaveGeneration_amp_mapped (double* __restrict__ frequency, float* __restrict__ amp_map_static,float* __restrict__ total_divider, double* pnOut,double** sumOut,double*phase_list);
extern __global__ void StaticWaveGeneration (double* __restrict__ frequency, double* pnOut,double** sumOut,double*phase_list);
extern __global__ void StaticWaveGeneration_amp (double* __restrict__ frequency, float* __restrict__ amp, double* pnOut,double** sumOut,double*phase_list);
extern __global__ void StaticWaveGeneration_single (double* __restrict__ frequency, double* pnOut,double** sumOut);
extern __global__ void StaticWaveGeneration_update (double* __restrict__ frequency, double* pnOut,double** sumOut,double* __restrict__ phase_list);
extern __global__ void printer (double* __restrict__ list,int count);
extern __global__ void printer (short* __restrict__ list,int count);

extern __global__ void StaticWaveGeneration_update_amp (double* __restrict__ frequency, float* __restrict__ amp, double* pnOut,double** sumOut,double* __restrict__ phase_list);

extern __global__ void phase_reorder_update(int* __restrict__ indexmap,double*__restrict__ newphaselist,double*phaselist,int length);
extern __global__ void StaticMux ( double** __restrict__ buffer,short* pnOut);

extern __device__ __constant__ unsigned int static_num_cuda[4];
extern __device__ __constant__ int    channel_num_cuda;
extern __device__ __constant__ double illSamplerate_cuda;
extern __device__ __constant__ double idynamic_bufferlength_cuda;
extern __device__ __constant__ float map_interval_cuda;
extern __device__ __constant__ size_t static_bufferlength_cuda;
extern __device__ __constant__ int    dynamic_num_cuda[4];
extern __device__ __constant__ double dynamic_bufferlength_cuda;
extern __device__ __constant__ double ipower_normalizer_cuda[4];
extern __device__ __constant__ double dynamic_loopcount_cuda;
extern cudaError_t eCudaErr;
extern __device__ double istatic_num_cuda[4];
extern double* summed_buffer[4];
extern double* saved_buffer[4];
extern short* dynamic_saved_buffer[4];
extern double** summed_buffer_cuda;
extern double** saved_buffer_cuda;
extern double* static_buffer_cuda;
extern double * real_static_freq_cuda;
extern double real_static_freq[16384];
extern unsigned int dynamic_total;
extern unsigned int static_total;
extern float * amp_map_cuda;
extern __device__ __constant__ unsigned int tone_count_cuda[5];
extern __device__ __constant__ unsigned int dynamic_tone_count_cuda[5];
// ------Dynamics----------------------------
extern double real_destination_freq[16384];
extern unsigned int dynamic_buffersize;
extern double* real_destination_freq_cuda;
extern int * dynamic_list_cuda;
extern int * static_list_cuda;
extern float * amp_list_cuda;
extern float * final_amp_list_cuda;
extern double * phase_list_cuda;
extern double * new_phase_list_cuda;
extern int * update_index_map_cuda;

extern int dynamic_loopcount;
extern bool not_arrived;
#endif