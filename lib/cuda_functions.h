#ifndef _cudaFunctions_included_
#define _cudaFunctions_included_

    // ----- CUDA includes -----
#include "parameters.h"
#include "../spcm_header/spcm_cuda_common.h"
#include <math.h>
#include <cuda_fp16.h>
#include "../spcm_header/dlltyp.h"
#include "../spcm_header/regs.h"
#include "../spcm_header/spcerr.h"
#include "../spcm_header/spcm_drv.h"
// ----- standard c include files -----
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream> 

extern drv_handle hCard;

extern void tone_counter(int dynamic);
extern void cuda_cleanup ();
extern int staticBufferMalloc();
extern int dynamicBufferMalloc();
extern __global__ void tester();
extern size_t static_length;
extern size_t lBytesPerChannelInNotifySize;
extern size_t int_temp;
extern double double_temp;
extern const int lNumCh;
extern __global__ void StaticWaveGeneration (double* __restrict__ frequency, double* pnOut,short** sumOut);
extern __global__ void StaticWaveGeneration_single (double* __restrict__ frequency, double* pnOut,short** sumOut);
// extern __global__ void StaticCombine(double*__restrict__ buffer,short**sum_buf);
extern __global__ void StaticMux ( short** __restrict__ buffer,short* pnOut);
extern __global__ void DynamicMux (unsigned int startPosition, short**__restrict__ buffer,short* pnOut);
extern __global__ void WaveformCopier (short* __restrict__ buffer,short* pnOut);
extern __global__ void Pre_computer(double * __restrict__ static_buf, int* __restrict__ static_list, double* __restrict__ ddest_freq, 
                            int*__restrict__ ddy_list, short* final_buf, short** dynamic_buf,double * __restrict__ dstartFreq);
extern __device__ __constant__ unsigned int static_num_cuda[4];
extern __device__ __constant__ int    channel_num_cuda;
extern __device__ __constant__ double llSamplerate_cuda;
extern __device__ __constant__ double illSamplerate_cuda;
extern __device__ __constant__ double idynamic_bufferlength_cuda;
extern __device__ __constant__ size_t static_bufferlength_cuda;
extern __device__ __constant__ int    dynamic_num_cuda[4];
extern __device__ __constant__ double dynamic_bufferlength_cuda;
extern __device__ __constant__ double dynamic_loopcount_cuda;
extern cudaError_t eCudaErr;
extern __device__ double istatic_num_cuda[4];
extern short* summed_buffer[4];
extern short* saved_buffer[4];
extern short* dynamic_saved_buffer[4];
extern short** summed_buffer_cuda;
extern short* final_buffer_cuda;
extern short** saved_buffer_cuda;
extern short** dynamic_saved_buffer_cuda;
extern double* static_buffer_cuda;
extern double * real_static_freq_cuda;
extern double real_static_freq[16384];
extern unsigned int dynamic_total;
extern unsigned int static_total;
extern __device__ __constant__ unsigned int tone_count_cuda[5];
extern __device__ __constant__ unsigned int dynamic_tone_count_cuda[5];
// ------Dynamics----------------------------
extern double real_destination_freq[1024];
extern unsigned int dynamic_buffersize;
extern double* real_destination_freq_cuda;
extern int * dynamic_list_cuda;
extern int * static_list_cuda;
extern int dynamic_loopcount;
extern bool not_arrived;

#endif