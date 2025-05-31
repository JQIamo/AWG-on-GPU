#ifndef _cudaFunctions_playback_included_
#define _cudaFunctions_playback_included_
    // ----- CUDA includes -----
#   include "cuda_functions.h"

extern int dynamicBufferInit();
extern void cuda_cleanup();
extern __global__ void DynamicMux (unsigned int startPosition, short**__restrict__ buffer,short* pnOut);
extern __global__ void WaveformCopier (short* __restrict__ buffer,short* pnOut);
extern __global__ void Pre_computer(double * __restrict__ static_buf, int* __restrict__ static_list, double* __restrict__ ddest_freq, 
                            int*__restrict__ ddy_list, short* final_buf, short** dynamic_buf,double * __restrict__ dstartFreq, double* __restrict__ phase,double * new_phase);
extern __global__ void Pre_computer_amp(double * __restrict__ static_buf, int* __restrict__ static_list, double* __restrict__ ddest_freq, 
                            int*__restrict__ ddy_list, short* final_buf, short** dynamic_buf, double * __restrict__ dstartFreq, double* __restrict__ phase,double * new_phase,
                            float* __restrict__ amp);
extern __global__ void Pre_computer_amp_modulated(double * __restrict__ static_buf, int* __restrict__ static_list, double* __restrict__ ddest_freq, 
                            int*__restrict__ ddy_list, short* final_buf, short** dynamic_buf, double * __restrict__ dstartFreq, double* __restrict__ phase,double * new_phase,
                            float* __restrict__ amp,float* __restrict__ new_amp);
extern __global__ void Pre_computer_amp_mapped(double * __restrict__ static_buf, int* __restrict__ static_list, double* __restrict__ ddest_freq, 
                            int*__restrict__ ddy_list, short* final_buf, short** dynamic_buf, double * __restrict__ dstartFreq, float*__restrict__ amp_map, double* __restrict__ phase,double * new_phase);
extern __global__ void Pre_computer_amp_mapped_modulated(double * __restrict__ static_buf, int* __restrict__ static_list, double* __restrict__ ddest_freq, 
                            int*__restrict__ ddy_list, short* final_buf, short** dynamic_buf, double * __restrict__ dstartFreq, float*__restrict__ amp_map, double* __restrict__ phase,double * new_phase,float* __restrict__ amp,float* __restrict__ new_amp);
extern short* final_buffer_cuda;
extern short** dynamic_saved_buffer_cuda;
#endif