#ifndef _cudaFunctions_playback_included_
#define _cudaFunctions_playback_included_
    // ----- CUDA includes -----
#   include "cuda_functions.h"

extern int dynamicBufferInit();
extern int rearrangeBufferInit();
extern void cuda_cleanup();

__device__ __forceinline__ double mini_jerk_coeff(double frequencyDiff, double ratio, double startFrequency){
    return __fma_rn(frequencyDiff* ratio *ratio *ratio , __fma_rn(ratio - 3. , ratio, 2.5), startFrequency);
}

__device__ __forceinline__ float mini_jerk_coefff(float frequencyDiff, float ratio, float startFrequency){
    return __fmaf_rn(frequencyDiff* ratio *ratio *ratio , __fmaf_rn(ratio - 3.f , ratio, 2.5f), startFrequency);
}

__device__ __forceinline__ double after_chirp_phase(double frequencyDiff, double phi, double ddynamic_bufferlength_cuda, double illSamplerate_cuda) {
    double res = __fma_rn(-0.5*frequencyDiff, ddynamic_bufferlength_cuda*illSamplerate_cuda, phi);
    return res - floor(res);
}

__device__ __forceinline__ float after_chirp_phasef(float frequencyDiff, double phi, double ddynamic_bufferlength_cuda, double illSamplerate_cuda) {
    float res = __fmaf_rn(-0.5f*frequencyDiff, __double2float_rn(ddynamic_bufferlength_cuda*illSamplerate_cuda), __double2float_rn(phi));
    return res - floorf(res);
}

__device__ __forceinline__ double after_accel_phase(double frequencyDiff, double phi, double ddynamic_bufferlength_cuda, double illSamplerate_cuda) {
    double res = __fma_rn(-0.375*frequencyDiff, ddynamic_bufferlength_cuda*illSamplerate_cuda, phi);
    return res - floor(res);
}

__device__ __forceinline__ float after_accel_phasef(float frequencyDiff, double phi, double ddynamic_bufferlength_cuda, double illSamplerate_cuda) {
    float res = __fmaf_rn(-0.375f*frequencyDiff, __double2float_rn(ddynamic_bufferlength_cuda*illSamplerate_cuda), __double2float_rn(phi));
    return res - floorf(res);
}

__device__ __forceinline__ double modfma(double x, double y, double z) {
    double frac_part = __fma_rn(x, y, z);
    return frac_part - floor(frac_part);
}

__device__ __forceinline__ float modfmaf(float x, float y, float z) {
    float freq_x = x * y;
    float frac_part = __fmaf_rn(x, y, -floorf(freq_x)) + z;
    return frac_part - floorf(frac_part);
}




extern __global__ void DynamicMux (unsigned int startPosition, short**__restrict__ buffer,short* pnOut);
extern __global__ void WaveformCopier (short* __restrict__ buffer,short* pnOut);
extern __global__ void Pre_computer(const double * __restrict__ static_buf, const int* __restrict__ static_list, const float* __restrict__ ddest_freq,
                            const int*__restrict__ ddy_list, short* final_buf, short** dynamic_buf, const float * __restrict__ dstartFreq, const float* __restrict__ phase,float * new_phase);
extern __global__ void Pre_computer_amp(const double * __restrict__ static_buf, const int* __restrict__ static_list, const float* __restrict__ ddest_freq,
                            const int*__restrict__ ddy_list, short* final_buf, short** dynamic_buf, const float * __restrict__ dstartFreq, const float* __restrict__ phase,float * new_phase,
                            const float* __restrict__ amp);
extern __global__ void Pre_computer_rearrange(int buffer_index, int static_num, int dynamic_num, const int * __restrict__ single_static_map_cuda, const int * __restrict__ single_dynamic_map_cuda, double ** summed_buf, double *  static_buf, const float* __restrict__ ddest_freq,
                            short* final_buf, short** dynamic_buf, const float * __restrict__ dstartFreq, const float* __restrict__ phase, float * new_phase, float * rearrange_freq_list_cuda);
extern __global__ void Pre_computer_rearrange_amp(int buffer_index, int static_num, int dynamic_num, const int * __restrict__ single_static_map_cuda, const int * __restrict__ single_dynamic_map_cuda, double ** summed_buf, double *  static_buf, const float* __restrict__ ddest_freq,
                            short* final_buf, short** dynamic_buf, const float * __restrict__ dstartFreq, const float* __restrict__ phase, float * new_phase, float * rearrange_freq_list_cuda,const float* __restrict__ amp);
extern __global__ void Pre_computer_accelerate(float frequencyDiff, int buffer_index, const double * __restrict__ static_buf, const int* __restrict__ static_list,
                            const int*__restrict__ ddy_list, short* final_buf, short** dynamic_buf, const float * __restrict__ dstartFreq, const float* __restrict__ phase,float * new_phase);
extern __global__ void Pre_computer_accelerate_amp(float frequencyDiff, int buffer_index, const double * __restrict__ static_buf, const int* __restrict__ static_list,
                            const int*__restrict__ ddy_list, short* final_buf, short** dynamic_buf, const float * __restrict__ dstartFreq, const float* __restrict__ phase,float * new_phase,
                            const float* __restrict__ amp);



// The following are the optimized float precision algorithm that is expected to run faster with satisfactory phase noise
extern __global__ void Pre_computerf(const float * __restrict__ static_buf, const int* __restrict__ static_list, const float* __restrict__ ddest_freq,
                            const int*__restrict__ ddy_list, short* final_buf, short** dynamic_buf, const float * __restrict__ dstartFreq, const float* __restrict__ phase,float * new_phase);
extern __global__ void Pre_computer_ampf(const float * __restrict__ static_buf, const int* __restrict__ static_list, const float* __restrict__ ddest_freq,
                            const int*__restrict__ ddy_list, short* final_buf, short** dynamic_buf, const float * __restrict__ dstartFreq, const float* __restrict__ phase,float * new_phase,
                            const float* __restrict__ amp);
extern __global__ void Pre_computer_rearrangef(int buffer_index, int static_num, int dynamic_num, const int * __restrict__ single_static_map_cuda, const int * __restrict__ single_dynamic_map_cuda, float ** summed_buf, float *  static_buf, const float* __restrict__ ddest_freq,
                            short* final_buf, short** dynamic_buf, const float * __restrict__ dstartFreq, const float* __restrict__ phase, float * new_phase, float * rearrange_freq_list_cuda);
extern __global__ void Pre_computer_rearrange_ampf(int buffer_index, int static_num, int dynamic_num, const int * __restrict__ single_static_map_cuda, const int * __restrict__ single_dynamic_map_cuda, double ** summed_buf, double *  static_buf, const float* __restrict__ ddest_freq,
                            short* final_buf, short** dynamic_buf, const float * __restrict__ dstartFreq, const float* __restrict__ phase, float * new_phase, float * rearrange_freq_list_cuda,const float* __restrict__ amp);
extern __global__ void Pre_computer_acceleratef(float frequencyDiff, int buffer_index, const float * __restrict__ static_buf, const int* __restrict__ static_list,
                            const int*__restrict__ ddy_list, short* final_buf, short** dynamic_buf, const float * __restrict__ dstartFreq, const float* __restrict__ phase,float * new_phase);
extern __global__ void Pre_computer_accelerate_ampf(float frequencyDiff, int buffer_index, const float * __restrict__ static_buf, const int* __restrict__ static_list,
                            const int*__restrict__ ddy_list, short* final_buf, short** dynamic_buf, const float * __restrict__ dstartFreq, const float* __restrict__ phase,float * new_phase,
                            const float* __restrict__ amp);





extern short* final_buffer_cuda;
extern short** dynamic_saved_buffer_cuda;


// Not used
// extern __global__ void Pre_computer_amp_modulated(double * __restrict__ static_buf, int* __restrict__ static_list, double* __restrict__ ddest_freq,
//                             int*__restrict__ ddy_list, short* final_buf, short** dynamic_buf, double * __restrict__ dstartFreq, double* __restrict__ phase,double * new_phase,
//                             float* __restrict__ amp,float* __restrict__ new_amp);
// extern __global__ void Pre_computer_amp_mapped(double * __restrict__ static_buf, int* __restrict__ static_list, double* __restrict__ ddest_freq,
//                             int*__restrict__ ddy_list, short* final_buf, short** dynamic_buf, double * __restrict__ dstartFreq, float*__restrict__ amp_map, double* __restrict__ phase,double * new_phase);
// extern __global__ void Pre_computer_amp_mapped_modulated(double * __restrict__ static_buf, int* __restrict__ static_list, double* __restrict__ ddest_freq,
//                             int*__restrict__ ddy_list, short* final_buf, short** dynamic_buf, double * __restrict__ dstartFreq, float*__restrict__ amp_map, double* __restrict__ phase,double * new_phase,float* __restrict__ amp,float* __restrict__ new_amp);

#endif