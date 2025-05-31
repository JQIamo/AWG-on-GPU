#ifndef _cudaFunctions_streaming_included_
#define _cudaFunctions_streaming_included_

#include "cuda_functions.h"
extern int dynamicBufferInit();
extern void cuda_cleanup();
extern double* dFreq_cuda;
extern double* dDiff_cuda;
extern double* dphi_cuda;
extern float* damp_cuda;

extern double* destination_buffer_cuda;
extern double ** temp_buffer_cuda;
extern double* temp_buffer[4];

extern __global__ void DynamicListWorker(double* __restrict__ staticFrequency,double* __restrict__ destinationFreq,int*dy_list,double *dstart,double*dDiff,double*dphi,double* __restrict__ phase_list);
extern __global__ void DynamicListWorker_amp(double*__restrict__ staticFrequency,double*__restrict__ destinationFreq,int* __restrict__ dy_list,double *dstart,double*dDiff,double*dphi,float*damp,double*__restrict__ phase_list,float*__restrict__ amp_list);
extern __global__ void Pre_AccelCombine(double** temp_save_buf,int last_start_index,int new_start_index,int last_counter, int new_counter,int* __restrict__ dy_list,double* __restrict__ static_buffer, double*__restrict__ startFreq, double* __restrict__ dest_frequency,double*__restrict__ phase_list);
extern __global__ void Pre_AccelCombine_amp(double** temp_save_buf,int last_start_index,int new_start_index,int last_counter, int new_counter,int* __restrict__ dy_list,double* __restrict__ static_buffer, double*__restrict__ startFreq, double* __restrict__ dest_frequency,double*__restrict__ phase_list,float*amplist);
extern __global__ void Pre_AccelCombine_amp_mapped(double** temp_save_buf,int last_start_index,int new_start_index,int last_counter, int new_counter,int* __restrict__ dy_list,double* __restrict__ static_buffer, double*__restrict__ startFreq, double* __restrict__ dest_frequency,double*__restrict__ phase_list,float*amp_map);
extern __global__ void Pre_AccelCombine_amp_amp_mapped(double** temp_save_buf,int last_start_index,int new_start_index,int last_counter, int new_counter,int* __restrict__ dy_list,double* __restrict__ static_buffer, double*__restrict__ startFreq, double* __restrict__ dest_frequency,double*__restrict__ phase_list,float*amp_map,float*amplist);
extern __global__ void AccelCombine(unsigned long long startPosition,double**__restrict__ save_buf,double** sum_buf,
                    double*__restrict__ dstartFrequency,double *__restrict__ dfrequencyDiff,double *__restrict__ dphi,int start_index, int maximal_index);
extern __global__ void AccelCombine_amp(unsigned long long startPosition,double**__restrict__ save_buf,double** sum_buf,
                    double*__restrict__ dstartFrequency,double *__restrict__ dfrequencyDiff,double *__restrict__ dphi,float*__restrict__ damp,int start_index, int maximal_index);
extern __global__ void AccelCombine_amp_modulated(unsigned long long startPosition,double**__restrict__ save_buf,double** sum_buf,
                    double*__restrict__ dstartFrequency,double *__restrict__ dfrequencyDiff,double *__restrict__ dphi,float*__restrict__ damp,float*__restrict__ namp,int start_index, int maximal_index);
extern __global__ void AccelCombine_amp_mapped(unsigned long long startPosition,double**__restrict__ save_buf,double** sum_buf,
                    double*__restrict__ dstartFrequency,double *__restrict__ dfrequencyDiff,double *__restrict__ dphi,float*__restrict__ amp_map,int start_index, int maximal_index);
extern __global__ void AccelCombine_amp_amp_mapped(unsigned long long startPosition,double**__restrict__ save_buf,double** sum_buf,
                    double*__restrict__ dstartFrequency,double *__restrict__ dfrequencyDiff,double *__restrict__ dphi,float*__restrict__ amp_map,float*__restrict__ damp,int start_index, int maximal_index);
extern __global__ void AccelCombine_amp_mapped_modulated(unsigned long long startPosition,double**__restrict__ save_buf,double** sum_buf,
                    double*__restrict__ dstartFrequency,double *__restrict__ dfrequencyDiff,double *__restrict__ dphi,float*__restrict__ amp_map,float*__restrict__ damp,float*__restrict__ namp,int start_index, int maximal_index);
extern __global__ void DoubleCopier(double**__restrict__ source,double**destination);                                                                                
#endif