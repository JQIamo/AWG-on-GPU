#   include "cuda_functions_playback.h"
short* final_buffer_cuda;
short** dynamic_saved_buffer_cuda;


__global__ void DynamicMux (unsigned int startPosition, short**__restrict__ buffer,short* pnOut)
    {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    size_t j = i+startPosition;
    if (channel_num_cuda>2){
        short4 *pnOut4 = reinterpret_cast<short4*>(pnOut);
        pnOut4[i] = {buffer[0][j],buffer[1][j],buffer[2][j],buffer[3][j]};
    }else if (channel_num_cuda==2){
        short2 *pnOut2 = reinterpret_cast<short2*>(pnOut);
        pnOut2[i] = {buffer[0][j],buffer[1][j]};
    }else{
        pnOut[i] = buffer[0][j];
    }
}

__global__ void WaveformCopier (short* __restrict__ buffer,short* pnOut)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    pnOut[i] = buffer[i];
}


__global__ void Pre_computer(double * __restrict__ static_buf, int* __restrict__ static_list, double* __restrict__ ddest_freq, 
                            int*__restrict__ ddy_list, short* final_buf, short** dynamic_buf, double * __restrict__ dstartFreq, double* __restrict__ phase,double * new_phase){
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    double dump;
    double sum;
    double sum1;
    size_t static_bufferlength_cuda2 = static_bufferlength_cuda / 2;
    double2* static_buf2 = reinterpret_cast<double2*>(static_buf);
    __shared__ double startFreq[1024];
    __shared__ double dest_freq[1024];
    __shared__ double phaselist[1024];
    for (int buffer_index=0;buffer_index<channel_num_cuda;buffer_index++){
        for (int iter =0;iter<= (dynamic_num_cuda[buffer_index]-1)/256+1;iter++){
            if(iter*256+threadIdx.x<dynamic_num_cuda[buffer_index]){
                size_t dynamic_index = dynamic_tone_count_cuda[buffer_index]+iter*256+threadIdx.x;
                int shared_index = ddy_list[dynamic_index];
                startFreq[iter*256+threadIdx.x] = dstartFreq[shared_index];
                dest_freq[iter*256+threadIdx.x] = ddest_freq[dynamic_index];
                phaselist[iter*256+threadIdx.x] = phase[shared_index];
            }
        }
        __syncthreads();

        double2 temp_sums;
        sum = 0;
        sum1 = 0;
        size_t static_tone_count_start = tone_count_cuda[buffer_index]-dynamic_tone_count_cuda[buffer_index];
        size_t static_tone_count_end = tone_count_cuda[buffer_index+1]-dynamic_tone_count_cuda[buffer_index+1];
        if (static_tone_count_end-static_tone_count_start>0){
            for(int iter=static_tone_count_start;iter<static_tone_count_end;iter++){
                temp_sums = static_buf2[static_bufferlength_cuda2*static_list[iter]+i];
                sum += temp_sums.x;
                sum1 += temp_sums.y;
            }
        }
        double temp_sumst = 0.;
        double temp_sumst1 = 0.;
        double ipower_normalizer = 32767. *ipower_normalizer_cuda[buffer_index];
        for (int j=0; j<dynamic_num_cuda[buffer_index];j++){
            double phi = __fma_rn(dest_freq[j],dynamic_loopcount_cuda*static_cast<double>(static_bufferlength_cuda)-0.5*dynamic_bufferlength_cuda,0.5*dynamic_bufferlength_cuda*startFreq[j]+phaselist[j]);
            if (i==0) new_phase[ddy_list[dynamic_tone_count_cuda[buffer_index]+j]] = 2.*modf(phi,&dump);
            temp_sumst +=  sinpi (2.*modf(__fma_rn(dest_freq[j] , 2.*static_cast<double>(i),phi)* illSamplerate_cuda,&dump));
            temp_sumst1 +=  sinpi (2.*modf(__fma_rn(dest_freq[j] , 2.*static_cast<double>(i)+1.,phi)* illSamplerate_cuda,&dump));
        }
        final_buf[buffer_index+2*i*channel_num_cuda] = static_cast<short>(fma(temp_sumst,ipower_normalizer,sum));
        final_buf[buffer_index+(2*i+1)*channel_num_cuda] = static_cast<short>(fma(temp_sumst1,ipower_normalizer,sum1));
        for (int counter = 0; counter< dynamic_loopcount_cuda*0.5; counter++){
            size_t index = counter*static_bufferlength_cuda+i;
            size_t checkpoint = index * 2 + 1;
            double position = static_cast<double>(checkpoint)-1.;
            float suml = static_cast<float>(sum1);
            float sums = static_cast<float>(sum);
            double ratio = position*idynamic_bufferlength_cuda;
            double ratio1 = ratio + idynamic_bufferlength_cuda;
            if (checkpoint < __double2ull_rn(dynamic_bufferlength_cuda)){
                for (int j=0; j<dynamic_num_cuda[buffer_index];j++){
                    double startFrequency = startFreq[j];
                    double freqeuncyDiff = dest_freq[j]-startFrequency;
                    double phi = phaselist[j];
                    double phase_c = position * illSamplerate_cuda * __fma_rn(freqeuncyDiff* ratio *ratio *ratio , __fma_rn(ratio-3. , ratio,2.5),startFrequency);
                    double pc = M_PI*2.*modf( phase_c+phi,&dump);
                    double phase_c1 = (position + 1.) * illSamplerate_cuda * __fma_rn(freqeuncyDiff* ratio1 *ratio1 *ratio1 , __fma_rn(ratio1-3. , ratio1,2.5),startFrequency);
                    double pc1 = M_PI*2.*modf( phase_c1+phi,&dump);
                    half2 temp = h2sin (__floats2half2_rn(pc,pc1));
                    sums +=  __low2float(temp)*32767.f;
                    suml +=  __high2float(temp)*32767.f;
                }
            } else if (checkpoint == __double2ull_rn(dynamic_bufferlength_cuda)){
                for (int j=0; j<dynamic_num_cuda[buffer_index];j++){
                    double startFrequency = startFreq[j];
                    double freqeuncyDiff = dest_freq[j]-startFrequency;
                    double phi = phaselist[j];
                    double phase_c = position * illSamplerate_cuda * __fma_rn(freqeuncyDiff* ratio *ratio *ratio , __fma_rn(ratio-3. , ratio,2.5),startFrequency);
                    double pc = M_PI*2.*modf( phase_c+phi,&dump);
                    double pc1 = M_PI*2.*modf(__fma_rn(__fma_rn(0.5,freqeuncyDiff,startFrequency)*dynamic_bufferlength_cuda,illSamplerate_cuda,phi),&dump);
                    half2 temp = h2sin (__floats2half2_rn(pc,pc1));
                    sums +=  __low2float(temp)*32767.f;
                    suml +=  __high2float(temp)*32767.f;
                }
            }else{
                for (int j=0; j<dynamic_num_cuda[buffer_index];j++){
                    double startFrequency = startFreq[j];
                    double freqeuncyDiff = dest_freq[j]-startFrequency;
                    double phi = phaselist[j];
                    double pc = M_PI*2.*modf( __fma_rn((startFrequency+freqeuncyDiff) *(position-dynamic_bufferlength_cuda) ,illSamplerate_cuda,__fma_rn(__fma_rn(0.5,freqeuncyDiff,startFrequency)*dynamic_bufferlength_cuda,illSamplerate_cuda,phi)),&dump);
                    double pc1 = M_PI*2.*modf( __fma_rn((startFrequency+freqeuncyDiff) *(position-dynamic_bufferlength_cuda+1.) ,illSamplerate_cuda,__fma_rn(__fma_rn(0.5,freqeuncyDiff,startFrequency)*dynamic_bufferlength_cuda,illSamplerate_cuda,phi)),&dump);
                    half2 temp = h2sin (__floats2half2_rn(pc,pc1));
                    sums +=  __low2float(temp)*32767.f;
                    suml +=  __high2float(temp)*32767.f;
                }
            }
            short2* dynamic_buf2 = reinterpret_cast<short2*>(dynamic_buf[buffer_index]);
            dynamic_buf2[index] = {static_cast<short>(sums),static_cast<short>(suml)};
        }
        __syncthreads();
    }
}




__global__ void Pre_computer_amp(double * __restrict__ static_buf, int* __restrict__ static_list, double* __restrict__ ddest_freq, 
                            int*__restrict__ ddy_list, short* final_buf, short** dynamic_buf, double * __restrict__ dstartFreq, double* __restrict__ phase,double * new_phase,
                            float* __restrict__ amp) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    double dump;
    double sum;
    double sum1;
    size_t static_bufferlength_cuda2 = static_bufferlength_cuda / 2;
    double2* static_buf2 = reinterpret_cast<double2*>(static_buf);
    __shared__ double startFreq[1024];
    __shared__ double dest_freq[1024];
    __shared__ float amplist[1024];
    __shared__ double phaselist[1024];
    for (int buffer_index=0;buffer_index<channel_num_cuda;buffer_index++){
        for (int iter =0;iter<= (dynamic_num_cuda[buffer_index]-1)/256+1;iter++){
            if(iter*256+threadIdx.x<dynamic_num_cuda[buffer_index]){
                size_t dynamic_index = dynamic_tone_count_cuda[buffer_index]+iter*256+threadIdx.x;
                int shared_index = ddy_list[dynamic_index];
                startFreq[iter*256+threadIdx.x] = dstartFreq[shared_index];
                dest_freq[iter*256+threadIdx.x] = ddest_freq[dynamic_index];
                phaselist[iter*256+threadIdx.x] = phase[shared_index];
                amplist[iter*256+threadIdx.x] = amp[shared_index];
            }
        }
        __syncthreads();

        double2 temp_sums;
        sum = 0.;
        sum1 = 0.;
        size_t static_tone_count_start = tone_count_cuda[buffer_index]-dynamic_tone_count_cuda[buffer_index];
        size_t static_tone_count_end = tone_count_cuda[buffer_index+1]-dynamic_tone_count_cuda[buffer_index+1];
        if (static_tone_count_end-static_tone_count_start>0){
            for(int iter=static_tone_count_start;iter<static_tone_count_end;iter++){
                temp_sums = static_buf2[static_bufferlength_cuda2*static_list[iter]+i];
                sum += temp_sums.x;
                sum1 += temp_sums.y;
            }
        }
        double temp_sumst = 0.;
        double temp_sumst1 = 0.;
        double ipower_normalizer = 32767. *ipower_normalizer_cuda[buffer_index];
        for (int j=0; j<dynamic_num_cuda[buffer_index];j++){
            double phi = __fma_rn(dest_freq[j],dynamic_loopcount_cuda*static_cast<double>(static_bufferlength_cuda)-0.5*dynamic_bufferlength_cuda,0.5*dynamic_bufferlength_cuda*startFreq[j]+phaselist[j]);
            if (i==0) new_phase[ddy_list[dynamic_tone_count_cuda[buffer_index]+j]] = 2.*modf(phi,&dump);
            double ampf = static_cast<double>(amplist[j]);
            temp_sumst += ampf * sinpi (2.*modf(__fma_rn(dest_freq[j] , 2.*static_cast<double>(i),phi)* illSamplerate_cuda,&dump));
            temp_sumst1 += ampf * sinpi (2.*modf(__fma_rn(dest_freq[j] , 2.*static_cast<double>(i)+1.,phi)* illSamplerate_cuda,&dump));
        }
        final_buf[buffer_index+2*i*channel_num_cuda] = static_cast<short>(fma(temp_sumst,ipower_normalizer,sum));
        final_buf[buffer_index+(2*i+1)*channel_num_cuda] = static_cast<short>(fma(temp_sumst1,ipower_normalizer,sum1));
        for (int counter = 0; counter< dynamic_loopcount_cuda*0.5; counter++){
            size_t index = counter*static_bufferlength_cuda+i;
            size_t checkpoint = index * 2 + 1;
            double position = static_cast<double>(checkpoint)-1.;
            float suml = static_cast<float>(sum1);
            float sums = static_cast<float>(sum);
            double ratio = position*idynamic_bufferlength_cuda;
            double ratio1 = ratio + idynamic_bufferlength_cuda;
            if (checkpoint < __double2ull_rn(dynamic_bufferlength_cuda)){
                for (int j=0; j<dynamic_num_cuda[buffer_index];j++){
                    double startFrequency = startFreq[j];
                    float ampf = amplist[j]*32767.f;
                    double freqeuncyDiff = dest_freq[j]-startFrequency;
                    double phi = phaselist[j];
                    double phase_c = position * illSamplerate_cuda * __fma_rn(freqeuncyDiff* ratio *ratio *ratio , __fma_rn(ratio-3. , ratio,2.5),startFrequency);
                    double pc = M_PI*2.*modf( phase_c+phi,&dump);
                    double phase_c1 = (position + 1.) * illSamplerate_cuda * __fma_rn(freqeuncyDiff* ratio1 *ratio1 *ratio1 , __fma_rn(ratio1-3. , ratio1,2.5),startFrequency);
                    double pc1 = M_PI*2.*modf( phase_c1+phi,&dump);
                    half2 temp = h2sin (__floats2half2_rn(pc,pc1));
                    sums +=  __low2float(temp)*ampf;
                    suml +=  __high2float(temp)*ampf;
                }
            } else if (checkpoint == __double2ull_rn(dynamic_bufferlength_cuda)){
                for (int j=0; j<dynamic_num_cuda[buffer_index];j++){
                    double startFrequency = startFreq[j];
                    float ampf = amplist[j]*32767.f;
                    double freqeuncyDiff = dest_freq[j]-startFrequency;
                    double phi = phaselist[j];
                    double phase_c = position * illSamplerate_cuda * __fma_rn(freqeuncyDiff* ratio *ratio *ratio , __fma_rn(ratio-3. , ratio,2.5),startFrequency);
                    double pc = M_PI*2.*modf( phase_c+phi,&dump);
                    double pc1 = M_PI*2.*modf(__fma_rn(__fma_rn(0.5,freqeuncyDiff,startFrequency)*dynamic_bufferlength_cuda,illSamplerate_cuda,phi),&dump);
                    half2 temp = h2sin (__floats2half2_rn(pc,pc1));
                    sums +=  __low2float(temp)*ampf;
                    suml +=  __high2float(temp)*ampf;
                }
            }else{
                for (int j=0; j<dynamic_num_cuda[buffer_index];j++){
                    double startFrequency = startFreq[j];
                    double freqeuncyDiff = dest_freq[j]-startFrequency;
                    float ampf = amplist[j]*32767.f;
                    double phi = phaselist[j];
                    double pc = M_PI*2.*modf( __fma_rn((startFrequency+freqeuncyDiff) *(position-dynamic_bufferlength_cuda) ,illSamplerate_cuda,__fma_rn(__fma_rn(0.5,freqeuncyDiff,startFrequency)*dynamic_bufferlength_cuda,illSamplerate_cuda,phi)),&dump);
                    double pc1 = M_PI*2.*modf( __fma_rn((startFrequency+freqeuncyDiff) *(position-dynamic_bufferlength_cuda+1.) ,illSamplerate_cuda,__fma_rn(__fma_rn(0.5,freqeuncyDiff,startFrequency)*dynamic_bufferlength_cuda,illSamplerate_cuda,phi)),&dump);
                    half2 temp = h2sin (__floats2half2_rn(pc,pc1));
                    sums +=  __low2float(temp)*ampf;
                    suml +=  __high2float(temp)*ampf;
                }
            }
            short2* dynamic_buf2 = reinterpret_cast<short2*>(dynamic_buf[buffer_index]);
            dynamic_buf2[index] = {static_cast<short>(sums),static_cast<short>(suml)};
        }
        __syncthreads();
    }
}



__global__ void Pre_computer_amp_modulated(double * __restrict__ static_buf, int* __restrict__ static_list, double* __restrict__ ddest_freq, 
                            int*__restrict__ ddy_list, short* final_buf, short** dynamic_buf, double * __restrict__ dstartFreq, double* __restrict__ phase,double * new_phase,
                            float* __restrict__ amp,float* __restrict__ new_amp) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    double dump;
    double sum;
    double sum1;
    size_t static_bufferlength_cuda2 = static_bufferlength_cuda / 2;
    double2* static_buf2 = reinterpret_cast<double2*>(static_buf);
    __shared__ double startFreq[1024];
    __shared__ double dest_freq[1024];
    __shared__ float newamp[1024];
    __shared__ float oldamp[1024];
    __shared__ double phaselist[1024];
    for (int buffer_index=0;buffer_index<channel_num_cuda;buffer_index++){
        for (int iter =0;iter<= (dynamic_num_cuda[buffer_index]-1)/256+1;iter++){
            if(iter*256+threadIdx.x<dynamic_num_cuda[buffer_index]){
                size_t dynamic_index = dynamic_tone_count_cuda[buffer_index]+iter*256+threadIdx.x;
                int shared_index = ddy_list[dynamic_index];
                startFreq[iter*256+threadIdx.x] = dstartFreq[shared_index];
                double df = ddest_freq[dynamic_index];
                dest_freq[iter*256+threadIdx.x] = df;
                phaselist[iter*256+threadIdx.x] = phase[shared_index];
                newamp[iter*256+threadIdx.x] = new_amp[dynamic_index];
                oldamp[iter*256+threadIdx.x] = amp[shared_index];
            }
        }
        __syncthreads();

        double2 temp_sums;
        sum = 0;
        sum1 = 0;
        size_t static_tone_count_start = tone_count_cuda[buffer_index]-dynamic_tone_count_cuda[buffer_index];
        size_t static_tone_count_end = tone_count_cuda[buffer_index+1]-dynamic_tone_count_cuda[buffer_index+1];
        if (static_tone_count_end-static_tone_count_start>0){
            for(int iter=static_tone_count_start;iter<static_tone_count_end;iter++){
                temp_sums = static_buf2[static_bufferlength_cuda2*static_list[iter]+i];
                sum += temp_sums.x;
                sum1 += temp_sums.y;
            }
        }
        double temp_sumst = 0.;
        double temp_sumst1 = 0.;
        double ipower_normalizer = 32767. *ipower_normalizer_cuda[buffer_index];
        for (int j=0; j<dynamic_num_cuda[buffer_index];j++){
            double phi = __fma_rn(dest_freq[j],dynamic_loopcount_cuda*static_cast<double>(static_bufferlength_cuda)-0.5*dynamic_bufferlength_cuda,0.5*dynamic_bufferlength_cuda*startFreq[j]+phaselist[j]);
            if (i==0) new_phase[ddy_list[dynamic_tone_count_cuda[buffer_index]+j]] = 2.*modf(phi,&dump);
            temp_sumst += sinpi (2.*modf(__fma_rn(dest_freq[j] , 2.*static_cast<double>(i),phi)* illSamplerate_cuda,&dump))*newamp[j];
            temp_sumst1 += sinpi (2.*modf(__fma_rn(dest_freq[j] , 2.*static_cast<double>(i)+1.,phi)* illSamplerate_cuda,&dump))*newamp[j];
        }
        final_buf[buffer_index+2*i*channel_num_cuda] = static_cast<short>(fma(temp_sumst,ipower_normalizer,sum));
        final_buf[buffer_index+(2*i+1)*channel_num_cuda] = static_cast<short>(fma(temp_sumst1,ipower_normalizer,sum1));
        float suml = static_cast<float>(sum1);
        float sums = static_cast<float>(sum);
        for (int counter = 0; counter< dynamic_loopcount_cuda*0.5; counter++){
            size_t index = counter*static_bufferlength_cuda+i;
            size_t checkpoint = index * 2 + 1;
            double position = static_cast<double>(checkpoint)-1.;
            float temp_sum=0.f;
            float temp_sum1 = 0.f;
            double ratio = position*idynamic_bufferlength_cuda;
            double ratio1 = ratio + idynamic_bufferlength_cuda;
            if (checkpoint < __double2ull_rn(dynamic_bufferlength_cuda)){
                for (int j=0; j<dynamic_num_cuda[buffer_index];j++){
                    float old_amp = oldamp[j];
                    float amp_ramper = __fmaf_rn(newamp[j] - old_amp, static_cast<float>(ratio),old_amp);
                    double startFrequency = startFreq[j];
                    double freqeuncyDiff = dest_freq[j]-startFrequency;
                    double phi = phaselist[j];
                    double phase_c = position * illSamplerate_cuda * __fma_rn(freqeuncyDiff* ratio *ratio *ratio , __fma_rn(ratio-3. , ratio,2.5),startFrequency);
                    double pc = M_PI*2.*modf( phase_c+phi,&dump);
                    double phase_c1 = (position + 1.) * illSamplerate_cuda * __fma_rn(freqeuncyDiff* ratio1 *ratio1 *ratio1 , __fma_rn(ratio1-3. , ratio1,2.5),startFrequency);
                    double pc1 = M_PI*2.*modf( phase_c1+phi,&dump);
                    half2 temp = h2sin (__floats2half2_rn(pc,pc1));
                    temp_sum +=  __low2float(temp)*amp_ramper;
                    temp_sum1 +=  __high2float(temp)*amp_ramper;
                }
            } else if (checkpoint == __double2ull_rn(dynamic_bufferlength_cuda)){
                for (int j=0; j<dynamic_num_cuda[buffer_index];j++){
                    double startFrequency = startFreq[j];
                    double freqeuncyDiff = dest_freq[j]-startFrequency;
                    double phi = phaselist[j];
                    double phase_c = position * illSamplerate_cuda * __fma_rn(freqeuncyDiff* ratio *ratio *ratio , __fma_rn(ratio-3. , ratio,2.5),startFrequency);
                    double pc = M_PI*2.*modf( phase_c+phi,&dump);
                    double pc1 = M_PI*2.*modf(__fma_rn(__fma_rn(0.5,freqeuncyDiff,startFrequency)*dynamic_bufferlength_cuda,illSamplerate_cuda,phi),&dump);
                    half2 temp = h2sin (__floats2half2_rn(pc,pc1));
                    temp_sum +=  __low2float(temp)*newamp[j];
                    temp_sum1 +=  __high2float(temp)*newamp[j];
                }
            }else{
                for (int j=0; j<dynamic_num_cuda[buffer_index];j++){
                    double startFrequency = startFreq[j];
                    double freqeuncyDiff = dest_freq[j]-startFrequency;
                    double phi = phaselist[j];
                    double pc = M_PI*2.*modf( __fma_rn((startFrequency+freqeuncyDiff) *(position-dynamic_bufferlength_cuda) ,illSamplerate_cuda,__fma_rn(__fma_rn(0.5,freqeuncyDiff,startFrequency)*dynamic_bufferlength_cuda,illSamplerate_cuda,phi)),&dump);
                    double pc1 = M_PI*2.*modf( __fma_rn((startFrequency+freqeuncyDiff) *(position-dynamic_bufferlength_cuda+1.) ,illSamplerate_cuda,__fma_rn(__fma_rn(0.5,freqeuncyDiff,startFrequency)*dynamic_bufferlength_cuda,illSamplerate_cuda,phi)),&dump);
                    half2 temp = h2sin (__floats2half2_rn(pc,pc1));
                    temp_sum +=  __low2float(temp)*newamp[j];
                    temp_sum1 +=  __high2float(temp)*newamp[j];
                }
            }
            float ipower_normalizerf = static_cast<float>(ipower_normalizer)*32767.f;
            short2* dynamic_buf2 = reinterpret_cast<short2*>(dynamic_buf[buffer_index]);
            dynamic_buf2[index] = {static_cast<short>(__fmaf_rn(temp_sum,ipower_normalizerf,sums)),static_cast<short>(__fmaf_rn(temp_sum1,ipower_normalizerf,suml))};
        }
        __syncthreads();
    }
}




__global__ void Pre_computer_amp_mapped(double * __restrict__ static_buf, int* __restrict__ static_list, double* __restrict__ ddest_freq, 
                            int*__restrict__ ddy_list, short* final_buf, short** dynamic_buf, double * __restrict__ dstartFreq, float*__restrict__ amp_map, double* __restrict__ phase,double * new_phase) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    double dump;
    double sum;
    double sum1;
    size_t static_bufferlength_cuda2 = static_bufferlength_cuda / 2;
    double2* static_buf2 = reinterpret_cast<double2*>(static_buf);
    __shared__ double startFreq[1024];
    __shared__ double dest_freq[1024];
    __shared__ float ampmap[1024];
    __shared__ double phaselist[1024];
    for (int buffer_index=0;buffer_index<channel_num_cuda;buffer_index++){
        for (int iter =0;iter<= (dynamic_num_cuda[buffer_index]-1)/256+1;iter++){
            if(iter*256+threadIdx.x<dynamic_num_cuda[buffer_index]){
                size_t dynamic_index = dynamic_tone_count_cuda[buffer_index]+iter*256+threadIdx.x;
                int shared_index = ddy_list[dynamic_index];
                startFreq[iter*256+threadIdx.x] = dstartFreq[shared_index];
                double df = ddest_freq[dynamic_index];
                dest_freq[iter*256+threadIdx.x] = df;
                phaselist[iter*256+threadIdx.x] = phase[shared_index];
                ampmap[iter*256+threadIdx.x] = amp_map[ __float2uint_rn(static_cast<float>(df)*map_interval_cuda)];
            }
        }
        __syncthreads();

        double2 temp_sums;
        sum = 0;
        sum1 = 0;
        int dynamic_num = dynamic_num_cuda[buffer_index];
        float scaler = static_cast<float>(dynamic_num) * static_cast<float>(istatic_num_cuda[buffer_index]);
        size_t static_tone_count_start = tone_count_cuda[buffer_index]-dynamic_tone_count_cuda[buffer_index];
        size_t static_tone_count_end = tone_count_cuda[buffer_index+1]-dynamic_tone_count_cuda[buffer_index+1];
        if (static_tone_count_end-static_tone_count_start>0){
            for(int iter=static_tone_count_start;iter<static_tone_count_end;iter++){
                int static_index = static_list[iter];
                temp_sums = static_buf2[static_bufferlength_cuda2*static_index+i];
                float amp_factor_sums = ampmap[__float2int_rn(static_cast<float>(dstartFreq[static_index])*map_interval_cuda)];
                sum += temp_sums.x;
                sum1 += temp_sums.y;
            }
        }
        double temp_sumst = 0.;
        double temp_sumst1 = 0.;
        float coefficient_correcter = 0;
        for (int j=0; j<dynamic_num;j++){
            double phi = __fma_rn(dest_freq[j],dynamic_loopcount_cuda*static_cast<double>(static_bufferlength_cuda)-0.5*dynamic_bufferlength_cuda,0.5*dynamic_bufferlength_cuda*startFreq[j]+phaselist[j]);
            if (i==0) new_phase[ddy_list[dynamic_tone_count_cuda[buffer_index]+j]] = 2.*modf(phi,&dump);
            double ampf = static_cast<double>(ampmap[j]);
            temp_sumst += ampf * sinpi (2.*modf(__fma_rn(dest_freq[j] , 2.*static_cast<double>(i),phi)* illSamplerate_cuda,&dump));
            temp_sumst1 += ampf * sinpi (2.*modf(__fma_rn(dest_freq[j] , 2.*static_cast<double>(i)+1.,phi)* illSamplerate_cuda,&dump));
            coefficient_correcter += ampmap[j];
        }
        coefficient_correcter = 32767.f/coefficient_correcter*scaler;
        final_buf[buffer_index+2*i*channel_num_cuda] = static_cast<short>(fma(temp_sumst,static_cast<double>(coefficient_correcter),sum));
        final_buf[buffer_index+(2*i+1)*channel_num_cuda] = static_cast<short>(fma(temp_sumst1,static_cast<double>(coefficient_correcter),sum1));
        float suml = static_cast<float>(sum1);
        float sums = static_cast<float>(sum);
        for (int counter = 0; counter< dynamic_loopcount_cuda*0.5; counter++){
            size_t index = counter*static_bufferlength_cuda+i;
            size_t checkpoint = index * 2 + 1;
            double position = static_cast<double>(checkpoint)-1;
            float temp_sum=0.f;
            coefficient_correcter = 0.f;
            float temp_sum1 = 0.f;
            double ratio = position*idynamic_bufferlength_cuda;
            double ratio1 = ratio + idynamic_bufferlength_cuda;
            if (checkpoint < __double2ull_rn(dynamic_bufferlength_cuda)){
                for (int j=0; j<dynamic_num;j++){
                    double startFrequency = startFreq[j];
                    double freqeuncyDiff = dest_freq[j]-startFrequency;
                    double phi = phaselist[j];
                    double phase_c = position * illSamplerate_cuda * __fma_rn(freqeuncyDiff* ratio *ratio *ratio , __fma_rn(ratio-3. , ratio,2.5),startFrequency);
                    double pc = M_PI*2.*modf( phase_c+phi,&dump);
                    float amp_factor = amp_map[__float2int_rn(static_cast<float>(__fma_rn(freqeuncyDiff* ratio *ratio *ratio , __fma_rn(__fma_rn(6.,ratio,-15.) , ratio,10.),startFrequency))*map_interval_cuda)];
                    double phase_c1 = (position + 1.) * illSamplerate_cuda * __fma_rn(freqeuncyDiff* ratio1 *ratio1 *ratio1 , __fma_rn(ratio1-3. , ratio1,2.5),startFrequency);
                    double pc1 = M_PI*2.*modf( phase_c1+phi,&dump);
                    half2 temp = h2sin (__floats2half2_rn(pc,pc1));
                    temp_sum +=  __low2float(temp)*amp_factor;
                    temp_sum1 +=  __high2float(temp)*amp_factor;
                    coefficient_correcter += amp_factor;
                }
            } else if (checkpoint == __double2ull_rn(dynamic_bufferlength_cuda)){
                for (int j=0; j<dynamic_num;j++){
                    double startFrequency = startFreq[j];
                    double freqeuncyDiff = dest_freq[j]-startFrequency;
                    double phi = phaselist[j];
                    double phase_c = position * illSamplerate_cuda * __fma_rn(freqeuncyDiff* ratio *ratio *ratio , __fma_rn(ratio-3. , ratio,2.5),startFrequency);
                    double pc = M_PI*2.*modf( phase_c+phi,&dump);
                    double pc1 = M_PI*2.*modf(__fma_rn(__fma_rn(0.5,freqeuncyDiff,startFrequency)*dynamic_bufferlength_cuda,illSamplerate_cuda,phi),&dump);
                    float amp_factor = amp_map[__float2int_rn(static_cast<float>(startFrequency+freqeuncyDiff)*map_interval_cuda)];
                    half2 temp = h2sin (__floats2half2_rn(pc,pc1));
                    temp_sum +=  __low2float(temp)*amp_factor;
                    temp_sum1 +=  __high2float(temp)*amp_factor;
                    coefficient_correcter += amp_factor;
                }
            }else{
                for (int j=0; j<dynamic_num;j++){
                    double startFrequency = startFreq[j];
                    double freqeuncyDiff = dest_freq[j]-startFrequency;
                    double phi = phaselist[j];
                    double pc = M_PI*2.*modf( __fma_rn((startFrequency+freqeuncyDiff) *(position-dynamic_bufferlength_cuda) ,illSamplerate_cuda,__fma_rn(__fma_rn(0.5,freqeuncyDiff,startFrequency)*dynamic_bufferlength_cuda,illSamplerate_cuda,phi)),&dump);
                    float amp_factor = amp_map[__float2int_rn(static_cast<float>(startFrequency+freqeuncyDiff)*map_interval_cuda)];
                    double pc1 = M_PI*2.*modf( __fma_rn((startFrequency+freqeuncyDiff) *(position-dynamic_bufferlength_cuda+1.) ,illSamplerate_cuda,__fma_rn(__fma_rn(0.5,freqeuncyDiff,startFrequency)*dynamic_bufferlength_cuda,illSamplerate_cuda,phi)),&dump);
                    half2 temp = h2sin (__floats2half2_rn(pc,pc1));
                    temp_sum +=  __low2float(temp)*amp_factor;
                    temp_sum1 +=  __high2float(temp)*amp_factor;
                    coefficient_correcter += amp_factor;
                }
            }
            coefficient_correcter = 32767.f/coefficient_correcter*scaler;
            short2* dynamic_buf2 = reinterpret_cast<short2*>(dynamic_buf[buffer_index]);
            dynamic_buf2[index] = {static_cast<short>(__fmaf_rn(temp_sum,coefficient_correcter,sums)),static_cast<short>(__fmaf_rn(temp_sum1,coefficient_correcter,suml))};
        }
        __syncthreads();
    }
}





__global__ void Pre_computer_amp_mapped_modulated(double * __restrict__ static_buf, int* __restrict__ static_list, double* __restrict__ ddest_freq, 
                            int*__restrict__ ddy_list, short* final_buf, short** dynamic_buf, double * __restrict__ dstartFreq, float*__restrict__ amp_map, double* __restrict__ phase,double * new_phase,
                            float* __restrict__ amp,float* __restrict__ new_amp) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    double dump;
    double sum;
    double sum1;
    size_t static_bufferlength_cuda2 = static_bufferlength_cuda / 2;
    double2* static_buf2 = reinterpret_cast<double2*>(static_buf);
    __shared__ double startFreq[1024];
    __shared__ double dest_freq[1024];
    __shared__ float ampmap[1024];
    __shared__ float oldamp[1024];
    __shared__ float newamp[1024];
    __shared__ double phaselist[1024];
    for (int buffer_index=0;buffer_index<channel_num_cuda;buffer_index++){
        for (int iter =0;iter<= (dynamic_num_cuda[buffer_index]-1)/256+1;iter++){
            if(iter*256+threadIdx.x<dynamic_num_cuda[buffer_index]){
                size_t dynamic_index = dynamic_tone_count_cuda[buffer_index]+iter*256+threadIdx.x;
                int shared_index = ddy_list[dynamic_index];
                startFreq[iter*256+threadIdx.x] = dstartFreq[shared_index];
                double df = ddest_freq[dynamic_index];
                dest_freq[iter*256+threadIdx.x] = df;
                phaselist[iter*256+threadIdx.x] = phase[shared_index];
                ampmap[iter*256+threadIdx.x] = amp_map[ __float2uint_rn(static_cast<float>(df)*map_interval_cuda)];
                newamp[iter*256+threadIdx.x] = new_amp[dynamic_index];
                oldamp[iter*256+threadIdx.x] = amp[shared_index];
            }
        }
        __syncthreads();

        double2 temp_sums;
        int dynamic_num = dynamic_num_cuda[buffer_index];
        float scaler = static_cast<float>(dynamic_num) * static_cast<float>(istatic_num_cuda[buffer_index]);
        sum = 0;
        sum1 = 0;
        size_t static_tone_count_start = tone_count_cuda[buffer_index]-dynamic_tone_count_cuda[buffer_index];
        size_t static_tone_count_end = tone_count_cuda[buffer_index+1]-dynamic_tone_count_cuda[buffer_index+1];
        if (static_tone_count_end-static_tone_count_start>0){
            for(int iter=static_tone_count_start;iter<static_tone_count_end;iter++){
                temp_sums = static_buf2[static_bufferlength_cuda2*static_list[iter]+i];
                sum += temp_sums.x;
                sum1 += temp_sums.y;
            }
        }
        double temp_sumst = 0.;
        double temp_sumst1 = 0.;
        float coefficient_correcter = 0;
        for (int j=0; j<dynamic_num;j++){
            double phi = __fma_rn(dest_freq[j],dynamic_loopcount_cuda*static_cast<double>(static_bufferlength_cuda)-0.5*dynamic_bufferlength_cuda,0.5*dynamic_bufferlength_cuda*startFreq[j]+phaselist[j]);
            if (i==0) new_phase[ddy_list[dynamic_tone_count_cuda[buffer_index]+j]] = 2.*modf(phi,&dump);
            double ampf = static_cast<double>(ampmap[j]);
            temp_sumst += ampf * sinpi (2.*modf(__fma_rn(dest_freq[j] , 2.*static_cast<double>(i),phi)* illSamplerate_cuda,&dump))*newamp[j];
            temp_sumst1 += ampf * sinpi (2.*modf(__fma_rn(dest_freq[j] , 2.*static_cast<double>(i)+1.,phi)* illSamplerate_cuda,&dump))*newamp[j];
            coefficient_correcter += ampmap[j];
        }
        coefficient_correcter = 32767.f/coefficient_correcter*scaler;
        final_buf[buffer_index+2*i*channel_num_cuda] = static_cast<short>(fma(temp_sumst,static_cast<double>(coefficient_correcter),sum));
        final_buf[buffer_index+(2*i+1)*channel_num_cuda] = static_cast<short>(fma(temp_sumst1,static_cast<double>(coefficient_correcter),sum1));
        float suml = static_cast<float>(sum1);
        float sums = static_cast<float>(sum);
        for (int counter = 0; counter< dynamic_loopcount_cuda*0.5; counter++){
            size_t index = counter*static_bufferlength_cuda+i;
            size_t checkpoint = index * 2 + 1;
            double position = static_cast<double>(checkpoint)-1.;
            float temp_sum=0.f;
            coefficient_correcter = 0.f;
            float temp_sum1 = 0.f;
            double ratio = position*idynamic_bufferlength_cuda;
            double ratio1 = ratio + idynamic_bufferlength_cuda;
            if (checkpoint < __double2ull_rn(dynamic_bufferlength_cuda)){
                for (int j=0; j<dynamic_num;j++){
                    float old_amp = oldamp[j];
                    float amp_ramper = __fmaf_rn(newamp[j] - old_amp, static_cast<float>(ratio),old_amp);
                    double startFrequency = startFreq[j];
                    double freqeuncyDiff = dest_freq[j]-startFrequency;
                    double phi = phaselist[j];
                    double phase_c = position * illSamplerate_cuda * __fma_rn(freqeuncyDiff* ratio *ratio *ratio , __fma_rn(ratio-3. , ratio,2.5),startFrequency);
                    double pc = M_PI*2.*modf( phase_c+phi,&dump);
                    float amp_factor = amp_map[__float2int_rn(static_cast<float>(__fma_rn(freqeuncyDiff* ratio *ratio *ratio , __fma_rn(__fma_rn(6.,ratio,-15.) , ratio,10.),startFrequency))*map_interval_cuda)];
                    double phase_c1 = (position + 1.) * illSamplerate_cuda * __fma_rn(freqeuncyDiff* ratio1 *ratio1 *ratio1 , __fma_rn(ratio1-3. , ratio1,2.5),startFrequency);
                    double pc1 = M_PI*2.*modf( phase_c1+phi,&dump);
                    half2 temp = h2sin (__floats2half2_rn(pc,pc1));
                    temp_sum +=  __low2float(temp)*amp_factor*amp_ramper;
                    temp_sum1 +=  __high2float(temp)*amp_factor*amp_ramper;
                    coefficient_correcter += amp_factor;
                }
            } else if (checkpoint == __double2ull_rn(dynamic_bufferlength_cuda)){
                for (int j=0; j<dynamic_num;j++){
                    double startFrequency = startFreq[j];
                    double freqeuncyDiff = dest_freq[j]-startFrequency;
                    double phi = phaselist[j];
                    double phase_c = position * illSamplerate_cuda * __fma_rn(freqeuncyDiff* ratio *ratio *ratio , __fma_rn(ratio-3. , ratio,2.5),startFrequency);
                    double pc = M_PI*2.*modf( phase_c+phi,&dump);
                    double pc1 = M_PI*2.*modf(__fma_rn(__fma_rn(0.5,freqeuncyDiff,startFrequency)*dynamic_bufferlength_cuda,illSamplerate_cuda,phi),&dump);
                    float amp_factor = amp_map[__float2int_rn(static_cast<float>(startFrequency+freqeuncyDiff)*map_interval_cuda)];
                    half2 temp = h2sin (__floats2half2_rn(pc,pc1));
                    temp_sum +=  __low2float(temp)*amp_factor*newamp[j];
                    temp_sum1 +=  __high2float(temp)*amp_factor*newamp[j];
                    coefficient_correcter += amp_factor;
                }
            }else{
                for (int j=0; j<dynamic_num;j++){
                    double startFrequency = startFreq[j];
                    double freqeuncyDiff = dest_freq[j]-startFrequency;
                    double phi = phaselist[j];
                    double pc = M_PI*2.*modf( __fma_rn((startFrequency+freqeuncyDiff) *(position-dynamic_bufferlength_cuda) ,illSamplerate_cuda,__fma_rn(__fma_rn(0.5,freqeuncyDiff,startFrequency)*dynamic_bufferlength_cuda,illSamplerate_cuda,phi)),&dump);
                    float amp_factor = amp_map[__float2int_rn(static_cast<float>(startFrequency+freqeuncyDiff)*map_interval_cuda)];
                    double pc1 = M_PI*2.*modf( __fma_rn((startFrequency+freqeuncyDiff) *(position-dynamic_bufferlength_cuda+1.) ,illSamplerate_cuda,__fma_rn(__fma_rn(0.5,freqeuncyDiff,startFrequency)*dynamic_bufferlength_cuda,illSamplerate_cuda,phi)),&dump);
                    half2 temp = h2sin (__floats2half2_rn(pc,pc1));
                    temp_sum +=  __low2float(temp)*amp_factor*newamp[j];
                    temp_sum1 +=  __high2float(temp)*amp_factor*newamp[j];
                    coefficient_correcter += amp_factor;
                }
            }
            coefficient_correcter = 32767.f/coefficient_correcter*scaler;
            short2* dynamic_buf2 = reinterpret_cast<short2*>(dynamic_buf[buffer_index]);
            dynamic_buf2[index] = {static_cast<short>(__fmaf_rn(temp_sum,coefficient_correcter,sums)),static_cast<short>(__fmaf_rn(temp_sum1,coefficient_correcter,suml))};
        }
        __syncthreads();
    }
}


void cuda_cleanup ()
    {
        cudaDeviceSynchronize();
        for (int i = 0; i < lNumCh; ++i){
            cudaFree (summed_buffer[i]);
            summed_buffer[i] = NULL;
            cudaFree (saved_buffer[i]);
            saved_buffer[i] = NULL;
            cudaFree (dynamic_saved_buffer[i]);
            dynamic_saved_buffer[i] = NULL;
        }
        cudaFree(summed_buffer_cuda);
        summed_buffer_cuda = NULL;
        cudaFree(final_buffer_cuda);
        final_buffer_cuda = NULL;
        cudaFree(saved_buffer_cuda);
        saved_buffer_cuda = NULL;
        cudaFree(dynamic_saved_buffer_cuda);
        dynamic_saved_buffer_cuda = NULL;
        cudaFree(dynamic_list_cuda);
        dynamic_list_cuda = NULL;
        cudaFree(amp_list_cuda);
        amp_list_cuda = NULL;
        cudaFree(final_amp_list_cuda);
        final_amp_list_cuda = NULL;
        cudaFree(static_list_cuda);
        static_list_cuda = NULL;
        cudaFree(real_destination_freq_cuda);
        real_destination_freq_cuda = NULL;
        cudaFree(real_static_freq_cuda);
        real_static_freq_cuda = NULL;
        cudaFree(static_buffer_cuda);
        static_buffer_cuda = NULL;
        if (!update_flag){
            cudaFree(phase_list_cuda);
            phase_list_cuda = NULL;
            cudaFree(new_phase_list_cuda);
            new_phase_list_cuda = NULL;
        }
        if (if_mapped){
            cudaFree(amp_map_static_cuda);
            amp_map_static_cuda = NULL;
            cudaFree(total_divider_cuda);
            total_divider_cuda = NULL;
        }
        cudaFree(update_index_map_cuda);
        update_index_map_cuda = NULL;
        if (eCudaErr=cudaPeekAtLastError()) printf("Cuda Buffer Clean Failed: %s\n",cudaGetErrorString(eCudaErr));
        cudaDeviceSynchronize();
    }

int dynamicBufferInit(){
    for (int i = 0; i < lNumCh; i++){
        eCudaErr = cudaMalloc ((void **)&saved_buffer[i], sizeof(double)/sizeof(short)*lBytesPerChannelInNotifySize); //Configure software buffer
        if (eCudaErr != cudaSuccess)
            {
            printf ("Allocating saved_buffer on GPU failed: %s\n",cudaGetErrorString(eCudaErr));
            return 1;   
            }

        eCudaErr = cudaMalloc ((void **)&dynamic_saved_buffer[i], dynamic_loopcount*lBytesPerChannelInNotifySize); //Configure software buffer
        if (eCudaErr != cudaSuccess)
            {
            printf ("Allocating dynamic_saved_buffer on GPU failed: %s\n",cudaGetErrorString(eCudaErr));
            return 1;   
            }
    }
    
    eCudaErr = cudaMalloc ((void**)&real_destination_freq_cuda, dynamic_total*sizeof(double)); //Configure software buffer
    if (eCudaErr != cudaSuccess)
        {
        printf ("Allocating real_destination_freq_cuda on GPU failed: %s\n",cudaGetErrorString(eCudaErr));

        return 1;
        }    
    eCudaErr = cudaMemcpy(real_destination_freq_cuda,real_destination_freq,dynamic_total*sizeof(double),cudaMemcpyHostToDevice);
    if (eCudaErr != cudaSuccess)
        {
        printf ("cudaMemcpy real_destination_freq_cuda on GPU failed: %s\n",cudaGetErrorString(eCudaErr));

        return 1;
        } 

    eCudaErr = cudaMalloc ((void**)&dynamic_list_cuda, dynamic_total*sizeof(int)); //Configure software buffer
    if (eCudaErr != cudaSuccess)
        {
        printf ("Allocating dynamic_list_cuda on GPU failed: %s\n",cudaGetErrorString(eCudaErr));

        return EXIT_FAILURE;
        }

    eCudaErr = cudaMalloc ((void**)&static_list_cuda, sizeof(static_list)); //Configure software buffer
    if (eCudaErr != cudaSuccess)
        {
        printf ("Allocating static_list_cuda on GPU failed: %s\n",cudaGetErrorString(eCudaErr));

        return EXIT_FAILURE;
        }
    eCudaErr = cudaMalloc ((void**)&final_buffer_cuda, lNumCh*lBytesPerChannelInNotifySize); //Configure software buffer
    if (eCudaErr != cudaSuccess)
        {
        printf ("Allocating final_buffer_cuda on GPU failed: %s\n",cudaGetErrorString(eCudaErr));

        return EXIT_FAILURE;
        }
    eCudaErr = cudaMalloc ((void**)&saved_buffer_cuda, sizeof(saved_buffer)); //Configure software buffer
    if (eCudaErr != cudaSuccess)
        {
        printf ("Allocating saved_buffer_cuda on GPU failed: %s\n",cudaGetErrorString(eCudaErr));

        return EXIT_FAILURE;
        }
    eCudaErr = cudaMalloc ((void**)&dynamic_saved_buffer_cuda, sizeof(dynamic_saved_buffer)); //Configure software buffer
    if (eCudaErr != cudaSuccess)
        {
        printf ("Allocating dynamic_saved_buffer_cuda on GPU failed: %s\n",cudaGetErrorString(eCudaErr));

        return EXIT_FAILURE;
        }

    eCudaErr = cudaMemcpy (static_list_cuda, static_list,sizeof(static_list),cudaMemcpyHostToDevice); //Configure software buffer
    if (eCudaErr != cudaSuccess)
        {
        printf ("cudaMemcpy static_list_cuda on GPU failed: %s\n",cudaGetErrorString(eCudaErr));

        return 1;
        }            
    eCudaErr = cudaMemcpy (dynamic_list_cuda, dynamic_list,dynamic_total*sizeof(int),cudaMemcpyHostToDevice); //Configure software buffer
    if (eCudaErr != cudaSuccess)
        {
        printf ("cudaMemcpy dynamic_list_cuda on GPU failed: %s\n",cudaGetErrorString(eCudaErr));

        return 1;
        }    
    eCudaErr = cudaMemcpy (saved_buffer_cuda, saved_buffer,sizeof(saved_buffer),cudaMemcpyHostToDevice); //Configure software buffer
    if (eCudaErr != cudaSuccess)
        {
        printf ("cudaMemcpy saved_buffer_cuda on GPU failed: %s\n",cudaGetErrorString(eCudaErr));

        return 1;
        }
    eCudaErr = cudaMemcpy (dynamic_saved_buffer_cuda, dynamic_saved_buffer,sizeof(dynamic_saved_buffer),cudaMemcpyHostToDevice); //Configure software buffer
    if (eCudaErr != cudaSuccess)
        {
        printf ("cudaMemcpy dynamic_saved_buffer_cuda on GPU failed: %s\n",cudaGetErrorString(eCudaErr));

        return 1;
        }
    eCudaErr = cudaMemcpyToSymbol(dynamic_num_cuda, &dynamic_num, sizeof(dynamic_num));
    if (eCudaErr != cudaSuccess)
        {
        printf ("cudaMemcpyToSymbol dynamic_num_cuda on GPU failed: %s\n",cudaGetErrorString(eCudaErr));

        return EXIT_FAILURE;
        }
    double dylength = dynamic_buffersize/2;
    eCudaErr = cudaMemcpyToSymbol(dynamic_bufferlength_cuda, &dylength, sizeof(dylength));
    if (eCudaErr != cudaSuccess)
        {
        printf ("cudaMemcpyToSymbol dynamic_bufferlength_cuda on GPU failed: %s\n",cudaGetErrorString(eCudaErr));

        return EXIT_FAILURE;
        }
    double_temp = 1./dylength;
    eCudaErr = cudaMemcpyToSymbol (idynamic_bufferlength_cuda, &double_temp, sizeof(double)); //Configure software buffer
    if (eCudaErr != cudaSuccess)
        {
        printf ("cudaMemcpyToSymbol idynamic_bufferlength_cuda on GPU failed: %s\n",cudaGetErrorString(eCudaErr));

        return 1;
        }
    double_temp = dynamic_loopcount;
    eCudaErr = cudaMemcpyToSymbol (dynamic_loopcount_cuda, &double_temp, sizeof(double_temp)); //Configure software buffer
    if (eCudaErr != cudaSuccess)
        {
        printf ("cudaMemcpyToSymbol dynamic_loopcount_cuda on GPU failed: %s\n",cudaGetErrorString(eCudaErr));

        return 1;
        }
    return 0;
}