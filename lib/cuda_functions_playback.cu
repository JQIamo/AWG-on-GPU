#   include "cuda_functions_playback.h"

short* final_buffer_cuda=nullptr;
short** dynamic_saved_buffer_cuda=nullptr;


__global__ void DynamicMux (unsigned int startPosition, short**__restrict__ buffer,short* pnOut)
    {
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int j = i+startPosition;
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


__global__ void Pre_computer(const double * __restrict__ static_buf, const int* __restrict__ static_list, const float* __restrict__ ddest_freq,
                            const int*__restrict__ ddy_list, short* final_buf, short** dynamic_buf, const float * __restrict__ dstartFreq, const float* __restrict__ phase,float * new_phase){
    // Compute the synthesized waveform containing dynamic tones with uniform amplitude.
    // The computation is executed in a doublet mannar to maximize the memory bandwidth for storing short-type data.
    // It is done this way to use the maximized 128-byte per warp (32-threads, so 32-bit/thread) memory transaction.
    // TODO: Consider the following to further optimize:
    //      1. Make the function buffer_index (channel) dependent and call the function separately for each channel.
    //      2. Use 2D block to unroll the for(int j=0;j<dynamic_num_cuda[buffer_index];j++) loop (only after 1. done).
    //      3. Unroll the for(int counter=0;counter<dynamic_loopcount_cuda/2;counter++) loop as dynamic_loopcount_cuda/2 is known.

    // static_buf: [Input] pointers to the static/initial tones
    // static_list: [Input] pointers to the indexes of static tones (should be sorted in advance to ensure memory coalescing)
    // ddest_freq: [Input] pointers to the destination frequencies
    // ddy_list: [Input] pointers to the mapping of dynamic tone indices to static tone indices
    // final_buf: [Output] pointers to the chirped (static) tones
    // dynamic_buf: [Output] pointers to the dynamic tones
    // dstartFreq: [Input] pointers to the static/initial frequencies
    // phase: [Input] pointers to the Schroeder phases
    // new_phase: [Output] pointers to the new phases after chirping


    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; // using uint to allow chirp time of seconds
    float sum;
    float sum1;
    unsigned int static_bufferlength_cuda2 = static_bufferlength_cuda / 2;
    const double2* static_buf2 = reinterpret_cast<const double2*>(static_buf);
    __shared__ float startFreq[128];
    __shared__ float dest_freq[128];
    __shared__ float phaselist[128];
    double ddynamic_bufferlength_cuda = __uint2double_rn(dynamic_bufferlength_cuda);

    //  Process data on different AWG channels separately
    for (int buffer_index=0;buffer_index<channel_num_cuda;buffer_index++){
        // Load input buffers into the shared memory for fast access per thread
        if(threadIdx.x<dynamic_num_cuda[buffer_index]){
            unsigned int dynamic_index = dynamic_tone_count_cuda[buffer_index]+threadIdx.x;
            int shared_index = ddy_list[dynamic_index];
            startFreq[threadIdx.x] = dstartFreq[shared_index];
            dest_freq[threadIdx.x] = ddest_freq[dynamic_index];
            phaselist[threadIdx.x] = phase[shared_index];
        }
        __syncthreads();

        // Compute the waveforms
        double2 temp_sums;
        sum = 0;
        sum1 = 0;
        // Extract the tones that are not chirped
        unsigned int static_tone_count_start = tone_count_cuda[buffer_index]-dynamic_tone_count_cuda[buffer_index];
        unsigned int static_tone_count_end = tone_count_cuda[buffer_index+1]-dynamic_tone_count_cuda[buffer_index+1];
        if (static_tone_count_end-static_tone_count_start>0){
            for(int iter=static_tone_count_start;iter<static_tone_count_end;iter++){
                temp_sums = static_buf2[static_bufferlength_cuda2*static_list[iter]+i];
                sum += __double2float_rn(temp_sums.x);
                sum1 += __double2float_rn(temp_sums.y);
            }
        }

        // Compute the static waveform after the chirping and store in final_buf
        float temp_sumst = 0.;
        float temp_sumst1 = 0.;
        float ipower_normalizer = 32767.f *imaximal_tweezer_number_cuda[buffer_index];
        for (int j=0; j<dynamic_num_cuda[buffer_index];j++){
            double phi = modfma(static_cast<double>(dest_freq[j]*illSamplerate_cuda),
                    __fma_rn(0.5*ddynamic_bufferlength_cuda,__double2float_rn(__fmaf_rn(0.5f,__frcp_rn(dest_freq[j])*startFreq[j],-1.f)),
                        __uint2double_rn(static_bufferlength_cuda*dynamic_loopcount_cuda)),phaselist[j]);
            if (i==0) new_phase[ddy_list[dynamic_tone_count_cuda[buffer_index]+j]] = phi;
            temp_sumst +=  __double2float_rn(sinpi(2.*modfma(dest_freq[j]* illSamplerate_cuda , 2.*__uint2double_rn(i),phi)));
            temp_sumst1 +=  __double2float_rn(sinpi(2.*modfma(dest_freq[j]* illSamplerate_cuda , __fma_rn(2.,__uint2double_rn(i),1.),phi)));
        }
        final_buf[buffer_index+2*i*channel_num_cuda] = static_cast<short>(__fmaf_rn(temp_sumst,ipower_normalizer,__double2float_rn(sum)));
        final_buf[buffer_index+(2*i+1)*channel_num_cuda] = static_cast<short>(__fmaf_rn(temp_sumst1,ipower_normalizer,__double2float_rn(sum1)));

        // Compute the waveform during the dynamic process and store in dynamic_buf
        unsigned int index = i;
        unsigned int position = i * 2; // the 2 is because of the doublet format
        for (int counter = 0; counter< dynamic_loopcount_cuda/2; counter++){
            // Loop in chunks of static bufferlength to match with the buffer pipelining/thread blocksize
            float suml = __double2float_rn(sum1);
            float sums = __double2float_rn(sum);
            double ratio = __uint2double_rn(position)*idynamic_bufferlength_cuda;
            double ratio1 = ratio + idynamic_bufferlength_cuda;
            for (int j=0; j<dynamic_num_cuda[buffer_index];j++){
                double startFrequency = startFreq[j];
                double frequencyDiff = dest_freq[j]-startFrequency;
                double phi = phaselist[j];
                double pc;
                double pc1;
                if (position < dynamic_bufferlength_cuda){
                // When the position is before the point where the chirping ends, compute the waveform as a minimum-jerk trajectory
                    pc = M_2PI*modfma(__uint2double_rn(position), illSamplerate_cuda * mini_jerk_coeff(frequencyDiff, ratio, startFrequency), phi);
                    pc1 = M_2PI*modfma(__uint2double_rn(position+1), illSamplerate_cuda * mini_jerk_coeff(frequencyDiff, ratio1, startFrequency), phi);
                }else{
                // When the position is after the point where the chirping ends, compute the waveform as static waveform with phase offset
                    pc = M_2PI * modfma((startFrequency+frequencyDiff)*illSamplerate_cuda,__uint2double_rn(position),
                        after_chirp_phase(frequencyDiff, phi, ddynamic_bufferlength_cuda, illSamplerate_cuda));
                    pc1 = M_2PI * modfma((startFrequency+frequencyDiff)*illSamplerate_cuda,__uint2double_rn(position+1),
                        after_chirp_phase(frequencyDiff, phi, ddynamic_bufferlength_cuda, illSamplerate_cuda));
                }
                half2 temp = h2sin (__floats2half2_rn(__double2float_rn(pc),__double2float_rn(pc1)));
                sums +=  __low2float(temp)*ipower_normalizer;
                suml +=  __high2float(temp)*ipower_normalizer;
            }
            short2* dynamic_buf2 = reinterpret_cast<short2*>(dynamic_buf[buffer_index]);
            dynamic_buf2[index] = {static_cast<short>(sums),static_cast<short>(suml)};
            index += static_bufferlength_cuda;
            position += static_bufferlength_cuda * 2;
        }
        __syncthreads();
    }
}




__global__ void Pre_computer_amp(const double * __restrict__ static_buf, const int* __restrict__ static_list, const float* __restrict__ ddest_freq,
                            const int*__restrict__ ddy_list, short* final_buf, short** dynamic_buf, const float * __restrict__ dstartFreq,
                            const float* __restrict__ phase, float * new_phase, const float* __restrict__ amp) {
    // Compute the synthesized waveform containing dynamic tones with amplitude factors.
    // The computation is executed in a doublet mannar to maximize the memory bandwidth for storing short-type data.
    // It is done this way to use the maximized 128-byte per warp (32-threads, so 32-bit/thread) memory transaction.
    // TODO: Consider the following to further optimize:
    //      1. Make the function buffer_index (channel) dependent and call the function separately for each channel.
    //      2. Use 2D block to unroll the for(int j=0;j<dynamic_num_cuda[buffer_index];j++) loop (only after 1. done).
    //      3. Unroll the for(int counter=0;counter<dynamic_loopcount_cuda/2;counter++) loop as dynamic_loopcount_cuda/2 is known.

    // static_buf: [Input] pointers to the static/initial tones
    // static_list: [Input] pointers to the indexes of static tones (should be sorted in advance to ensure memory coalescing)
    // ddest_freq: [Input] pointers to the destination frequencies
    // ddy_list: [Input] pointers to the mapping of dynamic tone indices to static tone indices
    // final_buf: [Output] pointers to the chirped (static) tones
    // dynamic_buf: [Output] pointers to the dynamic tones
    // dstartFreq: [Input] pointers to the static/initial frequencies
    // phase: [Input] pointers to the Schroeder phases
    // new_phase: [Output] pointers to the new phases after chirping
    // amp: [Input] pointers to the amplitude factors


    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    float sum;
    float sum1;
    unsigned int static_bufferlength_cuda2 = static_bufferlength_cuda / 2;
    const double2* static_buf2 = reinterpret_cast<const double2*>(static_buf);
    __shared__ float startFreq[128];
    __shared__ float dest_freq[128];
    __shared__ float amplist[128];
    __shared__ float phaselist[128];
    double ddynamic_bufferlength_cuda = __uint2double_rn(dynamic_bufferlength_cuda);

    //  Process data on different AWG channels separately
    for (int buffer_index=0;buffer_index<channel_num_cuda;buffer_index++){
        // Load input buffers into the shared memory for fast access per thread
        if(threadIdx.x<dynamic_num_cuda[buffer_index]){
            unsigned int dynamic_index = dynamic_tone_count_cuda[buffer_index]+threadIdx.x;
            int shared_index = ddy_list[dynamic_index];
            startFreq[threadIdx.x] = dstartFreq[shared_index];
            dest_freq[threadIdx.x] = ddest_freq[dynamic_index];
            phaselist[threadIdx.x] = phase[shared_index];
            amplist[threadIdx.x] = amp[shared_index];
        }
        __syncthreads();

        // Compute the waveforms
        double2 temp_sums;
        sum = 0.;
        sum1 = 0.;
        // Extract the tones that are not chirped
        unsigned int static_tone_count_start = tone_count_cuda[buffer_index]-dynamic_tone_count_cuda[buffer_index];
        unsigned int static_tone_count_end = tone_count_cuda[buffer_index+1]-dynamic_tone_count_cuda[buffer_index+1];
        if (static_tone_count_end-static_tone_count_start>0){
            for(int iter=static_tone_count_start;iter<static_tone_count_end;iter++){
                temp_sums = static_buf2[static_bufferlength_cuda2*static_list[iter]+i];
                sum += __double2float_rn(temp_sums.x);
                sum1 += __double2float_rn(temp_sums.y);
            }
        }

        // Compute the static waveform after the chirping and store in final_buf
        float temp_sumst = 0.;
        float temp_sumst1 = 0.;
        float ipower_normalizer = 32767.f *imaximal_tweezer_number_cuda[buffer_index];
        for (int j=0; j<dynamic_num_cuda[buffer_index];j++){
            double phi = modfma(static_cast<double>(dest_freq[j]*illSamplerate_cuda),
                    __fma_rn(0.5*ddynamic_bufferlength_cuda,__double2float_rn(__fmaf_rn(0.5f,__frcp_rn(dest_freq[j])*startFreq[j],-1.f)),
                        __uint2double_rn(static_bufferlength_cuda*dynamic_loopcount_cuda)),phaselist[j]);
            if (i==0) new_phase[ddy_list[dynamic_tone_count_cuda[buffer_index]+j]] = phi;
            float ampf = amplist[j];
            temp_sumst += ampf * __double2float_rn(sinpi(2.*modfma(dest_freq[j]* illSamplerate_cuda , 2.*__uint2double_rn(i),phi)));
            temp_sumst1 += ampf * __double2float_rn(sinpi(2.*modfma(dest_freq[j]* illSamplerate_cuda , __fma_rn(2.,__uint2double_rn(i),1.),phi)));
        }
        final_buf[buffer_index+2*i*channel_num_cuda] = static_cast<short>(__fmaf_rn(temp_sumst,ipower_normalizer,__double2float_rn(sum)));
        final_buf[buffer_index+(2*i+1)*channel_num_cuda] = static_cast<short>(__fmaf_rn(temp_sumst1,ipower_normalizer,__double2float_rn(sum1)));

        // Compute the waveform during the dynamic process and store in dynamic_buf
        unsigned int index = i;
        unsigned int position = i * 2; // the 2 is because of the doublet format
        for (int counter = 0; counter< dynamic_loopcount_cuda*0.5; counter++){
            // Loop in chunks of static bufferlength to match with the buffer pipelining/thread blocksize
            float suml = sum1;
            float sums = sum;
            double ratio = __uint2double_rn(position)*idynamic_bufferlength_cuda;
            double ratio1 = ratio + idynamic_bufferlength_cuda;
            for (int j=0; j<dynamic_num_cuda[buffer_index];j++) {
                double startFrequency = startFreq[j];
                float ampf = amplist[j]*ipower_normalizer;
                double frequencyDiff = dest_freq[j]-startFrequency;
                double phi = phaselist[j];
                double pc;
                double pc1;
                if (position < dynamic_bufferlength_cuda){
                    // When the position is before the point where the chirping ends, compute the waveform as a minimum-jerk trajectory
                    pc = M_2PI*modfma(__uint2double_rn(position), illSamplerate_cuda * mini_jerk_coeff(frequencyDiff, ratio, startFrequency), phi);
                    pc1 = M_2PI*modfma(__uint2double_rn(position+1), illSamplerate_cuda * mini_jerk_coeff(frequencyDiff, ratio1, startFrequency), phi);
                }else{
                    // When the position is after the point where the chirping ends, compute the waveform as static waveform with phase offset
                    pc = M_2PI * modfma((startFrequency+frequencyDiff)*illSamplerate_cuda,__uint2double_rn(position),
                        after_chirp_phase(frequencyDiff, phi, ddynamic_bufferlength_cuda, illSamplerate_cuda));
                    pc1 = M_2PI * modfma((startFrequency+frequencyDiff)*illSamplerate_cuda,__uint2double_rn(position+1),
                        after_chirp_phase(frequencyDiff, phi, ddynamic_bufferlength_cuda, illSamplerate_cuda));
                }
                half2 temp = h2sin(__floats2half2_rn(__double2float_rn(pc),__double2float_rn(pc1)));
                sums +=  __low2float(temp)*ampf;
                suml +=  __high2float(temp)*ampf;
            }
            short2* dynamic_buf2 = reinterpret_cast<short2*>(dynamic_buf[buffer_index]);
            dynamic_buf2[index] = {static_cast<short>(sums),static_cast<short>(suml)};
            index += static_bufferlength_cuda;
            position += static_bufferlength_cuda * 2;
        }
        __syncthreads();
    }
}



__global__ void Pre_computer_rearrange(int buffer_index, int static_num, int dynamic_num, const int * __restrict__ single_static_map_cuda,
                            const int * __restrict__ single_dynamic_map_cuda, double ** summed_buf, double * static_buf,
                            const float* __restrict__ ddest_freq, short* final_buf, short** dynamic_buf, const float * __restrict__ dstartFreq,
                            const float* __restrict__ phase,float * new_phase, float * rearrange_freq_list_cuda)
{
    // Update the pre-existing waveform and perform chirping for rearrangement on specified channel; pipeline waveforms on other channels
    // The computation is executed in a doublet mannar to maximize the memory bandwidth for storing short-type data.
    // It is done this way to use the maximized 128-byte per warp (32-threads, so 32-bit/thread) memory transaction.
    // TODO: Consider the following to further optimize:
    //      1. Use 2D block to unroll the for(int j=0;j<dynamic_num;j++) loop.
    //      2. Unroll the for(int counter=0;counter<dynamic_loopcount_cuda;counter++) loop as dynamic_loopcount_cuda is known.

    // buffer_index: [Input] the index of the buffer being rearranged
    // static_num: [Input] the number of kept tones
    // dynamic_num: [Input] the number of chirped tones
    // single_static_map_cuda: [Input] pointers to the mapping of static tone indices to buffer indices
    // single_dynamic_map_cuda: [Input] pointers to the mapping of dynamic tone indices to buffer indices
    // summed_buf: [Input] pointers to the summed buffer
    // static_buf: [Input] pointers to the static/initial tones
    // ddest_freq: [Input] pointers to the destination frequencies
    // final_buf: [Output] pointers to the chirped (static) tones
    // dynamic_buf: [Output] pointers to the dynamic tones
    // dstartFreq: [Input] pointers to the static/initial frequencies
    // phase: [Input] pointers to the Schroeder phases
    // new_phase: [Output] pointers to the new phases after chirping
    // rearrange_freq_list_cuda: [Output] pointers to the new frequencies after chirping


    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    float sum;
    float sum1;
    unsigned int static_bufferlength_cuda2 = static_bufferlength_cuda / 2;
    double2* static_buf2 = reinterpret_cast<double2*>(static_buf);
    __shared__ float startFreq[128];
    __shared__ float dest_freq[128];
    __shared__ float phaselist[128];
    __shared__ int dynamic_index_list[128];

    // Load the input buffers
    double ddynamic_bufferlength_cuda = __uint2double_rn(dynamic_bufferlength_cuda);
    if(threadIdx.x<dynamic_num){
        unsigned int dynamic_index = single_dynamic_map_cuda[threadIdx.x];
        dynamic_index_list[threadIdx.x] = dynamic_index;
        float ddest_freq_temp = ddest_freq[threadIdx.x];
        dest_freq[threadIdx.x] = ddest_freq_temp;
        phaselist[threadIdx.x] = phase[dynamic_index];
        startFreq[threadIdx.x] = dstartFreq[dynamic_index];
        if (blockDim.x * blockIdx.x==0) rearrange_freq_list_cuda[dynamic_index] = ddest_freq_temp;
    }
    __syncthreads();

    double2 temp_sums;
    sum = 0;
    sum1 = 0;
    // Load the kept static tones
    for(int iter=0;iter<static_num-dynamic_num;iter++){
        temp_sums = static_buf2[static_bufferlength_cuda2*single_static_map_cuda[iter]+i];
        sum += __double2float_rn(temp_sums.x);
        sum1 += __double2float_rn(temp_sums.y);
    }
    float temp_sumst = 0.;
    float temp_sumst1 = 0.;
    float ipower_normalizer = 32767.f*imaximal_tweezer_number_cuda[buffer_index];

    // Compute the chirped static tones
    for (int j=0; j<dynamic_num;j++){
        double phi = modfma(static_cast<double>(dest_freq[j]*illSamplerate_cuda),
                    __fma_rn(0.5*ddynamic_bufferlength_cuda,__double2float_rn(__fmaf_rn(0.5f,__frcp_rn(dest_freq[j])*startFreq[j],-1.f)),
                        __uint2double_rn(static_bufferlength_cuda*dynamic_loopcount_cuda)),phaselist[j]);
        if (i==0) new_phase[dynamic_index_list[j]] = phi;
        float temp_data =  ipower_normalizer * __double2float_rn(sinpi(2.*modfma(dest_freq[j]* illSamplerate_cuda , 2.*__uint2double_rn(i),phi)));
        float temp_data1 =  ipower_normalizer * __double2float_rn(sinpi(2.*modfma(dest_freq[j]* illSamplerate_cuda , __fma_rn(2.,__uint2double_rn(i),1.),phi)));
        temp_sumst +=  temp_data;
        temp_sumst1 +=  temp_data1;
        static_buf2[static_bufferlength_cuda2*dynamic_index_list[j]+i] =  {temp_data,temp_data1};
    }

    // Compute the chirped dynamic tones
    for (int ch=0; ch<channel_num_cuda;ch++){
        if (ch==buffer_index){
            // Compute the rearrangement on the specified channel
            double ttemp = temp_sumst+sum;
            double ttemp1 = temp_sumst1+sum1;
            final_buf[ch+2*i*channel_num_cuda] = static_cast<short>(ttemp);
            final_buf[ch+(2*i+1)*channel_num_cuda] = static_cast<short>(ttemp1);
            summed_buf[ch][2*i] = ttemp;
            summed_buf[ch][2*i+1] = ttemp1;
            unsigned int index = i;
            unsigned int position = i * 2;
            for (int counter = 0; counter< dynamic_loopcount_cuda; counter++){
                float suml = static_cast<float>(sum1);
                float sums = static_cast<float>(sum);
                double ratio = __uint2double_rn(position)*idynamic_bufferlength_cuda;
                double ratio1 = ratio + idynamic_bufferlength_cuda;
                for (int j=0; j<dynamic_num;j++){
                    double startFrequency = startFreq[j];
                    double frequencyDiff = dest_freq[j]-startFrequency;
                    if (i==0) printf("frequencyDiff=%f\n",frequencyDiff);
                    double phi = phaselist[j];
                    double pc;
                    double pc1;
                    if (position < dynamic_bufferlength_cuda){
                        // When the position is before the point where the chirping ends, compute the waveform as a minimum-jerk trajectory
                        pc = M_2PI*modfma(__uint2double_rn(position), illSamplerate_cuda * mini_jerk_coeff(frequencyDiff, ratio, startFrequency), phi);
                        pc1 = M_2PI*modfma(__uint2double_rn(position+1), illSamplerate_cuda * mini_jerk_coeff(frequencyDiff, ratio1, startFrequency), phi);
                    }else{
                        // When the position is after the point where the chirping ends, compute the waveform as static waveform with phase offset
                        pc = M_2PI * modfma((startFrequency+frequencyDiff)*illSamplerate_cuda,__uint2double_rn(position),
                            after_chirp_phase(frequencyDiff, phi, ddynamic_bufferlength_cuda, illSamplerate_cuda));
                        pc1 = M_2PI * modfma((startFrequency+frequencyDiff)*illSamplerate_cuda,__uint2double_rn(position+1),
                            after_chirp_phase(frequencyDiff, phi, ddynamic_bufferlength_cuda, illSamplerate_cuda));
                    }
                    half2 temp = h2sin (__floats2half2_rn(__double2float_rn(pc),__double2float_rn(pc1)));
                    sums +=  __low2float(temp)*ipower_normalizer;
                    suml +=  __high2float(temp)*ipower_normalizer;
                }
                short2* dynamic_buf2 = reinterpret_cast<short2*>(dynamic_buf[ch]);
                dynamic_buf2[index] = {static_cast<short>(sums),static_cast<short>(suml)};
                index += static_bufferlength_cuda2;
                position += static_bufferlength_cuda2 * 2;
            }
        }else{
            // Pipeline the waveforms on the unselected channels
            short out = static_cast<short>(summed_buf[ch][2*i]);
            short out1 = static_cast<short>(summed_buf[ch][2*i+1]);
            final_buf[ch+2*i*channel_num_cuda] = out;
            final_buf[ch+(2*i+1)*channel_num_cuda] = out1;
            unsigned int index = i;
            for (int counter = 0; counter< dynamic_loopcount_cuda; counter++){
                short2* dynamic_buf2 = reinterpret_cast<short2*>(dynamic_buf[ch]);
                dynamic_buf2[index] = {out,out1};
                index += static_bufferlength_cuda2;
            }
        }
    }
    __syncthreads();
}


__global__ void Pre_computer_rearrange_amp(int buffer_index, int static_num, int dynamic_num, const int * __restrict__ single_static_map_cuda,
                            const int * __restrict__ single_dynamic_map_cuda, double ** summed_buf, double * static_buf,
                            const float* __restrict__ ddest_freq, short* final_buf, short** dynamic_buf, const float * __restrict__ dstartFreq,
                            const float* __restrict__ phase,float * new_phase, float * rearrange_freq_list_cuda, const float* __restrict__ amp)
{
    // Update the pre-existing waveform and perform chirping for rearrangement on specified channel; pipeline waveforms on other channels
    // The computation is executed in a doublet mannar to maximize the memory bandwidth for storing short-type data.
    // It is done this way to use the maximized 128-byte per warp (32-threads, so 32-bit/thread) memory transaction.
    // TODO: Consider the following to further optimize:
    //      1. Use 2D block to unroll the for(int j=0;j<dynamic_num;j++) loop.
    //      2. Unroll the for(int counter=0;counter<dynamic_loopcount_cuda;counter++) loop as dynamic_loopcount_cuda is known.

    // buffer_index: [Input] the index of the buffer being rearranged
    // static_num: [Input] the number of kept tones
    // dynamic_num: [Input] the number of chirped tones
    // single_static_map_cuda: [Input] pointers to the mapping of static tone indices to buffer indices
    // single_dynamic_map_cuda: [Input] pointers to the mapping of dynamic tone indices to buffer indices
    // summed_buf: [Input] pointers to the summed buffer
    // static_buf: [Input] pointers to the static/initial tones
    // ddest_freq: [Input] pointers to the destination frequencies
    // final_buf: [Output] pointers to the chirped (static) tones
    // dynamic_buf: [Output] pointers to the dynamic tones
    // dstartFreq: [Input] pointers to the static/initial frequencies
    // phase: [Input] pointers to the Schroeder phases
    // new_phase: [Output] pointers to the new phases after chirping
    // rearrange_freq_list_cuda: [Output] pointers to the new frequencies after chirping


    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    float sum;
    float sum1;
    unsigned int static_bufferlength_cuda2 = static_bufferlength_cuda / 2;
    double2* static_buf2 = reinterpret_cast<double2*>(static_buf);
    __shared__ float startFreq[128];
    __shared__ float dest_freq[128];
    __shared__ float amplist[128];
    __shared__ float phaselist[128];
    __shared__ int dynamic_index_list[128];

    // Load the input buffers
    double ddynamic_bufferlength_cuda = __uint2double_rn(dynamic_bufferlength_cuda);
    if(threadIdx.x<dynamic_num){
        unsigned int dynamic_index = single_dynamic_map_cuda[threadIdx.x];
        dynamic_index_list[threadIdx.x] = dynamic_index;
        float ddest_freq_temp = ddest_freq[threadIdx.x];
        dest_freq[threadIdx.x] = ddest_freq_temp;
        phaselist[threadIdx.x] = phase[dynamic_index];
        startFreq[threadIdx.x] = dstartFreq[dynamic_index];
        amplist[threadIdx.x] = amp[dynamic_index];
        if (blockDim.x * blockIdx.x==0) rearrange_freq_list_cuda[dynamic_index] = ddest_freq_temp;
    }
    __syncthreads();

    double2 temp_sums;
    sum = 0;
    sum1 = 0;
    // Load the kept static tones
    for(int iter=0;iter<static_num-dynamic_num;iter++){
        temp_sums = static_buf2[static_bufferlength_cuda2*single_static_map_cuda[iter]+i];
        sum += __double2float_rn(temp_sums.x);
        sum1 += __double2float_rn(temp_sums.y);
    }
    double temp_sumst = 0.;
    double temp_sumst1 = 0.;
    float ipower_normalizer = 32767.f*imaximal_tweezer_number_cuda[buffer_index];

    // Compute the chirped static tones
    for (int j=0; j<dynamic_num;j++){
        double phi = modfma(static_cast<double>(dest_freq[j]*illSamplerate_cuda),
                    __fma_rn(0.5*ddynamic_bufferlength_cuda,__double2float_rn(__fmaf_rn(0.5f,__frcp_rn(dest_freq[j])*startFreq[j],-1.f)),
                        __uint2double_rn(static_bufferlength_cuda*dynamic_loopcount_cuda)),phaselist[j]);
        if (i==0) new_phase[dynamic_index_list[j]] = phi;
        float ampf = amplist[j] * __double2float_rn(ipower_normalizer);
        float temp_data =  ampf * __double2float_rn(sinpi(2.*modfma(dest_freq[j]* illSamplerate_cuda , 2.*__uint2double_rn(i),phi)));
        float temp_data1 =  ampf * __double2float_rn(sinpi(2.*modfma(dest_freq[j]* illSamplerate_cuda , __fma_rn(2.,__uint2double_rn(i),1.),phi)));
        temp_sumst +=  temp_data;
        temp_sumst1 +=  temp_data1;
        static_buf2[static_bufferlength_cuda2*dynamic_index_list[j]+i] =  {temp_data,temp_data1};
    }

    // Compute the chirped dynamic tones
    for (int ch=0; ch<channel_num_cuda;ch++){
        if (ch==buffer_index){
            // Compute the rearrangement on the specified channel
            double ttemp = temp_sumst+sum;
            double ttemp1 = temp_sumst1+sum1;
            final_buf[ch+2*i*channel_num_cuda] = static_cast<short>(ttemp);
            final_buf[ch+(2*i+1)*channel_num_cuda] = static_cast<short>(ttemp1);
            summed_buf[ch][2*i] = ttemp;
            summed_buf[ch][2*i+1] = ttemp1;
            unsigned int index = i;
            unsigned int position = i * 2;
            for (int counter = 0; counter< dynamic_loopcount_cuda; counter++){
                float suml = static_cast<float>(sum1);
                float sums = static_cast<float>(sum);
                double ratio = __uint2double_rn(position)*idynamic_bufferlength_cuda;
                double ratio1 = ratio + idynamic_bufferlength_cuda;
                for (int j=0; j<dynamic_num;j++){
                    double startFrequency = startFreq[j];
                    double frequencyDiff = dest_freq[j]-startFrequency;
                    double phi = phaselist[j];
                    float ampf = amplist[j]*ipower_normalizer;
                    double pc;
                    double pc1;
                    if (position < dynamic_bufferlength_cuda){
                        // When the position is before the point where the chirping ends, compute the waveform as a minimum-jerk trajectory
                        pc = M_2PI*modfma(__uint2double_rn(position), illSamplerate_cuda * mini_jerk_coeff(frequencyDiff, ratio, startFrequency), phi);
                        pc1 = M_2PI*modfma(__uint2double_rn(position+1), illSamplerate_cuda * mini_jerk_coeff(frequencyDiff, ratio1, startFrequency), phi);
                    }else{
                        // When the position is after the point where the chirping ends, compute the waveform as static waveform with phase offset
                        pc = M_2PI * modfma((startFrequency+frequencyDiff)*illSamplerate_cuda,__uint2double_rn(position),
                            after_chirp_phase(frequencyDiff, phi, ddynamic_bufferlength_cuda, illSamplerate_cuda));
                        pc1 = M_2PI * modfma((startFrequency+frequencyDiff)*illSamplerate_cuda,__uint2double_rn(position+1),
                            after_chirp_phase(frequencyDiff, phi, ddynamic_bufferlength_cuda, illSamplerate_cuda));
                    }
                    half2 temp = h2sin (__floats2half2_rn(__double2float_rn(pc),__double2float_rn(pc1)));
                    sums += ampf * __low2float(temp);
                    suml += ampf * __high2float(temp);
                }
                short2* dynamic_buf2 = reinterpret_cast<short2*>(dynamic_buf[ch]);
                dynamic_buf2[index] = {static_cast<short>(sums),static_cast<short>(suml)};
                index += static_bufferlength_cuda2;
                position += static_bufferlength_cuda;
            }
        }else{
            // Pipeline the waveforms on the unselected channels
            short out = static_cast<short>(summed_buf[ch][2*i]);
            short out1 = static_cast<short>(summed_buf[ch][2*i+1]);
            final_buf[ch+2*i*channel_num_cuda] = out;
            final_buf[ch+(2*i+1)*channel_num_cuda] = out1;
            unsigned int index = i;
            for (int counter = 0; counter< dynamic_loopcount_cuda; counter++){
                short2* dynamic_buf2 = reinterpret_cast<short2*>(dynamic_buf[ch]);
                dynamic_buf2[index] = {out,out1};
                index += static_bufferlength_cuda2;
            }
        }
    }
    __syncthreads();
}



__global__ void Pre_computer_accelerate(float frequencyDiff, int buffer_index, const double * __restrict__ static_buf, const int* __restrict__ static_list,
                            const int*__restrict__ ddy_list, short* final_buf, short** dynamic_buf, const float * __restrict__ dstartFreq, const float* __restrict__ phase,float * new_phase)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; // using uint to allow chirp time of seconds
    float sum;
    float sum1;
    unsigned int static_bufferlength_cuda2 = static_bufferlength_cuda / 2;
    const double2* static_buf2 = reinterpret_cast<const double2*>(static_buf);
    __shared__ float startFreq[32];
    __shared__ float phaselist[32];
    double ddynamic_bufferlength_cuda = __uint2double_rn(dynamic_bufferlength_cuda);
    //  Process data on different AWG channels separately


    // Load input buffers into the shared memory for fast access per thread
    if(threadIdx.x<dynamic_num_cuda[buffer_index]){
        // unsigned int dynamic_index = dynamic_tone_count_cuda[buffer_index]+threadIdx.x;
        int shared_index = ddy_list[dynamic_tone_count_cuda[buffer_index]+threadIdx.x];
        startFreq[threadIdx.x] = dstartFreq[shared_index];
        phaselist[threadIdx.x] = phase[shared_index];
    }
    __syncthreads();


    for (int ch = 0; ch < channel_num_cuda; ch++){
            // Compute the waveforms
        double2 temp_sums;
        sum = 0;
        sum1 = 0;
        // Extract the tones that are not chirped
        unsigned int static_tone_count_start = tone_count_cuda[buffer_index]-dynamic_tone_count_cuda[buffer_index];
        unsigned int static_tone_count_end = tone_count_cuda[buffer_index+1]-dynamic_tone_count_cuda[buffer_index+1];
        if (static_tone_count_end-static_tone_count_start>0){
            for(int iter=static_tone_count_start;iter<static_tone_count_end;iter++){
                temp_sums = static_buf2[static_bufferlength_cuda2*static_list[iter]+i];
                sum += __double2float_rn(temp_sums.x);
                sum1 += __double2float_rn(temp_sums.y);
            }
        }

        if (ch == buffer_index) {
            // Compute the static waveform after the chirping and store in final_buf
            float temp_sumst = 0.;
            float temp_sumst1 = 0.;
            float ipower_normalizer = 32767.f *imaximal_tweezer_number_cuda[buffer_index];
            for (int j=0; j<dynamic_num_cuda[buffer_index];j++){
                float dest_freq = startFreq[j]+frequencyDiff;
                float phi = __double2float_rn(modfma(static_cast<double>(dest_freq*__double2float_rn(illSamplerate_cuda)),
                                __fma_rn(0.5*ddynamic_bufferlength_cuda,static_cast<double>(__fmaf_rn(0.5f,__frcp_rn(dest_freq)*startFreq[j],-1.f)),
                                    __uint2double_rn(static_bufferlength_cuda*dynamic_loopcount_cuda)),phaselist[j]));
                if (i==0) new_phase[ddy_list[dynamic_tone_count_cuda[buffer_index]+j]] = phi;
                temp_sumst +=  sinpif(2.f*modfmaf(dest_freq* __double2float_rn(illSamplerate_cuda) , 2.f*__uint2float_rn(i),phi));
                temp_sumst1 +=  sinpif(2.f*modfmaf(dest_freq* __double2float_rn(illSamplerate_cuda) , __fmaf_rn(2.f,__uint2float_rn(i),1.f),phi));
            }
            final_buf[buffer_index+2*i*channel_num_cuda] = static_cast<short>(__fmaf_rn(temp_sumst,ipower_normalizer,sum));
            final_buf[buffer_index+(2*i+1)*channel_num_cuda] = static_cast<short>(__fmaf_rn(temp_sumst1,ipower_normalizer,sum1));

            // Compute the waveform during the dynamic process and store in dynamic_buf
            unsigned int index = i;
            unsigned int position = i * 2; // the 2 is because of the doublet format
            for (int counter = 0; counter< dynamic_loopcount_cuda/2; counter++){
                // Loop in chunks of static bufferlength to match with the buffer pipelining/thread blocksize
                float suml = sum1;
                float sums = sum;
                float ratio = 0.5f*__double2float_rn(__uint2double_rn(position)*idynamic_bufferlength_cuda); // No larger than 1
                float ratio1 = __fmaf_rn(0.5f, __double2float_rn(idynamic_bufferlength_cuda), ratio); // No larger than 1
                for (int j=0; j<dynamic_num_cuda[buffer_index];j++){
                    float startFrequency = startFreq[j];
                    double phi = phaselist[j]; // double precision is needed for any operation directly related to `position`
                    float pc;
                    float pc1;
                    if (position < dynamic_bufferlength_cuda){
                        // When the `position` is before the point where the chirping ends, compute the waveform as a minimum-jerk trajectory
                        // Note `position` can be a very large number so we need to use double precision for any operation directly related to it
                        pc = M_2PI_f*__double2float_rn(modfma(__uint2double_rn(position), illSamplerate_cuda * mini_jerk_coefff(2.f*frequencyDiff, ratio, startFrequency), phi));
                        pc1 = M_2PI_f*__double2float_rn(modfma(__uint2double_rn(position+1), illSamplerate_cuda * mini_jerk_coefff(2.f*frequencyDiff, ratio1, startFrequency), phi));
                    }else{
                        // When the position is after the point where the chirping ends, compute the waveform as static waveform with phase offset
                        // Note `position - dynamic_bufferlength_cuda` is a relative small value so we can use float precision
                        pc = M_2PI_f * modfmaf((startFrequency+frequencyDiff)*__double2float_rn(illSamplerate_cuda),__uint2float_rn(position),
                            after_accel_phasef(frequencyDiff, phi, ddynamic_bufferlength_cuda, illSamplerate_cuda));
                        pc = M_2PI_f * modfmaf((startFrequency+frequencyDiff)*__double2float_rn(illSamplerate_cuda),__uint2float_rn(position+1),
                            after_accel_phasef(frequencyDiff, phi, ddynamic_bufferlength_cuda, illSamplerate_cuda));
                    }
                    half2 temp = h2sin (__floats2half2_rn(pc,pc1));
                    sums +=  __low2float(temp)*ipower_normalizer;
                    suml +=  __high2float(temp)*ipower_normalizer;
                }
                short2* dynamic_buf2 = reinterpret_cast<short2*>(dynamic_buf[buffer_index]);
                dynamic_buf2[index] = {static_cast<short>(sums),static_cast<short>(suml)};
                index += static_bufferlength_cuda;
                position += static_bufferlength_cuda * 2;
            }
        } else {
            // Pipeline the waveforms on the unselected channels
            final_buf[ch+2*i*channel_num_cuda] = static_cast<short>(sum);
            final_buf[ch+(2*i+1)*channel_num_cuda] = static_cast<short>(sum1);
            unsigned int index = i;
            for (int counter = 0; counter< dynamic_loopcount_cuda; counter++){
                short2* dynamic_buf2 = reinterpret_cast<short2*>(dynamic_buf[ch]);
                dynamic_buf2[index] = {static_cast<short>(sum),static_cast<short>(sum1)};
                index += static_bufferlength_cuda2;
            }
        }
    }
    __syncthreads();
}



__global__ void Pre_computer_accelerate_amp(float frequencyDiff, int buffer_index, const double * __restrict__ static_buf, const int* __restrict__ static_list,
                            const int*__restrict__ ddy_list, short* final_buf, short** dynamic_buf, const float * __restrict__ dstartFreq, const float* __restrict__ phase,float * new_phase,
                            const float* __restrict__ amp)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; // using uint to allow chirp time of seconds
    float sum;
    float sum1;
    unsigned int static_bufferlength_cuda2 = static_bufferlength_cuda / 2;
    const double2* static_buf2 = reinterpret_cast<const double2*>(static_buf);
    __shared__ float startFreq[32];
    __shared__ float phaselist[32];
    __shared__ float amplist[32];
    double ddynamic_bufferlength_cuda = __uint2double_rn(dynamic_bufferlength_cuda);
    //  Process data on different AWG channels separately


    // Load input buffers into the shared memory for fast access per thread
    if(threadIdx.x<dynamic_num_cuda[buffer_index]){
        // unsigned int dynamic_index = dynamic_tone_count_cuda[buffer_index]+threadIdx.x;
        int shared_index = ddy_list[dynamic_tone_count_cuda[buffer_index]+threadIdx.x];
        startFreq[threadIdx.x] = dstartFreq[shared_index];
        phaselist[threadIdx.x] = phase[shared_index];
        amplist[threadIdx.x] = amp[shared_index];
    }
    __syncthreads();


    for (int ch = 0; ch < channel_num_cuda; ch++){
            // Compute the waveforms
        double2 temp_sums;
        sum = 0;
        sum1 = 0;
        // Extract the tones that are not chirped
        unsigned int static_tone_count_start = tone_count_cuda[buffer_index]-dynamic_tone_count_cuda[buffer_index];
        unsigned int static_tone_count_end = tone_count_cuda[buffer_index+1]-dynamic_tone_count_cuda[buffer_index+1];
        if (static_tone_count_end-static_tone_count_start>0){
            for(int iter=static_tone_count_start;iter<static_tone_count_end;iter++){
                temp_sums = static_buf2[static_bufferlength_cuda2*static_list[iter]+i];
                sum += __double2float_rn(temp_sums.x);
                sum1 += __double2float_rn(temp_sums.y);
            }
        }

        if (ch == buffer_index) {
            // Compute the static waveform after the chirping and store in final_buf
            float temp_sumst = 0.;
            float temp_sumst1 = 0.;
            float ipower_normalizer = 32767.f *imaximal_tweezer_number_cuda[buffer_index];
            for (int j=0; j<dynamic_num_cuda[buffer_index];j++){
                float dest_freq = startFreq[j]+frequencyDiff;
                float phi = __double2float_rn(modfma(static_cast<double>(dest_freq*__double2float_rn(illSamplerate_cuda)),
                                __fma_rn(0.5*ddynamic_bufferlength_cuda,static_cast<double>(__fmaf_rn(0.5f,__frcp_rn(dest_freq)*startFreq[j],-1.f)),
                                    __uint2double_rn(static_bufferlength_cuda*dynamic_loopcount_cuda)),phaselist[j]));
                if (i==0) new_phase[ddy_list[dynamic_tone_count_cuda[buffer_index]+j]] = phi;
                temp_sumst +=  amplist[j]*sinpif(2.f*modfmaf(dest_freq* __double2float_rn(illSamplerate_cuda) , 2.f*__uint2float_rn(i),phi));
                temp_sumst1 +=  amplist[j]*sinpif(2.f*modfmaf(dest_freq* __double2float_rn(illSamplerate_cuda) , __fmaf_rn(2.f,__uint2float_rn(i),1.f),phi));
            }
            final_buf[buffer_index+2*i*channel_num_cuda] = static_cast<short>(__fmaf_rn(temp_sumst,ipower_normalizer,sum));
            final_buf[buffer_index+(2*i+1)*channel_num_cuda] = static_cast<short>(__fmaf_rn(temp_sumst1,ipower_normalizer,sum1));

            // Compute the waveform during the dynamic process and store in dynamic_buf
            unsigned int index = i;
            unsigned int position = i * 2; // the 2 is because of the doublet format
            for (int counter = 0; counter< dynamic_loopcount_cuda/2; counter++){
                // Loop in chunks of static bufferlength to match with the buffer pipelining/thread blocksize
                float suml = sum1;
                float sums = sum;
                float ratio = 0.5f*__double2float_rn(__uint2double_rn(position)*idynamic_bufferlength_cuda); // No larger than 1
                float ratio1 = __fmaf_rn(0.5f, __double2float_rn(idynamic_bufferlength_cuda), ratio); // No larger than 1
                for (int j=0; j<dynamic_num_cuda[buffer_index];j++){
                    float startFrequency = startFreq[j];
                    double phi = phaselist[j]; // double precision is needed for any operation directly related to `position`
                    float ampl = amplist[j]*ipower_normalizer;
                    float pc;
                    float pc1;
                    if (position < dynamic_bufferlength_cuda){
                        // When the `position` is before the point where the chirping ends, compute the waveform as a minimum-jerk trajectory
                        // Note `position` can be a very large number so we need to use double precision for any operation directly related to it
                        pc = M_2PI_f*__double2float_rn(modfma(__uint2double_rn(position), illSamplerate_cuda * mini_jerk_coefff(2.f*frequencyDiff, ratio, startFrequency), phi));
                        pc1 = M_2PI_f*__double2float_rn(modfma(__uint2double_rn(position+1), illSamplerate_cuda * mini_jerk_coefff(2.f*frequencyDiff, ratio1, startFrequency), phi));
                    }else{
                        // When the position is after the point where the chirping ends, compute the waveform as static waveform with phase offset
                        // Note `position - dynamic_bufferlength_cuda` is a relative small value so we can use float precision
                        pc = M_2PI_f * modfmaf((startFrequency+frequencyDiff)*__double2float_rn(illSamplerate_cuda),__uint2float_rn(position),
                            after_accel_phasef(frequencyDiff, phi, ddynamic_bufferlength_cuda, illSamplerate_cuda));
                        pc = M_2PI_f * modfmaf((startFrequency+frequencyDiff)*__double2float_rn(illSamplerate_cuda),__uint2float_rn(position+1),
                            after_accel_phasef(frequencyDiff, phi, ddynamic_bufferlength_cuda, illSamplerate_cuda));
                    }
                    half2 temp = h2sin (__floats2half2_rn(pc,pc1));
                    sums +=  __low2float(temp)*ampl;
                    suml +=  __high2float(temp)*ampl;
                }
                short2* dynamic_buf2 = reinterpret_cast<short2*>(dynamic_buf[buffer_index]);
                dynamic_buf2[index] = {static_cast<short>(sums),static_cast<short>(suml)};
                index += static_bufferlength_cuda;
                position += static_bufferlength_cuda * 2;
            }
        } else {
            // Pipeline the waveforms on the unselected channels
            // short out = static_cast<short>(summed_buf[ch][2*i]);
            // short out1 = static_cast<short>(summed_buf[ch][2*i+1]);
            final_buf[ch+2*i*channel_num_cuda] = static_cast<short>(sum);
            final_buf[ch+(2*i+1)*channel_num_cuda] = static_cast<short>(sum1);
            unsigned int index = i;
            for (int counter = 0; counter< dynamic_loopcount_cuda; counter++){
                short2* dynamic_buf2 = reinterpret_cast<short2*>(dynamic_buf[ch]);
                dynamic_buf2[index] = {static_cast<short>(sum),static_cast<short>(sum1)};
                index += static_bufferlength_cuda2;
            }
        }
    }
    __syncthreads();
}




__global__ void Pre_computerf(const float * __restrict__ static_buf, const int* __restrict__ static_list, const float* __restrict__ ddest_freq,
                            const int*__restrict__ ddy_list, short* final_buf, short** dynamic_buf, const float * __restrict__ dstartFreq, const float* __restrict__ phase,float * new_phase)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; // using uint to allow chirp time of seconds
    float sum;
    float sum1;
    unsigned int static_bufferlength_cuda2 = static_bufferlength_cuda / 2;
    const float2* static_buf2 = reinterpret_cast<const float2*>(static_buf);
    __shared__ float startFreq[128];
    __shared__ float dest_freq[128];
    __shared__ float phaselist[128];
    double ddynamic_bufferlength_cuda = __uint2double_rn(dynamic_bufferlength_cuda);
    //  Process data on different AWG channels separately
    for (int buffer_index=0;buffer_index<channel_num_cuda;buffer_index++){
        // Load input buffers into the shared memory for fast access per thread
        if(threadIdx.x<dynamic_num_cuda[buffer_index]){
            unsigned int dynamic_index = dynamic_tone_count_cuda[buffer_index]+threadIdx.x;
            int shared_index = ddy_list[dynamic_index];
            startFreq[threadIdx.x] = dstartFreq[shared_index];
            dest_freq[threadIdx.x] = ddest_freq[dynamic_index];
            phaselist[threadIdx.x] = phase[shared_index];
        }
        __syncthreads();

        // Compute the waveforms
        float2 temp_sums;
        sum = 0;
        sum1 = 0;
        // Extract the tones that are not chirped
        unsigned int static_tone_count_start = tone_count_cuda[buffer_index]-dynamic_tone_count_cuda[buffer_index];
        unsigned int static_tone_count_end = tone_count_cuda[buffer_index+1]-dynamic_tone_count_cuda[buffer_index+1];
        if (static_tone_count_end-static_tone_count_start>0){
            for(int iter=static_tone_count_start;iter<static_tone_count_end;iter++){
                temp_sums = static_buf2[static_bufferlength_cuda2*static_list[iter]+i];
                sum += temp_sums.x;
                sum1 += temp_sums.y;
            }
        }

        // Compute the static waveform after the chirping and store in final_buf
        float temp_sumst = 0.;
        float temp_sumst1 = 0.;
        float ipower_normalizer = 32767.f *imaximal_tweezer_number_cuda[buffer_index];
        for (int j=0; j<dynamic_num_cuda[buffer_index];j++){
            float phi = __double2float_rn(modfma(static_cast<double>(dest_freq[j]*__double2float_rn(illSamplerate_cuda)),
                            __fma_rn(0.5*ddynamic_bufferlength_cuda,static_cast<double>(__fmaf_rn(0.5f,__frcp_rn(dest_freq[j])*startFreq[j],-1.f)),
                                __uint2double_rn(static_bufferlength_cuda*dynamic_loopcount_cuda)),phaselist[j]));
            if (i==0) new_phase[ddy_list[dynamic_tone_count_cuda[buffer_index]+j]] = phi;
            temp_sumst +=  sinpif(2.f*modfmaf(dest_freq[j]* __double2float_rn(illSamplerate_cuda) , 2.f*__uint2float_rn(i),phi));
            temp_sumst1 +=  sinpif(2.f*modfmaf(dest_freq[j]* __double2float_rn(illSamplerate_cuda) , __fmaf_rn(2.f,__uint2float_rn(i),1.f),phi));
        }
        final_buf[buffer_index+2*i*channel_num_cuda] = static_cast<short>(__fmaf_rn(temp_sumst,ipower_normalizer,sum));
        final_buf[buffer_index+(2*i+1)*channel_num_cuda] = static_cast<short>(__fmaf_rn(temp_sumst1,ipower_normalizer,sum1));

        // Compute the waveform during the dynamic process and store in dynamic_buf
        unsigned int index = i;
        unsigned int position = i * 2; // the 2 is because of the doublet format
        for (int counter = 0; counter< dynamic_loopcount_cuda/2; counter++){
            // Loop in chunks of static bufferlength to match with the buffer pipelining/thread blocksize
            float suml = sum1;
            float sums = sum;
            float ratio = __double2float_rn(__uint2double_rn(position)*idynamic_bufferlength_cuda); // No larger than 1
            float ratio1 = ratio + __double2float_rn(idynamic_bufferlength_cuda); // No larger than 1
            for (int j=0; j<dynamic_num_cuda[buffer_index];j++){
                float startFrequency = startFreq[j];
                float frequencyDiff = dest_freq[j]-startFrequency;
                double phi = phaselist[j]; // double precision is needed for any operation directly related to `position`
                float pc;
                float pc1;
                if (position < dynamic_bufferlength_cuda){
                    // When the `position` is before the point where the chirping ends, compute the waveform as a minimum-jerk trajectory
                    // Note `position` can be a very large number so we need to use double precision for any operation directly related to it
                    pc = M_2PI_f*__double2float_rn(modfma(__uint2double_rn(position), illSamplerate_cuda * mini_jerk_coefff(frequencyDiff, ratio, startFrequency), phi));
                    pc1 = M_2PI_f*__double2float_rn(modfma(__uint2double_rn(position+1), illSamplerate_cuda * mini_jerk_coefff(frequencyDiff, ratio1, startFrequency), phi));
                }else{
                    // When the position is after the point where the chirping ends, compute the waveform as static waveform with phase offset
                    // Note `position - dynamic_bufferlength_cuda` is a relative small value so we can use float precision
                    pc = M_2PI_f * modfmaf((startFrequency+frequencyDiff)*__double2float_rn(illSamplerate_cuda),__uint2float_rn(position-dynamic_bufferlength_cuda),
                        after_chirp_phasef(frequencyDiff, phi, ddynamic_bufferlength_cuda, illSamplerate_cuda));
                    pc1 = M_2PI_f * modfmaf((startFrequency+frequencyDiff)*__double2float_rn(illSamplerate_cuda),__uint2float_rn(position+1-dynamic_bufferlength_cuda),
                        after_chirp_phasef(frequencyDiff, phi, ddynamic_bufferlength_cuda, illSamplerate_cuda));
                }
                half2 temp = h2sin (__floats2half2_rn(pc,pc1));
                sums +=  __low2float(temp)*ipower_normalizer;
                suml +=  __high2float(temp)*ipower_normalizer;
            }
            short2* dynamic_buf2 = reinterpret_cast<short2*>(dynamic_buf[buffer_index]);
            dynamic_buf2[index] = {static_cast<short>(sums),static_cast<short>(suml)};
            index += static_bufferlength_cuda;
            position += static_bufferlength_cuda * 2;
        }
        __syncthreads();
    }
}



__global__ void Pre_computer_ampf(const float * __restrict__ static_buf, const int* __restrict__ static_list, const float* __restrict__ ddest_freq,
                            const int*__restrict__ ddy_list, short* final_buf, short** dynamic_buf, const float * __restrict__ dstartFreq, const float* __restrict__ phase,float * new_phase,
                            const float* __restrict__ amp)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; // using uint to allow chirp time of seconds
    float sum;
    float sum1;
    unsigned int static_bufferlength_cuda2 = static_bufferlength_cuda / 2;
    const float2* static_buf2 = reinterpret_cast<const float2*>(static_buf);
    __shared__ float startFreq[128];
    __shared__ float dest_freq[128];
    __shared__ float phaselist[128];
    __shared__ float amplist[128];
    double ddynamic_bufferlength_cuda = __uint2double_rn(dynamic_bufferlength_cuda);

    //  Process data on different AWG channels separately
    for (int buffer_index=0;buffer_index<channel_num_cuda;buffer_index++){
        // Load input buffers into the shared memory for fast access per thread
        if(threadIdx.x<dynamic_num_cuda[buffer_index]){
            unsigned int dynamic_index = dynamic_tone_count_cuda[buffer_index]+threadIdx.x;
            int shared_index = ddy_list[dynamic_index];
            startFreq[threadIdx.x] = dstartFreq[shared_index];
            dest_freq[threadIdx.x] = ddest_freq[dynamic_index];
            phaselist[threadIdx.x] = phase[shared_index];
            amplist[threadIdx.x] = amp[shared_index];
        }
        __syncthreads();

        // Compute the waveforms
        float2 temp_sums;
        sum = 0;
        sum1 = 0;
        // Extract the tones that are not chirped
        unsigned int static_tone_count_start = tone_count_cuda[buffer_index]-dynamic_tone_count_cuda[buffer_index];
        unsigned int static_tone_count_end = tone_count_cuda[buffer_index+1]-dynamic_tone_count_cuda[buffer_index+1];
        if (static_tone_count_end-static_tone_count_start>0){
            for(int iter=static_tone_count_start;iter<static_tone_count_end;iter++){
                temp_sums = static_buf2[static_bufferlength_cuda2*static_list[iter]+i];
                sum += temp_sums.x;
                sum1 += temp_sums.y;
            }
        }

        // Compute the static waveform after the chirping and store in final_buf
        float temp_sumst = 0.;
        float temp_sumst1 = 0.;
        float ipower_normalizer = 32767.f *imaximal_tweezer_number_cuda[buffer_index];
        for (int j=0; j<dynamic_num_cuda[buffer_index];j++){
            float phi = __double2float_rn(modfma(static_cast<double>(dest_freq[j]*__double2float_rn(illSamplerate_cuda)),
                            __fma_rn(0.5*ddynamic_bufferlength_cuda,static_cast<double>(__fmaf_rn(0.5f,__frcp_rn(dest_freq[j])*startFreq[j],-1.f)),
                                __uint2double_rn(static_bufferlength_cuda*dynamic_loopcount_cuda)),phaselist[j]));
            if (i==0) new_phase[ddy_list[dynamic_tone_count_cuda[buffer_index]+j]] = phi;
            float ampf = amplist[j];
            temp_sumst +=  ampf * sinpif(2.f*modfmaf(dest_freq[j]* __double2float_rn(illSamplerate_cuda) , 2.f*__uint2float_rn(i),phi));
            temp_sumst1 +=  ampf * sinpif(2.f*modfmaf(dest_freq[j]* __double2float_rn(illSamplerate_cuda) , __fmaf_rn(2.f,__uint2float_rn(i),1.f),phi));
        }
        final_buf[buffer_index+2*i*channel_num_cuda] = static_cast<short>(__fmaf_rn(temp_sumst,ipower_normalizer,sum));
        final_buf[buffer_index+(2*i+1)*channel_num_cuda] = static_cast<short>(__fmaf_rn(temp_sumst1,ipower_normalizer,sum1));

        // Compute the waveform during the dynamic process and store in dynamic_buf
        unsigned int index = i;
        unsigned int position = i * 2; // the 2 is because of the doublet format
        for (int counter = 0; counter< dynamic_loopcount_cuda/2; counter++){
            // Loop in chunks of static bufferlength to match with the buffer pipelining/thread blocksize
            float suml = sum1;
            float sums = sum;
            float ratio = __double2float_rn(__uint2double_rn(position)*idynamic_bufferlength_cuda); // No larger than 1
            float ratio1 = ratio + __double2float_rn(idynamic_bufferlength_cuda); // No larger than 1
            for (int j=0; j<dynamic_num_cuda[buffer_index];j++){
                float startFrequency = startFreq[j];
                float frequencyDiff = dest_freq[j]-startFrequency;
                double phi = phaselist[j]; // double precision is needed for any operation directly related to `position`
                float ampf = amplist[j]*ipower_normalizer;
                float pc;
                float pc1;
                if (position < dynamic_bufferlength_cuda){
                    // When the `position` is before the point where the chirping ends, compute the waveform as a minimum-jerk trajectory
                    // Note `position` can be a very large number so we need to use double precision for any operation directly related to it
                    pc = M_2PI_f*__double2float_rn(modfma(__uint2double_rn(position), illSamplerate_cuda * mini_jerk_coefff(frequencyDiff, ratio, startFrequency), phi));
                    pc1 = M_2PI_f*__double2float_rn(modfma(__uint2double_rn(position+1), illSamplerate_cuda * mini_jerk_coefff(frequencyDiff, ratio1, startFrequency), phi));
                }else{
                    // When the position is after the point where the chirping ends, compute the waveform as static waveform with phase offset
                    // Note position-dynamic_bufferlength_cuda is a relative small value so we can use float precision
                    pc = M_2PI_f * modfmaf((startFrequency+frequencyDiff)*__double2float_rn(illSamplerate_cuda),__uint2float_rn(position-dynamic_bufferlength_cuda),
                        after_chirp_phasef(frequencyDiff, phi, ddynamic_bufferlength_cuda, illSamplerate_cuda));
                    pc1 = M_2PI_f * modfmaf((startFrequency+frequencyDiff)*__double2float_rn(illSamplerate_cuda),__uint2float_rn(position+1-dynamic_bufferlength_cuda),
                        after_chirp_phasef(frequencyDiff, phi, ddynamic_bufferlength_cuda, illSamplerate_cuda));
                }
                half2 temp = h2sin (__floats2half2_rn(pc,pc1));
                sums +=  __low2float(temp)*ampf;
                suml +=  __high2float(temp)*ampf;
            }
            short2* dynamic_buf2 = reinterpret_cast<short2*>(dynamic_buf[buffer_index]);
            dynamic_buf2[index] = {static_cast<short>(sums),static_cast<short>(suml)};
            index += static_bufferlength_cuda;
            position += static_bufferlength_cuda * 2;
        }
        __syncthreads();
    }
}


__global__ void Pre_computer_rearrangef(int buffer_index, int static_num, int dynamic_num, const int * __restrict__ single_static_map_cuda, const int * __restrict__ single_dynamic_map_cuda, float ** summed_buf, float *  static_buf, const float* __restrict__ ddest_freq,
                            short* final_buf, short** dynamic_buf, const float * __restrict__ dstartFreq, const float* __restrict__ phase, float * new_phase, float * rearrange_freq_list_cuda)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    float sum;
    float sum1;
    unsigned int static_bufferlength_cuda2 = static_bufferlength_cuda / 2;
    float2* static_buf2 = reinterpret_cast<float2*>(static_buf);
    __shared__ float startFreq[128];
    __shared__ float dest_freq[128];
    __shared__ float phaselist[128];
    __shared__ int dynamic_index_list[128];

    // Load the input buffers
    double ddynamic_bufferlength_cuda = __uint2double_rn(dynamic_bufferlength_cuda);
    if(threadIdx.x<dynamic_num){
        unsigned int dynamic_index = single_dynamic_map_cuda[threadIdx.x];
        dynamic_index_list[threadIdx.x] = dynamic_index;
        float ddest_freq_temp = ddest_freq[threadIdx.x];
        dest_freq[threadIdx.x] = ddest_freq_temp;
        phaselist[threadIdx.x] = phase[dynamic_index];
        startFreq[threadIdx.x] = dstartFreq[dynamic_index];
        if (blockDim.x * blockIdx.x==0) rearrange_freq_list_cuda[dynamic_index] = ddest_freq_temp;
    }
    __syncthreads();

    float2 temp_sums;
    sum = 0;
    sum1 = 0;
    // Load the kept static tones
    for(int iter=0;iter<static_num-dynamic_num;iter++){
        temp_sums = static_buf2[static_bufferlength_cuda2*single_static_map_cuda[iter]+i];
        sum += temp_sums.x;
        sum1 += temp_sums.y;
    }
    float temp_sumst = 0.;
    float temp_sumst1 = 0.;
    float ipower_normalizer = 32767.f*imaximal_tweezer_number_cuda[buffer_index];

    // Compute the chirped static tones
    for (int j=0; j<dynamic_num;j++){
        float phi = __double2float_rn(modfma(static_cast<double>(dest_freq[j]*__double2float_rn(illSamplerate_cuda)),
                            __fma_rn(0.5*ddynamic_bufferlength_cuda,static_cast<double>(__fmaf_rn(0.5f,__frcp_rn(dest_freq[j])*startFreq[j],-1.f)),
                                __uint2double_rn(static_bufferlength_cuda*dynamic_loopcount_cuda)),phaselist[j]));
        if (i==0) new_phase[dynamic_index_list[j]] = phi;
        float temp_data = ipower_normalizer * sinpif(2.f*modfmaf(dest_freq[j]* __double2float_rn(illSamplerate_cuda) , 2.f*__uint2float_rn(i),phi));
        float temp_data1 = ipower_normalizer * sinpif(2.f*modfmaf(dest_freq[j]* __double2float_rn(illSamplerate_cuda) , __fmaf_rn(2.f,__uint2float_rn(i),1.f),phi));
        temp_sumst +=  temp_data;
        temp_sumst1 +=  temp_data1;
        static_buf2[static_bufferlength_cuda2*dynamic_index_list[j]+i] =  {temp_data,temp_data1};
    }

    // Compute the chirped dynamic tones
    for (int ch=0; ch<channel_num_cuda;ch++){
        if (ch==buffer_index){
            // Compute the rearrangement on the specified channel
            float ttemp = temp_sumst+sum;
            float ttemp1 = temp_sumst1+sum1;
            final_buf[ch+2*i*channel_num_cuda] = static_cast<short>(ttemp);
            final_buf[ch+(2*i+1)*channel_num_cuda] = static_cast<short>(ttemp1);
            summed_buf[ch][2*i] = ttemp;
            summed_buf[ch][2*i+1] = ttemp1;
            unsigned int index = i;
            unsigned int position = i * 2;
            for (int counter = 0; counter< dynamic_loopcount_cuda; counter++){
                float suml = sum1;
                float sums = sum;
                float ratio = __double2float_rn(__uint2double_rn(position)*idynamic_bufferlength_cuda);
                float ratio1 = ratio + __double2float_rn(idynamic_bufferlength_cuda);
                for (int j=0; j<dynamic_num;j++){
                    float startFrequency = startFreq[j];
                    float frequencyDiff = dest_freq[j]-startFrequency;
                    double phi = phaselist[j]; // double precision is needed for any operation directly related to `position`
                    float pc;
                    float pc1;
                    if (position < dynamic_bufferlength_cuda){
                        // When the `position` is before the point where the chirping ends, compute the waveform as a minimum-jerk trajectory
                        // Note `position` can be a very large number so we need to use double precision for any operation directly related to it
                        pc = M_2PI_f*__double2float_rn(modfma(__uint2double_rn(position), illSamplerate_cuda * mini_jerk_coefff(frequencyDiff, ratio, startFrequency), phi));
                        pc1 = M_2PI_f*__double2float_rn(modfma(__uint2double_rn(position+1), illSamplerate_cuda * mini_jerk_coefff(frequencyDiff, ratio1, startFrequency), phi));
                    }else{
                        // When the position is after the point where the chirping ends, compute the waveform as static waveform with phase offset
                        // Note `position - dynamic_bufferlength_cuda` is a relative small value so we can use float precision
                        pc = M_2PI_f * modfmaf((startFrequency+frequencyDiff)*__double2float_rn(illSamplerate_cuda),__uint2float_rn(position-dynamic_bufferlength_cuda),
                            after_chirp_phasef(frequencyDiff, phi, ddynamic_bufferlength_cuda, illSamplerate_cuda));
                        pc1 = M_2PI_f * modfmaf((startFrequency+frequencyDiff)*__double2float_rn(illSamplerate_cuda),__uint2float_rn(position+1-dynamic_bufferlength_cuda),
                            after_chirp_phasef(frequencyDiff, phi, ddynamic_bufferlength_cuda, illSamplerate_cuda));
                    }
                    half2 temp = h2sin (__floats2half2_rn(pc,pc1));
                    sums +=  __low2float(temp)*ipower_normalizer;
                    suml +=  __high2float(temp)*ipower_normalizer;
                }
                short2* dynamic_buf2 = reinterpret_cast<short2*>(dynamic_buf[ch]);
                dynamic_buf2[index] = {static_cast<short>(sums),static_cast<short>(suml)};
                index += static_bufferlength_cuda2;
                position += static_bufferlength_cuda2 * 2;
            }
        }else{
            // Pipeline the waveforms on the unselected channels
            short out = static_cast<short>(summed_buf[ch][2*i]);
            short out1 = static_cast<short>(summed_buf[ch][2*i+1]);
            final_buf[ch+2*i*channel_num_cuda] = out;
            final_buf[ch+(2*i+1)*channel_num_cuda] = out1;
            unsigned int index = i;
            for (int counter = 0; counter< dynamic_loopcount_cuda; counter++){
                short2* dynamic_buf2 = reinterpret_cast<short2*>(dynamic_buf[ch]);
                dynamic_buf2[index] = {out,out1};
                index += static_bufferlength_cuda2;
            }
        }
    }
    __syncthreads();
}


__global__ void Pre_computer_rearrange_ampf(int buffer_index, int static_num, int dynamic_num, const int * __restrict__ single_static_map_cuda, const int * __restrict__ single_dynamic_map_cuda, float ** summed_buf, float *  static_buf, const float* __restrict__ ddest_freq,
                            short* final_buf, short** dynamic_buf, const float * __restrict__ dstartFreq, const float* __restrict__ phase, float * new_phase, float * rearrange_freq_list_cuda, const float* __restrict__ amp)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    float sum;
    float sum1;
    unsigned int static_bufferlength_cuda2 = static_bufferlength_cuda / 2;
    float2* static_buf2 = reinterpret_cast<float2*>(static_buf);
    __shared__ float startFreq[128];
    __shared__ float dest_freq[128];
    __shared__ float phaselist[128];
    __shared__ float amplist[128];
    __shared__ int dynamic_index_list[128];

    // Load the input buffers
    double ddynamic_bufferlength_cuda = __uint2double_rn(dynamic_bufferlength_cuda);
    if(threadIdx.x<dynamic_num){
        unsigned int dynamic_index = single_dynamic_map_cuda[threadIdx.x];
        dynamic_index_list[threadIdx.x] = dynamic_index;
        float ddest_freq_temp = ddest_freq[threadIdx.x];
        dest_freq[threadIdx.x] = ddest_freq_temp;
        phaselist[threadIdx.x] = phase[dynamic_index];
        startFreq[threadIdx.x] = dstartFreq[dynamic_index];
        amplist[threadIdx.x] = amp[dynamic_index];
        if (blockDim.x * blockIdx.x==0) rearrange_freq_list_cuda[dynamic_index] = ddest_freq_temp;
    }
    __syncthreads();

    float2 temp_sums;
    sum = 0;
    sum1 = 0;
    // Load the kept static tones
    for(int iter=0;iter<static_num-dynamic_num;iter++){
        temp_sums = static_buf2[static_bufferlength_cuda2*single_static_map_cuda[iter]+i];
        sum += temp_sums.x;
        sum1 += temp_sums.y;
    }
    float temp_sumst = 0.;
    float temp_sumst1 = 0.;
    float ipower_normalizer = 32767.f*imaximal_tweezer_number_cuda[buffer_index];

    // Compute the chirped static tones
    for (int j=0; j<dynamic_num;j++){
        float phi = __double2float_rn(modfma(static_cast<double>(dest_freq[j]*__double2float_rn(illSamplerate_cuda)),
                            __fma_rn(0.5*ddynamic_bufferlength_cuda,static_cast<double>(__fmaf_rn(0.5f,__frcp_rn(dest_freq[j])*startFreq[j],-1.f)),
                                __uint2double_rn(static_bufferlength_cuda*dynamic_loopcount_cuda)),phaselist[j]));
        if (i==0) new_phase[dynamic_index_list[j]] = phi;
        float ampf = ipower_normalizer * amplist[j];
        float temp_data = ampf * sinpif(2.f*modfmaf(dest_freq[j]* __double2float_rn(illSamplerate_cuda) , 2.f*__uint2float_rn(i),phi));
        float temp_data1 = ampf * sinpif(2.f*modfmaf(dest_freq[j]* __double2float_rn(illSamplerate_cuda) , __fmaf_rn(2.f,__uint2float_rn(i),1.f),phi));
        temp_sumst +=  temp_data;
        temp_sumst1 +=  temp_data1;
        static_buf2[static_bufferlength_cuda2*dynamic_index_list[j]+i] =  {temp_data,temp_data1};
    }

    // Compute the chirped dynamic tones
    for (int ch=0; ch<channel_num_cuda;ch++){
        if (ch==buffer_index){
            // Compute the rearrangement on the specified channel
            float ttemp = temp_sumst+sum;
            float ttemp1 = temp_sumst1+sum1;
            final_buf[ch+2*i*channel_num_cuda] = static_cast<short>(ttemp);
            final_buf[ch+(2*i+1)*channel_num_cuda] = static_cast<short>(ttemp1);
            summed_buf[ch][2*i] = ttemp;
            summed_buf[ch][2*i+1] = ttemp1;
            unsigned int index = i;
            unsigned int position = i * 2;
            for (int counter = 0; counter< dynamic_loopcount_cuda; counter++){
                float suml = sum1;
                float sums = sum;
                float ratio = __double2float_rn(__uint2double_rn(position)*idynamic_bufferlength_cuda);
                float ratio1 = ratio + __double2float_rn(idynamic_bufferlength_cuda);
                for (int j=0; j<dynamic_num;j++){
                    float startFrequency = startFreq[j];
                    float frequencyDiff = dest_freq[j]-startFrequency;
                    double phi = phaselist[j]; // double precision is needed for any operation directly related to `position`
                    float ampf = amplist[j]*ipower_normalizer;
                    float pc;
                    float pc1;
                    if (position < dynamic_bufferlength_cuda){
                        // When the `position` is before the point where the chirping ends, compute the waveform as a minimum-jerk trajectory
                        // Note `position` can be a very large number so we need to use double precision for any operation directly related to it
                        pc = M_2PI_f*__double2float_rn(modfma(__uint2double_rn(position), illSamplerate_cuda * mini_jerk_coefff(frequencyDiff, ratio, startFrequency), phi));
                        pc1 = M_2PI_f*__double2float_rn(modfma(__uint2double_rn(position+1), illSamplerate_cuda * mini_jerk_coefff(frequencyDiff, ratio1, startFrequency), phi));
                    }else{
                        // When the position is after the point where the chirping ends, compute the waveform as static waveform with phase offset
                        // Note `position - dynamic_bufferlength_cuda` is a relative small value so we can use float precision
                        pc = M_2PI_f * modfmaf((startFrequency+frequencyDiff)*__double2float_rn(illSamplerate_cuda),__uint2float_rn(position-dynamic_bufferlength_cuda),
                            after_chirp_phasef(frequencyDiff, phi, ddynamic_bufferlength_cuda, illSamplerate_cuda));
                        pc1 = M_2PI_f * modfmaf((startFrequency+frequencyDiff)*__double2float_rn(illSamplerate_cuda),__uint2float_rn(position+1-dynamic_bufferlength_cuda),
                            after_chirp_phasef(frequencyDiff, phi, ddynamic_bufferlength_cuda, illSamplerate_cuda));
                    }
                    half2 temp = h2sin (__floats2half2_rn(pc,pc1));
                    sums +=  __low2float(temp)*ampf;
                    suml +=  __high2float(temp)*ampf;
                }
                short2* dynamic_buf2 = reinterpret_cast<short2*>(dynamic_buf[ch]);
                dynamic_buf2[index] = {static_cast<short>(sums),static_cast<short>(suml)};
                index += static_bufferlength_cuda2;
                position += static_bufferlength_cuda2 * 2;
            }
        }else{
            // Pipeline the waveforms on the unselected channels
            short out = static_cast<short>(summed_buf[ch][2*i]);
            short out1 = static_cast<short>(summed_buf[ch][2*i+1]);
            final_buf[ch+2*i*channel_num_cuda] = out;
            final_buf[ch+(2*i+1)*channel_num_cuda] = out1;
            unsigned int index = i;
            for (int counter = 0; counter< dynamic_loopcount_cuda; counter++){
                short2* dynamic_buf2 = reinterpret_cast<short2*>(dynamic_buf[ch]);
                dynamic_buf2[index] = {out,out1};
                index += static_bufferlength_cuda2;
            }
        }
    }
    __syncthreads();
}


__global__ void Pre_computer_acceleratef(float frequencyDiff, int buffer_index, const float * __restrict__ static_buf, const int* __restrict__ static_list,
                            const int*__restrict__ ddy_list, short* final_buf, short** dynamic_buf, const float * __restrict__ dstartFreq, const float* __restrict__ phase,float * new_phase)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; // using uint to allow chirp time of seconds
    float sum;
    float sum1;
    unsigned int static_bufferlength_cuda2 = static_bufferlength_cuda / 2;
    const float2* static_buf2 = reinterpret_cast<const float2*>(static_buf);
    __shared__ float startFreq[32];
    __shared__ float phaselist[32];
    double ddynamic_bufferlength_cuda = __uint2double_rn(dynamic_bufferlength_cuda);
    //  Process data on different AWG channels separately


    // Load input buffers into the shared memory for fast access per thread
    if(threadIdx.x<dynamic_num_cuda[buffer_index]){
        // unsigned int dynamic_index = dynamic_tone_count_cuda[buffer_index]+threadIdx.x;
        int shared_index = ddy_list[dynamic_tone_count_cuda[buffer_index]+threadIdx.x];
        startFreq[threadIdx.x] = dstartFreq[shared_index];
        phaselist[threadIdx.x] = phase[shared_index];
    }
    __syncthreads();


    for (int ch = 0; ch < channel_num_cuda; ch++){
            // Compute the waveforms
        float2 temp_sums;
        sum = 0;
        sum1 = 0;
        // Extract the tones that are not chirped
        unsigned int static_tone_count_start = tone_count_cuda[buffer_index]-dynamic_tone_count_cuda[buffer_index];
        unsigned int static_tone_count_end = tone_count_cuda[buffer_index+1]-dynamic_tone_count_cuda[buffer_index+1];
        if (static_tone_count_end-static_tone_count_start>0){
            for(int iter=static_tone_count_start;iter<static_tone_count_end;iter++){
                temp_sums = static_buf2[static_bufferlength_cuda2*static_list[iter]+i];
                sum += temp_sums.x;
                sum1 += temp_sums.y;
            }
        }

        if (ch == buffer_index) {
            // Compute the static waveform after the chirping and store in final_buf
            float temp_sumst = 0.;
            float temp_sumst1 = 0.;
            float ipower_normalizer = 32767.f *imaximal_tweezer_number_cuda[buffer_index];
            for (int j=0; j<dynamic_num_cuda[buffer_index];j++){
                float dest_freq = startFreq[j]+frequencyDiff;
                float phi = __double2float_rn(modfma(static_cast<double>(dest_freq*__double2float_rn(illSamplerate_cuda)),
                                __fma_rn(0.5*ddynamic_bufferlength_cuda,static_cast<double>(__fmaf_rn(0.5f,__frcp_rn(dest_freq)*startFreq[j],-1.f)),
                                    __uint2double_rn(static_bufferlength_cuda*dynamic_loopcount_cuda)),phaselist[j]));
                if (i==0) new_phase[ddy_list[dynamic_tone_count_cuda[buffer_index]+j]] = phi;
                temp_sumst +=  sinpif(2.f*modfmaf(dest_freq* __double2float_rn(illSamplerate_cuda) , 2.f*__uint2float_rn(i),phi));
                temp_sumst1 +=  sinpif(2.f*modfmaf(dest_freq* __double2float_rn(illSamplerate_cuda) , __fmaf_rn(2.f,__uint2float_rn(i),1.f),phi));
            }
            final_buf[buffer_index+2*i*channel_num_cuda] = static_cast<short>(__fmaf_rn(temp_sumst,ipower_normalizer,sum));
            final_buf[buffer_index+(2*i+1)*channel_num_cuda] = static_cast<short>(__fmaf_rn(temp_sumst1,ipower_normalizer,sum1));

            // Compute the waveform during the dynamic process and store in dynamic_buf
            unsigned int index = i;
            unsigned int position = i * 2; // the 2 is because of the doublet format
            for (int counter = 0; counter< dynamic_loopcount_cuda/2; counter++){
                // Loop in chunks of static bufferlength to match with the buffer pipelining/thread blocksize
                float suml = sum1;
                float sums = sum;
                float ratio = 0.5f*__double2float_rn(__uint2double_rn(position)*idynamic_bufferlength_cuda); // No larger than 1
                float ratio1 = __fmaf_rn(0.5f, __double2float_rn(idynamic_bufferlength_cuda), ratio); // No larger than 1
                for (int j=0; j<dynamic_num_cuda[buffer_index];j++){
                    float startFrequency = startFreq[j];
                    double phi = phaselist[j]; // double precision is needed for any operation directly related to `position`
                    float pc;
                    float pc1;
                    if (position < dynamic_bufferlength_cuda){
                        // When the `position` is before the point where the chirping ends, compute the waveform as a minimum-jerk trajectory
                        // Note `position` can be a very large number so we need to use double precision for any operation directly related to it
                        pc = M_2PI_f*__double2float_rn(modfma(__uint2double_rn(position), illSamplerate_cuda * mini_jerk_coefff(2.f*frequencyDiff, ratio, startFrequency), phi));
                        pc1 = M_2PI_f*__double2float_rn(modfma(__uint2double_rn(position+1), illSamplerate_cuda * mini_jerk_coefff(2.f*frequencyDiff, ratio1, startFrequency), phi));
                    }else{
                        // When the position is after the point where the chirping ends, compute the waveform as static waveform with phase offset
                        // Note `position - dynamic_bufferlength_cuda` is a relative small value so we can use float precision
                        pc = M_2PI_f * modfmaf((startFrequency+frequencyDiff)*__double2float_rn(illSamplerate_cuda),__uint2float_rn(position),
                            after_accel_phasef(frequencyDiff, phi, ddynamic_bufferlength_cuda, illSamplerate_cuda));
                        pc = M_2PI_f * modfmaf((startFrequency+frequencyDiff)*__double2float_rn(illSamplerate_cuda),__uint2float_rn(position+1),
                            after_accel_phasef(frequencyDiff, phi, ddynamic_bufferlength_cuda, illSamplerate_cuda));
                    }
                    half2 temp = h2sin (__floats2half2_rn(pc,pc1));
                    sums +=  __low2float(temp)*ipower_normalizer;
                    suml +=  __high2float(temp)*ipower_normalizer;
                }
                short2* dynamic_buf2 = reinterpret_cast<short2*>(dynamic_buf[buffer_index]);
                dynamic_buf2[index] = {static_cast<short>(sums),static_cast<short>(suml)};
                index += static_bufferlength_cuda;
                position += static_bufferlength_cuda * 2;
            }
        } else {
            // Pipeline the waveforms on the unselected channels
            // short out = static_cast<short>(summed_buf[ch][2*i]);
            // short out1 = static_cast<short>(summed_buf[ch][2*i+1]);
            final_buf[ch+2*i*channel_num_cuda] = static_cast<short>(sum);
            final_buf[ch+(2*i+1)*channel_num_cuda] = static_cast<short>(sum1);
            unsigned int index = i;
            for (int counter = 0; counter< dynamic_loopcount_cuda; counter++){
                short2* dynamic_buf2 = reinterpret_cast<short2*>(dynamic_buf[ch]);
                dynamic_buf2[index] = {static_cast<short>(sum),static_cast<short>(sum1)};
                index += static_bufferlength_cuda2;
            }
        }
    }
    __syncthreads();
}



__global__ void Pre_computer_accelerate_ampf(float frequencyDiff, int buffer_index, const float * __restrict__ static_buf, const int* __restrict__ static_list,
                            const int*__restrict__ ddy_list, short* final_buf, short** dynamic_buf, const float * __restrict__ dstartFreq, const float* __restrict__ phase,float * new_phase,
                            const float* __restrict__ amp)
{
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; // using uint to allow chirp time of seconds
    float sum;
    float sum1;
    unsigned int static_bufferlength_cuda2 = static_bufferlength_cuda / 2;
    const float2* static_buf2 = reinterpret_cast<const float2*>(static_buf);
    __shared__ float startFreq[32];
    __shared__ float phaselist[32];
    __shared__ float amplist[32];
    double ddynamic_bufferlength_cuda = __uint2double_rn(dynamic_bufferlength_cuda);
    //  Process data on different AWG channels separately


    // Load input buffers into the shared memory for fast access per thread
    if(threadIdx.x<dynamic_num_cuda[buffer_index]){
        // unsigned int dynamic_index = dynamic_tone_count_cuda[buffer_index]+threadIdx.x;
        int shared_index = ddy_list[dynamic_tone_count_cuda[buffer_index]+threadIdx.x];
        startFreq[threadIdx.x] = dstartFreq[shared_index];
        phaselist[threadIdx.x] = phase[shared_index];
        amplist[threadIdx.x] = amp[shared_index];
    }
    __syncthreads();


    for (int ch = 0; ch < channel_num_cuda; ch++){
            // Compute the waveforms
        float2 temp_sums;
        sum = 0;
        sum1 = 0;
        // Extract the tones that are not chirped
        unsigned int static_tone_count_start = tone_count_cuda[buffer_index]-dynamic_tone_count_cuda[buffer_index];
        unsigned int static_tone_count_end = tone_count_cuda[buffer_index+1]-dynamic_tone_count_cuda[buffer_index+1];
        if (static_tone_count_end-static_tone_count_start>0){
            for(int iter=static_tone_count_start;iter<static_tone_count_end;iter++){
                temp_sums = static_buf2[static_bufferlength_cuda2*static_list[iter]+i];
                sum += temp_sums.x;
                sum1 += temp_sums.y;
            }
        }

        if (ch == buffer_index) {
            // Compute the static waveform after the chirping and store in final_buf
            float temp_sumst = 0.;
            float temp_sumst1 = 0.;
            float ipower_normalizer = 32767.f *imaximal_tweezer_number_cuda[buffer_index];
            for (int j=0; j<dynamic_num_cuda[buffer_index];j++){
                float dest_freq = startFreq[j]+frequencyDiff;
                float phi = __double2float_rn(modfma(static_cast<double>(dest_freq*__double2float_rn(illSamplerate_cuda)),
                                __fma_rn(0.5*ddynamic_bufferlength_cuda,static_cast<double>(__fmaf_rn(0.5f,__frcp_rn(dest_freq)*startFreq[j],-1.f)),
                                    __uint2double_rn(static_bufferlength_cuda*dynamic_loopcount_cuda)),phaselist[j]));
                if (i==0) new_phase[ddy_list[dynamic_tone_count_cuda[buffer_index]+j]] = phi;
                temp_sumst +=  amplist[j]*sinpif(2.f*modfmaf(dest_freq* __double2float_rn(illSamplerate_cuda) , 2.f*__uint2float_rn(i),phi));
                temp_sumst1 +=  amplist[j]*sinpif(2.f*modfmaf(dest_freq* __double2float_rn(illSamplerate_cuda) , __fmaf_rn(2.f,__uint2float_rn(i),1.f),phi));
            }
            final_buf[buffer_index+2*i*channel_num_cuda] = static_cast<short>(__fmaf_rn(temp_sumst,ipower_normalizer,sum));
            final_buf[buffer_index+(2*i+1)*channel_num_cuda] = static_cast<short>(__fmaf_rn(temp_sumst1,ipower_normalizer,sum1));

            // Compute the waveform during the dynamic process and store in dynamic_buf
            unsigned int index = i;
            unsigned int position = i * 2; // the 2 is because of the doublet format
            for (int counter = 0; counter< dynamic_loopcount_cuda/2; counter++){
                // Loop in chunks of static bufferlength to match with the buffer pipelining/thread blocksize
                float suml = sum1;
                float sums = sum;
                float ratio = 0.5f*__double2float_rn(__uint2double_rn(position)*idynamic_bufferlength_cuda); // No larger than 1
                float ratio1 = __fmaf_rn(0.5f, __double2float_rn(idynamic_bufferlength_cuda), ratio); // No larger than 1
                for (int j=0; j<dynamic_num_cuda[buffer_index];j++){
                    float startFrequency = startFreq[j];
                    double phi = phaselist[j]; // double precision is needed for any operation directly related to `position`
                    float ampl = amplist[j]*ipower_normalizer;
                    float pc;
                    float pc1;
                    if (position < dynamic_bufferlength_cuda){
                        // When the `position` is before the point where the chirping ends, compute the waveform as a minimum-jerk trajectory
                        // Note `position` can be a very large number so we need to use double precision for any operation directly related to it
                        pc = M_2PI_f*__double2float_rn(modfma(__uint2double_rn(position), illSamplerate_cuda * mini_jerk_coefff(2.f*frequencyDiff, ratio, startFrequency), phi));
                        pc1 = M_2PI_f*__double2float_rn(modfma(__uint2double_rn(position+1), illSamplerate_cuda * mini_jerk_coefff(2.f*frequencyDiff, ratio1, startFrequency), phi));
                    }else{
                        // When the position is after the point where the chirping ends, compute the waveform as static waveform with phase offset
                        // Note `position - dynamic_bufferlength_cuda` is a relative small value so we can use float precision
                        pc = M_2PI_f * modfmaf((startFrequency+frequencyDiff)*__double2float_rn(illSamplerate_cuda),__uint2float_rn(position),
                            after_accel_phasef(frequencyDiff, phi, ddynamic_bufferlength_cuda, illSamplerate_cuda));
                        pc = M_2PI_f * modfmaf((startFrequency+frequencyDiff)*__double2float_rn(illSamplerate_cuda),__uint2float_rn(position+1),
                            after_accel_phasef(frequencyDiff, phi, ddynamic_bufferlength_cuda, illSamplerate_cuda));
                    }
                    half2 temp = h2sin (__floats2half2_rn(pc,pc1));
                    sums +=  __low2float(temp)*ampl;
                    suml +=  __high2float(temp)*ampl;
                }
                short2* dynamic_buf2 = reinterpret_cast<short2*>(dynamic_buf[buffer_index]);
                dynamic_buf2[index] = {static_cast<short>(sums),static_cast<short>(suml)};
                index += static_bufferlength_cuda;
                position += static_bufferlength_cuda * 2;
            }
        } else {
            // Pipeline the waveforms on the unselected channels
            // short out = static_cast<short>(summed_buf[ch][2*i]);
            // short out1 = static_cast<short>(summed_buf[ch][2*i+1]);
            final_buf[ch+2*i*channel_num_cuda] = static_cast<short>(sum);
            final_buf[ch+(2*i+1)*channel_num_cuda] = static_cast<short>(sum1);
            unsigned int index = i;
            for (int counter = 0; counter< dynamic_loopcount_cuda; counter++){
                short2* dynamic_buf2 = reinterpret_cast<short2*>(dynamic_buf[ch]);
                dynamic_buf2[index] = {static_cast<short>(sum),static_cast<short>(sum1)};
                index += static_bufferlength_cuda2;
            }
        }
    }
    __syncthreads();
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
        CUDA_CHECK(cudaMalloc ((void **)&saved_buffer[i], sizeof(double)/sizeof(short)*lBytesPerChannelInNotifySize)); //Configure software buffer
        CUDA_CHECK(cudaMalloc ((void **)&dynamic_saved_buffer[i], dynamic_loopcount*lBytesPerChannelInNotifySize)); //Configure software buffer
    }
    
    CUDA_CHECK(cudaMalloc ((void**)&real_destination_freq_cuda, dynamic_total*sizeof(float))); //Configure software buffer
    CUDA_CHECK(cudaMemcpy(real_destination_freq_cuda,real_destination_freq,dynamic_total*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc ((void**)&dynamic_list_cuda, dynamic_total*sizeof(int))); //Configure software buffer
    CUDA_CHECK(cudaMalloc ((void**)&static_list_cuda, sizeof(static_list))); //Configure software buffer
    CUDA_CHECK(cudaMalloc ((void**)&final_buffer_cuda, lNumCh*lBytesPerChannelInNotifySize)); //Configure software buffer
    CUDA_CHECK(cudaMalloc ((void**)&saved_buffer_cuda, sizeof(saved_buffer))); //Configure software buffer
    CUDA_CHECK(cudaMalloc ((void**)&dynamic_saved_buffer_cuda, sizeof(dynamic_saved_buffer))); //Configure software buffer
    CUDA_CHECK(cudaMemcpy (static_list_cuda, static_list,sizeof(static_list),cudaMemcpyHostToDevice)); //Configure software buffer
    CUDA_CHECK(cudaMemcpy (dynamic_list_cuda, dynamic_list,dynamic_total*sizeof(int),cudaMemcpyHostToDevice)); //Configure software buffer
    CUDA_CHECK(cudaMemcpy (saved_buffer_cuda, saved_buffer,sizeof(saved_buffer),cudaMemcpyHostToDevice)); //Configure software buffer
    CUDA_CHECK(cudaMemcpy (dynamic_saved_buffer_cuda, dynamic_saved_buffer,sizeof(dynamic_saved_buffer),cudaMemcpyHostToDevice)); //Configure software buffer
    CUDA_CHECK(cudaMemcpyToSymbol(dynamic_num_cuda, &dynamic_num, sizeof(dynamic_num)));
    unsigned int dylength = dynamic_buffersize/2;
    CUDA_CHECK(cudaMemcpyToSymbol(dynamic_bufferlength_cuda, &dylength, sizeof(dylength)));
    double_temp = 1./static_cast<double>(dylength);
    CUDA_CHECK(cudaMemcpyToSymbol (idynamic_bufferlength_cuda, &double_temp, sizeof(double))); //Configure software buffer
    CUDA_CHECK(cudaMemcpyToSymbol (dynamic_loopcount_cuda, &dynamic_loopcount, sizeof(dynamic_loopcount))); //Configure software buffer
    return 0;
}


void rearrangeCleanUp(){
    for (int i = 0; i < lNumCh; i++){
        if (dynamic_saved_buffer[i] != nullptr) {
            cudaFree(dynamic_saved_buffer[i]);
            dynamic_saved_buffer[i] = nullptr;
        }
    }
    if (real_destination_freq_cuda != nullptr) {
        cudaFree(real_destination_freq_cuda);
        real_destination_freq_cuda = nullptr;
    }
    if (rearrange_freq_list_cuda != nullptr) {
        cudaFree(rearrange_freq_list_cuda);
        rearrange_freq_list_cuda = nullptr;
    }
    if (final_buffer_cuda != nullptr) {
        cudaFree(final_buffer_cuda);
        final_buffer_cuda = nullptr;
    }
    if (dynamic_saved_buffer_cuda != nullptr) {
        cudaFree(dynamic_saved_buffer_cuda);
        dynamic_saved_buffer_cuda = nullptr;
    }
    if (single_static_map_cuda != nullptr) {
        cudaFree(single_static_map_cuda);
        single_static_map_cuda = nullptr;
    }
    if (single_dynamic_map_cuda != nullptr) {
        cudaFree(single_dynamic_map_cuda);
        single_dynamic_map_cuda = nullptr;
    }
}

int rearrangeBufferInit(){
    rearrangeCleanUp();
    // CUDA_CHECK(cudaMemcpyToSymbol(tone_count_cuda,tone_count,5*sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpyToSymbol(static_num_cuda,static_num,sizeof(static_num)));
    double double_array_temp[4];
    for (int i = 0; i < 4; i++) {
        if (static_num[i]) {double_array_temp[i] = 1./static_num[i];}
        else{double_array_temp[i]=0;}
    }
    
    // CUDA_CHECK(cudaMemcpyToSymbol(dynamic_tone_count_cuda,&dynamic_tone_count,5*sizeof(unsigned int)));
    float ipower_normalizer[4];
    for (int i = 0; i < 4; i++) {
        if (power_normalizer[i]<2.f) {power_normalizer[i]=2.f;}
        ipower_normalizer[i] = 1./power_normalizer[i];
        } 
    CUDA_CHECK(cudaMemcpyToSymbol(ipower_normalizer_cuda,&ipower_normalizer,sizeof(ipower_normalizer)));
    CUDA_CHECK(cudaMemcpyToSymbol(istatic_num_cuda,double_array_temp,4*sizeof(double)));

    for (int i = 0; i < lNumCh; i++){
        // CUDA_CHECK(cudaMalloc ((void **)&saved_buffer[i], sizeof(double)/sizeof(short)*lBytesPerChannelInNotifySize)); //Configure software buffer
        CUDA_CHECK(cudaMalloc ((void **)&dynamic_saved_buffer[i], dynamic_loopcount*lBytesPerChannelInNotifySize)); //Configure software buffer
    }
    
    CUDA_CHECK(cudaMalloc ((void**)&real_destination_freq_cuda, dynamic_num[single_dynamic_channel]*sizeof(float))); //Configure software buffer
    CUDA_CHECK(cudaMemcpy(real_destination_freq_cuda,real_destination_freq,dynamic_num[single_dynamic_channel]*sizeof(float),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc ((void**)&rearrange_freq_list_cuda, static_total*sizeof(float))); //Configure software buffer
    CUDA_CHECK(cudaMemcpy(rearrange_freq_list_cuda,real_static_freq_cuda,static_total*sizeof(float),cudaMemcpyHostToDevice));
    // CUDA_CHECK(cudaMalloc ((void**)&dynamic_list_cuda, dynamic_total*sizeof(int))); //Configure software buffer
    // CUDA_CHECK(cudaMalloc ((void**)&static_list_cuda, sizeof(static_list))); //Configure software buffer
    CUDA_CHECK(cudaMalloc ((void**)&final_buffer_cuda, lNumCh*lBytesPerChannelInNotifySize)); //Configure software buffer
    // CUDA_CHECK(cudaMalloc ((void**)&saved_buffer_cuda, sizeof(saved_buffer))); //Configure software buffer
    CUDA_CHECK(cudaMalloc ((void**)&dynamic_saved_buffer_cuda, sizeof(dynamic_saved_buffer))); //Configure software buffer
    CUDA_CHECK(cudaMalloc ((void**)&single_static_map_cuda, sizeof(static_single_index))); //Configure software buffer
    CUDA_CHECK(cudaMalloc ((void**)&single_dynamic_map_cuda, sizeof(dynamic_single_index))); //Configure software buffer
    CUDA_CHECK(cudaMemcpy(real_destination_freq_cuda,real_destination_freq,dynamic_num[single_dynamic_channel]*sizeof(float),cudaMemcpyHostToDevice));

    unsigned int dylength = dynamic_buffersize/2;
    CUDA_CHECK(cudaMemcpyToSymbol(dynamic_bufferlength_cuda, &dylength, sizeof(int)));
    double_temp = 1./static_cast<double>(dylength);
    CUDA_CHECK(cudaMemcpyToSymbol (idynamic_bufferlength_cuda, &double_temp, sizeof(double))); //Configure software buffer
    CUDA_CHECK(cudaMemcpyToSymbol (dynamic_loopcount_cuda, &dynamic_loopcount, sizeof(dynamic_loopcount))); //Configure software buffer
    CUDA_CHECK(cudaMemcpy (dynamic_saved_buffer_cuda, dynamic_saved_buffer,sizeof(dynamic_saved_buffer),cudaMemcpyHostToDevice)); //Configure software buffer
    CUDA_CHECK(cudaMemcpy(single_static_map_cuda,static_single_index,sizeof(int)*(static_num[single_dynamic_channel]-dynamic_num[single_dynamic_channel]),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(single_dynamic_map_cuda,dynamic_single_index,sizeof(int)*dynamic_num[single_dynamic_channel],cudaMemcpyHostToDevice));
    return 0;
}

