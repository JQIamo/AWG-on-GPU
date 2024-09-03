/*
Waveform_synthesis_streaming.cu
Author: Juntian Tu
Date: 2024.09.03

This is the implementation for the streaming pathway. Change the source file name in the Makefile to compile the program.
*/
#include "lib/cuda_functions.h"
#include <atomic>
//  Timer includes
#   include <iostream>
#   include <thread>

using namespace std;
int loop_counter=0;
volatile std::atomic<bool> init_flag(true);
volatile std::atomic<bool> static_flag(true);
char        szErrorTextBuffer[ERRORTEXTLEN];
uint32      dwError;
int32       lUserPos;
// ------Dynamics----------------------------
int16* destination_buffer_cuda;
double* dFreq_cuda;
double* dDiff_cuda;
double* dphi_cuda;
double* damp_cuda;
double* ratio_cuda;

void* DMABuffer = NULL;
 // Depending on the GPU used
void reset_amp(){
    if (lMaxOutputLevel>amplitude_limit){lMaxOutputLevel=amplitude_limit;}
    for (int i = 0; i < lNumCh; ++i){
        spcm_dwSetParam_i32 (hCard, SPC_AMP0       + i * (SPC_AMP1        - SPC_AMP0),      lMaxOutputLevel);
    }
}
void varReset(){
    dynamic_total = 0;
    static_total = 0;
    not_arrived = 1;
    loop_counter=0;
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
        cudaFree(update_index_map_cuda);
        update_index_map_cuda = NULL;
        cudaFree(destination_buffer_cuda);
        destination_buffer_cuda = NULL;
        cudaFree(ratio_cuda);
        ratio_cuda = NULL;
        cudaFree(damp_cuda);
        damp_cuda = NULL;
        cudaFree(dphi_cuda);
        dphi_cuda = NULL;
        cudaFree(dFreq_cuda);
        dFreq_cuda = NULL;
        cudaFree(dDiff_cuda);
        dDiff_cuda = NULL;
        if (eCudaErr=cudaPeekAtLastError()) printf("Cuda Buffer Clean Failed: %s\n",cudaGetErrorString(eCudaErr));
        cudaDeviceSynchronize();
    }




int block_size;
// settings for the FIFO mode buffer handling
uint32       lNotifySize =  MEGA_B(2); // The size of data the card will execute each time before signaling to the GPU
uint32       lBufferSize =  MEGA_B(64);
uint64       HBufferSize =  MEGA_B(64); // The actual buffer used on the AWG; must be a power of 2 and should be no more than 4 GB (lower size reduces delay)

// Parameter settings   
int32       lMaxOutputLevel = amplitude_limit; // +-1 Volt


// Test params
int16* final_buffer[4];

__device__ __constant__ int half_static_bufferlength_cuda;
unsigned long long sPointerPosition=0;


void static_looper(){
    while (!stop_flag){
        while (static_flag && !stop_flag){
            while (!static_endflag.load() && !stop_flag){
                if ((dwError = spcm_dwSetParam_i32 (hCard, SPC_M2CMD, M2CMD_DATA_WAITDMA)) != ERR_OK)
                {
                    if (dwError == ERR_TIMEOUT)
                        printf ("\n... Timeout\n");
                    else{
                        spcm_dwGetErrorInfo_i32 (hCard, NULL, NULL, szErrorTextBuffer);
                        printf ("\n... Error: %u (%s)\n", dwError,szErrorTextBuffer);
                        stop_flag = true;
                        break;}
                }
                else
                {             
                    dwError = spcm_dwSetParam_i32 (hCard, SPC_DATA_AVAIL_CARD_LEN,  lNotifySize);
                    if (dwError!=ERR_OK){
                        spcm_dwGetErrorInfo_i32 (hCard, NULL, NULL, szErrorTextBuffer);
                        printf("\n... Error in Setting CardAval: %u (%s)\n", dwError,szErrorTextBuffer);
                        stop_flag = true;
                        break;
                    }
                }
            }
            static_endflag=false;
            static_pulseflag = true;
        }
    }
}
std::thread staticThread;




/*
*********************************(loaded_atoms[index+i]-position-i)*****************************************
// Calculate and multiplexing the waveforms with CUDA kernel; positionBuff is used in case the waveform is not periodic at lNotify length
// Code the output waveform here
**************************************************************************
*/


__global__ void FinalWaveGeneration (double* startFreq, double* dest_frequency, int* dy_list,double* phase_list,double* newphase_list,short* pnOut)
    {
        size_t i = blockDim.x * blockIdx.x + threadIdx.x;
        double dump;
        for (int buffer_index=0;buffer_index<channel_num_cuda;buffer_index++){
            for (size_t j=0;j<dynamic_num_cuda[buffer_index];j++){
                int tone_index = j+dynamic_tone_count_cuda[buffer_index];
                double phi = (2.0 * dest_frequency[tone_index] * (dynamic_loopcount_cuda*static_bufferlength_cuda-dynamic_bufferlength_cuda) +(startFreq[dy_list[tone_index]] 
                + dest_frequency[tone_index])*dynamic_bufferlength_cuda)/ llSamplerate_cuda +phase_list[dy_list[tone_index]];
                pnOut[tone_index*static_bufferlength_cuda+i] = static_cast<short>(32767. * sinpi (2.0 * dest_frequency[tone_index] * i / llSamplerate_cuda+phi)/power_normalizer_cuda[buffer_index]);
                if (i==0) newphase_list[dy_list[tone_index]] = 2.*modf(phi,&dump);
            }            
        }
    }


__global__ void FinalWaveGeneration_amp (double* startFreq, double* dest_frequency, int* dy_list,double* amp_list,double* phase_list,double* newphase_list,short* pnOut)
    {
        size_t i = blockDim.x * blockIdx.x + threadIdx.x;
        double dump;
        for (int buffer_index=0;buffer_index<channel_num_cuda;buffer_index++){
            for (size_t j=0;j<dynamic_num_cuda[buffer_index];j++){
                int tone_index = j+dynamic_tone_count_cuda[buffer_index];
                double phi = (2.0 * dest_frequency[tone_index] * (dynamic_loopcount_cuda*static_bufferlength_cuda-dynamic_bufferlength_cuda) +(startFreq[dy_list[tone_index]] 
                + dest_frequency[tone_index])*dynamic_bufferlength_cuda)/ llSamplerate_cuda +phase_list[dy_list[tone_index]];
                pnOut[tone_index*static_bufferlength_cuda+i] = amp_list[dy_list[tone_index]] * static_cast<short>(32767. * sinpi (2.0 * dest_frequency[tone_index] * i / llSamplerate_cuda+phi)/power_normalizer_cuda[buffer_index]);
                if (i==0) newphase_list[dy_list[tone_index]] = 2.*modf(phi,&dump);
            }            
        }
    }


__global__ void SavedStaticCombine(double*buffer,short**sum_buf,short** save_buf,int*dy_list){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    for (int buffer_index=0;buffer_index<channel_num_cuda;buffer_index++){
        double sum = sum_buf[buffer_index][i];
        for (int j=0; j<dynamic_num_cuda[buffer_index];j++){
            sum -= buffer[(size_t)dy_list[j+dynamic_tone_count_cuda[buffer_index]]*static_bufferlength_cuda+i];
        }
        save_buf[buffer_index][i] = static_cast<short> (sum);
    }
}

__global__ void DynamicListWorker(double*staticFrequency,double*destinationFreq,int*dy_list,double *dstart,double*dDiff,double*dphi,double*phase_list){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    for (int buffer_index=0;buffer_index<channel_num_cuda;buffer_index++){
        if(i<dynamic_num_cuda[buffer_index]){
            int tone_index = i+dynamic_tone_count_cuda[buffer_index];
            double temp = staticFrequency[dy_list[tone_index]];
            dstart[tone_index] = temp;
            dDiff[tone_index] = destinationFreq[tone_index]-temp;
            dphi[tone_index] = phase_list[dy_list[tone_index]];
        }        
    }
}

__global__ void DynamicListWorker_amp(double*__restrict__ staticFrequency,double*__restrict__ destinationFreq,int* __restrict__ dy_list,double *dstart,double*dDiff,double*dphi,double*damp,double*__restrict__ phase_list,double*__restrict__ amp_list){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    for (int buffer_index=0;buffer_index<channel_num_cuda;buffer_index++){
        if(i<dynamic_num_cuda[buffer_index]){
            int tone_index = i+dynamic_tone_count_cuda[buffer_index];
            double temp = staticFrequency[dy_list[tone_index]];
            dstart[tone_index] = temp;
            dDiff[tone_index] = destinationFreq[tone_index]-temp;
            dphi[tone_index] = phase_list[dy_list[tone_index]];
            damp[tone_index] = amp_list[dy_list[tone_index]];
        }        
    }

}

__global__ void ratio_calc(double*ratiolist){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    double ratio = i/static_cast<double>(dynamic_bufferlength_cuda);
    ratiolist[i] = (2.5  - 3. * ratio +  ratio *ratio)* ratio *ratio *ratio;
}


__global__ void AccelCombine(unsigned long long startPosition,short**__restrict__ save_buf,short** sum_buf,
                    double*__restrict__ dstartFrequency,double *__restrict__ dfreqeuncyDiff,double *__restrict__ dphi,double*__restrict__ ratio){
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ double startFrequency[256];
    __shared__ double freqeuncyDiff[256];
    __shared__ double phi[256];
    double k = static_cast<double>(i)+static_cast<double>(startPosition);
    double k1 = k + half_static_bufferlength_cuda;
    double s;
    for (int buffer_index=0;buffer_index<channel_num_cuda;buffer_index++){
        if(threadIdx.x<dynamic_num_cuda[buffer_index]){
            startFrequency[threadIdx.x] = dstartFrequency[threadIdx.x+dynamic_tone_count_cuda[buffer_index]];
            freqeuncyDiff[threadIdx.x] = dfreqeuncyDiff[threadIdx.x+dynamic_tone_count_cuda[buffer_index]];
            phi[threadIdx.x] = dphi[threadIdx.x+dynamic_tone_count_cuda[buffer_index]];
        }
        __syncthreads();
        double sum = save_buf[buffer_index][i];
        double sum1 = save_buf[buffer_index][i + half_static_bufferlength_cuda];
        if (k1<dynamic_bufferlength_cuda){
            for (size_t j = 0; j<dynamic_num_cuda[buffer_index]; j++){
                double phase_c =  k * illSamplerate_cuda * fma(freqeuncyDiff[j] , ratio[static_cast<size_t>(k)],startFrequency[j]);
                double phase_c1 = k1 * illSamplerate_cuda * fma( freqeuncyDiff[j] , ratio[static_cast<size_t>(k1)],startFrequency[j]);
                half2 temp = __hmul2 (__float2half2_rn(32767./power_normalizer_cuda[buffer_index]) , h2sin (__floats2half2_rn(2.*M_PI*modf(phase_c+phi[j],&s),2.*M_PI*modf(phase_c1+phi[j],&s))));
                sum1 += __high2float(temp);
                sum += __low2float(temp);
            }
        }else{ 
            for (size_t j = 0; j<dynamic_num_cuda[buffer_index]; j++){
                double pc = 2.*M_PI*modf(k * fma(illSamplerate_cuda , fma( freqeuncyDiff[j] , ratio[static_cast<size_t>(k)],startFrequency[j]),phi[j]),&s);
                double pc1 = M_PI*(2.0 * modf((startFrequency[j]+freqeuncyDiff[j]) * (k1-static_cast<double>(dynamic_bufferlength_cuda)) * illSamplerate_cuda+fma(freqeuncyDiff[j],0.5,startFrequency[j])*static_cast<double>(dynamic_bufferlength_cuda)*illSamplerate_cuda+phi[j],&s));
                half2 temp = __hmul2 (__float2half2_rn(32767./power_normalizer_cuda[buffer_index]) , h2sin (__floats2half2_rn(pc,pc1)));
                sum1 += __high2float(temp);
                sum += __low2float(temp);
            }
        }
        sum_buf[buffer_index][i] = static_cast<short>(sum);
        sum_buf[buffer_index][i + half_static_bufferlength_cuda] = static_cast<short>(sum1);
        __syncthreads();
    }
}

__global__ void AccelCombine_amp(unsigned long long startPosition,short**__restrict__ save_buf,short** sum_buf,
                    double*__restrict__ dstartFrequency,double *__restrict__ dfreqeuncyDiff,double *__restrict__ dphi,double*__restrict__ ratio,double*__restrict__ damp){
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ double startFrequency[256];
    __shared__ double freqeuncyDiff[256];
    __shared__ double phi[256];
    __shared__ double amp[256];
    double k = static_cast<double>(i)+static_cast<double>(startPosition);
    double k1 = k + half_static_bufferlength_cuda;
    double s;
    for (int buffer_index=0;buffer_index<channel_num_cuda;buffer_index++){
        if(threadIdx.x<dynamic_num_cuda[buffer_index]){
            startFrequency[threadIdx.x] = dstartFrequency[threadIdx.x+dynamic_tone_count_cuda[buffer_index]];
            freqeuncyDiff[threadIdx.x] = dfreqeuncyDiff[threadIdx.x+dynamic_tone_count_cuda[buffer_index]];
            phi[threadIdx.x] = dphi[threadIdx.x+dynamic_tone_count_cuda[buffer_index]];
            amp[threadIdx.x] = damp[threadIdx.x+dynamic_tone_count_cuda[buffer_index]];
        }
        __syncthreads();
        double sum = save_buf[buffer_index][i];
        double sum1 = save_buf[buffer_index][i + half_static_bufferlength_cuda];
        if (k1<dynamic_bufferlength_cuda){
            for (size_t j = 0; j<dynamic_num_cuda[buffer_index]; j++){
                double phase_c =  k * illSamplerate_cuda * fma(freqeuncyDiff[j] , ratio[static_cast<size_t>(k)],startFrequency[j]);
                double phase_c1 = k1 * illSamplerate_cuda * fma( freqeuncyDiff[j] , ratio[static_cast<size_t>(k1)],startFrequency[j]);
                half2 temp = __hmul2 (__float2half2_rn(amp[j]*32767./power_normalizer_cuda[buffer_index]) , h2sin (__floats2half2_rn(2.*M_PI*modf(phase_c+phi[j],&s),2.*M_PI*modf(phase_c1+phi[j],&s))));
                sum1 += __high2float(temp);
                sum += __low2float(temp);
            }
        }else{ 
            for (size_t j = 0; j<dynamic_num_cuda[buffer_index]; j++){
                double pc = 2.*M_PI*modf(k * fma(illSamplerate_cuda , fma( freqeuncyDiff[j] , ratio[static_cast<size_t>(k)],startFrequency[j]),phi[j]),&s);
                double pc1 = M_PI*(2.0 * modf((startFrequency[j]+freqeuncyDiff[j]) * (k1-static_cast<double>(dynamic_bufferlength_cuda)) * illSamplerate_cuda+fma(freqeuncyDiff[j],0.5,startFrequency[j])*static_cast<double>(dynamic_bufferlength_cuda)*illSamplerate_cuda+phi[j],&s));
                half2 temp = __hmul2 (__float2half2_rn(amp[j]*32767./power_normalizer_cuda[buffer_index]) , h2sin (__floats2half2_rn(pc,pc1)));
                sum1 += __high2float(temp);
                sum += __low2float(temp);
            }
        }
        sum_buf[buffer_index][i] = static_cast<short>(sum);
        sum_buf[buffer_index][i + half_static_bufferlength_cuda] = static_cast<short>(sum1);
        __syncthreads();
    }
}


__global__ void DynamicCombine(double*staticBuffer,short**sum_buf,short** save_buf,short* dynamicBuffer,int*dy_list){
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    int k = i%static_bufferlength_cuda;
    for (int buffer_index=0;buffer_index<channel_num_cuda;buffer_index++){
        double sum = sum_buf[buffer_index][k];
        for (size_t j=0; j<dynamic_num_cuda[buffer_index];j++){
            sum -= staticBuffer[(size_t)dy_list[j+dynamic_tone_count_cuda[buffer_index]]*static_bufferlength_cuda+k];
            sum += dynamicBuffer[(j+dynamic_tone_count_cuda[buffer_index])*static_cast<size_t>(dynamic_loopcount_cuda)*static_bufferlength_cuda+i];
        }
        save_buf[buffer_index][i] = static_cast<short> (sum);
    }
}

__global__ void FinalCombine(short**save_buf,short*destBuffer,short*sum_buf){
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    for (int buffer_index=0;buffer_index<channel_num_cuda;buffer_index++){
        short sum = save_buf[buffer_index][i];
        
        for (unsigned int j=0; j<dynamic_num_cuda[buffer_index];j++){
            sum += destBuffer[(j+dynamic_tone_count_cuda[buffer_index])*static_bufferlength_cuda+i];
        }
        sum_buf[buffer_index+channel_num_cuda*i] = sum;
    }
}

/*
**************************************************************************
// Thread functions
// ifkeypress() monitors ENTER press used to interpret the process
// GUI_server() runs the TCP server to receive data from the MainControlPC
**************************************************************************
*/


void instructionReceiver(){
    // Receive instructions for tweezer motion from MainControlPC
    if (TCP_server()){printf("TCP server terminated.\r\n");}
}

double freq_approx(double freq, int bufLen){
    return round(freq/llSamplerate*bufLen)*llSamplerate/bufLen;
}



/*
****************************************************************************************************************************************************************************
main 
****************************************************************************************************************************************************************************
*/

int main ()
    {
    int_temp=0;
    dynamic_buffersize = 2*round(ramp_time*llSamplerate);

    // ----- open Spectrum card -----
    hCard = spcm_hOpen ((char*)"/dev/spcm0");
    if (!hCard)
        {
        printf ("no card found...\r\n");
        return 0;
        }


    // ----- do a simple FIFO setup for 66xx -----
    spcm_dwSetParam_i32 (hCard, SPC_CHENABLE,       (0x1 << lNumCh) - 1);   // enable all channels
    // spcm_dwSetParam_i32 (hCard, SPC_CARDMODE,       SPC_REP_FIFO_GATE);     // gated FIFO mode
    spcm_dwSetParam_i32 (hCard, SPC_CARDMODE,       SPC_REP_FIFO_SINGLE);   // Test purpose
    spcm_dwSetParam_i32 (hCard, SPC_TRIG_ORMASK,    SPC_TMASK_SOFTWARE);  // TEst purporse
    spcm_dwSetParam_i64 (hCard, SPC_LOOPS,          0);                     // forever
    spcm_dwSetParam_i32 (hCard, SPC_CLOCKMODE,      SPC_CM_INTPLL);         // clock mode internal PLL
    // spcm_dwSetParam_i32 (hCard, SPC_FILTER0,      0);
    spcm_dwSetParam_i64 (hCard, SPC_SAMPLERATE,     llSamplerate);
    spcm_dwSetParam_i32 (hCard, SPC_TIMEOUT,        5*1000);             // Timeout if necessary
    for (int lChIdx = 0; lChIdx < lNumCh; ++lChIdx)
    {
        if (lMaxOutputLevel>amplitude_limit){lMaxOutputLevel=amplitude_limit;}
        spcm_dwSetParam_i32 (hCard, SPC_FILTER0 + lChIdx * (SPC_FILTER1 - SPC_FILTER0), 0);
        spcm_dwSetParam_i32 (hCard, SPC_ENABLEOUT0 + lChIdx * (SPC_ENABLEOUT1 - SPC_ENABLEOUT0), 1);
        spcm_dwSetParam_i32 (hCard, SPC_AMP0       + lChIdx * (SPC_AMP1        - SPC_AMP0),      lMaxOutputLevel);
    }

    spcm_dwSetParam_i64 (hCard, SPC_DATA_OUTBUFSIZE,  HBufferSize);         // Set actual buffer size on the AWG 
    spcm_dwSetParam_i32 (hCard, SPC_M2CMD, M2CMD_CARD_WRITESETUP);          // Write in the configs

    // Card Setup finished
    // ------------------------------------------------------------------------


    // ----- DMA BUFFER SETUP -----
    // ----- get buffer on GPU that will be used as source for RDMA transfer -----
    int lCUDADeviceIdx = 0;         // index of used CUDA device
    static_length = lNotifySize / lNumCh / sizeof(short);
    DMABuffer = pvGetRDMABuffer (lCUDADeviceIdx, lBufferSize); // Configure GPU Buffer
    if (DMABuffer == NULL)
        {
        printf("FAILED TO GET GPUBUFFER\r\n");
        spcm_vClose (hCard);
        return EXIT_FAILURE;
        }

    
    lBytesPerChannelInNotifySize = lNotifySize / lNumCh;
    dynamic_loopcount = (int)ceil((double)dynamic_buffersize/lBytesPerChannelInNotifySize);
    std::thread serverThread(instructionReceiver);
    while (!stop_flag){
        while (!server_flag && !stop_flag){volatile int nulvar = server_flag;}
        start_and_reset: 
        if (stop_flag){break;} 
        static_flag = false;
        static_endflag = false;
        cuda_cleanup();
        varReset();
        server_flag = false;
        continue_flag=false;
        printf("Start generating waveform\n");
    
        for (int ch = 0; ch < 4; ch++){
            dynamic_total += dynamic_num[ch];
            static_total += static_num[ch];
        }
        double dyn_checker = (double)dynamic_loopcount*dynamic_total*lBytesPerChannelInNotifySize/1024/1024/1024;
        double stt_checker = (double) (dynamic_total+4*static_total)*lBytesPerChannelInNotifySize/1024/1024/1024;
        printf("Buffersize for single dynamic tweezer: %f MiB\n",dyn_checker * 1024 / dynamic_total);
        printf("Buffersize for all dynamic tweezer: %f GiB\n",dyn_checker);
        printf("Buffersize for all static tweezer: %f GiB\n",stt_checker);
        printf("Total buffersize: %f GiB\n",dyn_checker+stt_checker);
        if (dyn_checker+stt_checker > 22){
            printf("Buffer required exceeds GPU memory\n");
            spcm_vClose (hCard);
            return EXIT_FAILURE;
        }
        if (update_flag){
            cudaMalloc((void**)&update_index_map_cuda,static_total*sizeof(int));
            cudaMemcpy(update_index_map_cuda,update_index_map,static_total*sizeof(int),cudaMemcpyHostToDevice);
            phase_reorder_update<<<(int)ceil((float)static_total/lThreadsPerBlock),lThreadsPerBlock>>>(update_index_map_cuda,new_phase_list_cuda,phase_list_cuda,static_total);
        }
        cudaDeviceSynchronize();
        tone_counter(dynamic_total);
        for (int ch = 0; ch < lNumCh; ch++){
            for (int i = 0; i < static_num[ch]; i++){
                double freq_approxed = freq_approx(static_freq[ch][i],static_length);
                if (freq_approxed < frequency_limits[ch/2*2]|| freq_approxed > frequency_limits[ch/2*2+1]){
                    cerr<<"Frequency out of range: "<< freq_approxed <<":" << ch <<":" << i << endl;
                    spcm_vClose (hCard);
                    cudaDeviceReset();
                    return EXIT_FAILURE;
                }else{
                    real_static_freq[tone_count[ch]+i] = freq_approxed;
                }
            }        
            for (int i = 0; i < dynamic_num[ch]; i++){
                double freq_approxed = freq_approx(destination_freq[ch][i],static_length);
                if (freq_approxed < frequency_limits[ch/2*2]|| freq_approxed > frequency_limits[ch/2*2+1]){
                    cerr<<"Frequency out of range"<< freq_approxed << endl;
                    spcm_vClose (hCard);
                    cudaDeviceReset();
                    return EXIT_FAILURE;
                }else{
                    real_destination_freq[dynamic_tone_count[ch]+i] = freq_approxed;
                }
            }
        }
        cudaDeviceSynchronize();

        if (staticBufferMalloc()){
            spcm_vClose (hCard);
            cudaDeviceReset();
            return EXIT_FAILURE;
        }
        if (dynamic_total){
            if (dynamicBufferMalloc()){
                spcm_vClose (hCard);
                cudaDeviceReset();
                return EXIT_FAILURE;
            }
            eCudaErr = cudaMalloc ((void**)&destination_buffer_cuda, lBytesPerChannelInNotifySize*dynamic_total); //Configure software buffer
            if (eCudaErr != cudaSuccess)
                {
                printf ("Allocating destination_buffer_cuda on GPU failed: %s\n",cudaGetErrorString(eCudaErr));
                spcm_vClose (hCard);
                cuda_cleanup();
                return 1;
                }
            eCudaErr = cudaMalloc ((void**)&ratio_cuda, sizeof(double)*dynamic_loopcount*static_length); //Configure software buffer
            if (eCudaErr != cudaSuccess)
                {
                printf ("Allocating ratio_cuda on GPU failed: %s\n",cudaGetErrorString(eCudaErr));
                spcm_vClose (hCard);
                cuda_cleanup();
                return EXIT_FAILURE;
                }
            eCudaErr = cudaMalloc ((void**)&dFreq_cuda, dynamic_total*sizeof(double)); //Configure software buffer
                if (eCudaErr != cudaSuccess)
                    {
                    printf ("Allocating dFreq_cuda on GPU failed: %s\n",cudaGetErrorString(eCudaErr));
                    spcm_vClose (hCard);
                    cuda_cleanup();
                    return EXIT_FAILURE;
                    }
            
            eCudaErr = cudaMalloc ((void**)&dDiff_cuda, dynamic_total*sizeof(double)); //Configure software buffer
                if (eCudaErr != cudaSuccess)
                    {
                    printf ("Allocating dDiff_cuda on GPU failed: %s\n",cudaGetErrorString(eCudaErr));
                    spcm_vClose (hCard);
                    cuda_cleanup();
                    return EXIT_FAILURE;
                    }
            eCudaErr = cudaMalloc ((void**)&dphi_cuda, dynamic_total*sizeof(double)); //Configure software buffer
                if (eCudaErr != cudaSuccess)
                    {
                    printf ("Allocating dphi_cuda on GPU failed: %s\n",cudaGetErrorString(eCudaErr));
                    spcm_vClose (hCard);
                    cuda_cleanup();
                    return EXIT_FAILURE;
                    }
            eCudaErr = cudaMalloc ((void**)&damp_cuda, dynamic_total*sizeof(double)); //Configure software buffer
                if (eCudaErr != cudaSuccess)
                    {
                    printf ("Allocating damp_cuda on GPU failed: %s\n",cudaGetErrorString(eCudaErr));
                    spcm_vClose (hCard);
                    cuda_cleanup();
                    return EXIT_FAILURE;
                    }
        }

        
        if (dynamic_total){
            int_temp = static_length/2;
            eCudaErr = cudaMemcpyToSymbol(half_static_bufferlength_cuda, &int_temp, sizeof(int));
            if (eCudaErr != cudaSuccess)
                {
                printf ("cudaMemcpyToSymbol half_static_bufferlength_cuda on GPU failed: %s\n",cudaGetErrorString(eCudaErr));
                spcm_vClose (hCard);
                cuda_cleanup();
                return EXIT_FAILURE;
                }
        }

        for (int ch = 0; ch < lNumCh; ch++){
            for (int i = 0; i < static_num[ch]; i++){
                static_freq[ch][i] = new_static_freq[ch][i];
            }
        }
        if (update_flag){
            if (!amp_flag)StaticWaveGeneration_update<<<(static_length/lThreadsPerBlock),lThreadsPerBlock>>>(real_static_freq_cuda,static_buffer_cuda,summed_buffer_cuda,phase_list_cuda);
            else StaticWaveGeneration_update_amp<<<(static_length/lThreadsPerBlock),lThreadsPerBlock>>>(real_static_freq_cuda,amp_list_cuda,static_buffer_cuda,summed_buffer_cuda,phase_list_cuda);
        }else{
            if (!amp_flag)StaticWaveGeneration<<<(static_length/lThreadsPerBlock),lThreadsPerBlock>>>(real_static_freq_cuda,static_buffer_cuda,summed_buffer_cuda,phase_list_cuda);
            else StaticWaveGeneration_amp<<<(static_length/lThreadsPerBlock),lThreadsPerBlock>>>(real_static_freq_cuda,amp_list_cuda,static_buffer_cuda,summed_buffer_cuda,phase_list_cuda);
        }
        cudaDeviceSynchronize();
        cudaMemcpy(new_phase_list_cuda,phase_list_cuda,static_total*sizeof(double),cudaMemcpyDeviceToDevice);

        
        // ----- setup DMA transfer from GPU to Spectrum card -----
        // ----- fill the software buffer before we start the card -----
        if (dynamic_total){
            if (!amp_flag)FinalWaveGeneration<<<static_length/lThreadsPerBlock,lThreadsPerBlock>>>(real_static_freq_cuda,real_destination_freq_cuda,dynamic_list_cuda,phase_list_cuda,new_phase_list_cuda,destination_buffer_cuda);
            else FinalWaveGeneration_amp<<<static_length/lThreadsPerBlock,lThreadsPerBlock>>>(real_static_freq_cuda,real_destination_freq_cuda,dynamic_list_cuda,amp_list_cuda,phase_list_cuda,new_phase_list_cuda,destination_buffer_cuda);
            cudaDeviceSynchronize(); 
            SavedStaticCombine <<< static_length/ lThreadsPerBlock, lThreadsPerBlock >>> (static_buffer_cuda,summed_buffer_cuda,saved_buffer_cuda,dynamic_list_cuda);
            cudaDeviceSynchronize();
            if (!amp_flag)DynamicListWorker<<<ceil(dynamic_total/32.),32>>>(real_static_freq_cuda,real_destination_freq_cuda,dynamic_list_cuda,dFreq_cuda,dDiff_cuda,dphi_cuda,phase_list_cuda);
            else DynamicListWorker_amp<<<ceil(dynamic_total/32.),32>>>(real_static_freq_cuda,real_destination_freq_cuda,dynamic_list_cuda,dFreq_cuda,dDiff_cuda,dphi_cuda,damp_cuda,phase_list_cuda,amp_list_cuda);
            ratio_calc <<<dynamic_loopcount*static_length/lThreadsPerBlock,lThreadsPerBlock>>> (ratio_cuda);
            FinalCombine <<< static_length/lThreadsPerBlock, lThreadsPerBlock >>> (saved_buffer_cuda,destination_buffer_cuda,final_buffer_cuda);
            cudaDeviceSynchronize();   
        }
        if (init_flag) spcm_dwDefTransfer_i64 (hCard, SPCM_BUF_DATA, SPCM_DIR_GPUTOCARD, lNotifySize, DMABuffer, 0, lBufferSize);
    
        
        for (int32 lPosInBuf = 0; lPosInBuf < lBufferSize; lPosInBuf += lNotifySize)
            {
            StaticMux <<< static_length / lThreadsPerBlock, lThreadsPerBlock >>> (summed_buffer_cuda,(int16*)((char*)DMABuffer + lPosInBuf));
            }
        cudaDeviceSynchronize();
        
        if (init_flag){
            printf("\r\nCalculated: Init\r\n");
            // mark data as valid
            dwError = spcm_dwSetParam_i32 (hCard, SPC_DATA_AVAIL_CARD_LEN,  lBufferSize);
            if (dwError != ERR_OK){
                spcm_dwGetErrorInfo_i32 (hCard, NULL, NULL, szErrorTextBuffer);
                printf ("Error on SPC_DATA_AVAIL_CARD_LEN: %u (%s)\n", dwError, szErrorTextBuffer);
                spcm_vClose (hCard);
                cudaDeviceReset();
                return EXIT_FAILURE;
            }
            
            // ----- start transfer from GPU into card and wait until it has finished -----
            dwError = spcm_dwSetParam_i32 (hCard, SPC_M2CMD, M2CMD_DATA_STARTDMA | M2CMD_DATA_WAITDMA);   
            if (dwError != ERR_OK)
            {
                spcm_dwGetErrorInfo_i32 (hCard, NULL, NULL, szErrorTextBuffer);
                printf ("Error on STARTDMA | WAITDMA: %u (%s)\n", dwError, szErrorTextBuffer);
                spcm_vClose (hCard);
                cudaDeviceReset();
                return EXIT_FAILURE;
            }

            // std::thread terminatorThread(ifkeypress,&iskeypressed);
            // ----- start everything -----
            dwError = spcm_dwSetParam_i32 (hCard, SPC_M2CMD, M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER);
            if (dwError != ERR_OK)
            {
                spcm_dwGetErrorInfo_i32 (hCard, NULL, NULL, szErrorTextBuffer);
                printf ("CARD_START failed: %u (%s)\n", dwError, szErrorTextBuffer);
                spcm_vClose (hCard);
                cudaDeviceReset();
                return EXIT_FAILURE;
            }
            staticThread =std::thread(static_looper);
            init_flag = false;
        }
        static_flag = true;
        static_endflag = true;
        printf("Looping ...\n");

        if (dynamic_total){
            while (!continue_flag && !server_flag && !stop_flag){} 
            if (server_flag){goto start_and_reset;}
            if (stop_flag) break;
            static_pulseflag=false;
            continue_flag = false;
            static_flag = false;
            
            static_endflag = true;
            
            while (!static_pulseflag){}
            static_pulseflag=false;
            sPointerPosition=0;
            for (int cnt = 0; cnt < dynamic_loopcount;cnt++){
                if ((dwError = spcm_dwSetParam_i32 (hCard, SPC_M2CMD, M2CMD_DATA_WAITDMA)) != ERR_OK)
                {
                    if (dwError == ERR_TIMEOUT)
                        printf ("\n... Timeout\n");
                    else
                        printf ("\n... Error: %u (%s)\n", dwError,szErrorTextBuffer);
                    spcm_vClose (hCard);
                    cuda_cleanup();
                    return EXIT_FAILURE;
                }else{                           
                    spcm_dwGetParam_i32 (hCard, SPC_DATA_AVAIL_USER_POS,  &lUserPos);                    
                    StaticMux <<< static_length / lThreadsPerBlock, lThreadsPerBlock >>> (summed_buffer_cuda,(int16*)((char*)DMABuffer + lUserPos));                    
                    cudaDeviceSynchronize();
                    
                    dwError = spcm_dwSetParam_i32 (hCard, SPC_DATA_AVAIL_CARD_LEN,  lNotifySize);
                    if (dwError!=ERR_OK){
                        printf("\n... Error in Setting CardAval: %u (%s)\n", dwError,szErrorTextBuffer);
                        spcm_vClose (hCard);
                        cuda_cleanup();
                        return EXIT_FAILURE;
                    }
                    if (!amp_flag)AccelCombine<<< static_length/lThreadsPerBlock/2, lThreadsPerBlock >>>(sPointerPosition,saved_buffer_cuda,summed_buffer_cuda,dFreq_cuda,dDiff_cuda,dphi_cuda,ratio_cuda);
                    else AccelCombine_amp<<< static_length/lThreadsPerBlock/2, lThreadsPerBlock >>>(sPointerPosition,saved_buffer_cuda,summed_buffer_cuda,dFreq_cuda,dDiff_cuda,dphi_cuda,ratio_cuda,damp_cuda);
                    cudaDeviceSynchronize();
                    sPointerPosition += static_length;
                }
            }
            
            printf("DESTINATION\n");
            while (!static_flag) // Terminated when key pressed; not working if RMA keeps waiting
            {        
                if ((dwError = spcm_dwSetParam_i32 (hCard, SPC_M2CMD, M2CMD_DATA_WAITDMA)) != ERR_OK)
                    {
                    if (dwError == ERR_TIMEOUT)
                        printf ("\n... Timeout\n");
                    else
                        printf ("\n... Error: %u\n", dwError);
                    break;
                    }
                else
                    {        
                        spcm_dwGetParam_i32 (hCard, SPC_DATA_AVAIL_USER_POS,  &lUserPos);
                        if (not_arrived){
                            StaticMux <<< static_length / lThreadsPerBlock, lThreadsPerBlock >>> (summed_buffer_cuda,(int16*)((char*)DMABuffer + lUserPos));
                            not_arrived=0;
                            loop_counter=0;
                            printf("Done\n");
                        }else if(loop_counter<lBufferSize/lNotifySize){
                            WaveformCopier <<< lNumCh * static_length / lThreadsPerBlock, lThreadsPerBlock >>> (final_buffer_cuda,(int16*)((char*)DMABuffer + lUserPos));
                            loop_counter++;
                        }else{
                            static_flag = true;
                            break;
                        }
                        cudaDeviceSynchronize();
                        dwError = spcm_dwSetParam_i32 (hCard, SPC_DATA_AVAIL_CARD_LEN,  lNotifySize);
                        if (dwError!=ERR_OK){
                            printf("\n... Error in Setting CardAval: %u (%s)\n", dwError,szErrorTextBuffer);
                            break;
                        }
                    }
            }
        }
        // send the stop command
        if (eCudaErr=cudaPeekAtLastError()) printf("CUDA Error Peek: %s\n",cudaGetErrorString(eCudaErr));
    }
    dwError = spcm_dwSetParam_i32 (hCard, SPC_M2CMD, M2CMD_CARD_STOP | M2CMD_DATA_STOPDMA);

    // clean up
    staticThread.join();
    serverThread.join();
    printf ("\nFinished...\n");
    spcm_vClose (hCard);
    cudaDeviceReset();
    return EXIT_SUCCESS;
    }

