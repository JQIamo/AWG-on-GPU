#include "lib/cuda_functions.h"
//  Timer includes
#   include <chrono>
#   include <iostream>
#   include <thread>
using namespace std;

char        szErrorTextBuffer[ERRORTEXTLEN];
uint32      dwError;
int32       lUserPos;
int loop_counter;

void* DMABuffer = NULL;
 // Depending on the GPU used

int block_size;
// settings for the FIFO mode buffer handling
uint32       lNotifySize =  MEGA_B(2); // The size of data the card will execute each time before signaling to the GPU
uint32       lBufferSize =  MEGA_B(64);
uint64       HBufferSize =  MEGA_B(64); // The actual buffer used on the AWG; must be a power of 2 and should be no more than 4 GB (lower size reduces delay)

// Parameter settings   
int32       lMaxOutputLevel = 1000; // +-1 Volt


// Test params
int16* final_buffer[4];

double* dFreq_cuda;
double* dDiff_cuda;
double* dphi_cuda;
double* ratio_cuda;
__device__ __constant__ int half_static_bufferlength_cuda;
unsigned long long sPointerPosition=0;


// ------Dynamics----------------------------
 int16* destination_buffer_cuda;


__global__ void FinalWaveGeneration (double* startFreq, double* dest_frequency, int* dy_list,short* pnOut)
    {
        size_t i = blockDim.x * blockIdx.x + threadIdx.x;
        for (int buffer_index=0;buffer_index<channel_num_cuda;buffer_index++){
            for (size_t j=0;j<dynamic_num_cuda[buffer_index];j++){
                int tone_index = j+dynamic_tone_count_cuda[buffer_index];
                double phi = (2.0 * dest_frequency[tone_index] * (dynamic_loopcount_cuda*static_bufferlength_cuda-dynamic_bufferlength_cuda) +(startFreq[dy_list[tone_index]] 
                + dest_frequency[tone_index])*dynamic_bufferlength_cuda)/ llSamplerate_cuda +2./static_num_cuda[buffer_index]*dy_list[tone_index]*dy_list[tone_index];
                pnOut[tone_index*static_bufferlength_cuda+i] = static_cast<short>(32767. * sinpi (2.0 * dest_frequency[tone_index] * i / llSamplerate_cuda+phi)/static_num_cuda[buffer_index]);
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

__global__ void DynamicListWorker(double*staticFrequency,double*destinationFreq,int*dy_list,double *dstart,double*dDiff,double*dphi){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    for (int buffer_index=0;buffer_index<channel_num_cuda;buffer_index++){
        if(i<dynamic_num_cuda[buffer_index]){
            int tone_index = i+dynamic_tone_count_cuda[buffer_index];
            double temp = staticFrequency[dy_list[tone_index]];
            dstart[tone_index] = temp;
            dDiff[tone_index] = destinationFreq[tone_index]-temp;
            dphi[tone_index] = 1./static_num_cuda[buffer_index]*dy_list[tone_index]*dy_list[tone_index];
        }        
    }

}

__global__ void ratio_calc(double*ratiolist){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    double ratio = i/static_cast<double>(dynamic_bufferlength_cuda);
    ratiolist[i] = (2.5  - 3. * ratio +  ratio *ratio)* ratio *ratio *ratio;
}


__global__ void AccelCombine(unsigned long long startPosition,short**__restrict__ save_buf,short** sum_buf,double*__restrict__ dstartFrequency,double *__restrict__ dfreqeuncyDiff,double *__restrict__ dphi,double*__restrict__ ratio){
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
            // printf("startFrequency %f freqeuncyDiff %f phi %f\n",startFrequency[threadIdx.x],freqeuncyDiff[threadIdx.x],phi[threadIdx.x]);
        }
        __syncthreads();
        double sum = save_buf[buffer_index][i];
        double sum1 = save_buf[buffer_index][i + half_static_bufferlength_cuda];
        if (k1<dynamic_bufferlength_cuda){
            for (size_t j = 0; j<dynamic_num_cuda[buffer_index]; j++){
                double phase_c =  k * illSamplerate_cuda * fma(freqeuncyDiff[j] , ratio[static_cast<size_t>(k)],startFrequency[j]);
                double phase_c1 = k1 * illSamplerate_cuda * fma( freqeuncyDiff[j] , ratio[static_cast<size_t>(k1)],startFrequency[j]);
                half2 temp = __hmul2 (__float2half2_rn(32767.*(istatic_num_cuda[buffer_index])) , h2sin (__floats2half2_rn(2.*M_PI*modf(phase_c+phi[j],&s),2.*M_PI*modf(phase_c1+phi[j],&s))));
                sum1 += __high2float(temp);
                sum += __low2float(temp);
            }
        }else{ 
            for (size_t j = 0; j<dynamic_num_cuda[buffer_index]; j++){
                double pc = 2.*M_PI*modf(k * fma(illSamplerate_cuda , fma( freqeuncyDiff[j] , ratio[static_cast<size_t>(k)],startFrequency[j]),phi[j]),&s);
                double pc1 = M_PI*(2.0 * modf((startFrequency[j]+freqeuncyDiff[j]) * (k1-static_cast<double>(dynamic_bufferlength_cuda)) * illSamplerate_cuda+fma(freqeuncyDiff[j],0.5,startFrequency[j])*static_cast<double>(dynamic_bufferlength_cuda)*illSamplerate_cuda+phi[j],&s));
                half2 temp = __hmul2 (__float2half2_rn(32767.*(istatic_num_cuda[buffer_index])) , h2sin (__floats2half2_rn(pc,pc1)));
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
**************************************************************************
*/


void ifkeypress(bool * iskeypressed){
    printf("Start working. Press ENTER to terminate.\r\n");
    getchar();
    * iskeypressed = true;
    getchar();
    * iskeypressed = true;
}

void instructionReceiver(){
    // Receive instructions for tweezer motion from MainControlPC
}

double freq_approx(double freq, int bufLen){
    return round(freq/llSamplerate*bufLen)*llSamplerate/bufLen;
}

void cuda_cleanup ()
    {
        cudaDeviceSynchronize();
        if (DMABuffer != NULL) {cudaFree (DMABuffer);}

        for (int i = 0; i < lNumCh; ++i){
            if (summed_buffer[i] != NULL) cudaFree (summed_buffer[i]);
            if (final_buffer[i] != NULL) cudaFree (final_buffer[i]);
            if (saved_buffer[i] != NULL) cudaFree (saved_buffer[i]);
            if (dynamic_saved_buffer[i] != NULL) cudaFree (dynamic_saved_buffer[i]);
        }
        if (destination_buffer_cuda != NULL) cudaFree(destination_buffer_cuda);
        // if (dynamic_buffer_cuda != NULL) cudaFree(dynamic_buffer_cuda);
        if (summed_buffer_cuda != NULL) cudaFree(summed_buffer_cuda);
        if (final_buffer_cuda != NULL) cudaFree(final_buffer_cuda);
        if (saved_buffer_cuda != NULL) cudaFree(saved_buffer_cuda);
        if (dynamic_saved_buffer_cuda != NULL) cudaFree(dynamic_saved_buffer_cuda);
        if (dynamic_list_cuda != NULL) cudaFree(dynamic_list_cuda);
        if (static_buffer_cuda != NULL) cudaFree(static_buffer_cuda);
        cudaDeviceSynchronize();
    }


/*
****************************************************************************************************************************************************************************
main 
****************************************************************************************************************************************************************************
*/

int main ()
    {
    bool        iskeypressed = false;
    double test_freq_step = 50. / static_num[0];
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
    spcm_dwSetParam_i32 (hCard, SPC_CARDMODE,       SPC_REP_FIFO_SINGLE);   // Test purpose
    spcm_dwSetParam_i32 (hCard, SPC_TRIG_ORMASK,    SPC_TMASK_SOFTWARE);  // TEst purporse
    spcm_dwSetParam_i64 (hCard, SPC_LOOPS,          0);                     // forever
    spcm_dwSetParam_i32 (hCard, SPC_CLOCKMODE,      SPC_CM_INTPLL);         // clock mode internal PLL
    spcm_dwSetParam_i32 (hCard, SPC_FILTER0,      0);
    spcm_dwSetParam_i64 (hCard, SPC_SAMPLERATE,     llSamplerate);
    spcm_dwSetParam_i32 (hCard, SPC_TIMEOUT,        5*1000);             // Timeout if necessary
    // spcm_dwSetParam_i32 (hCard, SPC_TIMEOUT,        0);
    for (int lChIdx = 0; lChIdx < lNumCh; ++lChIdx)
        {
        spcm_dwSetParam_i32 (hCard, SPC_ENABLEOUT0 + lChIdx * (SPC_ENABLEOUT1 - SPC_ENABLEOUT0), 1);
        spcm_dwSetParam_i32 (hCard, SPC_AMP0       + lChIdx * (SPC_AMP1        - SPC_AMP0),      lMaxOutputLevel);
        }

    spcm_dwSetParam_i64 (hCard, SPC_DATA_OUTBUFSIZE,  HBufferSize);         // Set actual buffer size on the AWG 
    spcm_dwSetParam_i32 (hCard, SPC_M2CMD, M2CMD_CARD_WRITESETUP);          // Write in the configs
    // spcm_dwSetParam_i32 (hCard, SPC_CH0_STOPLEVEL, SPCM_STOPLVL_HIGH);   // Set the idle state of the signal

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


    // ----- allocate memory for each channel on GPU host to use for copying the waveform data -----
    
    auto ts = std::chrono::high_resolution_clock::now();
    lBytesPerChannelInNotifySize = lNotifySize / lNumCh;
    dynamic_loopcount = (int)ceil((double)dynamic_buffersize/lBytesPerChannelInNotifySize);
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
    tone_counter(dynamic_total);
    for (int ch = 0; ch < 4; ch++){
        for (int i = 0; i < static_num[ch]; i++){
            real_static_freq[tone_count[ch]+i] = freq_approx(static_freq[ch][i],static_length);
        }        
        for (int i = 0; i < dynamic_num[ch]; i++){
            real_destination_freq[dynamic_tone_count[ch]+i] = freq_approx(destination_freq[ch][i],static_length);
        }
    }

    if (staticBufferMalloc()){
    spcm_vClose (hCard);
    cuda_cleanup();
    return EXIT_FAILURE;
    }
    if (dynamic_total){
        if (dynamicBufferMalloc()){
            spcm_vClose (hCard);
            cuda_cleanup();
            return EXIT_FAILURE;
        }
        eCudaErr = cudaMalloc ((void**)&destination_buffer_cuda, lBytesPerChannelInNotifySize*dynamic_total); //Configure software buffer
        if (eCudaErr != cudaSuccess)
            {
            printf ("Allocating destination_buffer_cudaon GPU failed: %s\n",cudaGetErrorString(eCudaErr));
            spcm_vClose (hCard);
            cuda_cleanup();
            return 1;
            }
        eCudaErr = cudaMalloc ((void**)&ratio_cuda, sizeof(double)*dynamic_loopcount*static_length); //Configure software buffer
        if (eCudaErr != cudaSuccess)
            {
            printf ("Allocating dynamic_list_cuda on GPU failed: %s\n",cudaGetErrorString(eCudaErr));
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
                printf ("Allocating dFreq_cuda on GPU failed: %s\n",cudaGetErrorString(eCudaErr));
                spcm_vClose (hCard);
                cuda_cleanup();
                return EXIT_FAILURE;
                }
        eCudaErr = cudaMalloc ((void**)&dphi_cuda, dynamic_total*sizeof(double)); //Configure software buffer
            if (eCudaErr != cudaSuccess)
                {
                printf ("Allocating dFreq_cuda on GPU failed: %s\n",cudaGetErrorString(eCudaErr));
                spcm_vClose (hCard);
                cuda_cleanup();
                return EXIT_FAILURE;
                }
    }


    int_temp = static_length/2;
    eCudaErr = cudaMemcpyToSymbol(half_static_bufferlength_cuda, &int_temp, sizeof(int));
    if (eCudaErr != cudaSuccess)
        {
        printf ("cudaMemcpyToSymbol static_bufferlength_cuda on GPU failed: %s\n",cudaGetErrorString(eCudaErr));
        spcm_vClose (hCard);
        cuda_cleanup();
        return EXIT_FAILURE;
        }
       
    StaticWaveGeneration<<<(static_length/lThreadsPerBlock),lThreadsPerBlock>>>(real_static_freq_cuda,static_buffer_cuda);
    cudaDeviceSynchronize();
    
    if (dynamic_total){
        FinalWaveGeneration<<<static_length/lThreadsPerBlock,lThreadsPerBlock>>>(real_static_freq_cuda,real_destination_freq_cuda,dynamic_list_cuda,destination_buffer_cuda);
        cudaDeviceSynchronize(); 
    }
    
    // ----- setup DMA transfer from GPU to Spectrum card -----
    spcm_dwDefTransfer_i64 (hCard, SPCM_BUF_DATA, SPCM_DIR_GPUTOCARD, lNotifySize, DMABuffer, 0, lBufferSize);
    // ----- fill the software buffer before we start the card -----
    auto tt3 = std::chrono::high_resolution_clock::now();
    StaticCombine <<< static_length/ lThreadsPerBlock, lThreadsPerBlock >>> (static_buffer_cuda,summed_buffer_cuda);
    cudaDeviceSynchronize();

    if (dynamic_total){
        SavedStaticCombine <<< static_length/ lThreadsPerBlock, lThreadsPerBlock >>> (static_buffer_cuda,summed_buffer_cuda,saved_buffer_cuda,dynamic_list_cuda);
        cudaDeviceSynchronize();
        if (eCudaErr=cudaPeekAtLastError()) printf("3CUDA Error Peek!!: %s\n",cudaGetErrorString(eCudaErr));
        DynamicListWorker<<<ceil(dynamic_total/32.),32>>>(real_static_freq_cuda,real_destination_freq_cuda,dynamic_list_cuda,dFreq_cuda,dDiff_cuda,dphi_cuda);
        ratio_calc <<<dynamic_loopcount*static_length/lThreadsPerBlock,lThreadsPerBlock>>> (ratio_cuda);
        FinalCombine <<< static_length/lThreadsPerBlock, lThreadsPerBlock >>> (saved_buffer_cuda,destination_buffer_cuda,final_buffer_cuda);
        cudaDeviceSynchronize();   
    }
 
    
    for (int32 lPosInBuf = 0; lPosInBuf < lBufferSize; lPosInBuf += lNotifySize)
        {
        StaticMux <<< static_length / lThreadsPerBlock, lThreadsPerBlock >>> (summed_buffer_cuda,(int16*)((char*)DMABuffer + lPosInBuf));
        cudaDeviceSynchronize();
        }
    
    printf("\r\nCalculated: Init\r\n");
    
    // send the stop command
    if (eCudaErr=cudaPeekAtLastError()) printf("CUDA Error Peek: %s\n",cudaGetErrorString(eCudaErr));

    // dwError = spcm_dwSetParam_i32 (hCard, SPC_M2CMD, M2CMD_CARD_STOP | M2CMD_DATA_STOPDMA);

    // clean up
    printf ("\nFinished...\n");
    spcm_vClose (hCard);
    // free CUDA buffers on GPU and host
    cuda_cleanup();
    return EXIT_SUCCESS;
    }

