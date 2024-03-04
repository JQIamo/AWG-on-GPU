/*
Waveform_synthesis_playback.cu
Author: Juntian Tu
Date: 2024.03.12

This is the implementation for the playback pathway. Change the source file name in the Makefile to compile the program.
*/

#include "lib/cuda_functions.h"
#   include <iostream>
#   include <thread>

using namespace std;

char        szErrorTextBuffer[ERRORTEXTLEN];
uint32      dwError;
int32       lUserPos;
int loop_counter=0;

void* DMABuffer = NULL;

void cuda_cleanup ()
    {
        cudaDeviceSynchronize();
        if (DMABuffer != NULL) {cudaFree (DMABuffer);}
        for (int i = 0; i < lNumCh; ++i){
            if (summed_buffer[i] != NULL) cudaFree (summed_buffer[i]);
            if (saved_buffer[i] != NULL) cudaFree (saved_buffer[i]);
            if (dynamic_saved_buffer[i] != NULL) cudaFree (dynamic_saved_buffer[i]);
        }
        if (summed_buffer_cuda != NULL) cudaFree(summed_buffer_cuda);
        if (final_buffer_cuda != NULL) cudaFree(final_buffer_cuda);
        if (saved_buffer_cuda != NULL) cudaFree(saved_buffer_cuda);
        if (dynamic_saved_buffer_cuda != NULL) cudaFree(dynamic_saved_buffer_cuda);
        if (dynamic_list_cuda != NULL) cudaFree(dynamic_list_cuda);
        if (static_buffer_cuda != NULL) cudaFree(static_buffer_cuda);
        cudaDeviceSynchronize();
    }


// settings for the FIFO mode buffer handling
uint64       lNotifySize =  MEGA_B(2); // The size of data the card will execute each time before signaling to the GPU
uint32       lBufferSize =  MEGA_B(64);
uint64       HBufferSize =  MEGA_B(64); // The actual buffer used on the AWG; must be a power of 2 and should be no more than 4 GB (lower size reduces delay)

// Parameter settings   
int32        lMaxOutputLevel = 1000; // +-1 Volt
unsigned long long sPointerPosition=0;
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

/*
****************************************************************************************************************************************************************************
main 
****************************************************************************************************************************************************************************
*/

int main ()
{
    
    bool        iskeypressed = false;    
    dynamic_buffersize = 2*round(ramp_time*llSamplerate);
    // ------------------------------------------------------------------------

    // ----- open Spectrum card -----
    hCard = spcm_hOpen ((char*)"/dev/spcm0");
    if (!hCard)
    {
        printf ("no card found...\r\n");
        return 0;
    }


    // ----- read type, function and sn and check for A/D card -----
    // ----- do a simple FIFO setup for 66xx -----
    spcm_dwSetParam_i32 (hCard, SPC_CHENABLE,       (0x1 << lNumCh) - 1);   // enable all channels
    // spcm_dwSetParam_i32 (hCard, SPC_CARDMODE,       SPC_REP_FIFO_GATE);     // gated FIFO mode
    spcm_dwSetParam_i32 (hCard, SPC_CARDMODE,       SPC_REP_FIFO_SINGLE);   // Test purpose
    spcm_dwSetParam_i32 (hCard, SPC_TRIG_ORMASK,    SPC_TMASK_SOFTWARE);  // TEst purporse
    spcm_dwSetParam_i64 (hCard, SPC_LOOPS,          0);                     // forever
    spcm_dwSetParam_i32 (hCard, SPC_CLOCKMODE,      SPC_CM_INTPLL);         // clock mode internal PLL
    spcm_dwSetParam_i32 (hCard, SPC_FILTER0,      0);
    spcm_dwSetParam_i64 (hCard, SPC_SAMPLERATE,     llSamplerate);
    spcm_dwSetParam_i32 (hCard, SPC_TIMEOUT,        5*1000);             // Timeout if necessary
    for (int lChIdx = 0; lChIdx < lNumCh; ++lChIdx)
    {
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


    // ----- allocate memory for each channel on GPU host to use for copying the waveform data -----
    
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
    };
    if (dynamic_total){
        if (dynamicBufferMalloc()){
            spcm_vClose (hCard);
            cuda_cleanup();
            return EXIT_FAILURE;
        }
    }
    auto tt = std::chrono::high_resolution_clock::now();
    StaticWaveGeneration_single<<<(static_length/lThreadsPerBlock),lThreadsPerBlock>>>(real_static_freq_cuda,static_buffer_cuda,summed_buffer_cuda);
    cudaDeviceSynchronize();
    auto tt1 = std::chrono::high_resolution_clock::now();
    if (eCudaErr=cudaPeekAtLastError()) printf("CUDA Error Peek!!: %s\n",cudaGetErrorString(eCudaErr));

    if (dynamic_total){
        Pre_computer<<<static_length/lThreadsPerBlock,lThreadsPerBlock>>>(static_buffer_cuda,static_list_cuda,real_destination_freq_cuda,
                    dynamic_list_cuda,final_buffer_cuda,dynamic_saved_buffer_cuda,real_static_freq_cuda);
    }
    // ----- setup DMA transfer from GPU to Spectrum card -----
    
    // ----- fill the software buffer before we start the card -----
    printf("static length: %d\n",static_length);
    auto duration1 = std::chrono::duration_cast<std::chrono::microseconds>(tt1 - tt).count();
    printf("Static time: %d\n", duration1);
    spcm_dwDefTransfer_i64 (hCard, SPCM_BUF_DATA, SPCM_DIR_GPUTOCARD, lNotifySize, DMABuffer, 0, lBufferSize);
    for (int32 lPosInBuf = 0; lPosInBuf < lBufferSize; lPosInBuf += lNotifySize)
    {
    StaticMux <<< static_length / lThreadsPerBlock, lThreadsPerBlock >>> (summed_buffer_cuda,(int16*)((char*)DMABuffer + lPosInBuf));
    }
    cudaDeviceSynchronize();
    
    printf("\r\nCalculated: Init\r\n");
    // mark data as valid
    dwError = spcm_dwSetParam_i32 (hCard, SPC_DATA_AVAIL_CARD_LEN,  lBufferSize);
    if (dwError != ERR_OK){
        spcm_dwGetErrorInfo_i32 (hCard, NULL, NULL, szErrorTextBuffer);
        printf ("Error on SPC_DATA_AVAIL_CARD_LEN: %u (%s)\n", dwError, szErrorTextBuffer);
        spcm_vClose (hCard);
        cuda_cleanup();
        return EXIT_FAILURE;
    }
    
    // ----- start transfer from GPU into card and wait until it has finished -----
    dwError = spcm_dwSetParam_i32 (hCard, SPC_M2CMD, M2CMD_DATA_STARTDMA | M2CMD_DATA_WAITDMA);   
    if (dwError != ERR_OK)
    {
        spcm_dwGetErrorInfo_i32 (hCard, NULL, NULL, szErrorTextBuffer);
        printf ("Error on STARTDMA | WAITDMA: %u (%s)\n", dwError, szErrorTextBuffer);
        spcm_vClose (hCard);
        cuda_cleanup();
        return EXIT_FAILURE;
    }


    std::thread terminatorThread(ifkeypress,&iskeypressed);
    // ----- start everything -----
    dwError = spcm_dwSetParam_i32 (hCard, SPC_M2CMD, M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER);
    if (dwError != ERR_OK)
    {
        spcm_dwGetErrorInfo_i32 (hCard, NULL, NULL, szErrorTextBuffer);
        printf ("CARD_START failed: %u (%s)\n", dwError, szErrorTextBuffer);
        spcm_vClose (hCard);
        cuda_cleanup();
        return EXIT_FAILURE;
    }

    std::thread serverThread(instructionReceiver);
    printf("Looping ...\n");
    while (iskeypressed == false) // Terminated when key pressed; not working if RMA keeps waiting
    {
        if ((dwError = spcm_dwSetParam_i32 (hCard, SPC_M2CMD, M2CMD_DATA_WAITDMA)) != ERR_OK)
        {
            if (dwError == ERR_TIMEOUT)
                printf ("\n... Timeout\n");
            else
                printf ("\n... Error: %u (%s)\n", dwError,szErrorTextBuffer);
            break;
        }
        else
        {             
            spcm_dwGetParam_i32 (hCard, SPC_DATA_AVAIL_USER_POS,  &lUserPos);
            // mark data as valid and transfer it to card
            dwError = spcm_dwSetParam_i32 (hCard, SPC_DATA_AVAIL_CARD_LEN,  lNotifySize);
            if (dwError!=ERR_OK){
                printf("\n... Error in Setting CardAval: %u (%s)\n", dwError,szErrorTextBuffer);
                break;
            }
        }
    }

    if (dynamic_total){
        iskeypressed = false;
        for (int cnt = 0; cnt < dynamic_loopcount && !iskeypressed;cnt++){
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
                DynamicMux <<< static_length / lThreadsPerBlock, lThreadsPerBlock >>> (sPointerPosition,dynamic_saved_buffer_cuda,(int16*)((char*)DMABuffer + lUserPos));
                cudaDeviceSynchronize();
                if (eCudaErr=cudaPeekAtLastError()) printf("CUDA Error Peek: %s\n",cudaGetErrorString(eCudaErr));
                dwError = spcm_dwSetParam_i32 (hCard, SPC_DATA_AVAIL_CARD_LEN,  lNotifySize);
                if (dwError!=ERR_OK){
                    printf("\n... Error in Setting CardAval: %u (%s)\n", dwError,szErrorTextBuffer);
                    spcm_vClose (hCard);
                    cuda_cleanup();
                    return EXIT_FAILURE;
                }
                sPointerPosition += static_length;
            }
        }

        printf("DESTINATION\n");
        while (iskeypressed == false) // Terminated when key pressed; not working if RMA keeps waiting
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
                if (loop_counter<lBufferSize/lNotifySize){
                    WaveformCopier <<< lNumCh * static_length / lThreadsPerBlock, lThreadsPerBlock >>> (final_buffer_cuda,(int16*)((char*)DMABuffer + lUserPos));
                    cudaDeviceSynchronize();
                    loop_counter++;
                }
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

    dwError = spcm_dwSetParam_i32 (hCard, SPC_M2CMD, M2CMD_CARD_STOP | M2CMD_DATA_STOPDMA);

    // clean up
    printf ("\nFinished...\n");
    spcm_vClose (hCard);
    cuda_cleanup();
    return EXIT_SUCCESS;
}