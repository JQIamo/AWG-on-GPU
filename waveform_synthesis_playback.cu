/*
Waveform_synthesis_playback.cu
Author: Juntian Tu
Date: 2024.09.03

This is the implementation for the playback pathway. Change the source file name in the Makefile to compile the program.
*/

#include "lib/cuda_functions.h"
#include <atomic>
#   include <iostream>
#   include <thread>

using namespace std;
char        szErrorTextBuffer[ERRORTEXTLEN];
uint32      dwError;
int32       lUserPos;
int loop_counter=0;
volatile std::atomic<bool> init_flag(true);
volatile std::atomic<bool> static_flag(true);

void* DMABuffer = NULL;

void reset_amp(){
    if (lMaxOutputLevel>amplitude_limit){lMaxOutputLevel=amplitude_limit;}
    for (int i = 0; i < lNumCh; ++i){
        spcm_dwSetParam_i32 (hCard, SPC_AMP0       + i * (SPC_AMP1        - SPC_AMP0),      lMaxOutputLevel);
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
        if (eCudaErr=cudaPeekAtLastError()) printf("Cuda Buffer Clean Failed: %s\n",cudaGetErrorString(eCudaErr));
        cudaDeviceSynchronize();
    }


// settings for the FIFO mode buffer handling
uint64       lNotifySize =  MEGA_B(2); // The size of data the card will execute each time before signaling to the GPU
uint32       lBufferSize =  MEGA_B(64);
uint64       HBufferSize =  MEGA_B(64); // The actual buffer used on the AWG; must be a power of 2 and should be no more than 4 GB (lower size reduces delay)
std::thread staticThread;
// Parameter settings   
int32       lMaxOutputLevel  = amplitude_limit; // +-1 Volt
unsigned long long sPointerPosition=0;

void instructionReceiver(){
    // Receive instructions for tweezer motion from MainControlPC
    if (TCP_server()){printf("TCP server terminated.\r\n");}
}

double freq_approx(double freq, int bufLen){
    return round(freq/llSamplerate*bufLen)*llSamplerate/bufLen;
}

void varReset(){
    dynamic_total = 0;
    static_total = 0;
    loop_counter=0;
}

void static_looper(){
    while (!stop_flag){
        while (static_flag && !stop_flag){
            while (!static_endflag.load() && !stop_flag){
                if ((dwError = spcm_dwSetParam_i32 (hCard, SPC_M2CMD, M2CMD_DATA_WAITDMA)) != ERR_OK)
                {
                    if (dwError == ERR_TIMEOUT)
                        printf ("\n... Timeout\n");
                    else
                        spcm_dwGetErrorInfo_i32 (hCard, NULL, NULL, szErrorTextBuffer);
                        printf ("\n... Error: %u (%s)\n", dwError,szErrorTextBuffer);
                        stop_flag = true;
                    break;
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

/*
****************************************************************************************************************************************************************************
main 
****************************************************************************************************************************************************************************
*/

int main ()
{
    
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
    spcm_dwSetParam_i64 (hCard, SPC_SAMPLERATE,     llSamplerate);
    spcm_dwSetParam_i32 (hCard, SPC_TIMEOUT,        5*1000);             // Timeout if necessary
    if (lMaxOutputLevel>amplitude_limit){lMaxOutputLevel=amplitude_limit;}
    for (int lChIdx = 0; lChIdx < lNumCh; ++lChIdx)
    {
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


    // ----- allocate memory for each channel on GPU host to use for copying the waveform data -----
    
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
        if (eCudaErr=cudaPeekAtLastError()) printf("QCUDA Error Peek: %s\n",cudaGetErrorString(eCudaErr));
        if (init_flag)printf("Start generating waveform\n");
        for (int ch = 0; ch < lNumCh; ch++){
            dynamic_total += dynamic_num[ch];
            static_total += static_num[ch];
        }
        double dyn_checker = (double)dynamic_loopcount*dynamic_total*lBytesPerChannelInNotifySize/1024/1024/1024;
        double stt_checker = (double) (dynamic_total+lNumCh*static_total)*lBytesPerChannelInNotifySize/1024/1024/1024;
        if (init_flag){
            printf("Buffersize for single dynamic tweezer: %f MiB\n",dyn_checker * 1024 / dynamic_total);
            printf("Buffersize for all dynamic tweezer: %f GiB\n",dyn_checker);
            printf("Buffersize for all static tweezer: %f GiB\n",stt_checker);
            printf("Total buffersize: %f GiB\n",dyn_checker+stt_checker);
        }
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
        };
        if (dynamic_total){
            if (dynamicBufferMalloc()){
                spcm_vClose (hCard);
                cudaDeviceReset();
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

        if (dynamic_total){
            if (!amp_flag){Pre_computer<<<static_length/lThreadsPerBlock,lThreadsPerBlock>>>(static_buffer_cuda,static_list_cuda,real_destination_freq_cuda,
                        dynamic_list_cuda,final_buffer_cuda,dynamic_saved_buffer_cuda,real_static_freq_cuda,phase_list_cuda,new_phase_list_cuda);
            }else{
                Pre_computer_amp<<<static_length/lThreadsPerBlock,lThreadsPerBlock>>>(static_buffer_cuda,static_list_cuda,real_destination_freq_cuda,
                        dynamic_list_cuda,final_buffer_cuda,dynamic_saved_buffer_cuda,real_static_freq_cuda,amp_list_cuda,phase_list_cuda,new_phase_list_cuda);
            }
        cudaDeviceSynchronize();
        }
        
        // ----- setup DMA transfer from GPU to Spectrum card -----
        // ----- fill the software buffer before we start the card -----
        
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
                    else{
                        spcm_dwGetErrorInfo_i32 (hCard, NULL, NULL, szErrorTextBuffer);
                        printf ("\n... Error: %u (%s)\n", dwError,szErrorTextBuffer);
                    }
                    spcm_vClose (hCard);
                    cudaDeviceReset();
                    return EXIT_FAILURE;
                }else{       
                    spcm_dwGetParam_i32 (hCard, SPC_DATA_AVAIL_USER_POS,  &lUserPos);
                    DynamicMux <<< static_length / lThreadsPerBlock, lThreadsPerBlock >>> (sPointerPosition,dynamic_saved_buffer_cuda,(int16*)((char*)DMABuffer + lUserPos));
                    cudaDeviceSynchronize();
                    dwError = spcm_dwSetParam_i32 (hCard, SPC_DATA_AVAIL_CARD_LEN,  lNotifySize);
                    if (dwError!=ERR_OK){
                        spcm_dwGetErrorInfo_i32 (hCard, NULL, NULL, szErrorTextBuffer);
                        printf("\n... Error in Setting CardAval1: %u (%s)\n", dwError,szErrorTextBuffer);
                        spcm_vClose (hCard);
                        cudaDeviceReset();
                        return EXIT_FAILURE;
                    }
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
                    {
                        spcm_dwGetErrorInfo_i32 (hCard, NULL, NULL, szErrorTextBuffer);
                        printf("\n... Error in Setting CardAval2: %u (%s)\n", dwError,szErrorTextBuffer);
                    }
                    spcm_vClose (hCard);
                    cudaDeviceReset();
                    return EXIT_FAILURE;
                    break;
                }
                else
                {        
                    spcm_dwGetParam_i32 (hCard, SPC_DATA_AVAIL_USER_POS,  &lUserPos);
                    if (loop_counter<lBufferSize/lNotifySize){
                        WaveformCopier <<< lNumCh * static_length / lThreadsPerBlock, lThreadsPerBlock >>> (final_buffer_cuda,(int16*)((char*)DMABuffer + lUserPos));
                        cudaDeviceSynchronize();
                        loop_counter++;
                    
                    dwError = spcm_dwSetParam_i32 (hCard, SPC_DATA_AVAIL_CARD_LEN,  lNotifySize);
                    if (dwError!=ERR_OK){
                        spcm_dwGetErrorInfo_i32 (hCard, NULL, NULL, szErrorTextBuffer);
                        printf("\n... Error in Setting CardAval: %u (%s)\n", dwError,szErrorTextBuffer);
                        break;
                    }
                    }else{
                    static_flag = true;
                    break;}
                }
            }
        }
        // send the stop command
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