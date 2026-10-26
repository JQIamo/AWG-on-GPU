/*
Waveform_synthesis_sequence.cu
Author: Juntian Tu
Date: 2025.09

This is the implementation for the sequential operation mode designed for specific experiments like parametric heating. Change the source file name in the Makefile to compile the program.
*/

#include "lib/cuda_functions_playback.h"
#include <atomic>
#   include <iostream>
#   include <thread>
#   include <signal.h>


using namespace std;
char        szErrorTextBuffer[ERRORTEXTLEN];
int32       lUserPos;
int loop_counter=0;
volatile std::atomic<bool> init_flag(true);
volatile std::atomic<bool> static_flag(true);
void* DMABuffer = NULL;
std::thread serverThread;
// std::thread staticThread;
void clean_exit(){
    stop_flag=true;
    printf ("\nExiting...\n");
    try{
        spcm_dwSetParam_i32 (hCard, SPC_M2CMD, M2CMD_CARD_STOP | M2CMD_DATA_STOPDMA);
    } catch (...) {}
    try{
        spcm_vClose (hCard);
    } catch (...) {}
    cudaDeviceReset();
    serverThread.join();
}

void handle_sigint(int sig) {
    if (sig == SIGINT) clean_exit();
}

void spcm_dwSetParam_i32_err(void* handle, long reg, long value)
{
    unsigned long error;
    error = spcm_dwSetParam_i32 (handle, reg, value);
    if (__builtin_expect(error,0)){
        spcm_dwGetErrorInfo_i32 (hCard, NULL, NULL, szErrorTextBuffer);
        printf ("Error when set %X with value %ld (%s)\n", reg, error, szErrorTextBuffer);
        throw std::runtime_error("Spcm Error\n");
    }
}

void spcm_dwSetParam_i64_err(void* handle, long reg, long long value)
{
    unsigned long error;
    error = spcm_dwSetParam_i64 (handle, reg, value);
    if (__builtin_expect(error,0)){
        spcm_dwGetErrorInfo_i32 (hCard, NULL, NULL, szErrorTextBuffer);
        printf ("Error when set %X with value %lld (%s)\n", reg, error, szErrorTextBuffer);
        throw std::runtime_error("Spcm Error\n");
    }
}

void reset_amp(int channel, int value){
    if (channel<0){
        for (int i = 0; i < lNumCh; ++i){
            int tempval = value;
            if (tempval>amplitude_limit[i]||tempval< 79){tempval=amplitude_limit[i];}
            spcm_dwSetParam_i32_err (hCard, SPC_AMP0       + i * (SPC_AMP1        - SPC_AMP0),      tempval);
        }
    }else{
        int tempval = value;
        if (tempval>amplitude_limit[channel]||tempval< 79){tempval=amplitude_limit[channel];}
        spcm_dwSetParam_i32_err (hCard, SPC_AMP0       + channel * (SPC_AMP1        - SPC_AMP0),      tempval);
    }
}



// settings for the FIFO mode buffer handling
const uint32       lNotifySize =  MEGA_B(2); // The size of data the card will execute each time before signaling to the GPU

// Parameter settings   
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

/*
****************************************************************************************************************************************************************************
main 
****************************************************************************************************************************************************************************
*/

int main ()
{
    signal(SIGINT, handle_sigint);
    bool        iskeypressed = false;    
    dynamic_buffersize = 2*round(ramp_time*llSamplerate);
    // ------------------------------------------------------------------------

    // ----- open Spectrum card -----
    hCard = spcm_hOpen ((char*)"/dev/spcm0");
    if (!hCard)
    {
        printf ("no card found...\r\n");
        cudaDeviceReset();
        return 0;
    }

    

    // ----- DMA BUFFER SETUP -----
    // ----- get buffer on GPU that will be used as source for RDMA transfer -----
    int lCUDADeviceIdx = 0;         // index of used CUDA device
    static_length = lNotifySize / lNumCh / sizeof(short);
    DMABuffer = pvGetRDMABuffer (lCUDADeviceIdx, lNotifySize); // Configure GPU Buffer
    if (DMABuffer == NULL)
    {
        printf("FAILED TO GET GPUBUFFER\r\n");
        spcm_vClose (hCard);
        cudaDeviceReset();
        return EXIT_FAILURE;
    }

    try{

        // ----- read type, function and sn and check for A/D card -----
        // ----- do a simple FIFO setup for 66xx -----
        spcm_dwSetParam_i32_err (hCard, SPC_CHENABLE,       (0x1 << lNumCh) - 1);   // enable all channels
        spcm_dwSetParam_i32_err (hCard, SPC_CARDMODE,       SPC_REP_STD_SEQUENCE);   // Test purpose
        spcm_dwSetParam_i32_err (hCard, SPC_SEQMODE_MAXSEGMENTS,         2); // Divide on-board mem in two parts 
        spcm_dwSetParam_i32_err (hCard, SPC_SEQMODE_STARTSTEP,           0); // Step#0 is the first step after card start
        spcm_dwSetParam_i32_err (hCard, SPC_TRIG_EXT0_LEVEL0, 1800);
        spcm_dwSetParam_i32_err (hCard, SPC_TRIG_EXT0_MODE, SPC_TM_POS);
        spcm_dwSetParam_i32_err (hCard, SPC_TRIG_ORMASK,    SPC_TMASK_EXT0);  // TEst purporse
        spcm_dwSetParam_i64_err (hCard, SPC_LOOPS,          0);                     // forever
        spcm_dwSetParam_i32_err (hCard, SPC_CLOCKMODE,      SPC_CM_INTPLL);         // clock mode internal PLL
        spcm_dwSetParam_i64_err (hCard, SPC_SAMPLERATE,     llSamplerate);
        spcm_dwSetParam_i32_err (hCard, SPC_TIMEOUT,        5*1000);             // Timeout if necessary
        reset_amp(-1, 2500);
        for (int lChIdx = 0; lChIdx < lNumCh; ++lChIdx)
        {
            spcm_dwSetParam_i32_err (hCard, SPC_FILTER0 + lChIdx * (SPC_FILTER1 - SPC_FILTER0), 0);
            spcm_dwSetParam_i32_err (hCard, SPC_ENABLEOUT0 + lChIdx * (SPC_ENABLEOUT1 - SPC_ENABLEOUT0), 1);
        }

        // Card Setup finished
        // ------------------------------------------------------------------------



        // ----- allocate memory for each channel on GPU host to use for copying the waveform data -----
        
        lBytesPerChannelInNotifySize = lNotifySize / lNumCh;
        serverThread = std::thread(instructionReceiver);
        while (1){
            printf("Waiting for instructions...\n");
            while (!server_flag && !stop_flag){volatile int nulvar = server_flag;}
            start_and_reset: 
            if (stop_flag){break;} 
            static_flag = false;
            static_endflag = false;
            cuda_cleanup();
            varReset();
            server_flag = false;
            continue_flag=false; //For testing
            printf("Sequential Mode\n");
            for (int ch = 0; ch < 4; ch++){
                static_total += static_num[ch];
            }
            double stt_checker = (double) sizeof(double)/sizeof(short)*(dynamic_total+static_total)*lBytesPerChannelInNotifySize/1024/1024/1024;
            printf("Buffersize for all static tweezer: %f GiB\n",stt_checker);
            if (stt_checker > 22.5){
                printf("Buffer required exceeds GPU memory\n");
                spcm_vClose (hCard);
                return EXIT_FAILURE;
            }
            shake_freq = freq_approx(shake_freq,static_length);
            tone_counter(0);
            for (int ch = 0; ch < 4; ch++){
                for (int i = 0; i < static_num[ch]; i++){
                    double freq_approxed = freq_approx(static_freq[ch][i],static_length);
                    if (freq_approxed < frequency_limits[ch/2*2]|| freq_approxed > frequency_limits[ch/2*2+1]){
                        cerr<<"Frequency out of range: "<< freq_approxed <<":" << ch <<":" << i << endl;
                        clean_exit();
                        return EXIT_FAILURE;
                    }else{
                        real_static_freq[tone_count[ch]+i] = freq_approxed;
                        printf("Static freq: %f\n",real_static_freq[tone_count[ch]+i]);
                    }
                }        

            }
            cudaDeviceSynchronize();

            if (staticBufferInit()){
                clean_exit();
                return EXIT_FAILURE;
            };

            
            for (int ch = 0; ch < 4; ch++){
                for (int i = 0; i < static_num[ch]; i++){
                    static_freq[ch][i] = new_static_freq[ch][i];
                }
            }


            if (!amp_flag)
                StaticWaveGeneration<<<(static_length/lThreadsPerBlock),lThreadsPerBlock>>>(real_static_freq_cuda,static_buffer_cuda,summed_buffer_cuda,phase_list_cuda);
            else 
                StaticWaveGeneration_amp<<<(static_length/lThreadsPerBlock),lThreadsPerBlock>>>(real_static_freq_cuda,amp_list_cuda,static_buffer_cuda,summed_buffer_cuda,phase_list_cuda);
        
            cudaDeviceSynchronize();
            
            // ----- setup DMA transfer from GPU to Spectrum card -----
            // ----- fill the software buffer before we start the card -----
            spcm_dwSetParam_i32_err (hCard, SPC_SEQMODE_WRITESEGMENT,   0); // set current configuration switch to segment 0 
            spcm_dwSetParam_i32_err (hCard, SPC_SEQMODE_SEGMENTSIZE, lNotifySize / sizeof(short)/lNumCh); // define size of current segment 0
            StaticMux <<< static_length / lThreadsPerBlock, lThreadsPerBlock >>> (summed_buffer_cuda,(int16*)((char*)DMABuffer));
            cudaDeviceSynchronize();
            
            // ----- start transfer from GPU into card and wait until it has finished -----
            if (init_flag) spcm_dwDefTransfer_i64 (hCard, SPCM_BUF_DATA, SPCM_DIR_GPUTOCARD, 0, DMABuffer, 0, lNotifySize);
            spcm_dwSetParam_i32_err (hCard, SPC_M2CMD, M2CMD_DATA_STARTDMA | M2CMD_DATA_WAITDMA);   

            printf("Shake Mode: shake_amp: %f, shake_freq: %f, unit_freq: %f\n" ,shake_amp,shake_freq,(double)llSamplerate/static_length);
            if (!amp_flag)
                ShakeWaveGeneration<<<(static_length/lThreadsPerBlock),lThreadsPerBlock>>>(real_static_freq_cuda,shake_amp,shake_freq,static_buffer_cuda,summed_buffer_cuda,phase_list_cuda);
            else 
                ShakeWaveGeneration_amp<<<(static_length/lThreadsPerBlock),lThreadsPerBlock>>>(real_static_freq_cuda,amp_list_cuda,shake_amp,shake_freq,static_buffer_cuda,summed_buffer_cuda,phase_list_cuda);
        
            cudaDeviceSynchronize();
            
            // ----- setup DMA transfer from GPU to Spectrum card -----
            // ----- fill the software buffer before we start the card -----
            spcm_dwSetParam_i32_err (hCard, SPC_SEQMODE_WRITESEGMENT,   1); // set current configuration switch to segment 0 
            spcm_dwSetParam_i32_err (hCard, SPC_SEQMODE_SEGMENTSIZE, lNotifySize / sizeof(short)/lNumCh); // define size of current segment 0
            StaticMux <<< static_length / lThreadsPerBlock, lThreadsPerBlock >>> (summed_buffer_cuda,(int16*)((char*)DMABuffer));
            cudaDeviceSynchronize();
            
            // ----- start transfer from GPU into card and wait until it has finished -----
            if (init_flag) spcm_dwDefTransfer_i64 (hCard, SPCM_BUF_DATA, SPCM_DIR_GPUTOCARD, 0, DMABuffer, 0, lNotifySize);
            spcm_dwSetParam_i32_err (hCard, SPC_M2CMD, M2CMD_DATA_STARTDMA | M2CMD_DATA_WAITDMA);   


            // Setting up the sequence memory (Only two steps used here as an example) 
            int32 lStep =       0;                    // current step is Step#0 
            int64 llSegment =   0;                    // associated with data memory segment 0 
            int64 llLoop =     1;                    // Pattern will be repeated 10 times 
            int64 llNext =      1;                    // Next step is Step#1 
            int64 llCondition = SPCSEQ_ENDLOOPONTRIG; // Unconditionally leave current step 
            // combine all the parameters to one int64 bit value 
            int64 llValue =   (llCondition << 32) | (llLoop << 32) | (llNext << 16) | (llSegment); 
            spcm_dwSetParam_i64_err (hCard, SPC_SEQMODE_STEPMEM0 + lStep, llValue); 
            lStep =       1;                    // current step is Step#1 
            llSegment =   1;                    // associated with data memory segment 1 
            llNext =      0;                    // Next step is Step#0 
            llValue =   (llCondition << 32) | (llLoop << 32) | (llNext << 16) | (llSegment); 
            spcm_dwSetParam_i64_err (hCard, SPC_SEQMODE_STEPMEM0 + lStep, llValue);

            // std::thread terminatorThread(ifkeypress,&iskeypressed);
            // ----- start everything -----
            if (init_flag){
                spcm_dwSetParam_i32_err (hCard, SPC_M2CMD, M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER);
            }

            static_flag = true;
            static_endflag = true;
            int sequence_index;
            if (init_flag) spcm_dwSetParam_i32_err (hCard, SPC_M2CMD, M2CMD_CARD_FORCETRIGGER);
            else{
                spcm_dwGetParam_i32(hCard,SPC_SEQMODE_STATUS,&sequence_index);
                if (sequence_index == 1) spcm_dwSetParam_i32_err (hCard, SPC_M2CMD, M2CMD_CARD_FORCETRIGGER);
                init_flag = false;
            }
        }
        clean_exit();
        return EXIT_SUCCESS;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        clean_exit();  // run your cleanup code
    }
    catch (...) {
        std::cerr << "Unknown exception caught!" << std::endl;
        clean_exit();  // fallback cleanup for everything else
    }
}