/*
Waveform_synthesis_playback.cu
Author: Juntian Tu
Date: 2025.01

This is the implementation for the playback pathway. Change the source file name in the Makefile to compile the program.
*/

#include "lib/cuda_functions_playback.h"
#include <atomic>
#   include <iostream>
#   include <thread>
#   include <signal.h>
#ifdef NO_OUTPUT_TESTING
#include "lib/spectrum_analyzer.h"
float spectrum_analyzer_buffer[4*fft_length];
#endif

using namespace std;
char        szErrorTextBuffer[ERRORTEXTLEN];
volatile uint32      dwError;
int32       lUserPos;
int loop_counter=0;
volatile std::atomic<bool> init_flag(true);
volatile std::atomic<bool> static_flag(true);
volatile std::atomic<bool> clean_flag(false);
std::string spectra_file;

void* DMABuffer = NULL;
std::thread serverThread;
std::thread staticThread;
void clean_exit(){
    if (clean_flag) return;
    clean_flag = true;
    stop_flag=true;
    printf ("\nExiting...\n");
    #ifndef NO_OUTPUT_TESTING
    try{
        spcm_dwSetParam_i32 (hCard, SPC_M2CMD, M2CMD_CARD_STOP | M2CMD_DATA_STOPDMA);
    } catch (...) {}
    try{
        spcm_vClose (hCard);
    } catch (...) {}
    #endif
    cudaDeviceReset();
    serverThread.join();
}


void handle_sigint(int sig) {
    if (sig == SIGQUIT) {
        clean_exit();
    }
}


void spcm_dwSetParam_i32_err(void* handle, long reg, long value)
{
    unsigned long error;
    error = spcm_dwSetParam_i32 (handle, reg, value);
    if (__builtin_expect(error,0)){
        spcm_dwGetErrorInfo_i32 (hCard, NULL, NULL, szErrorTextBuffer);
        printf ("ERROR when set %X with value %lu (%s)\n", static_cast<unsigned int>(reg), error, szErrorTextBuffer);
        throw std::runtime_error("Spcm Error\n");
    }
}

void spcm_dwSetParam_i64_err(void* handle, long reg, long long value)
{
    unsigned long error;
    error = spcm_dwSetParam_i64 (handle, reg, value);
    if (__builtin_expect(error,0)){
        spcm_dwGetErrorInfo_i32 (hCard, NULL, NULL, szErrorTextBuffer);
        printf ("ERROR when set %X with value %lu (%s)\n", static_cast<unsigned int>(reg), error, szErrorTextBuffer);
        throw std::runtime_error("Spcm Error\n");
    }
}


void reset_amp(int channel, int value){
    if (channel<0){
        for (int i = 0; i < lNumCh; ++i){
            int tempval = value;
            if (tempval>amplitude_limit[i]){tempval=amplitude_limit[i];}
            // std::cout << tempval << std::endl;
            spcm_dwSetParam_i32_err (hCard, SPC_AMP0       + i * (SPC_AMP1        - SPC_AMP0),      tempval);
        }
    }else{
        int tempval = value;
        if (tempval>amplitude_limit[channel]){tempval=amplitude_limit[channel];}
        spcm_dwSetParam_i32_err (hCard, SPC_AMP0       + channel * (SPC_AMP1        - SPC_AMP0),      tempval);
    }
}




// settings for the FIFO mode buffer handling
// uint64       lNotifySize =  MEGA_B(2); // The size of data the card will execute each time before signaling to the GPU
uint64       lNotifySize =  KILO_B(512); // The size of data the card will execute each time before signaling to the GPU
uint32       lBufferSize =  MEGA_B(2);
uint64       HBufferSize =  MEGA_B(2); // The actual buffer used on the AWG; must be a power of 2 and should be no more than 4 GB (lower size reduces delay)

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

void static_looper(){
    while (!stop_flag){
        while (static_flag && !stop_flag){
            static_pulseflag = false;
            while (!static_endflag.load() && !stop_flag){
                #ifndef NO_OUTPUT_TESTING
                if ((dwError = spcm_dwSetParam_i32 (hCard, SPC_M2CMD, M2CMD_DATA_WAITDMA)) != ERR_OK)
                {
                    if (dwError == ERR_TIMEOUT)
                        printf ("\nTIMEOUT in static looper\n");
                    else
                        spcm_dwGetErrorInfo_i32 (hCard, NULL, NULL, szErrorTextBuffer);
                        printf ("\nERROR in static looper0: %u (%s)\n", dwError,szErrorTextBuffer);
                    stop_flag = true;
                    break;
                }
                else
                {             
                    dwError = spcm_dwSetParam_i32 (hCard, SPC_DATA_AVAIL_CARD_LEN,  lNotifySize);
                    if (dwError!=ERR_OK){
                        spcm_dwGetErrorInfo_i32 (hCard, NULL, NULL, szErrorTextBuffer);
                        printf("\nERROR in static looper1: %u (%s)\n", dwError,szErrorTextBuffer);
                        stop_flag = true;
                        break;
                    }
                }
                #endif
            }
            static_endflag=false;
        }
        static_pulseflag = true;
        // printf("Static looping stopped\n");
    }
    std::cout << "Static looper terminated." << std::endl;
}

/*
****************************************************************************************************************************************************************************
main 
****************************************************************************************************************************************************************************
*/

int main ()
{
    signal(SIGQUIT, handle_sigint);
    std::cout<< "AWG Program Initializing..." << std::endl;
    std::cout<< "Press Ctrl + \\ to terminate" << std::endl;
    bool        iskeypressed = false;
    // Modified to force the dynamic_bufferlength to be an even number to reduce the logic complexity in CUDA kernels
    dynamic_buffersize = 4*static_cast<unsigned int>(round(0.5*ramp_time*static_cast<double>(llSamplerate)));
    // ------------------------------------------------------------------------

    // ----- open Spectrum card -----
    #ifndef NO_OUTPUT_TESTING
    hCard = spcm_hOpen ((char*)"/dev/spcm0");
    if (!hCard)
    {
        printf ("no card found...\r\n");
        return 0;
    }
    #else
    const std::time_t now = std::time(nullptr);
    const std::tm* local = std::localtime(&now);
    if (local == nullptr) {
        throw std::runtime_error("Cannot obtain local time");
    }
    char timestamp[32];
    if (std::strftime(timestamp, sizeof(timestamp),
                      "%Y-%m-%d_%H-%M-%S", local) == 0) {
        throw std::runtime_error("Cannot format timestamp");
    }
    spectra_file = std::string("test/spectrum_analyzer_output_") + timestamp + ".bin";
    std::ofstream out(spectra_file, std::ios::binary | std::ios::trunc);
    #endif

    try{
        #ifndef NO_OUTPUT_TESTING
        // ----- read type, function and sn and check for A/D card -----
        // ----- do a simple FIFO setup for 66xx -----
        spcm_dwSetParam_i32_err (hCard, SPC_CHENABLE,       (0x1 << lNumCh) - 1);   // enable all channels
        // spcm_dwSetParam_i32_err (hCard, SPC_CARDMODE,       SPC_REP_FIFO_GATE);     // gated FIFO mode
        spcm_dwSetParam_i32_err (hCard, SPC_CARDMODE,       SPC_REP_FIFO_SINGLE);   // Test purpose
        spcm_dwSetParam_i32_err (hCard, SPC_TRIG_ORMASK,    SPC_TMASK_SOFTWARE);  // TEst purporse
        spcm_dwSetParam_i64_err (hCard, SPC_LOOPS,          0);                     // forever
        spcm_dwSetParam_i32_err (hCard, SPC_CLOCKMODE,      SPC_CM_INTPLL);         // clock mode internal PLL
        spcm_dwSetParam_i64_err (hCard, SPC_SAMPLERATE,     llSamplerate);
        spcm_dwSetParam_i32_err (hCard, SPC_TIMEOUT,        5*1000);             // Timeout if necessary
        spcm_dwSetParam_i32_err (hCard, SPCM_X0_MODE, SPCM_XMODE_ASYNCOUT);
        spcm_dwSetParam_i32_err (hCard, SPCM_X1_MODE, SPCM_XMODE_ASYNCIN);
        reset_amp(-1, 2500);
        for (int lChIdx = 0; lChIdx < lNumCh; ++lChIdx)
        {
            spcm_dwSetParam_i32_err (hCard, SPC_FILTER0 + lChIdx * (SPC_FILTER1 - SPC_FILTER0), 0);
            spcm_dwSetParam_i32_err (hCard, SPC_ENABLEOUT0 + lChIdx * (SPC_ENABLEOUT1 - SPC_ENABLEOUT0), 1);
        }

        spcm_dwSetParam_i64_err (hCard, SPC_DATA_OUTBUFSIZE,  HBufferSize);         // Set actual buffer size on the AWG 
        spcm_dwSetParam_i32_err (hCard, SPCM_XX_ASYNCIO, 1);
        spcm_dwSetParam_i32_err (hCard, SPC_M2CMD, M2CMD_CARD_WRITESETUP);          // Write in the configs
        #endif

        // Card Setup finished
        // ------------------------------------------------------------------------


        // ----- DMA BUFFER SETUP -----
        // ----- get buffer on GPU that will be used as source for RDMA transfer -----
        int lCUDADeviceIdx = 0;         // index of used CUDA device
        static_length = static_cast<int>(lNotifySize / lNumCh / sizeof(short));
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
        serverThread = std::thread(instructionReceiver);
        while (1){
            // spcm_dwSetParam_i32_err (hCard, SPCM_XX_ASYNCIO, 1);
            while (!server_flag && !stop_flag){volatile int nulvar = server_flag;}
            start_and_reset: 
            if (stop_flag){break;} 
            static_flag = false;
            static_endflag = false;
            server_flag = false;
            continue_flag=false; //For testing
            switch (mode_flag) {
                case MODE_INIT:
                {
                    cuda_cleanup();
                    varReset();
                    if (init_flag)printf("Start generating waveform\n");
                    for (int ch = 0; ch < 4; ch++){
                        dynamic_total += dynamic_num[ch];
                        static_total += static_num[ch];
                    }
                    double dyn_checker = (double)dynamic_loopcount*dynamic_total*lBytesPerChannelInNotifySize/1024/1024/1024;
                    double stt_checker = (double) sizeof(double)/sizeof(short)*(dynamic_total+static_total)*lBytesPerChannelInNotifySize/1024/1024/1024;
                    if (init_flag){
                        printf("Buffersize for single dynamic tweezer: %f MiB\n",dyn_checker * 1024 / dynamic_total);
                        printf("Buffersize for all dynamic tweezer: %f GiB\n",dyn_checker);
                        printf("Buffersize for all static tweezer: %f GiB\n",stt_checker);
                        printf("Total buffersize: %f GiB\n",dyn_checker+stt_checker);
                    }
                    if (dyn_checker+stt_checker > 22.5){
                        printf("Buffer required exceeds GPU memory\n");
                        spcm_vClose (hCard);
                        return EXIT_FAILURE;
                    }
                    // if (update_flag){
                    //     cudaMalloc((void**)&update_index_map_cuda,static_total*sizeof(int));
                    //     cudaMemcpy(update_index_map_cuda,update_index_map,static_total*sizeof(int),cudaMemcpyHostToDevice);
                    //     phase_reorder_update<<<(int)ceil((float)static_total/lThreadsPerBlock),lThreadsPerBlock>>>(update_index_map_cuda,new_phase_list_cuda,phase_list_cuda,static_total);
                    // }
                    cudaDeviceSynchronize();
                    tone_counter(dynamic_total);
                    for (int ch = 0; ch < 4; ch++){
                        for (int i = 0; i < static_num[ch]; i++){
                            double freq_approxed = freq_approx(static_freq[ch][i],static_length);
                            if (freq_approxed < frequency_limits[ch/2*2]|| freq_approxed > frequency_limits[ch/2*2+1]){
                                cerr<<"Frequency out of range: "<< freq_approxed <<":" << ch <<":" << i << endl;
                                clean_exit();
                                return EXIT_FAILURE;
                            }else{
                                real_static_freq[tone_count[ch]+i] = freq_approxed;
                            }
                        }
                        for (int i = 0; i < dynamic_num[ch]; i++){
                            double freq_approxed = freq_approx(destination_freq[ch][i],static_length);
                            if (freq_approxed < frequency_limits[ch/2*2]|| freq_approxed > frequency_limits[ch/2*2+1]){
                                cerr<<"Frequency out of range:"<< freq_approxed << endl;
                                clean_exit();
                                return EXIT_FAILURE;
                            }else{
                                real_destination_freq[dynamic_tone_count[ch]+i] = freq_approxed;
                            }
                        }
                    }
                    cudaDeviceSynchronize();
                    // if (if_mapped){
                    //     StaticAmpMapper( real_static_freq, amp_freq_map,  amp_map_static, total_divider);
                    // }
                    if (staticBufferInit()){
                        clean_exit();
                        return EXIT_FAILURE;
                    };
                    if (dynamic_total){
                        if (dynamicBufferInit()){
                            clean_exit();
                            return EXIT_FAILURE;
                        }
                    }


                    for (int ch = 0; ch < 4; ch++){
                        for (int i = 0; i < static_num[ch]; i++){
                            static_freq[ch][i] = new_static_freq[ch][i];
                        }
                    }

                    if (!amp_flag) StaticWaveGeneration<<<(static_length/lThreadsPerBlock),lThreadsPerBlock>>>(real_static_freq_cuda,static_buffer_cuda,summed_buffer_cuda,phase_list_cuda);
                    else StaticWaveGeneration_amp<<<(static_length/lThreadsPerBlock),lThreadsPerBlock>>>(real_static_freq_cuda,amp_list_cuda,static_buffer_cuda,summed_buffer_cuda,phase_list_cuda);
                    cudaDeviceSynchronize();
                    cudaMemcpy(new_phase_list_cuda,phase_list_cuda,static_total*sizeof(float),cudaMemcpyDeviceToDevice);

                    if (dynamic_total){
                        if (!amp_flag){
                            Pre_computer<<<static_length/lThreadsPerBlock/2,lThreadsPerBlock>>>(static_buffer_cuda,static_list_cuda,real_destination_freq_cuda,
                                    dynamic_list_cuda,final_buffer_cuda,dynamic_saved_buffer_cuda,real_static_freq_cuda,phase_list_cuda,new_phase_list_cuda);
                        }else{
                            Pre_computer_amp<<<static_length/lThreadsPerBlock/2,lThreadsPerBlock>>>(static_buffer_cuda,static_list_cuda,real_destination_freq_cuda,
                                    dynamic_list_cuda,final_buffer_cuda,dynamic_saved_buffer_cuda,real_static_freq_cuda,phase_list_cuda,new_phase_list_cuda,amp_list_cuda);
                        }
                        cudaDeviceSynchronize();
                    }

                    // ----- setup DMA transfer from GPU to Spectrum card -----
                    // ----- fill the software buffer before we start the card -----

                    #ifndef NO_OUTPUT_TESTING
                    if (init_flag) spcm_dwDefTransfer_i64 (hCard, SPCM_BUF_DATA, SPCM_DIR_GPUTOCARD, lNotifySize, DMABuffer, 0, lBufferSize);
                    #endif
                    for (int32 lPosInBuf = 0; lPosInBuf < lBufferSize; lPosInBuf += lNotifySize)
                    {
                        StaticMux <<< static_length / lThreadsPerBlock, lThreadsPerBlock >>> (summed_buffer_cuda,(int16*)((char*)DMABuffer + lPosInBuf));
                    }
                    cudaDeviceSynchronize();
                    std::cout<< "Static MUX finished" << std::endl;
                    #ifdef NO_OUTPUT_TESTING
                    cudaMemcpy((void*)spectrum_analyzer_buffer,DMABuffer, 4*fft_length*sizeof(float), cudaMemcpyDeviceToHost);
                    short temp_buf[4*fft_length];
                    cudaMemcpy((void*)temp_buf,(void*)((short*)DMABuffer), 4*fft_length*sizeof(short), cudaMemcpyDeviceToHost);
                    for (int i = 0; i < 4*fft_length; i++) spectrum_analyzer_buffer[i] = static_cast<float>(temp_buf[i]);
                    // std::copy((char*)DMABuffer+4*sample_start, (char*)DMABuffer + 4*(sample_start + fft_length), spectrum_analyzer_buffer);
                    write_magnitude_file(spectra_file, fft_4ch_muxed_real(spectrum_analyzer_buffer),
                    fftfreq(fft_length, llSamplerate), 0);
                    std::cout<< "Spectra saved" << std::endl;
                    #endif


                    if (eCudaErr=cudaPeekAtLastError()) printf("Static Mux error: %s\n",cudaGetErrorString(eCudaErr));
                    if (init_flag){
                        printf("\r\nCalculated: Init\r\n");
                        // mark data as valid
                        #ifndef NO_OUTPUT_TESTING
                        spcm_dwSetParam_i32_err (hCard, SPC_DATA_AVAIL_CARD_LEN,  lBufferSize);

                        // ----- start transfer from GPU into card and wait until it has finished -----
                        spcm_dwSetParam_i32_err (hCard, SPC_M2CMD, M2CMD_DATA_STARTDMA | M2CMD_DATA_WAITDMA);

                        // std::thread terminatorThread(ifkeypress,&iskeypressed);
                        // ----- start everything -----
                        spcm_dwSetParam_i32_err (hCard, SPC_M2CMD, M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER);
                        #endif
                        staticThread =std::thread(static_looper);
                        init_flag = false;
                    }
                    static_flag = true;
                    static_endflag = true;
                    if (eCudaErr=cudaPeekAtLastError()) printf("Init error: %s\n",cudaGetErrorString(eCudaErr));
                    printf("Looping ...\n");
                    if (dynamic_total){
                        while (!continue_flag && !server_flag && !stop_flag){}
                        if (server_flag){goto start_and_reset;}
                        if (stop_flag) break;
                        static_pulseflag=false;
                        continue_flag = false; // For testing
                        static_flag = false;

                        static_endflag = true;

                        while (!static_pulseflag){}
                        static_pulseflag=false;
                        sPointerPosition=0;
                        #ifndef NO_OUTPUT_TESTING
                        for (int cnt = 0; cnt < dynamic_loopcount && !iskeypressed;cnt++){
                            spcm_dwSetParam_i32_err (hCard, SPC_M2CMD, M2CMD_DATA_WAITDMA);
                            spcm_dwGetParam_i32 (hCard, SPC_DATA_AVAIL_USER_POS,  &lUserPos);
                            DynamicMux <<< static_length / lThreadsPerBlock, lThreadsPerBlock >>>
                                (sPointerPosition,dynamic_saved_buffer_cuda,(int16*)((char*)DMABuffer + lUserPos));
                            cudaDeviceSynchronize();
                            spcm_dwSetParam_i32_err (hCard, SPC_DATA_AVAIL_CARD_LEN,  lNotifySize);
                            sPointerPosition += static_length;

                        }
                        #else
                        double scan_timer=0.;
                        for (int cnt = 0; cnt < dynamic_loopcount && !iskeypressed;cnt++){
                            DynamicMux <<< static_length / lThreadsPerBlock, lThreadsPerBlock >>>
                                (sPointerPosition,dynamic_saved_buffer_cuda,(int16*)((char*)DMABuffer));
                            cudaDeviceSynchronize();
                            const double timer_f =  static_cast<double>(sPointerPosition+1)/static_cast<double>(llSamplerate);
                            while (scan_timer < timer_f){
                                const unsigned long long sample_start = static_cast<unsigned long long>(scan_timer * llSamplerate)-sPointerPosition;
                                short temp_buf[4*fft_length];
                                cudaMemcpy((void*)temp_buf,(void*)((short*)DMABuffer+sample_start), 4*fft_length*sizeof(short), cudaMemcpyDeviceToHost);
                                for (int i = 0; i < 4*fft_length; i++) spectrum_analyzer_buffer[i] = static_cast<float>(temp_buf[i]);
                                // std::copy((char*)DMABuffer+4*sample_start, (char*)DMABuffer + 4*(sample_start + fft_length), spectrum_analyzer_buffer);
                                write_magnitude_file(spectra_file, fft_4ch_muxed_real(spectrum_analyzer_buffer),
                                    fftfreq(fft_length, llSamplerate), scan_timer);
                                scan_timer += fft_sample_interval;
                            }
                            sPointerPosition += static_length;
                        }
                        #endif

                        printf("DESTINATION\n");
                        while (!static_flag) // Terminated when key pressed; not working if RMA keeps waiting
                        {
                            #ifndef NO_OUTPUT_TESTING
                            spcm_dwSetParam_i32_err (hCard, SPC_M2CMD, M2CMD_DATA_WAITDMA);
                            spcm_dwGetParam_i32 (hCard, SPC_DATA_AVAIL_USER_POS,  &lUserPos);
                            if (loop_counter<lBufferSize/lNotifySize){
                                WaveformCopier <<< lNumCh * static_length / lThreadsPerBlock, lThreadsPerBlock >>> (final_buffer_cuda,(int16*)((char*)DMABuffer + lUserPos));
                                cudaDeviceSynchronize();
                                loop_counter++;
                                spcm_dwSetParam_i32_err (hCard, SPC_DATA_AVAIL_CARD_LEN,  lNotifySize);
                            }else{
                                static_flag = true;
                                break;
                            }
                            #else
                            if (loop_counter<lBufferSize/lNotifySize){
                                WaveformCopier <<< lNumCh * static_length / lThreadsPerBlock, lThreadsPerBlock >>> (final_buffer_cuda,(int16*)((char*)DMABuffer + lUserPos));
                                cudaDeviceSynchronize();
                                loop_counter++;
                            }else{
                                static_flag = true;
                                break;
                            }
                            #endif
                        }
                    }
                    break;
                }

                case MODE_REARRANGE:
                {
                    mode_flag = MODE_INIT;
                    // spcm_dwSetParam_i32_err (hCard, SPCM_XX_ASYNCIO, 0);
                    if (error_flag){
                        error_flag = false;
                    }else{
                        dynamic_total = dynamic_num[single_dynamic_channel];
                        for (int i = 0; i < dynamic_num[single_dynamic_channel]; i++){
                            double freq_approxed = freq_approx(destination_freq[single_dynamic_channel][i],static_length);
                            if (freq_approxed < frequency_limits[single_dynamic_channel/2*2]|| freq_approxed > frequency_limits[single_dynamic_channel/2*2+1]){
                                cerr<<"Frequency out of range:"<< freq_approxed << endl;
                                clean_exit();
                                return EXIT_FAILURE;
                            }else{
                                real_destination_freq[i] = freq_approxed;
                            }
                        }
                        if (eCudaErr=cudaPeekAtLastError()) printf("Cuda Buffer Clean Failed: %s\n",cudaGetErrorString(eCudaErr));
                        if (rearrangeBufferInit()){
                            clean_exit();
                            return EXIT_FAILURE;
                        };
                        cudaDeviceSynchronize();
                        // SingleStaticRecombiner<<<static_length/lThreadsPerBlock,lThreadsPerBlock>>>(single_dynamic_channel, single_static_map_cuda,static_buffer_cuda,summed_buffer_cuda);
                        // cudaDeviceSynchronize();
                        if (!amp_flag) {
                            Pre_computer_rearrange<<<static_length/lThreadsPerBlock/2,lThreadsPerBlock>>>(single_dynamic_channel, static_num[single_dynamic_channel], dynamic_num[single_dynamic_channel], single_static_map_cuda, single_dynamic_map_cuda,summed_buffer_cuda,
                                            static_buffer_cuda, real_destination_freq_cuda, final_buffer_cuda, dynamic_saved_buffer_cuda, real_static_freq_cuda,phase_list_cuda,new_phase_list_cuda,rearrange_freq_list_cuda);
                        }else {
                            Pre_computer_rearrange_amp<<<static_length/lThreadsPerBlock/2,lThreadsPerBlock>>>(single_dynamic_channel, static_num[single_dynamic_channel], dynamic_num[single_dynamic_channel], single_static_map_cuda, single_dynamic_map_cuda,summed_buffer_cuda,
                                            static_buffer_cuda, real_destination_freq_cuda, final_buffer_cuda, dynamic_saved_buffer_cuda, real_static_freq_cuda,phase_list_cuda,new_phase_list_cuda,rearrange_freq_list_cuda,amp_list_cuda);
                        }
                        cudaDeviceSynchronize();
                        if (eCudaErr=cudaPeekAtLastError()) printf("Cuda Rearrange Failed: %s\n",cudaGetErrorString(eCudaErr));
                        cudaMemcpyAsync(real_static_freq_cuda,rearrange_freq_list_cuda,static_total*sizeof(float),cudaMemcpyDeviceToDevice);
                        static_pulseflag=false;
                        static_flag = false;
                        static_endflag = true;
                        while (!static_pulseflag){}
                        static_pulseflag=false;
                        sPointerPosition=0;
                        #ifndef NO_OUTPUT_TESTING
                        for (int cnt = 0; cnt < dynamic_loopcount && !iskeypressed;cnt++){
                            spcm_dwSetParam_i32_err (hCard, SPC_M2CMD, M2CMD_DATA_WAITDMA);
                            spcm_dwGetParam_i32 (hCard, SPC_DATA_AVAIL_USER_POS,  &lUserPos);
                            DynamicMux <<< static_length / lThreadsPerBlock, lThreadsPerBlock >>>
                                (sPointerPosition,dynamic_saved_buffer_cuda,(int16*)((char*)DMABuffer + lUserPos));
                            cudaDeviceSynchronize();
                            spcm_dwSetParam_i32_err (hCard, SPC_DATA_AVAIL_CARD_LEN,  lNotifySize);
                            DynamicMux <<< static_length / lThreadsPerBlock, lThreadsPerBlock >>>
                                (sPointerPosition,dynamic_saved_buffer_cuda,(int16*)((char*)DMABuffer + lUserPos));
                            cudaDeviceSynchronize();
                            sPointerPosition += static_length;
                        }
                        #else
                        double scan_timer=0.;
                        for (int cnt = 0; cnt < dynamic_loopcount && !iskeypressed;cnt++){
                            DynamicMux <<< static_length / lThreadsPerBlock, lThreadsPerBlock >>>
                                (sPointerPosition,dynamic_saved_buffer_cuda,(int16*)((char*)DMABuffer));
                            cudaDeviceSynchronize();
                            const double timer_f =  static_cast<double>(sPointerPosition+1)/static_cast<double>(llSamplerate);
                            while (scan_timer < timer_f){
                                const unsigned long long sample_start = static_cast<unsigned long long>(scan_timer * llSamplerate)-sPointerPosition;
                                cudaMemcpy((void*)spectrum_analyzer_buffer,DMABuffer, 4*fft_length*sizeof(float), cudaMemcpyDeviceToHost);
                                // std::copy((char*)DMABuffer+4*sample_start, (char*)DMABuffer + 4*(sample_start + fft_length), spectrum_analyzer_buffer);
                                write_magnitude_file(spectra_file, fft_4ch_muxed_real(spectrum_analyzer_buffer),
                                    fftfreq(fft_length, llSamplerate), scan_timer);
                                scan_timer += fft_sample_interval;
                            }
                            sPointerPosition += static_length;
                        }
                        #endif
                        printf("DESTINATION\n");
                        while (!static_flag) // Terminated when key pressed; not working if RMA keeps waiting
                        {
                            #ifndef NO_OUTPUT_TESTING
                            spcm_dwSetParam_i32_err (hCard, SPC_M2CMD, M2CMD_DATA_WAITDMA);
                            spcm_dwGetParam_i32 (hCard, SPC_DATA_AVAIL_USER_POS,  &lUserPos);
                            if (loop_counter<lBufferSize/lNotifySize){
                                WaveformCopier <<< lNumCh * static_length / lThreadsPerBlock, lThreadsPerBlock >>>
                                    (final_buffer_cuda,(int16*)((char*)DMABuffer + lUserPos));
                                cudaDeviceSynchronize();
                                loop_counter++;
                                spcm_dwSetParam_i32_err (hCard, SPC_DATA_AVAIL_CARD_LEN,  lNotifySize);
                            }else{
                                static_flag = true;
                                break;
                            }
                            #else
                            if (loop_counter<lBufferSize/lNotifySize){
                                WaveformCopier <<< lNumCh * static_length / lThreadsPerBlock, lThreadsPerBlock >>>
                                    (final_buffer_cuda,(int16*)((char*)DMABuffer + lUserPos));
                                cudaDeviceSynchronize();
                                loop_counter++;
                            }else{
                                static_flag = true;
                                break;
                            }
                            #endif
                        }
                    }
                    #ifndef NO_OUTPUT_TESTING
                    spcm_dwSetParam_i32_err (hCard, SPCM_XX_ASYNCIO, 0);
                    struct timespec req;
                    struct timespec rem;
                    req.tv_sec = 0;
                    req.tv_nsec = 100;
                    nanosleep(&req, &rem);
                    spcm_dwSetParam_i32_err (hCard, SPCM_XX_ASYNCIO, 1);
                    #endif
                    break;
                }


                default:
                    throw std::runtime_error("Unknown mode_flag");
            }
            // send the stop command
        }
        clean_exit();
        if (stop_flag){
            std::cout << "TERMINATED due to Stop Flag." << std::endl;
        }

        return EXIT_SUCCESS;
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        clean_exit();  // run your cleanup code
    }
    catch (...) {
        std::cerr << "UNKNOWN EXCEPTION CAUGHT!" << std::endl;
        clean_exit();  // fallback cleanup for everything else
    }
}