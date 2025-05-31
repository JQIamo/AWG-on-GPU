/*
Waveform_synthesis_streaming.cu
Author: Juntian Tu
Date: 2025.01

This is the implementation for the streaming pathway. Change the source file name in the Makefile to compile the program.
*/
#include "lib/cuda_functions_streaming.h"
//  Timer includes
#   include <iostream>
#   include <signal.h>
#   include <thread>

using namespace std;
volatile std::atomic<bool> init_flag(true);
volatile std::atomic<bool> static_flag(true);
char        szErrorTextBuffer[ERRORTEXTLEN];
uint32      dwError;
int32       lUserPos;
int         loop_counter = 0;
bool        job_distirbuter_flag = false;
bool first_loop_flag = true;

std::thread serverThread;
std::thread staticThread;

void clean_exit(){
    stop_flag=true;
    printf ("\nExiting...\n");
    spcm_vClose (hCard);
    cudaDeviceReset();
}

void handle_sigint(int sig) {
    if (sig == SIGINT) clean_exit();
}


void* DMABuffer = NULL;
 // Depending on the GPU used
void reset_amp(int channel, int value){
    if (channel<0){
        for (int i = 0; i < lNumCh; ++i){
            int tempval = value;
            if (tempval>amplitude_limit[i]){tempval=amplitude_limit[i];}
            spcm_dwSetParam_i32 (hCard, SPC_AMP0       + i * (SPC_AMP1        - SPC_AMP0),      tempval);
        }
    }else{
        int tempval = value;
        if (tempval>amplitude_limit[channel]){tempval=amplitude_limit[channel];}
        spcm_dwSetParam_i32 (hCard, SPC_AMP0       + channel * (SPC_AMP1        - SPC_AMP0),      tempval);
    }
}

void varReset(){
    dynamic_total = 0;
    static_total = 0;
    not_arrived = 1;
    loop_counter=0;
}

int job_distributer(unsigned int * dynamic_num, int current_index,int maximal_jobcount,int last_moved,int* new_perloop,int* last_perloop){
    int counter = -1;
    int ind0 = dynamic_num[0] - current_index;
    int ind1 = dynamic_num[1] - current_index;
    int ind2 = dynamic_num[2] - current_index;
    int ind3 = dynamic_num[3] - current_index;
    int foresighter = 0;
    while (foresighter <= maximal_jobcount){
        counter += 1;
        if (ind0<=0 && ind1<=0 &&ind2<=0 &&ind3<=0) {
            job_distirbuter_flag = true;
            break;
        }
        if (ind0 > 0){
            foresighter += 1;
            ind0 -= 1;
        }
        if (ind1 > 0){
            foresighter += 1;
            ind1 -= 1;
        }
        if (ind2 > 0){
            foresighter += 1;
            ind2 -= 1;
        }
        if (ind3 > 0){
            foresighter += 1;
            ind3 -= 1;
        }
    }
    *new_perloop = (counter+dynamic_loopcount-1) /  dynamic_loopcount;
    if (first_loop_flag) last_moved = counter;
    *last_perloop = (last_moved+dynamic_loopcount-1) /  dynamic_loopcount;
    return counter;
}



int block_size;
// settings for the FIFO mode buffer handling
uint32       lNotifySize =  MEGA_B(2); // The size of data the card will execute each time before signaling to the GPU
uint32       lBufferSize =  MEGA_B(4);
uint64       HBufferSize =  MEGA_B(64); // The actual buffer used on the AWG; must be a power of 2 and should be no more than 4 GB (lower size reduces delay)

// Parameter settings   
int32       lMaxOutputLevel = 2500; // +-1 Volt


// Test params

unsigned long long sPointerPosition=0;

void static_looper(){
    while (!stop_flag){
        while (static_flag && !stop_flag){
            static_pulseflag = false;
            while (!static_endflag && !stop_flag){
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
        }
        static_pulseflag = true;
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
    int_temp=0;

    dynamic_buffersize = 2*round(ramp_time*llSamplerate);

    cudaDeviceReset();
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
    reset_amp(-1, lMaxOutputLevel);
    for (int lChIdx = 0; lChIdx < lNumCh; ++lChIdx)
    {
        spcm_dwSetParam_i32 (hCard, SPC_FILTER0 + lChIdx * (SPC_FILTER1 - SPC_FILTER0), 0);
        spcm_dwSetParam_i32 (hCard, SPC_ENABLEOUT0 + lChIdx * (SPC_ENABLEOUT1 - SPC_ENABLEOUT0), 1);
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
    serverThread = std::thread(TCP_server);
    while (1){
        while (!server_flag && !stop_flag){volatile int nulvar = server_flag;}
        start_and_reset: 
        job_distirbuter_flag = false;
        if (stop_flag){break;} 
        cuda_cleanup();
        varReset();
        server_flag = false;
        continue_flag=false; // For testing
        printf("Start generating waveform\n");
    
        for (int ch = 0; ch < 4; ch++){
            dynamic_total += dynamic_num[ch];
            static_total += static_num[ch];
        }
        double stt_checker = (double) sizeof(double)/sizeof(short)*(dynamic_total+static_total)*lBytesPerChannelInNotifySize/1024/1024/1024;
        printf("Total buffersize: %f GiB\n",stt_checker);
        if (stt_checker > 21.5){
            printf("Buffer required exceeds GPU memory\n");
            spcm_vClose (hCard);
            return EXIT_FAILURE;
        }
        if (update_flag){
            cudaMalloc((void**)&update_index_map_cuda,static_total*sizeof(int));
            cudaMemcpy(update_index_map_cuda,update_index_map,static_total*sizeof(int),cudaMemcpyHostToDevice);
            phase_reorder_update<<<(int)ceil((float)static_total/lThreadsPerBlock),lThreadsPerBlock>>>(update_index_map_cuda,new_phase_list_cuda,phase_list_cuda,static_total);
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
        if (if_mapped){
            StaticAmpMapper( real_static_freq, amp_freq_map,  amp_map_static, total_divider);
        }
        if (staticBufferInit()){
            clean_exit();
            return EXIT_FAILURE;
        }
        if (dynamic_total){
            if (dynamicBufferInit()){
                clean_exit();
                return EXIT_FAILURE;
            }
        }


        if (update_flag){
            if (!amp_flag)StaticWaveGeneration_update<<<(static_length/lThreadsPerBlock),lThreadsPerBlock>>>(real_static_freq_cuda,static_buffer_cuda,summed_buffer_cuda,phase_list_cuda);
            else StaticWaveGeneration_update_amp<<<(static_length/lThreadsPerBlock),lThreadsPerBlock>>>(real_static_freq_cuda,amp_list_cuda,static_buffer_cuda,summed_buffer_cuda,phase_list_cuda);
        }else{
            if (!amp_flag){
                if (if_mapped){
                    printf("MAPPED!!\n");
                    StaticWaveGeneration_amp_mapped<<<(static_length/lThreadsPerBlock),lThreadsPerBlock>>>(real_static_freq_cuda,amp_map_cuda,total_divider,static_buffer_cuda,summed_buffer_cuda,phase_list_cuda);
                }else{
                    StaticWaveGeneration<<<(static_length/lThreadsPerBlock),lThreadsPerBlock>>>(real_static_freq_cuda,static_buffer_cuda,summed_buffer_cuda,phase_list_cuda);
                }
            }
            else StaticWaveGeneration_amp<<<(static_length/lThreadsPerBlock),lThreadsPerBlock>>>(real_static_freq_cuda,amp_list_cuda,static_buffer_cuda,summed_buffer_cuda,phase_list_cuda);
        }

        // ----- setup DMA transfer from GPU to Spectrum card -----
        if (init_flag) spcm_dwDefTransfer_i64 (hCard, SPCM_BUF_DATA, SPCM_DIR_GPUTOCARD, lNotifySize, DMABuffer, 0, lBufferSize);
        // ----- fill the software buffer before we start the card -----

        if (dynamic_total){
            if (!amp_flag)DynamicListWorker<<<ceil(dynamic_total/32.),32>>>(real_static_freq_cuda,real_destination_freq_cuda,dynamic_list_cuda,dFreq_cuda,dDiff_cuda,dphi_cuda,phase_list_cuda);
            else DynamicListWorker_amp<<<ceil(dynamic_total/32.),32>>>(real_static_freq_cuda,real_destination_freq_cuda,dynamic_list_cuda,dFreq_cuda,dDiff_cuda,dphi_cuda,damp_cuda,phase_list_cuda,amp_list_cuda);
            DoubleCopier <<< static_length/lThreadsPerBlock, lThreadsPerBlock >>> (summed_buffer_cuda,saved_buffer_cuda);
            cudaDeviceSynchronize();   
        }

        
        for (int32 lPosInBuf = 0; lPosInBuf < lBufferSize; lPosInBuf += lNotifySize)
            {
            StaticMux <<< static_length / lThreadsPerBlock, lThreadsPerBlock >>> (summed_buffer_cuda,(int16*)((char*)DMABuffer + lPosInBuf));
            cudaDeviceSynchronize();
            }
        
        printf("\r\nCalculated: Init\r\n");
        if (init_flag) {
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
            // std::thread terminatorThread(ifkeypress,&iskeypressed);
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
        printf("Looping ...\n");
        staticThread =std::thread(static_looper);
        init_flag = false;
        }else{
            static_endflag = false;
        }

        if (dynamic_total){
            while (!continue_flag && !server_flag && !stop_flag){} 
            if (server_flag){goto start_and_reset;}
            if (stop_flag) break;
            int counterprint = 0;
            continue_flag = false; // For testing
            static_endflag = true;
            static_flag = false;
            while (!static_pulseflag){}
            int moved_tweezers = 0;
            int last_moving_num = 0;
            int new_counter = 0;
            int last_counter = 0;
            while (!job_distirbuter_flag){
                int last_record;
                int moving_num = job_distributer(dynamic_num,moved_tweezers,max_streaming_tweezers,last_moving_num,&new_counter,&last_counter);
                sPointerPosition=0;
                bool not_finish_gen_flag = true;
                if (first_loop_flag){
                    Pre_AccelCombine<<< static_length/ lThreadsPerBlock, lThreadsPerBlock >>> (saved_buffer_cuda,0,0,0,moving_num,dynamic_list_cuda,static_buffer_cuda,real_static_freq_cuda,real_destination_freq_cuda,phase_list_cuda);
                    cudaDeviceSynchronize();
                    DoubleCopier<<< static_length/ lThreadsPerBlock, lThreadsPerBlock >>> (saved_buffer_cuda,temp_buffer_cuda);
                    last_record = moved_tweezers-last_moving_num;
                } else{
                    last_record = moved_tweezers;
                }
                int new_record = moved_tweezers+moving_num;
                for (int cnt = 0; cnt < dynamic_loopcount && !iskeypressed;cnt++){
                    if ((dwError = spcm_dwSetParam_i32 (hCard, SPC_M2CMD, M2CMD_DATA_WAITDMA)) != ERR_OK)
                    {
                        if (dwError == ERR_TIMEOUT)
                            printf ("\n... Timeout %d\n",counterprint);
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
                        if (not_finish_gen_flag){
                            if (cnt==0){
                                if (first_loop_flag){
                                    first_loop_flag = false;
                                } else {
                                    DoubleCopier<<< static_length/ lThreadsPerBlock, lThreadsPerBlock >>> (temp_buffer_cuda,saved_buffer_cuda);
                                    cudaDeviceSynchronize();
                                }
                            }
                            Pre_AccelCombine<<< static_length/ lThreadsPerBlock, lThreadsPerBlock >>> (temp_buffer_cuda,last_record,new_record,last_counter,new_counter,dynamic_list_cuda,static_buffer_cuda,real_static_freq_cuda,real_destination_freq_cuda,phase_list_cuda);
                            cudaDeviceSynchronize();
                        }
                        if (!amp_flag)AccelCombine<<< static_length/lThreadsPerBlock/2, lThreadsPerBlock >>>(sPointerPosition,saved_buffer_cuda,summed_buffer_cuda,dFreq_cuda,dDiff_cuda,dphi_cuda,moved_tweezers,moving_num);
                        else AccelCombine_amp<<< static_length/lThreadsPerBlock/2, lThreadsPerBlock >>>(sPointerPosition,saved_buffer_cuda,summed_buffer_cuda,dFreq_cuda,dDiff_cuda,dphi_cuda,damp_cuda,moved_tweezers,moving_num);
                        cudaDeviceSynchronize();
                        sPointerPosition += static_length;
                    }
                    if (not_finish_gen_flag){
                        last_record += last_counter;
                        new_record += new_counter;
                        if (last_record >= moved_tweezers+moving_num && new_record >=moved_tweezers+2*moving_num) {
                            not_finish_gen_flag=false;
                        }
                    }
                }
                moved_tweezers += moving_num;
                last_moving_num = moving_num;
                counterprint+=1;
            }
            DoubleCopier<<< static_length/ lThreadsPerBlock, lThreadsPerBlock >>> (temp_buffer_cuda,summed_buffer_cuda);
            // DoubletoShortCopier<<< static_length/ lThreadsPerBlock, lThreadsPerBlock >>> (temp_buffer_cuda,final_buffer_cuda);
            cudaDeviceSynchronize();

            printf("DESTINATION\n");

            while (!static_flag) // Terminated when key pressed; not working if RMA keeps waiting
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
                        if (not_arrived){
                            StaticMux <<< static_length / lThreadsPerBlock, lThreadsPerBlock >>> (summed_buffer_cuda,(int16*)((char*)DMABuffer + lUserPos));
                            not_arrived=0;
                            loop_counter=0;
                            printf("Done\n");
                        }else if(loop_counter<lBufferSize/lNotifySize){
                            StaticMux <<< static_length / lThreadsPerBlock, lThreadsPerBlock >>> (temp_buffer_cuda,(int16*)((char*)DMABuffer + lUserPos));
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
    }
    dwError = spcm_dwSetParam_i32 (hCard, SPC_M2CMD, M2CMD_CARD_STOP | M2CMD_DATA_STOPDMA);

    // clean up
    staticThread.join();
    serverThread.join();
    clean_exit();
    return EXIT_SUCCESS;
}

