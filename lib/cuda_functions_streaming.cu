#   include "cuda_functions_streaming.h"

double* dFreq_cuda;
double* dDiff_cuda;
double* dphi_cuda;
float* damp_cuda;

double* destination_buffer_cuda;
double ** temp_buffer_cuda;
double* temp_buffer[4];

__global__ void DynamicListWorker(double* __restrict__ staticFrequency,double* __restrict__ destinationFreq,int*dy_list,double *dstart,double*dDiff,double*dphi,double* __restrict__ phase_list){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    for (int buffer_index=0;buffer_index<channel_num_cuda;buffer_index++){
        if(i<dynamic_num_cuda[buffer_index]){
            int tone_index = i+dynamic_tone_count_cuda[buffer_index];
            int dy_index = dy_list[tone_index];
            double temp = staticFrequency[dy_index];
            dstart[tone_index] = temp;
            dDiff[tone_index] = destinationFreq[tone_index]-temp;
            dphi[tone_index] = phase_list[dy_index];
        }        
    }
}

__global__ void DynamicListWorker_amp(double*__restrict__ staticFrequency,double*__restrict__ destinationFreq,int* __restrict__ dy_list,double *dstart,double*dDiff,double*dphi,float*damp,double*__restrict__ phase_list,float*__restrict__ amp_list){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    for (int buffer_index=0;buffer_index<channel_num_cuda;buffer_index++){
        if(i<dynamic_num_cuda[buffer_index]){
            int tone_index = i+dynamic_tone_count_cuda[buffer_index];
            int dy_index = dy_list[tone_index];
            double temp = staticFrequency[dy_index];
            dstart[tone_index] = temp;
            dDiff[tone_index] = destinationFreq[tone_index]-temp;
            dphi[tone_index] = phase_list[dy_index];
            damp[tone_index] = amp_list[dy_index];
        }        
    }
}


__global__ void Pre_AccelCombine(double** temp_save_buf,int last_start_index,int new_start_index,int last_counter, int new_counter,int* __restrict__ dy_list,double* __restrict__ static_buffer, double*__restrict__ startFreq, double* __restrict__ dest_frequency,double*__restrict__ phase_list){
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    double dump;
    for (int buffer_index=0;buffer_index<channel_num_cuda;buffer_index++){
        double sum = temp_save_buf[buffer_index][i];
        double temp = 0.0;
        int dynamic_count = static_cast<int>(dynamic_tone_count_cuda[buffer_index]);
        if (new_start_index !=0){
            double temp_factor = fma(dynamic_loopcount_cuda,static_cast<double>(static_bufferlength_cuda),static_cast<double>(i));
            for (int j = last_start_index; (j<last_start_index+last_counter&&j<dynamic_num_cuda[buffer_index]); j++){
                int tone_index = j+dynamic_count;
                double dest_freq = dest_frequency[tone_index];
                double phi_dest = fma(fma((startFreq[tone_index] - dest_freq),0.5*dynamic_bufferlength_cuda,temp_factor * dest_freq), illSamplerate_cuda ,phase_list[tone_index]);
                temp +=  sin2pi_fast (phi_dest);
            }
        }
        sum += 32767.*istatic_num_cuda[buffer_index] *temp;
        for (int j = new_start_index; (j<new_start_index+new_counter && j<dynamic_num_cuda[buffer_index]); j++){
            sum -= static_buffer[dy_list[j+dynamic_count]*static_bufferlength_cuda+i];
        }
        temp_save_buf[buffer_index][i] = sum;
    }
}


__global__ void Pre_AccelCombine_amp(double** temp_save_buf,int last_start_index,int new_start_index,int last_counter, int new_counter,int* __restrict__ dy_list,double* __restrict__ static_buffer, double*__restrict__ startFreq, double* __restrict__ dest_frequency,double*__restrict__ phase_list,float*amplist){
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    double dump;
    for (int buffer_index=0;buffer_index<channel_num_cuda;buffer_index++){
        double sum = temp_save_buf[buffer_index][i];
        double temp = 0.0;
        int dynamic_count = static_cast<int>(dynamic_tone_count_cuda[buffer_index]);
        if (new_start_index !=0){
            double temp_factor = fma(dynamic_loopcount_cuda,static_cast<double>(static_bufferlength_cuda),static_cast<double>(i));
            for (int j = last_start_index; (j<last_start_index+last_counter&&j<dynamic_num_cuda[buffer_index]); j++){
                int tone_index = j+dynamic_count;
                double dest_freq = dest_frequency[tone_index];
                double phi_dest = fma(fma((startFreq[tone_index] - dest_freq),0.5*dynamic_bufferlength_cuda,temp_factor * dest_freq), illSamplerate_cuda ,phase_list[tone_index]);
                temp += amplist[tone_index] * sin2pi_fast (phi_dest);
            }
        }
        sum += 32767.*istatic_num_cuda[buffer_index] *temp;
        for (int j = new_start_index; (j<new_start_index+new_counter && j<dynamic_num_cuda[buffer_index]); j++){
            sum -= static_buffer[dy_list[j+dynamic_count]*static_bufferlength_cuda+i];
        }
        temp_save_buf[buffer_index][i] = sum;
    }
}


__global__ void Pre_AccelCombine_amp_mapped(double** temp_save_buf,int last_start_index,int new_start_index,int last_counter, int new_counter,int* __restrict__ dy_list,double* __restrict__ static_buffer, double*__restrict__ startFreq, double* __restrict__ dest_frequency,double*__restrict__ phase_list,float*amp_map){
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    double dump;
    for (int buffer_index=0;buffer_index<channel_num_cuda;buffer_index++){
        double sum = temp_save_buf[buffer_index][i];
        double temp = 0.0;
        float coefficient_corrector = 0.0f;
        int dynamic_num = dynamic_num_cuda[buffer_index];
        float scaler = static_cast<float>(dynamic_num) * static_cast<float>(istatic_num_cuda[buffer_index]);
        int dynamic_count = static_cast<int>(dynamic_tone_count_cuda[buffer_index]);
        if (new_start_index !=0){
            double temp_factor = fma(dynamic_loopcount_cuda,static_cast<double>(static_bufferlength_cuda),static_cast<double>(i));
            for (int j = last_start_index; (j<last_start_index+last_counter&&j<dynamic_num_cuda[buffer_index]); j++){
                int tone_index = j+dynamic_count;
                double dest_freq = dest_frequency[tone_index];
                float amp_factor = amp_map[static_cast<int>(fmaf(static_cast<float>(dest_freq),map_interval_cuda,0.5f))];
                double phi_dest = fma(fma((startFreq[tone_index] - dest_freq),0.5*dynamic_bufferlength_cuda,temp_factor * dest_freq), illSamplerate_cuda ,phase_list[tone_index]);
                temp += amp_factor * sin2pi_fast (phi_dest);
                coefficient_corrector+= amp_factor;
            }
        }
        sum += 32767.*istatic_num_cuda[buffer_index] *temp*static_cast<double>(scaler/coefficient_corrector);
        for (int j = new_start_index; (j<new_start_index+new_counter && j<dynamic_num_cuda[buffer_index]); j++){
            sum -= static_buffer[dy_list[j+dynamic_count]*static_bufferlength_cuda+i];
        }
        temp_save_buf[buffer_index][i] = sum;
    }
}

__global__ void Pre_AccelCombine_amp_amp_mapped(double** temp_save_buf,int last_start_index,int new_start_index,int last_counter, int new_counter,int* __restrict__ dy_list,double* __restrict__ static_buffer, double*__restrict__ startFreq, double* __restrict__ dest_frequency,double*__restrict__ phase_list,float*amp_map,float*amplist){
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    double dump;
    for (int buffer_index=0;buffer_index<channel_num_cuda;buffer_index++){
        double sum = temp_save_buf[buffer_index][i];
        double temp = 0.0;
        float coefficient_corrector = 0.0f;
        int dynamic_num = dynamic_num_cuda[buffer_index];
        float scaler = static_cast<float>(dynamic_num) * static_cast<float>(istatic_num_cuda[buffer_index]);
        int dynamic_count = static_cast<int>(dynamic_tone_count_cuda[buffer_index]);
        if (new_start_index !=0){
            double temp_factor = fma(dynamic_loopcount_cuda,static_cast<double>(static_bufferlength_cuda),static_cast<double>(i));
            for (int j = last_start_index; (j<last_start_index+last_counter&&j<dynamic_num_cuda[buffer_index]); j++){
                int tone_index = j+dynamic_count;
                double dest_freq = dest_frequency[tone_index];
                float amp_factor = amp_map[static_cast<int>(fmaf(static_cast<float>(dest_freq),map_interval_cuda,0.5f))];
                double phi_dest = fma(fma((startFreq[tone_index] - dest_freq),0.5*dynamic_bufferlength_cuda,temp_factor * dest_freq), illSamplerate_cuda ,phase_list[tone_index]);
                temp += amp_factor * sin2pi_fast (phi_dest)*amplist[tone_index];
                coefficient_corrector+= amp_factor;
            }
        }
        sum += 32767.*istatic_num_cuda[buffer_index] *temp*static_cast<double>(scaler/coefficient_corrector);
        for (int j = new_start_index; (j<new_start_index+new_counter && j<dynamic_num_cuda[buffer_index]); j++){
            sum -= static_buffer[dy_list[j+dynamic_count]*static_bufferlength_cuda+i];
        }
        temp_save_buf[buffer_index][i] = sum;
    }
}


__global__ void AccelCombine(unsigned long long startPosition,double**__restrict__ save_buf,double** sum_buf,
                    double*__restrict__ dstartFrequency,double *__restrict__ dfrequencyDiff,double *__restrict__ dphi,int start_index, int maximal_index){
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ double startFrequency[512];
    __shared__ double frequencyDiff[512];
    __shared__ double phi[512];
    size_t checkpoint = 2*i+startPosition+1;
    double k = static_cast<double>(checkpoint)-1.;
    // double s;
    double ratioks = k*idynamic_bufferlength_cuda;
    double ratiok = fma(ratioks, ratioks- 3.,2.5 )* ratioks *ratioks *ratioks;
    double ratioks1 = ratioks+ idynamic_bufferlength_cuda;
    double ratiok1 = fma(ratioks1, ratioks1- 3.,2.5 )* ratioks1 *ratioks1 *ratioks1;

    for (int buffer_index=0;buffer_index<channel_num_cuda;buffer_index++){
        for (int iter = 0;iter * 256 < dynamic_num_cuda[buffer_index]-start_index; iter++){
            int tinx = threadIdx.x + iter*256;
            if(tinx<dynamic_num_cuda[buffer_index]-start_index && tinx < maximal_index){
                int index = start_index+tinx+dynamic_tone_count_cuda[buffer_index];
                startFrequency[tinx] = dstartFrequency[index];
                frequencyDiff[tinx] = dfrequencyDiff[index];
                phi[tinx] = dphi[index];
            }
        }
        __syncthreads();
        double ampss = 32767.*istatic_num_cuda[buffer_index];
        double2* save_buf2 = reinterpret_cast<double2*>(save_buf[buffer_index]);
        double2 save2 = save_buf2[i];
        double sum = save2.x;
        double sum1 = save2.y;
        
        if (checkpoint<__double2uint_rn(dynamic_bufferlength_cuda)){
            for (int j = 0; (j<dynamic_num_cuda[buffer_index]- start_index && j<maximal_index); j++){
                double phase_c =  fma(k * illSamplerate_cuda , fma(frequencyDiff[j] , ratiok,startFrequency[j]),phi[j]);
                double phase_c1 = fma((k+1.) * illSamplerate_cuda , fma( frequencyDiff[j] , ratiok1,startFrequency[j]),phi[j]);
                half2 temp =  h2sin (__floats2half2_rn(M_2PI*modfeq(phase_c),M_2PI*modfeq(phase_c1)));
                sum1 += __high2float(temp);
                sum += __low2float(temp);
            }
        }else if(checkpoint==__double2uint_rn(dynamic_bufferlength_cuda)){
            for (int j = 0; (j<dynamic_num_cuda[buffer_index]- start_index && j<maximal_index); j++){
                double phase_c =  fma(k * illSamplerate_cuda , fma(frequencyDiff[j] , ratiok,startFrequency[j]),phi[j]);
                double phase_c1 = modfeq(fma(fma(frequencyDiff[j],0.5,startFrequency[j]),dynamic_bufferlength_cuda*illSamplerate_cuda,phi[j]));
                half2 temp =  h2sin (__floats2half2_rn(M_2PI*modfeq(phase_c),M_2PI*modfeq(phase_c1)));
                sum1 += __high2float(temp);
                sum += __low2float(temp);
            }
        }else{ 
            for (int j = 0; (j<dynamic_num_cuda[buffer_index]- start_index && j<maximal_index); j++){
                double temp_factor = (startFrequency[j]+frequencyDiff[j]) * illSamplerate_cuda;
                double phase_c = fma(temp_factor , (k-dynamic_bufferlength_cuda) ,fma(fma(frequencyDiff[j],0.5,startFrequency[j]),dynamic_bufferlength_cuda*illSamplerate_cuda,phi[j]));
                double phase_c1 = phase_c + temp_factor;
                half2 temp =  h2sin (__floats2half2_rn(M_2PI*modfeq(phase_c),M_2PI*modfeq(phase_c1)));
                sum1 += __high2float(temp);
                sum += __low2float(temp);
            }
        }
        double2* sum_buf2 = reinterpret_cast<double2*>(sum_buf[buffer_index]);
        sum_buf2[i] = {sum*ampss,sum1*ampss};
        __syncthreads();
    }
}



__global__ void AccelCombine_amp(unsigned long long startPosition,double**__restrict__ save_buf,double** sum_buf,
                    double*__restrict__ dstartFrequency,double *__restrict__ dfrequencyDiff,double *__restrict__ dphi,float*__restrict__ damp,int start_index, int maximal_index){
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ double startFrequency[512];
    __shared__ double frequencyDiff[512];
    __shared__ double phi[512];
    __shared__ float amp[512];
    size_t checkpoint = 2*i+startPosition+1;
    double k = static_cast<double>(checkpoint)-1.;
    // double s;
    double ratioks = k*idynamic_bufferlength_cuda;
    double ratiok = fma(ratioks, ratioks- 3.,2.5 )* ratioks *ratioks *ratioks;
    double ratioks1 = ratioks+ idynamic_bufferlength_cuda;
    double ratiok1 = fma(ratioks1, ratioks1- 3.,2.5 )* ratioks1 *ratioks1 *ratioks1;


    for (int buffer_index=0;buffer_index<channel_num_cuda;buffer_index++){
        for (int iter = 0;iter * 256 < dynamic_num_cuda[buffer_index]-start_index; iter++){
            int tinx = threadIdx.x + iter*256;
            if(tinx<dynamic_num_cuda[buffer_index]-start_index && tinx < maximal_index){
                int index = start_index+tinx+dynamic_tone_count_cuda[buffer_index];
                startFrequency[tinx] = dstartFrequency[index];
                frequencyDiff[tinx] = dfrequencyDiff[index];
                phi[tinx] = dphi[index];
                amp[tinx] = damp[index];
            }
        }
        __syncthreads();
        double ampss = 32767.*istatic_num_cuda[buffer_index];
        double2* save_buf2 = reinterpret_cast<double2*>(save_buf[buffer_index]);
        double2 save2 = save_buf2[i];
        double sum = save2.x;
        double sum1 = save2.y;
        
        if (checkpoint<__double2ull_rn(dynamic_bufferlength_cuda)){
            for (int j = 0; (j<dynamic_num_cuda[buffer_index]- start_index && j<maximal_index); j++){
                double phase_c =  fma(k * illSamplerate_cuda , fma(frequencyDiff[j] , ratiok,startFrequency[j]),phi[j]);
                double phase_c1 = fma((k+1.) * illSamplerate_cuda , fma( frequencyDiff[j] , ratiok1,startFrequency[j]),phi[j]);
                half2 temp =  h2sin (__floats2half2_rn(M_2PI*modfeq(phase_c),M_2PI*modfeq(phase_c1)));
                sum1 += __high2float(temp)*amp[j];
                sum += __low2float(temp)*amp[j];
            }
        }else if(checkpoint==__double2ull_rn(dynamic_bufferlength_cuda)){
            for (int j = 0; (j<dynamic_num_cuda[buffer_index]- start_index && j<maximal_index); j++){
                double phase_c =  fma(k * illSamplerate_cuda , fma(frequencyDiff[j] , ratiok,startFrequency[j]),phi[j]);
                double phase_c1 = modfeq(fma(fma(frequencyDiff[j],0.5,startFrequency[j]),dynamic_bufferlength_cuda*illSamplerate_cuda,phi[j]));
                half2 temp =  h2sin (__floats2half2_rn(M_2PI*modfeq(phase_c),M_2PI*modfeq(phase_c1)));
                sum1 += __high2float(temp)*amp[j];
                sum += __low2float(temp)*amp[j];
            }
        }else{ 
            for (int j = 0; (j<dynamic_num_cuda[buffer_index]- start_index && j<maximal_index); j++){
                double temp_factor = (startFrequency[j]+frequencyDiff[j]) * illSamplerate_cuda;
                double phase_c = fma(temp_factor , (k-dynamic_bufferlength_cuda) ,fma(fma(frequencyDiff[j],0.5,startFrequency[j]),dynamic_bufferlength_cuda*illSamplerate_cuda,phi[j]));
                double phase_c1 = phase_c + temp_factor;
                half2 temp =  h2sin (__floats2half2_rn(M_2PI*modfeq(phase_c),M_2PI*modfeq(phase_c1)));
                sum1 += __high2float(temp)*amp[j];
                sum += __low2float(temp)*amp[j];
            }
        }
        double2* sum_buf2 = reinterpret_cast<double2*>(sum_buf[buffer_index]);
        sum_buf2[i] = {sum*ampss,sum1*ampss};
        __syncthreads();
    }
}



__global__ void AccelCombine_amp_modulated(unsigned long long startPosition,double**__restrict__ save_buf,double** sum_buf,
                    double*__restrict__ dstartFrequency,double *__restrict__ dfrequencyDiff,double *__restrict__ dphi,float*__restrict__ damp,float*__restrict__ namp,int start_index, int maximal_index){
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ double startFrequency[512];
    __shared__ double frequencyDiff[512];
    __shared__ double phi[512];
    __shared__ float amp[512];
    __shared__ float newamp[512];
    size_t checkpoint = 2*i+startPosition+1;
    double k = static_cast<double>(checkpoint)-1.;
    // double s;
    double ratioks = k*idynamic_bufferlength_cuda;
    double ratiok = fma(ratioks, ratioks- 3.,2.5 )* ratioks *ratioks *ratioks;
    double ratioks1 = ratioks+ idynamic_bufferlength_cuda;
    double ratiok1 = fma(ratioks1, ratioks1- 3.,2.5 )* ratioks1 *ratioks1 *ratioks1;


    for (int buffer_index=0;buffer_index<channel_num_cuda;buffer_index++){
        for (int iter = 0;iter * 256 < dynamic_num_cuda[buffer_index]-start_index; iter++){
            int tinx = threadIdx.x + iter*256;
            if(tinx<dynamic_num_cuda[buffer_index]-start_index && tinx < maximal_index){
                int index = start_index+tinx+dynamic_tone_count_cuda[buffer_index];
                startFrequency[tinx] = dstartFrequency[index];
                frequencyDiff[tinx] = dfrequencyDiff[index];
                phi[tinx] = dphi[index];
                amp[tinx] = damp[index];
                newamp[tinx] = namp[index];
            }
        }
        __syncthreads();
        double ampss = 32767.*istatic_num_cuda[buffer_index];
        double2* save_buf2 = reinterpret_cast<double2*>(save_buf[buffer_index]);
        double2 save2 = save_buf2[i];
        double sum = save2.x;
        double sum1 = save2.y;
        
        if (checkpoint<__double2ull_rn(dynamic_bufferlength_cuda)){
            for (int j = 0; (j<dynamic_num_cuda[buffer_index]- start_index && j<maximal_index); j++){
                float old_amp = amp[j];
                float amp_ramper = __fmaf_rn(newamp[j] - old_amp, static_cast<float>(ratioks),old_amp);
                double phase_c =  fma(k * illSamplerate_cuda , fma(frequencyDiff[j] , ratiok,startFrequency[j]),phi[j]);
                double phase_c1 = fma((k+1.) * illSamplerate_cuda , fma( frequencyDiff[j] , ratiok1,startFrequency[j]),phi[j]);
                half2 temp =  h2sin (__floats2half2_rn(M_2PI*modfeq(phase_c),M_2PI*modfeq(phase_c1)));
                sum1 += __high2float(temp)*amp_ramper;
                sum += __low2float(temp)*amp_ramper;
            }
        }else if(checkpoint==__double2ull_rn(dynamic_bufferlength_cuda)){
            for (int j = 0; (j<dynamic_num_cuda[buffer_index]- start_index && j<maximal_index); j++){
                double phase_c =  fma(k * illSamplerate_cuda , fma(frequencyDiff[j] , ratiok,startFrequency[j]),phi[j]);
                double phase_c1 = modfeq(fma(fma(frequencyDiff[j],0.5,startFrequency[j]),dynamic_bufferlength_cuda*illSamplerate_cuda,phi[j]));
                half2 temp =  h2sin (__floats2half2_rn(M_2PI*modfeq(phase_c),M_2PI*modfeq(phase_c1)));
                sum1 += __high2float(temp)*newamp[j];
                sum += __low2float(temp)*newamp[j];
            }
        }else{ 
            for (int j = 0; (j<dynamic_num_cuda[buffer_index]- start_index && j<maximal_index); j++){
                double temp_factor = (startFrequency[j]+frequencyDiff[j]) * illSamplerate_cuda;
                double phase_c = fma(temp_factor , (k-dynamic_bufferlength_cuda) ,fma(fma(frequencyDiff[j],0.5,startFrequency[j]),dynamic_bufferlength_cuda*illSamplerate_cuda,phi[j]));
                double phase_c1 = phase_c + temp_factor;
                half2 temp =  h2sin (__floats2half2_rn(M_2PI*modfeq(phase_c),M_2PI*modfeq(phase_c1)));
                sum1 += __high2float(temp)*newamp[j];
                sum += __low2float(temp)*newamp[j];
            }
        }
        double2* sum_buf2 = reinterpret_cast<double2*>(sum_buf[buffer_index]);
        sum_buf2[i] = {sum*ampss,sum1*ampss};
        __syncthreads();
    }
}
                


__global__ void AccelCombine_amp_mapped(unsigned long long startPosition,double**__restrict__ save_buf,double** sum_buf,
                    double*__restrict__ dstartFrequency,double *__restrict__ dfrequencyDiff,double *__restrict__ dphi,float*__restrict__ amp_map,int start_index, int maximal_index){
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ double startFrequency[512];
    __shared__ double frequencyDiff[512];
    __shared__ double phi[512];
    size_t checkpoint = 2*i+startPosition+1;
    double k = static_cast<double>(checkpoint)-1.;
    // double s;
    double ratioks = k*idynamic_bufferlength_cuda;
    double ratiok = fma(ratioks, ratioks- 3.,2.5 )* ratioks *ratioks *ratioks;
    double ratioks1 = ratioks+ idynamic_bufferlength_cuda;
    double ratiok1 = fma(ratioks1, ratioks1- 3.,2.5 )* ratioks1 *ratioks1 *ratioks1;
    float coefficient_corrector;
    float temp_amp;
    float temp_amp1;

    for (int buffer_index=0;buffer_index<channel_num_cuda;buffer_index++){
        for (int iter = 0;iter * 256 < dynamic_num_cuda[buffer_index]-start_index; iter++){
            int tinx = threadIdx.x + iter*256;
            if(tinx<dynamic_num_cuda[buffer_index]-start_index && tinx < maximal_index){
                int index = start_index+tinx+dynamic_tone_count_cuda[buffer_index];
                startFrequency[tinx] = dstartFrequency[index];
                frequencyDiff[tinx] = dfrequencyDiff[index];
                phi[tinx] = dphi[index];
            }
        }
        __syncthreads();


        coefficient_corrector= 0.f;   
        int dynamic_num = dynamic_num_cuda[buffer_index];
        temp_amp = 0.f;
        temp_amp1 = 0.f;
        double2* save_buf2 = reinterpret_cast<double2*>(save_buf[buffer_index]);
        double2 save2 = save_buf2[i];
        double sum = save2.x;
        double sum1 = save2.y;
        
        if (checkpoint<__double2ull_rn(dynamic_bufferlength_cuda)){
                double freqr = ratioks *ratioks *ratioks * __fma_rn(__fma_rn(6.,ratioks,-15.) , ratioks,10.);
                for (int j = 0; (j<dynamic_num- start_index && j<maximal_index); j++){
                double phase_c =  fma(k * illSamplerate_cuda , fma(frequencyDiff[j] , ratiok,startFrequency[j]),phi[j]);
                double phase_c1 = fma((k+1.) * illSamplerate_cuda , fma( frequencyDiff[j] , ratiok1,startFrequency[j]),phi[j]);
                float amp_factor = amp_map[static_cast<int>(fmaf(fmaf(static_cast<float>(frequencyDiff[j]),static_cast<float>(freqr),static_cast<float>(startFrequency[j])),map_interval_cuda,0.5f))];
                half2 temp =  h2sin (__floats2half2_rn(M_2PI*modfeq(phase_c),M_2PI*modfeq(phase_c1)));
                temp_amp1 += __high2float(temp)*amp_factor;
                temp_amp += __low2float(temp)*amp_factor;
                coefficient_corrector += amp_factor;
            }
        }else if(checkpoint==__double2ull_rn(dynamic_bufferlength_cuda)){
            for (int j = 0; (j<dynamic_num- start_index && j<maximal_index); j++){
                float amp_factor = amp_map[static_cast<int>(fmaf(static_cast<float>(frequencyDiff[j])+static_cast<float>(startFrequency[j]),map_interval_cuda,0.5f))];
                double phase_c =  fma(k * illSamplerate_cuda , fma(frequencyDiff[j] , ratiok,startFrequency[j]),phi[j]);
                double phase_c1 = modfeq(fma(fma(frequencyDiff[j],0.5,startFrequency[j]),dynamic_bufferlength_cuda*illSamplerate_cuda,phi[j]));
                half2 temp =  h2sin (__floats2half2_rn(M_2PI*modfeq(phase_c),M_2PI*modfeq(phase_c1)));
                temp_amp1 += __high2float(temp)*amp_factor;
                temp_amp += __low2float(temp)*amp_factor;
                coefficient_corrector += amp_factor;
            }
        }else{ 
            for (int j = 0; (j<dynamic_num- start_index && j<maximal_index); j++){
                double temp_factor = (startFrequency[j]+frequencyDiff[j]) * illSamplerate_cuda;
                float amp_factor = amp_map[static_cast<int>(fmaf(static_cast<float>(frequencyDiff[j])+static_cast<float>(startFrequency[j]),map_interval_cuda,0.5f))];
                double phase_c = fma(temp_factor , (k-dynamic_bufferlength_cuda) ,fma(fma(frequencyDiff[j],0.5,startFrequency[j]),dynamic_bufferlength_cuda*illSamplerate_cuda,phi[j]));
                double phase_c1 = phase_c + temp_factor;
                half2 temp =  h2sin (__floats2half2_rn(M_2PI*modfeq(phase_c),M_2PI*modfeq(phase_c1)));
                temp_amp1 += __high2float(temp)*amp_factor;
                temp_amp += __low2float(temp)*amp_factor;
                coefficient_corrector += amp_factor;
            }
        }
        coefficient_corrector = 32767.f/coefficient_corrector*static_cast<float>(dynamic_num) * static_cast<float>(istatic_num_cuda[buffer_index]);
        double2* sum_buf2 = reinterpret_cast<double2*>(sum_buf[buffer_index]);
        sum_buf2[i] = {static_cast<double>(fmaf(temp_amp,coefficient_corrector,static_cast<float>(sum))),static_cast<double>(fmaf(temp_amp1,coefficient_corrector,static_cast<float>(sum1)))};
        __syncthreads();
    }
}



__global__ void AccelCombine_amp_amp_mapped(unsigned long long startPosition,double**__restrict__ save_buf,double** sum_buf,
                    double*__restrict__ dstartFrequency,double *__restrict__ dfrequencyDiff,double *__restrict__ dphi,float*__restrict__ amp_map,float*__restrict__ damp,int start_index, int maximal_index){
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ double startFrequency[512];
    __shared__ double frequencyDiff[512];
    __shared__ double phi[512];
    __shared__ float amp[512];
    size_t checkpoint = 2*i+startPosition+1;
    double k = static_cast<double>(checkpoint)-1.;
    // double s;
    double ratioks = k*idynamic_bufferlength_cuda;
    double ratiok = fma(ratioks, ratioks- 3.,2.5 )* ratioks *ratioks *ratioks;
    double ratioks1 = ratioks+ idynamic_bufferlength_cuda;
    double ratiok1 = fma(ratioks1, ratioks1- 3.,2.5 )* ratioks1 *ratioks1 *ratioks1;
    float coefficient_corrector;
    float temp_amp;
    float temp_amp1;

    for (int buffer_index=0;buffer_index<channel_num_cuda;buffer_index++){
        for (int iter = 0;iter * 256 < dynamic_num_cuda[buffer_index]-start_index; iter++){
            int tinx = threadIdx.x + iter*256;
            if(tinx<dynamic_num_cuda[buffer_index]-start_index && tinx < maximal_index){
                int index = start_index+tinx+dynamic_tone_count_cuda[buffer_index];
                startFrequency[tinx] = dstartFrequency[index];
                frequencyDiff[tinx] = dfrequencyDiff[index];
                phi[tinx] = dphi[index];
                amp[tinx] = damp[index];
            }
        }
        __syncthreads();


        coefficient_corrector= 0.f;   
        int dynamic_num = dynamic_num_cuda[buffer_index];
        temp_amp = 0.f;
        temp_amp1 = 0.f;
        double2* save_buf2 = reinterpret_cast<double2*>(save_buf[buffer_index]);
        double2 save2 = save_buf2[i];
        double sum = save2.x;
        double sum1 = save2.y;
        
        if (checkpoint<__double2ull_rn(dynamic_bufferlength_cuda)){
                double freqr = ratioks *ratioks *ratioks * __fma_rn(__fma_rn(6.,ratioks,-15.) , ratioks,10.);
                for (int j = 0; (j<dynamic_num- start_index && j<maximal_index); j++){
                double phase_c =  fma(k * illSamplerate_cuda , fma(frequencyDiff[j] , ratiok,startFrequency[j]),phi[j]);
                double phase_c1 = fma((k+1.) * illSamplerate_cuda , fma( frequencyDiff[j] , ratiok1,startFrequency[j]),phi[j]);
                float amp_factor = amp_map[static_cast<int>(fmaf(fmaf(static_cast<float>(frequencyDiff[j]),static_cast<float>(freqr),static_cast<float>(startFrequency[j])),map_interval_cuda,0.5f))];
                half2 temp =  h2sin (__floats2half2_rn(M_2PI*modfeq(phase_c),M_2PI*modfeq(phase_c1)));
                temp_amp1 += __high2float(temp)*amp_factor*amp[j];
                temp_amp += __low2float(temp)*amp_factor*amp[j];
                coefficient_corrector += amp_factor;
            }
        }else if(checkpoint==__double2ull_rn(dynamic_bufferlength_cuda)){
            for (int j = 0; (j<dynamic_num- start_index && j<maximal_index); j++){
                float amp_factor = amp_map[static_cast<int>(fmaf(static_cast<float>(frequencyDiff[j])+static_cast<float>(startFrequency[j]),map_interval_cuda,0.5f))];
                double phase_c =  fma(k * illSamplerate_cuda , fma(frequencyDiff[j] , ratiok,startFrequency[j]),phi[j]);
                double phase_c1 = modfeq(fma(fma(frequencyDiff[j],0.5,startFrequency[j]),dynamic_bufferlength_cuda*illSamplerate_cuda,phi[j]));
                half2 temp =  h2sin (__floats2half2_rn(M_2PI*modfeq(phase_c),M_2PI*modfeq(phase_c1)));
                temp_amp1 += __high2float(temp)*amp_factor*amp[j];
                temp_amp += __low2float(temp)*amp_factor*amp[j];
                coefficient_corrector += amp_factor;
            }
        }else{ 
            for (int j = 0; (j<dynamic_num- start_index && j<maximal_index); j++){
                double temp_factor = (startFrequency[j]+frequencyDiff[j]) * illSamplerate_cuda;
                float amp_factor = amp_map[static_cast<int>(fmaf(static_cast<float>(frequencyDiff[j])+static_cast<float>(startFrequency[j]),map_interval_cuda,0.5f))];
                double phase_c = fma(temp_factor , (k-dynamic_bufferlength_cuda) ,fma(fma(frequencyDiff[j],0.5,startFrequency[j]),dynamic_bufferlength_cuda*illSamplerate_cuda,phi[j]));
                double phase_c1 = phase_c + temp_factor;
                half2 temp =  h2sin (__floats2half2_rn(M_2PI*modfeq(phase_c),M_2PI*modfeq(phase_c1)));
                temp_amp1 += __high2float(temp)*amp_factor*amp[j];
                temp_amp += __low2float(temp)*amp_factor*amp[j];
                coefficient_corrector += amp_factor;
            }
        }
        coefficient_corrector = 32767.f/coefficient_corrector*static_cast<float>(dynamic_num) * static_cast<float>(istatic_num_cuda[buffer_index]);
        double2* sum_buf2 = reinterpret_cast<double2*>(sum_buf[buffer_index]);
        sum_buf2[i] = {static_cast<double>(fmaf(temp_amp,coefficient_corrector,static_cast<float>(sum))),static_cast<double>(fmaf(temp_amp1,coefficient_corrector,static_cast<float>(sum1)))};
        __syncthreads();
    }
}




__global__ void AccelCombine_amp_mapped_modulated(unsigned long long startPosition,double**__restrict__ save_buf,double** sum_buf,
                    double*__restrict__ dstartFrequency,double *__restrict__ dfrequencyDiff,double *__restrict__ dphi,float*__restrict__ amp_map,float*__restrict__ damp,float*__restrict__ namp,int start_index, int maximal_index){
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ double startFrequency[512];
    __shared__ double frequencyDiff[512];
    __shared__ double phi[512];
    __shared__ float amp[512];
    __shared__ float newamp[512];
    size_t checkpoint = 2*i+startPosition+1;
    double k = static_cast<double>(checkpoint)-1.;
    // double s;
    double ratioks = k*idynamic_bufferlength_cuda;
    double ratiok = fma(ratioks, ratioks- 3.,2.5 )* ratioks *ratioks *ratioks;
    double ratioks1 = ratioks+ idynamic_bufferlength_cuda;
    double ratiok1 = fma(ratioks1, ratioks1- 3.,2.5 )* ratioks1 *ratioks1 *ratioks1;
    float coefficient_corrector;
    float temp_amp;
    float temp_amp1;

    for (int buffer_index=0;buffer_index<channel_num_cuda;buffer_index++){
        for (int iter = 0;iter * 256 < dynamic_num_cuda[buffer_index]-start_index; iter++){
            int tinx = threadIdx.x + iter*256;
            if(tinx<dynamic_num_cuda[buffer_index]-start_index && tinx < maximal_index){
                int index = start_index+tinx+dynamic_tone_count_cuda[buffer_index];
                startFrequency[tinx] = dstartFrequency[index];
                frequencyDiff[tinx] = dfrequencyDiff[index];
                phi[tinx] = dphi[index];
                amp[tinx] = damp[index];
                newamp[tinx] = namp[index];
            }
        }
        __syncthreads();


        coefficient_corrector= 0.f;   
        int dynamic_num = dynamic_num_cuda[buffer_index];
        temp_amp = 0.f;
        temp_amp1 = 0.f;
        double2* save_buf2 = reinterpret_cast<double2*>(save_buf[buffer_index]);
        double2 save2 = save_buf2[i];
        double sum = save2.x;
        double sum1 = save2.y;
        
        if (checkpoint<__double2ull_rn(dynamic_bufferlength_cuda)){
                double freqr = ratioks *ratioks *ratioks * __fma_rn(__fma_rn(6.,ratioks,-15.) , ratioks,10.);
                for (int j = 0; (j<dynamic_num- start_index && j<maximal_index); j++){
                float old_amp = amp[j];
                float amp_ramper = __fmaf_rn(newamp[j] - old_amp, static_cast<float>(ratioks),old_amp);
                double phase_c =  fma(k * illSamplerate_cuda , fma(frequencyDiff[j] , ratiok,startFrequency[j]),phi[j]);
                double phase_c1 = fma((k+1.) * illSamplerate_cuda , fma( frequencyDiff[j] , ratiok1,startFrequency[j]),phi[j]);
                float amp_factor = amp_map[static_cast<int>(fmaf(fmaf(static_cast<float>(frequencyDiff[j]),static_cast<float>(freqr),static_cast<float>(startFrequency[j])),map_interval_cuda,0.5f))];
                half2 temp =  h2sin (__floats2half2_rn(M_2PI*modfeq(phase_c),M_2PI*modfeq(phase_c1)));
                temp_amp1 += __high2float(temp)*amp_factor*amp_ramper;
                temp_amp += __low2float(temp)*amp_factor*amp_ramper;
                coefficient_corrector += amp_factor;
            }
        }else if(checkpoint==__double2ull_rn(dynamic_bufferlength_cuda)){
            for (int j = 0; (j<dynamic_num- start_index && j<maximal_index); j++){
                float amp_factor = amp_map[static_cast<int>(fmaf(static_cast<float>(frequencyDiff[j])+static_cast<float>(startFrequency[j]),map_interval_cuda,0.5f))];
                double phase_c =  fma(k * illSamplerate_cuda , fma(frequencyDiff[j] , ratiok,startFrequency[j]),phi[j]);
                double phase_c1 = modfeq(fma(fma(frequencyDiff[j],0.5,startFrequency[j]),dynamic_bufferlength_cuda*illSamplerate_cuda,phi[j]));
                half2 temp =  h2sin (__floats2half2_rn(M_2PI*modfeq(phase_c),M_2PI*modfeq(phase_c1)));
                temp_amp1 += __high2float(temp)*amp_factor*newamp[j];
                temp_amp += __low2float(temp)*amp_factor*newamp[j];
                coefficient_corrector += amp_factor;
            }
        }else{ 
            for (int j = 0; (j<dynamic_num- start_index && j<maximal_index); j++){
                double temp_factor = (startFrequency[j]+frequencyDiff[j]) * illSamplerate_cuda;
                float amp_factor = amp_map[static_cast<int>(fmaf(static_cast<float>(frequencyDiff[j])+static_cast<float>(startFrequency[j]),map_interval_cuda,0.5f))];
                double phase_c = fma(temp_factor , (k-dynamic_bufferlength_cuda) ,fma(fma(frequencyDiff[j],0.5,startFrequency[j]),dynamic_bufferlength_cuda*illSamplerate_cuda,phi[j]));
                double phase_c1 = phase_c + temp_factor;
                half2 temp =  h2sin (__floats2half2_rn(M_2PI*modfeq(phase_c),M_2PI*modfeq(phase_c1)));
                temp_amp1 += __high2float(temp)*amp_factor*newamp[j];
                temp_amp += __low2float(temp)*amp_factor*newamp[j];
                coefficient_corrector += amp_factor;
            }
        }
        coefficient_corrector = 32767.f/coefficient_corrector*static_cast<float>(dynamic_num) * static_cast<float>(istatic_num_cuda[buffer_index]);
        double2* sum_buf2 = reinterpret_cast<double2*>(sum_buf[buffer_index]);
        sum_buf2[i] = {static_cast<double>(fmaf(temp_amp,coefficient_corrector,static_cast<float>(sum))),static_cast<double>(fmaf(temp_amp1,coefficient_corrector,static_cast<float>(sum1)))};
        __syncthreads();
    }
}




__global__ void DoubleCopier(double**__restrict__ source,double**destination){
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    for (int buffer_index=0;buffer_index<channel_num_cuda;buffer_index++){
        destination[buffer_index][i] = source[buffer_index][i];
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
        cudaFree(temp_buffer[i]);
        temp_buffer[i] = NULL;
    }
    cudaFree(dFreq_cuda);
    dFreq_cuda = NULL;
    cudaFree(destination_buffer_cuda);
    destination_buffer_cuda = NULL;
    cudaFree(dDiff_cuda);
    dDiff_cuda = NULL;
    cudaFree(dphi_cuda);
    dphi_cuda = NULL;
    cudaFree(damp_cuda);
    damp_cuda = NULL;
    cudaFree(summed_buffer_cuda);
    summed_buffer_cuda = NULL;
    cudaFree(saved_buffer_cuda);
    saved_buffer_cuda = NULL;
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
    
    CUDA_CHECK(cudaMalloc ((void**)&real_destination_freq_cuda, dynamic_total*sizeof(double))); //Configure software buffer
    CUDA_CHECK(cudaMemcpy(real_destination_freq_cuda,real_destination_freq,dynamic_total*sizeof(double),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc ((void**)&dynamic_list_cuda, dynamic_total*sizeof(int))); //Configure software buffer
    CUDA_CHECK(cudaMalloc ((void**)&static_list_cuda, sizeof(static_list))); //Configure software buffer
    CUDA_CHECK(cudaMalloc ((void**)&saved_buffer_cuda, sizeof(saved_buffer))); //Configure software buffer
    CUDA_CHECK(cudaMemcpy (static_list_cuda, static_list,sizeof(static_list),cudaMemcpyHostToDevice)); //Configure software buffer
    CUDA_CHECK(cudaMemcpy (dynamic_list_cuda, dynamic_list,dynamic_total*sizeof(int),cudaMemcpyHostToDevice)); //Configure software buffer
    CUDA_CHECK(cudaMemcpy (saved_buffer_cuda, saved_buffer,sizeof(saved_buffer),cudaMemcpyHostToDevice)); //Configure software buffer
    CUDA_CHECK(cudaMemcpyToSymbol(dynamic_num_cuda, &dynamic_num, sizeof(dynamic_num)));

    double dylength = dynamic_buffersize/2;
    CUDA_CHECK(cudaMemcpyToSymbol(dynamic_bufferlength_cuda, &dylength, sizeof(dylength)));
    double_temp = 1./dylength;
    CUDA_CHECK(cudaMemcpyToSymbol (idynamic_bufferlength_cuda, &double_temp, sizeof(double))); //Configure software buffer
    double_temp = dynamic_loopcount;
    CUDA_CHECK(cudaMemcpyToSymbol (dynamic_loopcount_cuda, &double_temp, sizeof(double_temp))); //Configure software buffer
    CUDA_CHECK(cudaMalloc ((void**)&dFreq_cuda, dynamic_total*sizeof(double))); //Configure software buffer
    CUDA_CHECK(cudaMalloc ((void**)&dDiff_cuda, dynamic_total*sizeof(double))); //Configure software buffer
    CUDA_CHECK(cudaMalloc ((void**)&dphi_cuda, dynamic_total*sizeof(double))); //Configure software buffer
    CUDA_CHECK(cudaMalloc ((void**)&damp_cuda, dynamic_total*sizeof(float))); //Configure software buffer
    for (int i =0; i< lNumCh;i++){
        CUDA_CHECK(cudaMalloc ((void **)&temp_buffer[i], sizeof(double)/sizeof(short)*lBytesPerChannelInNotifySize)); //Configure software buffer
    }
    CUDA_CHECK(cudaMalloc ((void**)&temp_buffer_cuda, sizeof(temp_buffer))); //Configure software buffer
    CUDA_CHECK(cudaMemcpy (temp_buffer_cuda, temp_buffer,sizeof(temp_buffer),cudaMemcpyHostToDevice)); //Configure software buffer
    return 0;
}