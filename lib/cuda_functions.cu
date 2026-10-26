#   include "cuda_functions.h"
__device__ __constant__ unsigned int static_num_cuda[4];
__device__ __constant__ int    channel_num_cuda;
__device__ __constant__ double llSamplerate_cuda;
__device__ __constant__ double illSamplerate_cuda;
__device__ __constant__ float ipower_normalizer_cuda[4];
__device__ __constant__ float imaximal_tweezer_number_cuda[4];
__device__ __constant__ double idynamic_bufferlength_cuda;
__device__ __constant__ unsigned int static_bufferlength_cuda;
__device__ __constant__ float map_interval_cuda;
__device__ __constant__ int    dynamic_num_cuda[4];
__device__ __constant__ unsigned int dynamic_bufferlength_cuda;
__device__ __constant__ int dynamic_loopcount_cuda;
__device__ __constant__ unsigned int tone_count_cuda[5];
__device__ __constant__ unsigned int dynamic_tone_count_cuda[5];
int * single_static_map_cuda=nullptr;
int * single_dynamic_map_cuda=nullptr;
__device__ double istatic_num_cuda[4];
unsigned int dynamic_total = 0;
unsigned int static_total = 0;
bool not_arrived = true;
int * update_index_map_cuda=nullptr;
drv_handle hCard = nullptr;
int static_length;
size_t lBytesPerChannelInNotifySize;
size_t int_temp;
double double_temp;
cudaError_t eCudaErr = cudaSuccess;

double* summed_buffer[4];
double* saved_buffer[4];
short* dynamic_saved_buffer[4]={nullptr,nullptr,nullptr,nullptr};
double** summed_buffer_cuda=nullptr;
double** saved_buffer_cuda=nullptr;
float * amp_map_cuda=nullptr;
float * amp_map_static_cuda=nullptr;
float total_divider[4];
float * total_divider_cuda=nullptr;
float amp_map_static[4*MAX_TWEEZERS_PER_CHANNEL];
double* static_buffer_cuda=nullptr;
float * real_static_freq_cuda=nullptr;
float * rearrange_freq_list_cuda=nullptr;
float real_static_freq[4*MAX_TWEEZERS_PER_CHANNEL];
// ------Dynamics----------------------------
float real_destination_freq[4*MAX_TWEEZERS_PER_CHANNEL];
float* real_destination_freq_cuda=nullptr;
int * dynamic_list_cuda=nullptr;
int * static_list_cuda=nullptr;
float * amp_list_cuda=nullptr;
float * final_amp_list_cuda=nullptr;
float * phase_list_cuda=nullptr;
float * new_phase_list_cuda=nullptr;
int dynamic_loopcount;
int new_index_map_cuda[MAX_TWEEZERS_PER_CHANNEL];


__device__ double modfeq(double x){
    return x-floor(x);
}
__device__ float modfeqf(float x){
    return x-floorf(x);
}

__global__ void printer (double* __restrict__ list,int count){
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i<count) printf("%lu,%f\n",i,list[i]);
}

__global__ void printer (short* __restrict__ list,int count){
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i<count) printf("%lu,%d\n",i,list[i]);
}

void StaticAmpMapper(float * __restrict__ frequency, const float * __restrict__ amp_map, float * amp_map_static, float *total_divider){
    for (int ch=0;ch<lNumCh;ch++){
        float sum = 0.;
        int tone_counter = static_cast<int>(tone_count[ch]);
        for (int j=0;j<static_num[ch];j++){
            int index = tone_counter+j;
            float mapped = amp_map[static_cast<int>(frequency[index]*map_interval+0.5)];
            sum += mapped;
            amp_map_static[index] = mapped;
        }
        total_divider[ch] = 1.f/sum;
    }
}

__global__ void StaticWaveGeneration (float* __restrict__ frequency, double* pnOut,double** sumOut,float*phase_list)
{
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    for (int ch=0;ch<channel_num_cuda;ch++){
        double sum = 0.;
        double ipower_normalizer = ipower_normalizer_cuda[ch];
        double amp_factor = 32767.*imaximal_tweezer_number_cuda[ch];
        double ratio = static_cast<double>(i)*illSamplerate_cuda;
        for (int j=0;j<static_num_cuda[ch];j++){
            int index = tone_count_cuda[ch]+j;
            double phi = -0.5*ipower_normalizer*static_cast<double>(j*j);
            if (i==0) phase_list[index] = phi;
            double ampl = sin2pi_fast(fma(static_cast<double>(frequency[index]) ,ratio ,phi))*amp_factor;
            pnOut[index*static_bufferlength_cuda+i] =ampl;
            sum += ampl;
        }
        sumOut[ch][i] = sum;
    }
}


__global__ void StaticWaveGeneration_amp (float* __restrict__ frequency, float* __restrict__ amp, double* pnOut,double** sumOut,float*phase_list)
{
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    // double dump;
    for (int ch=0;ch<channel_num_cuda;ch++){
        double sum = 0.;
        double ipower_normalizer = ipower_normalizer_cuda[ch];
        // double amp_factor = 32767. *ipower_normalizer;
        double amp_factor = 32767.*imaximal_tweezer_number_cuda[ch];
        double ratio = static_cast<double>(i)* illSamplerate_cuda;
        for (int j=0;j<static_num_cuda[ch];j++){
            int index = tone_count_cuda[ch]+j;
            double phi = -0.5*ipower_normalizer*static_cast<double>(j*j);
            if (i==0) {phase_list[index] = phi;}
            double ampl =   sin2pi_fast (fma(static_cast<double>(frequency[index]) ,ratio ,phi))*amp_factor*amp[index];
            pnOut[index*static_bufferlength_cuda+i] =ampl;
            sum += ampl;
        }
        sumOut[ch][i] = sum;
    }
}


__global__ void phase_reorder_update(int* __restrict__ indexmap,float*__restrict__ newphaselist,float*phaselist,int length){
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i<length)phaselist[i]=newphaselist[indexmap[i]];
}


__global__ void StaticMux ( double** __restrict__ buffer,short* pnOut)
    {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (channel_num_cuda==4){
        short4 *pnOut4 = reinterpret_cast<short4*>(pnOut);
        pnOut4[i] = {static_cast<short>(buffer[0][i]),static_cast<short>(buffer[1][i]),static_cast<short>(buffer[2][i]),static_cast<short>(buffer[3][i])};
    }else if (channel_num_cuda==2){
        short2 *pnOut2 = reinterpret_cast<short2*>(pnOut);
        pnOut2[i] = {static_cast<short>(buffer[0][i]),static_cast<short>(buffer[1][i])};
    }else{
        pnOut[i] = static_cast<short>(buffer[0][i]);
    }
}

__global__ void ShakeWaveGeneration (float* __restrict__ frequency, double shake_amp, double shake_freq, double* pnOut,double** sumOut,float*phase_list)
{
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    // double dump;
    for (int ch=0;ch<channel_num_cuda;ch++){
        double sum = 0.;
        double ipower_normalizer = ipower_normalizer_cuda[ch];
        double amp_factor = 32767. *imaximal_tweezer_number_cuda[ch];
        double ratio = static_cast<double>(i)*illSamplerate_cuda;
        for (int j=0;j<static_num_cuda[ch];j++){
            int index = tone_count_cuda[ch]+j;
            double phi = -0.5*ipower_normalizer*static_cast<double>(j*j);
            if (i==0) phase_list[index] = phi;
            double ampl =   sin2pi_fast (
                                    fma(shake_amp/shake_freq*0.5*M_1_PI,(1.-cos2pi_fast(shake_freq*ratio)),fma(static_cast<double>(frequency[index]) ,ratio ,phi))
                                    )*amp_factor;
            pnOut[index*static_bufferlength_cuda+i] =ampl;
            sum += ampl;
        }
        sumOut[ch][i] = sum;
    }
}

__global__ void ShakeWaveGeneration_amp (float* __restrict__ frequency, float* __restrict__ amp, double shake_amp, double shake_freq, double* pnOut,double** sumOut,float*phase_list)
{
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    // double dump;
    for (int ch=0;ch<channel_num_cuda;ch++){
        double sum = 0.;
        double ipower_normalizer = ipower_normalizer_cuda[ch];
        double amp_factor = 32767. *imaximal_tweezer_number_cuda[ch];
        double ratio = static_cast<double>(i)* illSamplerate_cuda;
        for (int j=0;j<static_num_cuda[ch];j++){
            int tone_count = static_cast<int>(tone_count_cuda[ch]);
            int index = tone_count+j;
            double phi = -0.5*ipower_normalizer*static_cast<double>(j*j);
            if (i==0) {phase_list[index] = phi;}
            double ampl =   sin2pi_fast (
                                    fma(shake_amp/shake_freq*0.5*M_1_PI,(1.-cos2pi_fast(shake_freq*ratio)),fma(static_cast<double>(frequency[index]) ,ratio ,phi))
                                    )*amp_factor*amp[index];
            pnOut[index*static_bufferlength_cuda+i] =ampl;
            sum += ampl;
        }
        sumOut[ch][i] = sum;
    }
}


__global__ void SingleStaticRecombiner(int ch, int *__restrict__ single_static_map_cuda, double* __restrict__ pnOut,double** sumOut)
{
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    double sum = 0.;
    for (int j=0;j<static_num_cuda[ch]-dynamic_num_cuda[ch];j++){
        int index = single_static_map_cuda[j];
        sum += pnOut[index*static_bufferlength_cuda+i];
    }
    sumOut[ch][i] = sum;
}

void tone_counter(int dynamic){
    unsigned int counter = 0;
    tone_count[0]=0;
    for (int i = 0; i < lNumCh; i++){
        counter += static_num[i];
        tone_count[i+1] = counter;
    }
    if (dynamic){
        counter = 0;
        dynamic_tone_count[0]=0;
        for (int i = 0; i < lNumCh; i++){
            counter += dynamic_num[i];
            dynamic_tone_count[i+1] = counter;
        }
    }
}

int staticBufferInit(){
    CUDA_CHECK(cudaMalloc ((void**)&real_static_freq_cuda, static_total*sizeof(float))); //Configure software buffer
    CUDA_CHECK(cudaMemcpy(real_static_freq_cuda,real_static_freq,static_total*sizeof(float),cudaMemcpyHostToDevice));
    for (int i = 0; i < lNumCh; i++){
        CUDA_CHECK(cudaMalloc ((void **)&summed_buffer[i], sizeof(double)/sizeof(short)*lBytesPerChannelInNotifySize)); //Configure software buffer
    }
    
    double_temp = 1./llSamplerate;
    CUDA_CHECK(cudaMemcpyToSymbol(illSamplerate_cuda,&double_temp,sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(tone_count_cuda,tone_count,5*sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpyToSymbol(static_num_cuda,static_num,sizeof(static_num)));
    double double_array_temp[4];
    for (int i = 0; i < 4; i++) {
        if (static_num[i]) {double_array_temp[i] = 1./static_num[i];}
        else{double_array_temp[i]=0;}
    }
    
    CUDA_CHECK(cudaMemcpyToSymbol(dynamic_tone_count_cuda,&dynamic_tone_count,5*sizeof(unsigned int)));
    float ipower_normalizer[4];
    for (int i = 0; i < 4; i++) {
        if (power_normalizer[i]<2.) {power_normalizer[i]=2.;}
        ipower_normalizer[i] = 1./power_normalizer[i];
        } 
    CUDA_CHECK(cudaMemcpyToSymbol(ipower_normalizer_cuda,&ipower_normalizer,sizeof(ipower_normalizer)));
    float imaximal_tweezer_number[4];
    for (int i = 0; i < 4; i++) {
        imaximal_tweezer_number[i] = 1./maximal_tweezer_number[i];
        } 
    CUDA_CHECK(cudaMemcpyToSymbol(imaximal_tweezer_number_cuda,&imaximal_tweezer_number,sizeof(imaximal_tweezer_number)));
    CUDA_CHECK(cudaMemcpyToSymbol(istatic_num_cuda,double_array_temp,4*sizeof(double)));
    CUDA_CHECK(cudaMalloc((void**)&static_buffer_cuda, (unsigned long long)sizeof(double)/sizeof(short)*static_total*lBytesPerChannelInNotifySize)); //Configure software buffer
    printf("Malloced size: %llu\n",(unsigned long long)sizeof(double)/sizeof(short)*static_total*lBytesPerChannelInNotifySize);
    CUDA_CHECK(cudaMalloc ((void**)&summed_buffer_cuda, sizeof(summed_buffer))); //Configure software buffer
    CUDA_CHECK(cudaMemcpy (summed_buffer_cuda, summed_buffer,sizeof(summed_buffer),cudaMemcpyHostToDevice)); //Configure software buffer
    if (phase_list_cuda == NULL){
        CUDA_CHECK(cudaMalloc ((void**)&phase_list_cuda, static_total*sizeof(float))); //Configure software buffer
        CUDA_CHECK(cudaMalloc ((void**)&new_phase_list_cuda, static_total*sizeof(float))); //Configure software buffer
    }
    CUDA_CHECK(cudaMemcpyToSymbol(channel_num_cuda,&lNumCh,sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(static_bufferlength_cuda, &static_length, sizeof(static_length)));
    if (amp_flag){
        CUDA_CHECK(cudaMalloc ((void**)&amp_list_cuda, static_total*sizeof(float))); //Configure software buffer
        CUDA_CHECK(cudaMemcpy (amp_list_cuda, amp_list,static_total*sizeof(float),cudaMemcpyHostToDevice)); //Configure software buffer
        CUDA_CHECK(cudaMalloc ((void**)&final_amp_list_cuda, static_total*sizeof(float))); //Configure software buffer
        CUDA_CHECK(cudaMemcpy (final_amp_list_cuda, final_amp_list,static_total*sizeof(float),cudaMemcpyHostToDevice)); //Configure software buffer
    }

    if (if_mapped){
        CUDA_CHECK(cudaMemcpyToSymbol(map_interval_cuda, &map_interval, sizeof(map_interval)));
        CUDA_CHECK(cudaMalloc((void**)&amp_map_cuda, sizeof(amp_freq_map)));
        CUDA_CHECK(cudaMemcpy (amp_map_cuda, amp_freq_map,sizeof(amp_freq_map),cudaMemcpyHostToDevice)); //Configure software buffer
        CUDA_CHECK(cudaMalloc ((void**)&amp_map_static_cuda, static_total*sizeof(float))); //Configure software buffer
        CUDA_CHECK(cudaMalloc ((void**)&total_divider_cuda,lNumCh*sizeof(float))); //Configure software buffer
        CUDA_CHECK(cudaMemcpy (amp_map_static_cuda, amp_map_static,static_total*sizeof(float),cudaMemcpyHostToDevice)); //Configure software buffer
        CUDA_CHECK(cudaMemcpy (total_divider_cuda, total_divider,lNumCh*sizeof(float),cudaMemcpyHostToDevice)); //Configure software buffer
    }
    return 0;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// The following are not used

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void StaticWaveGeneration_update (float* __restrict__ frequency, double* pnOut,double** sumOut,float* __restrict__ phase_list)
{
    // Not used
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    // double dump;
    for (int ch=0;ch<channel_num_cuda;ch++){
        double sum = 0.;
        double amp_factor = 32767. *imaximal_tweezer_number_cuda[ch];
        double ratio = static_cast<double>(i)* illSamplerate_cuda;
        for (int j=0;j<static_num_cuda[ch];j++){
            int index = tone_count_cuda[ch]+j;
            double ampl =   sin2pi_fast(fma(static_cast<double>(frequency[index]) ,ratio ,static_cast<double>(phase_list[index])))*amp_factor;
            pnOut[index*static_bufferlength_cuda+i] =ampl;
            sum += ampl;
        }
        sumOut[ch][i] = sum;
    }
}

__global__ void StaticWaveGeneration_amp_mapped (float* __restrict__ frequency, float* __restrict__ amp_map_static,float* __restrict__ total_divider, double* pnOut,double** sumOut,float*phase_list)
{
    // Not used yet. Need to be polished and tested
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    // double dump;
    for (int ch=0;ch<channel_num_cuda;ch++){
        double sum = 0.;
        double ipower_normalizer = ipower_normalizer_cuda[ch];
        double divider = 32767.*static_cast<double>(total_divider[ch]);
        double ratio = static_cast<double>(i)* illSamplerate_cuda;
        for (int j=0;j<static_num_cuda[ch];j++){
            int index = tone_count_cuda[ch]+j;
            double freq = frequency[index];
            double phi = -0.5*ipower_normalizer*static_cast<double>(j*j);
            if (i==0) phase_list[index] = phi;
            double ampl =   sin2pi_fast (fma(freq ,ratio ,phi))*static_cast<double>(amp_map_static[index])*divider;
            pnOut[index*static_bufferlength_cuda+i] =ampl;
            sum += ampl;
        }
        sumOut[ch][i] = sum;
    }
}

__global__ void StaticWaveGeneration_amp_amp_mapped (float* __restrict__ frequency, double* __restrict__ amp, float* __restrict__ amp_map_static,float* __restrict__ total_divider, double* pnOut,double** sumOut,float*phase_list)
{
    // Not used yet. Need to be polished and tested
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    // double dump;
    for (int ch=0;ch<channel_num_cuda;ch++){
        double sum = 0.;
        double ipower_normalizer = ipower_normalizer_cuda[ch];
        double divider = 32767.*static_cast<double>(total_divider[ch]);
        double ratio = static_cast<double>(i)* illSamplerate_cuda;
        for (int j=0;j<static_num_cuda[ch];j++){
            int tone_count = tone_count_cuda[ch];
            int index = tone_count+j;
            double freq = frequency[index];
            double phi = -0.5*ipower_normalizer*static_cast<double>(j*j);
            // double phi = 0.;
            // for (int k=0;k<j;k++){
            //     phi += (k-j)*amp[tone_count+k]*ipower_normalizer;
            // }
            if (i==0) phase_list[index] = phi;
            double ampl =   sin2pi_fast (fma(freq ,ratio ,phi))*static_cast<double>(amp_map_static[index])*divider*amp[index];
            pnOut[index*static_bufferlength_cuda+i] =ampl;
            sum += ampl;
        }
        sumOut[ch][i] = sum;
    }
}

__global__ void StaticWaveGeneration_update_amp (float* __restrict__ frequency, float* __restrict__ amp, double* pnOut,double** sumOut,float* __restrict__ phase_list)
{
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    // double dump;
    for (int ch=0;ch<channel_num_cuda;ch++){
        double sum = 0.;
        double amp_factor = 32767. *imaximal_tweezer_number_cuda[ch];
        double ratio = static_cast<double>(i)* illSamplerate_cuda;
        for (int j=0;j<static_num_cuda[ch];j++){
            int tone_count = static_cast<int>(tone_count_cuda[ch]);
            int index = tone_count+j;
            double ampl =   sin2pi_fast (fma(static_cast<double>(frequency[index]) ,ratio ,static_cast<double>(phase_list[index])))*amp_factor*amp[index];
            pnOut[index*static_bufferlength_cuda+i] =ampl;
            sum += ampl;
        }
        sumOut[ch][i] = sum;
    }
}
