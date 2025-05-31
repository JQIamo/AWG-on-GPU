#   include "cuda_functions.h"
__device__ __constant__ unsigned int static_num_cuda[4];
__device__ __constant__ int    channel_num_cuda;
__device__ __constant__ double llSamplerate_cuda;
__device__ __constant__ double illSamplerate_cuda;
__device__ __constant__ double ipower_normalizer_cuda[4];
__device__ __constant__ double idynamic_bufferlength_cuda;
__device__ __constant__ size_t static_bufferlength_cuda;
__device__ __constant__ float map_interval_cuda;
__device__ __constant__ int    dynamic_num_cuda[4];
__device__ __constant__ double dynamic_bufferlength_cuda;
__device__ __constant__ double dynamic_loopcount_cuda;
__device__ __constant__ unsigned int tone_count_cuda[5];
__device__ __constant__ unsigned int dynamic_tone_count_cuda[5];
__device__ double istatic_num_cuda[4];
unsigned int dynamic_total = 0;
unsigned int static_total = 0;
bool not_arrived = 1;
int * update_index_map_cuda=nullptr;
drv_handle hCard;
size_t static_length;
size_t lBytesPerChannelInNotifySize;
size_t int_temp;
double double_temp;
cudaError_t eCudaErr = cudaSuccess;

double* summed_buffer[4];
double* saved_buffer[4];
short* dynamic_saved_buffer[4];
double** summed_buffer_cuda;
double** saved_buffer_cuda;
float * amp_map_cuda;
float * amp_map_static_cuda;
float total_divider[4];
float * total_divider_cuda;
float amp_map_static[16384];
double* static_buffer_cuda;
double * real_static_freq_cuda;
double real_static_freq[16384];
// ------Dynamics----------------------------
double real_destination_freq[16384];
unsigned int dynamic_buffersize;
double* real_destination_freq_cuda;
int * dynamic_list_cuda;
int * static_list_cuda;
float * amp_list_cuda;
float * final_amp_list_cuda;
double * phase_list_cuda;
double * new_phase_list_cuda;
int dynamic_loopcount;


__global__ void printer (double* __restrict__ list,int count){
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i<count) printf("%d,%f\n",i,list[i]);
}

__global__ void printer (short* __restrict__ list,int count){
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i<count) printf("%d,%d\n",i,list[i]);
}

void StaticAmpMapper(double * __restrict__ frequency, const float * __restrict__ amp_map, float * amp_map_static, float *total_divider){
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


__global__ void StaticWaveGeneration (double* __restrict__ frequency, double* pnOut,double** sumOut,double*phase_list)
{
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    double dump;
    for (int ch=0;ch<channel_num_cuda;ch++){
        double sum = 0.;
        double ipower_normalizer = ipower_normalizer_cuda[ch];
        double amp_factor = 32767. *ipower_normalizer;
        double ratio = static_cast<double>(i);
        for (int j=0;j<static_num_cuda[ch];j++){
            int index = tone_count_cuda[ch]+j;
            double phi = -0.5*ipower_normalizer*static_cast<double>(j*j);
            if (i==0) phase_list[index] = phi;
            double ampl =   sinpi (2.*modf(fma(frequency[index] ,ratio ,phi),&dump))*amp_factor;
            pnOut[index*static_bufferlength_cuda+i] =ampl;
            sum += ampl;
        }
        sumOut[ch][i] = sum;
    }
}

__global__ void StaticWaveGeneration_update (double* __restrict__ frequency, double* pnOut,double** sumOut,double* __restrict__ phase_list)
{
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    double dump;
    for (int ch=0;ch<channel_num_cuda;ch++){
        double sum = 0.;
        double ipower_normalizer = ipower_normalizer_cuda[ch];
        double amp_factor = 32767. *ipower_normalizer;
        double ratio = static_cast<double>(i)* illSamplerate_cuda;
        for (int j=0;j<static_num_cuda[ch];j++){
                int index = tone_count_cuda[ch]+j;
                double ampl =   sinpi (2.*modf(fma(frequency[index] ,ratio ,phase_list[index]),&dump))*amp_factor;
                pnOut[index*static_bufferlength_cuda+i] =ampl;
                sum += ampl;
        }
        sumOut[ch][i] = sum;
    }
}

__global__ void StaticWaveGeneration_amp_mapped (double* __restrict__ frequency, float* __restrict__ amp_map_static,float* __restrict__ total_divider, double* pnOut,double** sumOut,double*phase_list)
{
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    double dump;
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
            double ampl =   sinpi (2.*modf(fma(freq ,ratio ,phi),&dump))*static_cast<double>(amp_map_static[index])*divider;
            pnOut[index*static_bufferlength_cuda+i] =ampl;
            sum += ampl;
        }
        sumOut[ch][i] = sum;
    }
}

__global__ void StaticWaveGeneration_amp (double* __restrict__ frequency, float* __restrict__ amp, double* pnOut,double** sumOut,double*phase_list)
{
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    double dump;
    for (int ch=0;ch<channel_num_cuda;ch++){
        double sum = 0.;
        double ipower_normalizer = ipower_normalizer_cuda[ch];
        double amp_factor = 32767. *ipower_normalizer;
        double ratio = static_cast<double>(i)* illSamplerate_cuda;
        for (int j=0;j<static_num_cuda[ch];j++){
            int tone_count = static_cast<int>(tone_count_cuda[ch]);
            int index = tone_count+j;
            double phi = 0.;
            for (int k=0;k<j;k++){
                phi += (k-j)*amp[tone_count+k]*ipower_normalizer;
            }
            if (i==0) {phase_list[index] = phi;}
            double ampl = sinpi (2.*modf(fma(frequency[index] ,ratio ,phi),&dump))*amp_factor*amp[index];
            pnOut[index*static_bufferlength_cuda+i] =ampl;
            sum += ampl;
        }
        sumOut[ch][i] = sum;
    }
}

__global__ void StaticWaveGeneration_amp_amp_mapped (double* __restrict__ frequency, double* __restrict__ amp, float* __restrict__ amp_map_static,float* __restrict__ total_divider, double* pnOut,double** sumOut,double*phase_list)
{
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    double dump;
    for (int ch=0;ch<channel_num_cuda;ch++){
        double sum = 0.;
        double ipower_normalizer = ipower_normalizer_cuda[ch];
        double divider = 32767.*static_cast<double>(total_divider[ch]);
        double ratio = static_cast<double>(i)* illSamplerate_cuda;
        for (int j=0;j<static_num_cuda[ch];j++){
            int tone_count = static_cast<int>(tone_count_cuda[ch]);
            int index = tone_count+j;
            double freq = frequency[index];
            double phi = 0.;
            for (int k=0;k<j;k++){
                phi += (k-j)*amp[tone_count+k]*ipower_normalizer;
            }
            if (i==0) phase_list[index] = phi;
            double ampl =   sinpi (2.*modf(fma(freq ,ratio ,phi),&dump))*static_cast<double>(amp_map_static[index])*divider*amp[index];
            pnOut[index*static_bufferlength_cuda+i] =ampl;
            sum += ampl;
        }
        sumOut[ch][i] = sum;
    }
}

__global__ void StaticWaveGeneration_update_amp (double* __restrict__ frequency, float* __restrict__ amp, double* pnOut,double** sumOut,double* __restrict__ phase_list)
{
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    double dump;
    for (int ch=0;ch<channel_num_cuda;ch++){
        double sum = 0.;
        double ipower_normalizer = ipower_normalizer_cuda[ch];
        double amp_factor = 32767. *ipower_normalizer;
        double ratio = static_cast<double>(i)* illSamplerate_cuda;
        for (int j=0;j<static_num_cuda[ch];j++){
            int tone_count = static_cast<int>(tone_count_cuda[ch]);
            int index = tone_count+j;
            double ampl = sinpi (2.*modf(fma(frequency[index] ,ratio ,phase_list[index]),&dump))*amp_factor*amp[index];
            pnOut[index*static_bufferlength_cuda+i] =ampl;
            sum += ampl;
        }
        sumOut[ch][i] = sum;
    }
}


__global__ void phase_reorder_update(int* __restrict__ indexmap,double*__restrict__ newphaselist,double*phaselist,int length){
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
    
    eCudaErr = cudaMalloc ((void**)&real_static_freq_cuda, static_total*sizeof(double)); //Configure software buffer
    if (eCudaErr != cudaSuccess)
        {
        printf ("Allocating real_static_freq_cuda on GPU failed: %s\n",cudaGetErrorString(eCudaErr));
        return 1;
        }    
    eCudaErr = cudaMemcpy(real_static_freq_cuda,real_static_freq,static_total*sizeof(double),cudaMemcpyHostToDevice);
    if (eCudaErr != cudaSuccess)
        {
        printf ("cudaMemcpy real_static_freq_cuda on GPU failed: %s\n",cudaGetErrorString(eCudaErr));
        return 1;
        }       
    for (int i = 0; i < lNumCh; i++){
        eCudaErr = cudaMalloc ((void **)&summed_buffer[i], sizeof(double)/sizeof(short)*lBytesPerChannelInNotifySize); //Configure software buffer
        if (eCudaErr != cudaSuccess)
            {
            printf ("Allocating summed_buffer on GPU failed: %s\n",cudaGetErrorString(eCudaErr));
            return 1;   
            }
    }
    
    double_temp = 1./llSamplerate;
    eCudaErr = cudaMemcpyToSymbol(illSamplerate_cuda,&double_temp,sizeof(double));
    if (eCudaErr != cudaSuccess)
        {
        printf ("cudaMemcpy illSamplerate_cuda on GPU failed: %s\n",cudaGetErrorString(eCudaErr));
        return 1;
        } 
    eCudaErr = cudaMemcpyToSymbol(tone_count_cuda,tone_count,5*sizeof(unsigned int));
    if (eCudaErr != cudaSuccess) 
        {
        printf ("cudaMemcpy tone_count_cuda on GPU failed: %s\n",cudaGetErrorString(eCudaErr));
        return 1;
        } 
    eCudaErr = cudaMemcpyToSymbol(static_num_cuda,static_num,sizeof(static_num));
    if (eCudaErr != cudaSuccess)
        {
        printf ("cudaMemcpy static_num_cuda on GPU failed: %s\n",cudaGetErrorString(eCudaErr));
        return 1;
        } 
    double double_array_temp[4];
    for (int i = 0; i < 4; i++) {
        if (static_num[i]) {double_array_temp[i] = 1./static_num[i];}
        else{double_array_temp[i]=0;}
    }
    
    eCudaErr = cudaMemcpyToSymbol(dynamic_tone_count_cuda,&dynamic_tone_count,5*sizeof(unsigned int));
    if (eCudaErr != cudaSuccess)
        {
        printf ("cudaMemcpy dynamic_tone_count_cuda on GPU failed: %s\n",cudaGetErrorString(eCudaErr));
        return 1;
        } 
    double ipower_normalizer[4];
    for (int i = 0; i < 4; i++) {ipower_normalizer[i] = 1./power_normalizer[i];} 
    eCudaErr = cudaMemcpyToSymbol(ipower_normalizer_cuda,&ipower_normalizer,sizeof(ipower_normalizer));
        if (eCudaErr != cudaSuccess)
            {
            printf ("cudaMemcpy ipower_normalizer_cuda on GPU failed: %s\n",cudaGetErrorString(eCudaErr));
            return 1;
            } 
    eCudaErr = cudaMemcpyToSymbol(istatic_num_cuda,double_array_temp,4*sizeof(double));
    if (eCudaErr != cudaSuccess)
        {
        printf ("cudaMemcpy istatic_num_cuda on GPU failed: %s\n",cudaGetErrorString(eCudaErr));
        return 1;
        } 
    eCudaErr = cudaMalloc ((void**)&static_buffer_cuda, (unsigned long long)sizeof(double)/sizeof(short)*static_total*lBytesPerChannelInNotifySize); //Configure software buffer
    if (eCudaErr != cudaSuccess)
        {
        printf ("Allocating static_buffer_cuda on GPU failed: %s\n",cudaGetErrorString(eCudaErr));
        return EXIT_FAILURE;
        }
        printf("Malloced size: %llu\n",(unsigned long long)sizeof(double)/sizeof(short)*static_total*lBytesPerChannelInNotifySize);
    eCudaErr = cudaMalloc ((void**)&summed_buffer_cuda, sizeof(summed_buffer)); //Configure software buffer
    if (eCudaErr != cudaSuccess)
        {
        printf ("Allocating summed_buffer_cuda on GPU failed: %s\n",cudaGetErrorString(eCudaErr));

        return EXIT_FAILURE;
        }
    eCudaErr = cudaMemcpy (summed_buffer_cuda, summed_buffer,sizeof(summed_buffer),cudaMemcpyHostToDevice); //Configure software buffer
    if (eCudaErr != cudaSuccess)
        {
        printf ("cudaMemcpy summed_buffer_cuda on GPU failed: %s\n",cudaGetErrorString(eCudaErr));

        return 1;
}
    if (phase_list_cuda == NULL){
        eCudaErr = cudaMalloc ((void**)&phase_list_cuda, static_total*sizeof(double)); //Configure software buffer
        if (eCudaErr != cudaSuccess)
            {
            printf ("Allocating phase_list_cuda on GPU failed: %s\n",cudaGetErrorString(eCudaErr));
            return EXIT_FAILURE;
            }
    
     eCudaErr = cudaMalloc ((void**)&new_phase_list_cuda, static_total*sizeof(double)); //Configure software buffer
    if (eCudaErr != cudaSuccess)
        {
        printf ("Allocating new_phase_list_cuda on GPU failed: %s\n",cudaGetErrorString(eCudaErr));
        return EXIT_FAILURE;
        }
        }
    eCudaErr = cudaMemcpyToSymbol(channel_num_cuda,&lNumCh,sizeof(int));
    if (eCudaErr != cudaSuccess)
        {
        printf ("cudaMemcpyToSymbol channel_num_cuda on GPU failed: %s\n",cudaGetErrorString(eCudaErr));

        return EXIT_FAILURE;
        }
    
    eCudaErr = cudaMemcpyToSymbol(static_bufferlength_cuda, &static_length, sizeof(static_length));
    if (eCudaErr != cudaSuccess)
        {
        printf ("cudaMemcpyToSymbol static_bufferlength_cuda on GPU failed: %s\n",cudaGetErrorString(eCudaErr));

        return EXIT_FAILURE;
        }
    if (amp_flag){
        eCudaErr = cudaMalloc ((void**)&amp_list_cuda, static_total*sizeof(float)); //Configure software buffer
        if (eCudaErr != cudaSuccess)
            {
            printf ("Allocating amp_list_cuda on GPU failed: %s\n",cudaGetErrorString(eCudaErr));
            return EXIT_FAILURE;
            }
        eCudaErr = cudaMemcpy (amp_list_cuda, amp_list,static_total*sizeof(float),cudaMemcpyHostToDevice); //Configure software buffer
        if (eCudaErr != cudaSuccess)
            {
            printf ("cudaMemcpy amp_list_cuda on GPU failed: %s\n",cudaGetErrorString(eCudaErr));
            return 1;
            }
        eCudaErr = cudaMalloc ((void**)&final_amp_list_cuda, static_total*sizeof(float)); //Configure software buffer
        if (eCudaErr != cudaSuccess)
            {
            printf ("Allocating final_amp_list_cuda on GPU failed: %s\n",cudaGetErrorString(eCudaErr));
            return EXIT_FAILURE;
            }
        eCudaErr = cudaMemcpy (final_amp_list_cuda, final_amp_list,static_total*sizeof(float),cudaMemcpyHostToDevice); //Configure software buffer
        if (eCudaErr != cudaSuccess)
            {
            printf ("cudaMemcpy final_amp_list_cuda on GPU failed: %s\n",cudaGetErrorString(eCudaErr));
            return 1;
            }
    }

    if (if_mapped){
        eCudaErr = cudaMemcpyToSymbol(map_interval_cuda, &map_interval, sizeof(map_interval));
        if (eCudaErr != cudaSuccess)
            {
            printf ("cudaMemcpyToSymbol map_interval_cuda on GPU failed: %s\n",cudaGetErrorString(eCudaErr));

            return EXIT_FAILURE;
            }
        eCudaErr = cudaMalloc((void**)&amp_map_cuda, sizeof(amp_freq_map));
        if (eCudaErr != cudaSuccess)
            {
            printf ("cudaMemcpyToSymbol amp_map_cuda on GPU failed: %s\n",cudaGetErrorString(eCudaErr));

            return EXIT_FAILURE;
            }
        eCudaErr = cudaMemcpy (amp_map_cuda, amp_freq_map,sizeof(amp_freq_map),cudaMemcpyHostToDevice); //Configure software buffer
        if (eCudaErr != cudaSuccess)
            {
            printf ("cudaMemcpy amp_map_cuda on GPU failed: %s\n",cudaGetErrorString(eCudaErr));

            return 1;
            }    
        eCudaErr = cudaMalloc ((void**)&amp_map_static_cuda, static_total*sizeof(float)); //Configure software buffer
        if (eCudaErr != cudaSuccess)
            {
            printf ("Allocating amp_map_static_cuda on GPU failed: %s\n",cudaGetErrorString(eCudaErr));
            return 1;
            }    
        eCudaErr = cudaMalloc ((void**)&total_divider_cuda,lNumCh*sizeof(float)); //Configure software buffer
        if (eCudaErr != cudaSuccess)
            {
            printf ("Allocating total_divider_cuda on GPU failed: %s\n",cudaGetErrorString(eCudaErr));
            return 1;
            }   
        eCudaErr = cudaMemcpy (amp_map_static_cuda, amp_map_static,static_total*sizeof(float),cudaMemcpyHostToDevice); //Configure software buffer
        if (eCudaErr != cudaSuccess)
            {
            printf ("cudaMemcpy amp_map_static_cuda on GPU failed: %s\n",cudaGetErrorString(eCudaErr));
    
            return 1;
            }    
        eCudaErr = cudaMemcpy (total_divider_cuda, total_divider,lNumCh*sizeof(float),cudaMemcpyHostToDevice); //Configure software buffer
        if (eCudaErr != cudaSuccess)
            {
            printf ("cudaMemcpy total_divider_cuda on GPU failed: %s\n",cudaGetErrorString(eCudaErr));
    
            return 1;
            }    
    }
    return 0;
}
