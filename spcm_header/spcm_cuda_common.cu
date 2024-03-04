#include "spcm_cuda_common.h"

#include <cstdio>

// ----- CUDA include -----
#   include <cuda_runtime.h>

// CUDA-C includes
#   include <cuda.h>


// ----- Init CUDA device without RDMA support (e.g. on Windows). -----
// -- lCUDADevIdx:  index of CUDA device to be used
// -- return:       true if initialization succeeded, false otherwise
bool bInitCUDADevice (int lCUDADevIdx)
    {
    // ----- check for CUDA-capable devices -----
    int lCUDADeviceCount = 0;
    cudaError_t eCudaErr = cudaGetDeviceCount (&lCUDADeviceCount);
    if (eCudaErr != cudaSuccess)
        {
        printf ("ERROR in cudaGetDeviceCount(): %s\n", cudaGetErrorString(eCudaErr));
        return false;
        }

    if (lCUDADeviceCount == 0)
        {
        printf ("ERROR: there are no available devices that support CUDA\n");
        return false;
        }
    if (lCUDADevIdx >= lCUDADeviceCount)
        {
        printf ("ERROR: requested device %d, but only %d CUDA device(s) available\n", lCUDADevIdx, lCUDADeviceCount);
        return false;
        }

    printf ("Detected %d CUDA Capable device(s).\n", lCUDADeviceCount);
    cudaSetDevice (lCUDADevIdx);
    cudaDeviceProp stCUDADeviceProp;
    cudaGetDeviceProperties (&stCUDADeviceProp, lCUDADevIdx);

    printf("\nUsing device %d: \"%s\"\n", 0, stCUDADeviceProp.name);

    return true;
    }

#ifndef WIN32

// ----- Sets the CUDA device (GPU) to be used and allocates a buffer that's usable for RDMA. -----
// -- lCUDADevIdx:           index of CUDA device to be used
// -- qwDMABufferSize_bytes: size of the buffer that should be allocated
// -- return:                pointer to buffer of requested size if initialization succeeded, NULL otherwise
void* pvGetRDMABuffer (int lCUDADevIdx, size_t qwDMABufferSize_bytes)
    {
    // ----- check for CUDA-capable devices -----
    int lCUDADeviceCount = 0;
    cudaError_t eCudaErr = cudaGetDeviceCount (&lCUDADeviceCount);
    if (eCudaErr != cudaSuccess)
        {
        printf ("ERROR in cudaGetDeviceCount(): %s\n", cudaGetErrorString(eCudaErr));
        return NULL;
        }

    if (lCUDADeviceCount == 0)
        {
        printf ("ERROR: there are no available devices that support CUDA\n");
        return NULL;
        }
    if (lCUDADevIdx >= lCUDADeviceCount)
        {
        printf ("ERROR: requested device %d, but only %d CUDA device(s) available\n", lCUDADevIdx, lCUDADeviceCount);
        return NULL;
        }

    printf ("Detected %d CUDA Capable device(s).\n", lCUDADeviceCount);
    cudaSetDevice (lCUDADevIdx);
    cudaDeviceProp stCUDADeviceProp;
    cudaGetDeviceProperties (&stCUDADeviceProp, lCUDADevIdx);
    printf("\nUsing device %d: \"%s\"\n", 0, stCUDADeviceProp.name);

    // ----- we require at least CUDA 5.0 -----
    if (stCUDADeviceProp.major < 5)
        {
        printf ("ERROR: RDMA requires at least CUDA compute capability 5.0 (found: %d.%d)\n", stCUDADeviceProp.major, stCUDADeviceProp.minor);
        return NULL;
        }

    int lRDMASupported = 0;
    cudaDeviceGetAttribute (&lRDMASupported, cudaDevAttrGPUDirectRDMASupported, lCUDADevIdx);
    if (lRDMASupported == 0)
        {
        printf ("ERROR: device \"%s\" does not support RDMA", stCUDADeviceProp.name);
        return NULL;
        }

    // ----- allocate DMA buffer on GPU -----
    void* pvDMABuffer_gpu = NULL;
    if (strncmp (stCUDADeviceProp.name, "Xavier", 6) == 0)
        eCudaErr = cudaHostAlloc (&pvDMABuffer_gpu, qwDMABufferSize_bytes, cudaHostAllocDefault); // required to use cudaHostAlloc on Jetson
    else
        {
        eCudaErr = cudaMalloc (&pvDMABuffer_gpu, qwDMABufferSize_bytes);
        }
    if (eCudaErr != cudaSuccess)
        {
        printf ("ERROR in cudaMalloc(): %s\n", cudaGetErrorString(eCudaErr));
        return NULL;
        }

    // in GPUDirect RDMA scope should always be CU_POINTER_ATTRIBUTE_SYNC_MEMOPS
    unsigned int dwFlag = 1;
    CUresult eResult = cuPointerSetAttribute (&dwFlag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, (CUdeviceptr)pvDMABuffer_gpu);
    if (eResult != CUDA_SUCCESS)
        {
        const char* szError;
        cuGetErrorString (eResult, &szError);
        printf ("ERROR in cuPointerSetAttribute(ATTRIBUTE_SYNC_MEMOPS): %s\n", szError);
        cudaFree (pvDMABuffer_gpu);
        return NULL;
        }

    return pvDMABuffer_gpu;
    }

#endif

// ----- Returns error description for CUDA FFT error code -----
const char* szCudaGetErrorText (cufftResult eError)
    {
    switch (eError)
        {
        case CUFFT_SUCCESS:         return "CUFFT_SUCCESS";
        case CUFFT_INVALID_PLAN:    return "CUFFT_INVALID_PLAN";
        case CUFFT_ALLOC_FAILED:    return "CUFFT_ALLOC_FAILED";
        case CUFFT_INVALID_TYPE:    return "CUFFT_INVALID_TYPE";
        case CUFFT_INVALID_VALUE:   return "CUFFT_INVALID_VALUE";
        case CUFFT_INTERNAL_ERROR:  return "CUFFT_INTERNAL_ERROR";
        case CUFFT_EXEC_FAILED:     return "CUFFT_EXEC_FAILED";
        case CUFFT_SETUP_FAILED:    return "CUFFT_SETUP_FAILED";
        case CUFFT_INVALID_SIZE:    return "CUFFT_INVALID_SIZE";
        case CUFFT_UNALIGNED_DATA:  return "CUFFT_UNALIGNED_DATA";
        }

    return "<unknown>";
    }
