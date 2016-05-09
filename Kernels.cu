// Includes CUDA
#include <cuda_runtime.h>
#include <cstdio>
#include <cub/cub.cuh>
#include <moderngpu/kernel_mergesort.hxx>
#include "kernels.h"

const int vectorPerBlock = 1024;
const int vectorSize = 25;

// popcll is slower, don't use it
/*__global__ void distanceKernel(unsigned short* keyOutput, unsigned int *valueOutput, unsigned int *query, cudaTextureObject_t texObj, unsigned int texHeight)
{
    int tu = blockIdx.x;
    int tv = threadIdx.x;

    if (tu <texHeight && tv <vectorPerBlock)
    {
        __shared__ unsigned int queryLocal[vectorSize];

        if (tv < vectorSize)
        {
            queryLocal[tv] = query[tv];
        }

        __syncthreads();

        unsigned short count = 0;
        unsigned int offset = tv * vectorSize;

        for (int i = 0; i < vectorSize ; i += 2)
        {
            unsigned int m[2] = {0};
            m[0] = tex2D<unsigned int>(texObj, offset + i, tu);
            m[1] = tex2D<unsigned int>(texObj, offset + i + 1, tu);
            count += __popcll(*((unsigned long long*)m) ^ *((unsigned long long*)&queryLocal[i]));
        }

        if (vectorSize % 2 == 1)
        {
            count += __popc(tex2D<unsigned int>(texObj, offset + vectorSize - 1, tu) ^ queryLocal[vectorSize - 1]);
        }

        unsigned int id = tu*vectorPerBlock + tv;

        keyOutput[id] = count;
        valueOutput[id] = id;
    }
}*/


__global__ void distanceKernel(unsigned short* keyOutput, unsigned int *valueOutput, unsigned int *query, cudaTextureObject_t texObj, unsigned int texHeight)
{
    int tu = blockIdx.x;
    int tv = threadIdx.x;

    if (tu <texHeight && tv <vectorPerBlock)
    {
        __shared__ unsigned int queryLocal[vectorSize];

        if (tv < vectorSize)
        {
            queryLocal[tv] = query[tv];
        }

        __syncthreads();

        unsigned short count = 0;

        for (int i = 0; i<vectorSize;++i)
        {
            unsigned int m = tex2D<unsigned int>(texObj, tv * vectorSize + i, tu);
            count += __popc(m ^ queryLocal[i]);

        }

        unsigned int id = tu*vectorPerBlock + tv;

        keyOutput[id] = count;
        valueOutput[id] = id;
    }
}

static enum SortAlgorithm sortAlgorithm = CUB_RADIX_SORT;
static int texHeight = 0;
static cudaArray *d_matrixArray = 0;
static cudaTextureObject_t matrixTexObj = 0;
static unsigned int *d_query = 0;
static unsigned short *d_keyResultArray = 0;
static unsigned int *d_valueResultArray = 0;
static size_t  tempStorageBytes  = 0;
static void    *tempStorage_device     = NULL;
static cub::DoubleBuffer<unsigned short> d_keys;
static cub::DoubleBuffer<unsigned int> d_values;
static mgpu::standard_context_t *context;
static unsigned int num_items = 0;


void kNN_cleanUp()
{
    if (matrixTexObj)
    {
        cudaDestroyTextureObject(matrixTexObj);
    }

    if (d_matrixArray)
    {
        cudaFreeArray(d_matrixArray);
    }

    if (d_query)
    {
        cudaFree(d_query);
    }

    if (d_keys.d_buffers[0] || d_keys.d_buffers[1])
    {
        cudaFree(d_keys.d_buffers[0]);
        cudaFree(d_keys.d_buffers[1]);
    }
    else if(d_keyResultArray)
    {
        cudaFree(d_keyResultArray);
    }

    if (d_values.d_buffers[0] || d_values.d_buffers[1])
    {
        cudaFree(d_values.d_buffers[0]);
        cudaFree(d_values.d_buffers[1]);
    }
    else if(d_valueResultArray)
    {
        cudaFree(d_valueResultArray);
    }

    if (tempStorage_device)
    {
        cudaFree(tempStorage_device);
    }

    if (context)
    {
        delete context;
    }

}


bool kNN_init(unsigned int *matrixBuffer, const unsigned int _num_items, enum SortAlgorithm _sortAlgorithm)
{
    cudaError error = cudaSuccess;

    num_items = _num_items;
    sortAlgorithm = _sortAlgorithm;
    texHeight = (num_items / vectorPerBlock) + ((num_items % vectorPerBlock)?1:0);

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0 , 0, 0, cudaChannelFormatKindUnsigned);


    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));

    error = cudaMallocArray(&d_matrixArray, &channelDesc, vectorSize * vectorPerBlock, texHeight);
    if (cudaSuccess != error)
    {
        printf("can't allocate matrixArray [%u]\n", error);
        kNN_cleanUp();
        return false;
    }

    error = cudaMemcpyToArray(d_matrixArray, 0, 0, matrixBuffer, num_items * vectorSize * sizeof(unsigned int), cudaMemcpyHostToDevice);
    if (cudaSuccess != error)
    {
        printf("can't memcpy matrixArray [%u]\n", error);
        kNN_cleanUp();
        return false;
    }

    // Specify texture
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = d_matrixArray;

    // Specify texture object parameters
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    // Create texture object
    error = cudaCreateTextureObject(&matrixTexObj, &resDesc, &texDesc, NULL);
    if (cudaSuccess != error)
    {
        printf("can't allocate texture object [%u]\n", error);
        kNN_cleanUp();
        return false;
    }

    error = cudaMalloc(&d_query, sizeof(unsigned int) * vectorSize);
    if (cudaSuccess != error)
    {
        printf("can't allocate for query\n");
        kNN_cleanUp();
        return false;
    }



    if (sortAlgorithm == CUB_RADIX_SORT)
    {
        cub::CachingDeviceAllocator cubAllocator(true);
        CubDebugExit(cubAllocator.DeviceAllocate((void**)&d_keys.d_buffers[0], sizeof(unsigned short) * num_items));
        CubDebugExit(cubAllocator.DeviceAllocate((void**)&d_values.d_buffers[0], sizeof(unsigned int) * num_items));
        CubDebugExit(cubAllocator.DeviceAllocate((void**)&d_keys.d_buffers[1], sizeof(unsigned short) * num_items));
        CubDebugExit(cubAllocator.DeviceAllocate((void**)&d_values.d_buffers[1], sizeof(unsigned int) * num_items));
        CubDebugExit(cub::DeviceRadixSort::SortPairs(tempStorage_device, tempStorageBytes, d_keys, d_values, num_items));

        d_keyResultArray = d_keys.d_buffers[0];
        d_valueResultArray = d_values.d_buffers[0];

        // Allocate temporary storage
        CubDebugExit(cubAllocator.DeviceAllocate(&tempStorage_device, tempStorageBytes));
    }
    else
    {
        context = new mgpu::standard_context_t;

        error = cudaMalloc(&d_keyResultArray, sizeof(unsigned short) * num_items);
        if (cudaSuccess != error)
        {
            printf("can't allocate for key\n");
            kNN_cleanUp();
            return false;
        }

        error = cudaMalloc(&d_valueResultArray, sizeof(unsigned short) * num_items);
        if (cudaSuccess != error)
        {
            printf("can't allocate for value\n");
            kNN_cleanUp();
            return false;
        }
    }

    return true;

}

bool kNN_query(unsigned int query[vectorSize], unsigned int *result, const unsigned int resultCount)
{
    cudaError error = cudaMemcpy(d_query, query, sizeof(unsigned int) * vectorSize, cudaMemcpyHostToDevice);
    if (cudaSuccess != error)
    {
        printf("can't memcpy query\n");
        kNN_cleanUp();
        return false;
    }

    distanceKernel<<<texHeight, vectorPerBlock>>>(d_keyResultArray, d_valueResultArray, d_query, matrixTexObj, texHeight);
    if (sortAlgorithm == MGPU_MERGE_SORT)
    {
        mgpu::mergesort(d_keyResultArray, d_valueResultArray, num_items, mgpu::less_t<unsigned int>(), *context);
        context->synchronize();
        cudaMemcpy(result,d_valueResultArray, sizeof(unsigned int) * resultCount, cudaMemcpyDeviceToHost);
    }
    else if (sortAlgorithm == CUB_RADIX_SORT)
    {
        d_keys.selector = d_values.selector = 0;
        // Run
        CubDebugExit(cub::DeviceRadixSort::SortPairs(tempStorage_device, tempStorageBytes, d_keys, d_values, num_items));
        cudaMemcpy(result, d_values.Current(), sizeof(unsigned int) * resultCount, cudaMemcpyDeviceToHost);
    }

    return true;
}
