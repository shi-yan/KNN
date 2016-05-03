// Includes CUDA
#include <cuda_runtime.h>
#include <cstdio>
#include <cub/cub.cuh>
#include <moderngpu/kernel_mergesort.hxx>
#include "kernels.h"

const int vectorPerBlock = 1024;
const int vectorSize = 25;

__global__ void distanceKernel(unsigned int* keyOutput, unsigned int *valueOutput, unsigned int *query, cudaTextureObject_t texObj, unsigned int texHeight)
{
    int tu = blockDim.x * blockIdx.x;
    int tv = threadIdx.x;

    if (tu <texHeight && tv <vectorPerBlock)
    {
        __shared__ unsigned int queryLocal[vectorSize];

        if (tv < vectorSize)
        {
            queryLocal[tv] = query[tv];
        }

        __syncthreads();

        unsigned int count = 0;

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
static cudaArray *matrixArray_device = 0;
static cudaTextureObject_t matrixTexObj = 0;
static unsigned int *query_device = 0;
static unsigned int *keyResultArray_device = 0;
static unsigned int *valueResultArray_device = 0;
static size_t  tempStorageBytes  = 0;
static void    *tempStorage_device     = NULL;
static cub::DoubleBuffer<unsigned int> keys_device;
static cub::DoubleBuffer<unsigned int> values_device;
static mgpu::standard_context_t *context;
static unsigned int num_items = 0;


__host__ void kNN_cleanUp()
{
    if (matrixTexObj)
    {
        cudaDestroyTextureObject(matrixTexObj);
    }

    if (matrixArray_device)
    {
        cudaFreeArray(matrixArray_device);
    }

    if (query_device)
    {
        cudaFree(query_device);
    }

    if (keys_device.d_buffers[0] || keys_device.d_buffers[1])
    {
        cudaFree(keys_device.d_buffers[0]);
        cudaFree(keys_device.d_buffers[1]);
    }
    else if(keyResultArray_device)
    {
        cudaFree(keyResultArray_device);
    }

    if (values_device.d_buffers[0] || values_device.d_buffers[1])
    {
        cudaFree(values_device.d_buffers[0]);
        cudaFree(values_device.d_buffers[1]);
    }
    else if(valueResultArray_device)
    {
        cudaFree(valueResultArray_device);
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


__host__ bool kNN_init(unsigned int *matrixBuffer, const unsigned int _num_items, enum SortAlgorithm _sortAlgorithm)
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

    error = cudaMallocArray(&matrixArray_device, &channelDesc, vectorSize * vectorPerBlock, texHeight);
    if (cudaSuccess != error)
    {
        printf("can't allocate matrixArray [%u]\n", error);
        kNN_cleanUp();
        return false;
    }

    error = cudaMemcpyToArray(matrixArray_device, 0, 0, matrixBuffer, num_items * vectorSize * sizeof(unsigned int), cudaMemcpyHostToDevice);
    if (cudaSuccess != error)
    {
        printf("can't memcpy matrixArray [%u]\n", error);
        kNN_cleanUp();
        return false;
    }

    // Specify texture
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = matrixArray_device;

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

    error = cudaMalloc(&query_device, sizeof(unsigned int) * vectorSize);
    if (cudaSuccess != error)
    {
        printf("can't allocate for query\n");
        kNN_cleanUp();
        return false;
    }



    if (sortAlgorithm == CUB_RADIX_SORT)
    {
        cub::CachingDeviceAllocator cubAllocator(true);
        CubDebugExit(cubAllocator.DeviceAllocate((void**)&keys_device.d_buffers[0], sizeof(unsigned int) * num_items));
        CubDebugExit(cubAllocator.DeviceAllocate((void**)&values_device.d_buffers[0], sizeof(unsigned int) * num_items));
        CubDebugExit(cubAllocator.DeviceAllocate((void**)&keys_device.d_buffers[1], sizeof(unsigned int) * num_items));
        CubDebugExit(cubAllocator.DeviceAllocate((void**)&values_device.d_buffers[1], sizeof(unsigned int) * num_items));
        CubDebugExit(cub::DeviceRadixSort::SortPairs(tempStorage_device, tempStorageBytes, keys_device, values_device, num_items));

        keyResultArray_device = keys_device.d_buffers[0];
        valueResultArray_device = values_device.d_buffers[0];

        // Allocate temporary storage
        CubDebugExit(cubAllocator.DeviceAllocate(&tempStorage_device, tempStorageBytes));
    }
    else
    {
        context = new mgpu::standard_context_t;

        error = cudaMalloc(&keyResultArray_device, sizeof(unsigned int) * num_items);
        if (cudaSuccess != error)
        {
            printf("can't allocate for key\n");
            kNN_cleanUp();
            return false;
        }

        error = cudaMalloc(&valueResultArray_device, sizeof(unsigned int) * num_items);
        if (cudaSuccess != error)
        {
            printf("can't allocate for value\n");
            kNN_cleanUp();
            return false;
        }
    }

    return true;

}

__host__ bool kNN_query(unsigned int query[vectorSize], unsigned int *result, const unsigned int resultCount)
{
    cudaError error = cudaMemcpy(query_device, query, sizeof(unsigned int) * vectorSize, cudaMemcpyHostToDevice);
    if (cudaSuccess != error)
    {
        printf("can't memcpy query\n");
        kNN_cleanUp();
        return false;
    }

    distanceKernel<<<texHeight, vectorPerBlock>>>(keyResultArray_device, valueResultArray_device, query_device, matrixTexObj, texHeight);
    if (sortAlgorithm == MGPU_MERGE_SORT)
    {
        mgpu::mergesort(keyResultArray_device, valueResultArray_device, num_items, mgpu::less_t<unsigned int>(), *context);
        cudaDeviceSynchronize();
        context->synchronize();
        cudaMemcpy(result,valueResultArray_device, sizeof(unsigned int) * resultCount, cudaMemcpyDeviceToHost);
    }
    else if (sortAlgorithm == CUB_RADIX_SORT)
    {
        keys_device.selector = values_device.selector = 0;
        // Run
        CubDebugExit(cub::DeviceRadixSort::SortPairs(tempStorage_device, tempStorageBytes, keys_device, values_device, num_items));
        cudaDeviceSynchronize();
        cudaMemcpy(result, values_device.Current(), sizeof(unsigned int) * resultCount, cudaMemcpyDeviceToHost);
    }

    return true;
}
