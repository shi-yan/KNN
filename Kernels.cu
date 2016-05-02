// Includes CUDA
#include <cuda_runtime.h>
#include <cstdio>
#include <cub/cub.cuh>

const int vectorSize = 25;
const int vectorPerBlock = 1024;

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

__host__ bool kNN(unsigned int *matrixBuffer, const unsigned int num_items, unsigned int query[vectorSize], unsigned int *result, const unsigned int resultCount)
{
    int texHeight = (num_items / vectorPerBlock) + ((num_items % vectorPerBlock)?1:0);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0 , 0, 0, cudaChannelFormatKindUnsigned);
    cudaArray *matrixArray_device = 0;
    cudaError error = cudaSuccess;
    bool success = false;
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    cudaTextureObject_t matrixTexObj = 0;
    unsigned int *query_device = 0;
    unsigned int *keyResultArray_device = 0;
    unsigned int *valueResultArray_device = 0;
    size_t  tempStorageBytes  = 0;
    void    *tempStorage_device     = NULL;
    cub::CachingDeviceAllocator  cubAllocator(true);
    cub::DoubleBuffer<unsigned int> keys_device;
    cub::DoubleBuffer<unsigned int> values_device;

    error = cudaMallocArray(&matrixArray_device, &channelDesc, vectorSize * vectorPerBlock, texHeight);
    if (cudaSuccess != error)
    {
        printf("can't allocate matrixArray [%u]\n", error);
        goto cleanup;
    }

    error = cudaMemcpyToArray(matrixArray_device, 0, 0, matrixBuffer, num_items * vectorSize * sizeof(unsigned int), cudaMemcpyHostToDevice);
    if (cudaSuccess != error)
    {
        printf("can't memcpy matrixArray [%u]\n", error);
        goto cleanup;
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
        goto cleanup;
    }

    error = cudaMalloc(&query_device, sizeof(unsigned int) * vectorSize);
    if (cudaSuccess != error)
    {
        printf("can't allocate for query\n");
        goto cleanup;
    }

    error = cudaMemcpy(query_device, query, sizeof(unsigned int) * vectorSize, cudaMemcpyHostToDevice);
    if (cudaSuccess != error)
    {
        printf("can't memcpy query\n");
        goto cleanup;
    }

    error = cudaMalloc(&keyResultArray_device, sizeof(unsigned int) * num_items);
    if (cudaSuccess != error)
    {
        printf("can't allocate for key\n");
        goto cleanup;
    }

    error = cudaMalloc(&valueResultArray_device, sizeof(unsigned int) * num_items);
    if (cudaSuccess != error)
    {
        printf("can't allocate for value\n");
        goto cleanup;
    }

    distanceKernel<<<texHeight, vectorPerBlock>>>(keyResultArray_device, valueResultArray_device, query_device, matrixTexObj, texHeight);

    /* debug code {
        unsigned int *debugKey = new unsigned int [2000];

        cudaMemcpy(debugKey, keyResultArray_device, 2000*sizeof(unsigned int), cudaMemcpyDeviceToHost);

        printf("debug: %d, %d, %d\n", debugKey[1025], debugKey[500], debugKey[502]);
    }*/

    keys_device.d_buffers[keys_device.selector] = keyResultArray_device;
    values_device.d_buffers[values_device.selector] = valueResultArray_device;
    CubDebugExit(cubAllocator.DeviceAllocate((void**)&keys_device.d_buffers[keys_device.selector ^ 1], sizeof(unsigned int) * num_items));
    CubDebugExit(cubAllocator.DeviceAllocate((void**)&values_device.d_buffers[values_device.selector ^ 1], sizeof(unsigned int) * num_items));
    // Allocate temporary storage
    CubDebugExit(cub::DeviceRadixSort::SortPairs(tempStorage_device, tempStorageBytes, keys_device, values_device, num_items));
    CubDebugExit(cubAllocator.DeviceAllocate(&tempStorage_device, tempStorageBytes));

    // Run
    CubDebugExit(cub::DeviceRadixSort::SortPairs(tempStorage_device, tempStorageBytes, keys_device, values_device, num_items));

    cudaMemcpy(result, values_device.Current(), sizeof(unsigned int) * resultCount, cudaMemcpyDeviceToHost);

    success = true;
cleanup:
    return true;
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

    return success;
}
