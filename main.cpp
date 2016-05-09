#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include "kernels.h"
#include <omp.h>

const int vectorSize = 25;
const int vectorPerBlock = 1024;
const int testMatrixHeight = 16384;

extern bool kNN_init(unsigned int *matrixBuffer, const unsigned int _num_items, enum SortAlgorithm _sortAlgorithm);
extern bool kNN_query(unsigned int query[vectorSize], unsigned int *result, const unsigned int resultCount);
extern void kNN_cleanUp();

int main(int argc, char *argv[])
{
    enum SortAlgorithm sortAlgorithm = CUB_RADIX_SORT;

    if (argc == 3 && strcmp(argv[1],"-a") == 0)
    {
        if (strcmp(argv[2],"cub") == 0)
        {
            sortAlgorithm = CUB_RADIX_SORT;
            printf("sort algorithm: CUB_RADIX_SORT\n");
        }
        else if(strcmp(argv[2],"mgpu") == 0)
        {
            sortAlgorithm = MGPU_MERGE_SORT;
            printf("sort algorithm: MGPU_MERGE_SORT\n");
        }
        else
        {
            printf("wrong commandline, default to CUB_RADIX_SORT\n");
            printf("supported commandline: ./KNN -a [cub/mgpu]\n");
        }
    }
    else
    {
        printf("no sort algorithm specified, default to CUB_RADIX_SORT\n");
        printf("supported commandline: ./KNN -a [cub/mgpu]\n");
    }

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    int device;
    for (device = 0; device < deviceCount; ++device)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        printf("Device %d has compute capability %d.%d.\n", device, deviceProp.major, deviceProp.minor);
        printf("Maximum threads per block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("Device texture alignment: %lu\n", deviceProp.textureAlignment);
        printf("Device texture dimension: %d X %d\n", deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1]);
    }

    size_t freeMem = 0;
    size_t totalMem = 0;
    cudaMemGetInfo(&freeMem, &totalMem);
    printf("available video memory: %ld, %ld (bytes)\n", freeMem, totalMem);


    const size_t bufferSize = vectorPerBlock * vectorSize * testMatrixHeight;
    unsigned int *matrixBuffer = new unsigned int[bufferSize];
    memset(matrixBuffer, 0xff, bufferSize * sizeof(unsigned int));

    // test data:
    int offset = 1050;
    unsigned int diff = 0x1;
    for(int i = 0; i < 32;++i)
    {
        diff = (diff << 1) | 0x1;
        for(int e = 0; e < 25; ++e)
        {
            matrixBuffer[25*(offset - i*7) + e] = 0;
        }

        matrixBuffer[25*(offset - i*7) + 8] = diff;

        //offset += 1024;
    }

    unsigned int query[vectorSize] = {0};
    query[10] = 0x1;
    // ------------

    double start = omp_get_wtime();
    if (kNN_init(matrixBuffer, vectorPerBlock  * testMatrixHeight, sortAlgorithm))
    {
        const int resultCount = 30;
        unsigned int result[resultCount] = {0};
        double allocatedTime = omp_get_wtime();
        for (int i = 0; i<100; ++i)
        {
            kNN_query(query, result, resultCount);
        }
        double end = omp_get_wtime();
        if (sortAlgorithm == MGPU_MERGE_SORT)
        {
            printf("sort using mgpu merge sort\n");
        }
        else
        {
            printf("sort using cub radix sort\n");
        }
        printf("find %d nearest neighbors for 100 times in %d vectors \n [%f secs for io, %f secs for each query]:\n", resultCount, vectorPerBlock  * testMatrixHeight,(allocatedTime - start), (end - allocatedTime) * 0.01);
        for(int i=0;i<resultCount;++i)
        {
            printf("[%d] %u\n", i, result[i]);
        }

        kNN_cleanUp();
    }

    cudaDeviceReset();
}
