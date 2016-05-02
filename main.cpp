#include <cuda_runtime.h>
#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <cstring>

const int vectorSize = 25;
const int vectorPerBlock = 1024;
const int testMatrixHeight = 16384;

extern __host__ bool kNN(unsigned int *matrixBuffer, const unsigned int matrixSize, unsigned int query[25], unsigned int *result, const unsigned int resultSize);

int main(int argc, char *argv[])
{
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
    int offset = 500;
    unsigned int diff = 0x1;
    for(int i = 0; i < 32;++i)
    {
        diff = (diff << 1) | 0x1;
        for(int e = 0; e < 25; ++e)
        {
            matrixBuffer[25*(offset - i*7) + e] = 0;
        }

        matrixBuffer[25*(offset - i*7) + 8] = diff;
    }

    unsigned int query[25] = {0};
    query[10] = 0x1;
    // ------------

    const int resultCount = 30;
    unsigned int result[resultCount] = {0};

    if (kNN(matrixBuffer, vectorPerBlock  * testMatrixHeight, query, result, resultCount)) {

        printf("%d nearest neighbors:\n", resultCount);
        for(int i=0;i<resultCount;++i)
        {
            printf("[%d] %u\n", i, result[i]);
        }
    }

    cudaDeviceReset();
}
