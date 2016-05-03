# knn on gpu

## How to Build

1. git clone --recursive https://github.com/shi-yan/KNN.git
2. cd KNN
3. make
4. export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-7.5/lib64
5. ./KNN -a cub  or   ./KNN -a mgpu

## Early Benchmark Results

sort using cub radix sort
find 30 nearest neighbors for 100 times in 16777216 vectors 
 [0.280620 secs for io, 0.032426 secs for each query]:

sort using mgpu merge sort
find 30 nearest neighbors for 100 times in 16777216 vectors 
 [0.281592 secs for io, 0.054164 secs for each query]:


## todo:

able to load partial data into gpu mem.