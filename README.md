= knn on gpu =

= How to build =

1. git clone --recursive https://github.com/shi-yan/KNN.git
2. cd KNN
3. make
4. export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-7.5/lib64
5. ./KNN

todo:

able to load partial data into gpu mem.