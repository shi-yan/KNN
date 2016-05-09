CONFIG += c++11

TARGET = KNN
CONFIG += console
CONFIG -= app_bundle

TEMPLATE = app

INCLUDEPATH += /usr/local/cuda-7.5/include \
               $$PWD/cub \
               $$PWD/moderngpu/src


CUDA_SOURCES += Kernels.cu

OTHER_FILES +=  Kernels.cu \
                README.md

NVCCFLAGS = --use_fast_math -arch sm_30 -m64 -std=c++11 --expt-extended-lambda
CUDA_INC = $$join(INCLUDEPATH,' -I',' -I',' ')

CONFIG(debug, debug|release) {
     cuda_d.input = CUDA_SOURCES
     cuda_d.output = $$PWD/debug/cuda/${QMAKE_FILE_BASE}_cuda.obj
     cuda_d.commands = /usr/local/cuda-7.5/bin/nvcc -D_DEBUG -c $$NVCCFLAGS $$CUDA_INC $$LIBS ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
     cuda_d.dependency_type = TYPE_C

     QMAKE_EXTRA_COMPILERS += cuda_d
}
else {
     cuda.commands = /usr/local/cuda-7.5/bin/nvcc -c $$NVCCFLAGS $$CUDA_INC $$LIBS ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
     cuda.input = CUDA_SOURCES
     cuda.output = $$PWD/release/cuda/${QMAKE_FILE_BASE}_cuda.obj
     cuda.dependency_type = TYPE_C
     QMAKE_EXTRA_COMPILERS += cuda
}

DISTFILES += \
    Kernels.cu \
    Makefile

LIBS += -L/usr/local/cuda-7.5/lib64 \
        -lcudart \
        -lgomp



SOURCES += main.cpp

HEADERS += \
    kernels.h
