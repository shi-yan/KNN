#############################################################################
# Makefile for building: KNN
# Generated by qmake (3.0) (Qt 5.6.0)
# Project:  KNN.pro
# Template: app
# Command: /home/shiy/Qt/5.6/gcc_64/bin/qmake -o Makefile KNN.pro
#############################################################################

MAKEFILE      = Makefile

####### Compiler, tools and options

CC            = gcc
CXX           = g++
DEFINES       = -DQT_NO_DEBUG -DQT_GUI_LIB -DQT_CORE_LIB
CFLAGS        = -pipe -O2 -Wall -W -D_REENTRANT -fPIC $(DEFINES)
CXXFLAGS      = -pipe -O2 -std=gnu++0x -Wall -W -D_REENTRANT -fPIC $(DEFINES)
INCPATH       = -I. -I/usr/local/cuda-7.5/include -Icub -Imoderngpu/src -I.
QMAKE         = /home/shiy/Qt/5.6/gcc_64/bin/qmake
DEL_FILE      = rm -f
CHK_DIR_EXISTS= test -d
MKDIR         = mkdir -p
COPY          = cp -f
COPY_FILE     = cp -f
COPY_DIR      = cp -f -R
INSTALL_FILE  = install -m 644 -p
INSTALL_PROGRAM = install -m 755 -p
INSTALL_DIR   = cp -f -R
DEL_FILE      = rm -f
SYMLINK       = ln -f -s
DEL_DIR       = rmdir
MOVE          = mv -f
TAR           = tar -cf
COMPRESS      = gzip -9f
DISTNAME      = KNN1.0.0
DISTDIR = /home/shiy/KNN/.tmp/KNN1.0.0
LINK          = g++
LFLAGS        = -Wl,-O1 -Wl,-z,origin -Wl,-rpath,\$$ORIGIN -Wl,-rpath,/usr/local/cuda-7.5/lib64
LIBS          = $(SUBLIBS) -L/usr/local/cuda-7.5/lib64 -lcudart -lgomp
AR            = ar cqs
RANLIB        = 
SED           = sed
STRIP         = strip

####### Output directory

OBJECTS_DIR   = ./

####### Files

SOURCES       = main.cpp 
OBJECTS       = release/cuda/Kernels_cuda.obj \
		main.o
DIST          = Kernels.cu \
		KNN.pro kernels.h main.cpp
QMAKE_TARGET  = KNN
DESTDIR       = 
TARGET        = KNN


first: all
####### Build rules

$(TARGET):  $(OBJECTS)  
	$(LINK) $(LFLAGS) -o $(TARGET) $(OBJECTS) $(OBJCOMP) $(LIBS)

qmake_all: FORCE


all: Makefile $(TARGET)

dist: distdir FORCE
	(cd `dirname $(DISTDIR)` && $(TAR) $(DISTNAME).tar $(DISTNAME) && $(COMPRESS) $(DISTNAME).tar) && $(MOVE) `dirname $(DISTDIR)`/$(DISTNAME).tar.gz . && $(DEL_FILE) -r $(DISTDIR)

distdir: FORCE
	@test -d $(DISTDIR) || mkdir -p $(DISTDIR)
	$(COPY_FILE) --parents $(DIST) $(DISTDIR)/
	$(COPY_FILE) --parents Kernels.cu $(DISTDIR)/
	$(COPY_FILE) --parents kernels.h $(DISTDIR)/
	$(COPY_FILE) --parents main.cpp $(DISTDIR)/


clean: compiler_clean 
	-$(DEL_FILE) $(OBJECTS)
	-$(DEL_FILE) *~ core *.core


distclean: clean 
	-$(DEL_FILE) $(TARGET) 
	-$(DEL_FILE) Makefile


####### Sub-libraries

mocclean: compiler_moc_header_clean compiler_moc_source_clean

mocables: compiler_moc_header_make_all compiler_moc_source_make_all

check: first

compiler_cuda_make_all: release/cuda/Kernels_cuda.obj
compiler_cuda_clean:
	-$(DEL_FILE) release/cuda/Kernels_cuda.obj
release/cuda/Kernels_cuda.obj: /usr/local/cuda-7.5/include/cuda_runtime.h \
		/usr/local/cuda-7.5/include/host_config.h \
		/usr/local/cuda-7.5/include/builtin_types.h \
		/usr/local/cuda-7.5/include/device_types.h \
		/usr/local/cuda-7.5/include/host_defines.h \
		/usr/local/cuda-7.5/include/driver_types.h \
		/usr/local/cuda-7.5/include/surface_types.h \
		/usr/local/cuda-7.5/include/texture_types.h \
		/usr/local/cuda-7.5/include/vector_types.h \
		/usr/local/cuda-7.5/include/channel_descriptor.h \
		/usr/local/cuda-7.5/include/cuda_runtime_api.h \
		/usr/local/cuda-7.5/include/cuda_device_runtime_api.h \
		/usr/local/cuda-7.5/include/driver_functions.h \
		/usr/local/cuda-7.5/include/vector_functions.h \
		/usr/local/cuda-7.5/include/vector_functions.hpp \
		/usr/local/cuda-7.5/include/device_functions.h \
		/usr/local/cuda-7.5/include/device_functions.hpp \
		/usr/local/cuda-7.5/include/device_atomic_functions.h \
		/usr/local/cuda-7.5/include/device_atomic_functions.hpp \
		/usr/local/cuda-7.5/include/device_double_functions.h \
		/usr/local/cuda-7.5/include/device_double_functions.hpp \
		/usr/local/cuda-7.5/include/sm_20_atomic_functions.h \
		/usr/local/cuda-7.5/include/sm_20_atomic_functions.hpp \
		/usr/local/cuda-7.5/include/sm_32_atomic_functions.h \
		/usr/local/cuda-7.5/include/sm_32_atomic_functions.hpp \
		/usr/local/cuda-7.5/include/sm_35_atomic_functions.h \
		/usr/local/cuda-7.5/include/sm_20_intrinsics.h \
		/usr/local/cuda-7.5/include/sm_20_intrinsics.hpp \
		/usr/local/cuda-7.5/include/sm_30_intrinsics.h \
		/usr/local/cuda-7.5/include/sm_30_intrinsics.hpp \
		/usr/local/cuda-7.5/include/sm_32_intrinsics.h \
		/usr/local/cuda-7.5/include/sm_32_intrinsics.hpp \
		/usr/local/cuda-7.5/include/sm_35_intrinsics.h \
		/usr/local/cuda-7.5/include/surface_functions.h \
		/usr/local/cuda-7.5/include/cuda_surface_types.h \
		/usr/local/cuda-7.5/include/surface_functions.hpp \
		/usr/local/cuda-7.5/include/texture_fetch_functions.h \
		/usr/local/cuda-7.5/include/cuda_texture_types.h \
		/usr/local/cuda-7.5/include/texture_fetch_functions.hpp \
		/usr/local/cuda-7.5/include/texture_indirect_functions.h \
		/usr/local/cuda-7.5/include/texture_indirect_functions.hpp \
		/usr/local/cuda-7.5/include/surface_indirect_functions.h \
		/usr/local/cuda-7.5/include/surface_indirect_functions.hpp \
		/usr/local/cuda-7.5/include/common_functions.h \
		/usr/local/cuda-7.5/include/math_functions.h \
		/usr/local/cuda-7.5/include/math_constants.h \
		/usr/local/cuda-7.5/include/device_functions_decls.h \
		/usr/local/cuda-7.5/include/crt/func_macro.h \
		/usr/local/cuda-7.5/include/math_functions.hpp \
		/usr/local/cuda-7.5/include/math_functions_dbl_ptx3.h \
		/usr/local/cuda-7.5/include/math_functions_dbl_ptx3.hpp \
		/usr/local/cuda-7.5/include/device_launch_parameters.h \
		cub/cub/cub.cuh \
		cub/cub/block/block_histogram.cuh \
		cub/cub/block/specializations/block_histogram_sort.cuh \
		cub/cub/block/block_radix_sort.cuh \
		cub/cub/block/block_exchange.cuh \
		cub/cub/util_ptx.cuh \
		cub/cub/util_type.cuh \
		cub/cub/util_macro.cuh \
		cub/cub/util_namespace.cuh \
		cub/cub/util_arch.cuh \
		cub/cub/util_debug.cuh \
		cub/cub/block/block_radix_rank.cuh \
		cub/cub/thread/thread_reduce.cuh \
		cub/cub/thread/thread_operators.cuh \
		cub/cub/thread/thread_scan.cuh \
		cub/cub/block/block_scan.cuh \
		cub/cub/block/specializations/block_scan_raking.cuh \
		cub/cub/block/block_raking_layout.cuh \
		cub/cub/warp/warp_scan.cuh \
		cub/cub/warp/specializations/warp_scan_shfl.cuh \
		cub/cub/warp/specializations/warp_scan_smem.cuh \
		cub/cub/thread/thread_load.cuh \
		/usr/local/cuda-7.5/include/cuda.h \
		cub/cub/thread/thread_store.cuh \
		cub/cub/block/specializations/block_scan_warp_scans.cuh \
		cub/cub/block/block_discontinuity.cuh \
		cub/cub/block/specializations/block_histogram_atomic.cuh \
		cub/cub/block/block_load.cuh \
		cub/cub/iterator/cache_modified_input_iterator.cuh \
		cub/cub/util_device.cuh \
		/usr/local/cuda-7.5/include/thrust/iterator/iterator_facade.h \
		/usr/local/cuda-7.5/include/thrust/detail/config.h \
		/usr/local/cuda-7.5/include/thrust/detail/config/config.h \
		/usr/local/cuda-7.5/include/thrust/detail/config/simple_defines.h \
		/usr/local/cuda-7.5/include/thrust/detail/config/compiler.h \
		/usr/local/cuda-7.5/include/thrust/detail/config/host_system.h \
		/usr/local/cuda-7.5/include/thrust/detail/config/device_system.h \
		/usr/local/cuda-7.5/include/thrust/detail/config/host_device.h \
		/usr/local/cuda-7.5/include/thrust/detail/config/debug.h \
		/usr/local/cuda-7.5/include/thrust/detail/config/compiler_fence.h \
		/usr/local/cuda-7.5/include/thrust/detail/config/forceinline.h \
		/usr/local/cuda-7.5/include/thrust/detail/config/exec_check_disable.h \
		/usr/local/cuda-7.5/include/thrust/detail/config/global_workarounds.h \
		/usr/local/cuda-7.5/include/thrust/detail/type_traits.h \
		/usr/local/cuda-7.5/include/thrust/detail/type_traits/has_trivial_assign.h \
		/usr/local/cuda-7.5/include/thrust/iterator/detail/iterator_facade_category.h \
		/usr/local/cuda-7.5/include/thrust/iterator/detail/host_system_tag.h \
		/usr/local/cuda-7.5/include/thrust/iterator/detail/device_system_tag.h \
		/usr/local/cuda-7.5/include/thrust/iterator/detail/any_system_tag.h \
		/usr/local/cuda-7.5/include/thrust/detail/execution_policy.h \
		/usr/local/cuda-7.5/include/thrust/iterator/iterator_categories.h \
		/usr/local/cuda-7.5/include/thrust/iterator/detail/iterator_category_with_system_and_traversal.h \
		/usr/local/cuda-7.5/include/thrust/iterator/detail/iterator_traversal_tags.h \
		/usr/local/cuda-7.5/include/thrust/iterator/detail/universal_categories.h \
		/usr/local/cuda-7.5/include/thrust/iterator/detail/is_iterator_category.h \
		/usr/local/cuda-7.5/include/thrust/iterator/detail/iterator_category_to_traversal.h \
		/usr/local/cuda-7.5/include/thrust/iterator/detail/iterator_category_to_system.h \
		/usr/local/cuda-7.5/include/thrust/iterator/detail/distance_from_result.h \
		/usr/local/cuda-7.5/include/thrust/iterator/iterator_traits.h \
		/usr/local/cuda-7.5/include/thrust/iterator/detail/iterator_traits.inl \
		cub/cub/block/block_reduce.cuh \
		cub/cub/block/specializations/block_reduce_raking.cuh \
		cub/cub/warp/warp_reduce.cuh \
		cub/cub/warp/specializations/warp_reduce_shfl.cuh \
		cub/cub/warp/specializations/warp_reduce_smem.cuh \
		cub/cub/block/specializations/block_reduce_raking_commutative_only.cuh \
		cub/cub/block/specializations/block_reduce_warp_reductions.cuh \
		cub/cub/block/block_store.cuh \
		cub/cub/device/device_histogram.cuh \
		cub/cub/device/dispatch/dispatch_histogram.cuh \
		cub/cub/agent/agent_histogram.cuh \
		cub/cub/grid/grid_queue.cuh \
		cub/cub/thread/thread_search.cuh \
		cub/cub/device/device_partition.cuh \
		cub/cub/device/dispatch/dispatch_select_if.cuh \
		cub/cub/device/dispatch/dispatch_scan.cuh \
		cub/cub/agent/agent_scan.cuh \
		cub/cub/agent/single_pass_scan_operators.cuh \
		cub/cub/agent/agent_select_if.cuh \
		cub/cub/device/device_radix_sort.cuh \
		cub/cub/device/dispatch/dispatch_radix_sort.cuh \
		cub/cub/agent/agent_radix_sort_upsweep.cuh \
		cub/cub/agent/agent_radix_sort_downsweep.cuh \
		cub/cub/grid/grid_even_share.cuh \
		cub/cub/device/device_reduce.cuh \
		cub/cub/device/dispatch/dispatch_reduce.cuh \
		cub/cub/agent/agent_reduce.cuh \
		cub/cub/grid/grid_mapping.cuh \
		cub/cub/iterator/arg_index_input_iterator.cuh \
		/usr/local/cuda-7.5/include/thrust/version.h \
		cub/cub/device/dispatch/dispatch_reduce_by_key.cuh \
		cub/cub/agent/agent_reduce_by_key.cuh \
		cub/cub/iterator/constant_input_iterator.cuh \
		cub/cub/device/device_run_length_encode.cuh \
		cub/cub/device/dispatch/dispatch_rle.cuh \
		cub/cub/agent/agent_rle.cuh \
		cub/cub/device/device_scan.cuh \
		cub/cub/device/device_segmented_radix_sort.cuh \
		cub/cub/device/device_segmented_reduce.cuh \
		cub/cub/device/device_select.cuh \
		cub/cub/device/device_spmv.cuh \
		cub/cub/device/dispatch/dispatch_spmv_orig.cuh \
		cub/cub/agent/agent_segment_fixup.cuh \
		cub/cub/agent/agent_spmv_orig.cuh \
		cub/cub/iterator/counting_input_iterator.cuh \
		cub/cub/iterator/tex_ref_input_iterator.cuh \
		cub/cub/iterator/cache_modified_output_iterator.cuh \
		cub/cub/iterator/tex_obj_input_iterator.cuh \
		cub/cub/iterator/transform_input_iterator.cuh \
		cub/cub/util_allocator.cuh \
		cub/cub/host/mutex.cuh \
		moderngpu/src/moderngpu/kernel_mergesort.hxx \
		moderngpu/src/moderngpu/transform.hxx \
		moderngpu/src/moderngpu/launch_box.hxx \
		moderngpu/src/moderngpu/context.hxx \
		moderngpu/src/moderngpu/util.hxx \
		moderngpu/src/moderngpu/types.hxx \
		moderngpu/src/moderngpu/meta.hxx \
		moderngpu/src/moderngpu/operators.hxx \
		moderngpu/src/moderngpu/launch_params.hxx \
		moderngpu/src/moderngpu/tuple.hxx \
		moderngpu/src/moderngpu/kernel_merge.hxx \
		moderngpu/src/moderngpu/cta_merge.hxx \
		moderngpu/src/moderngpu/loadstore.hxx \
		moderngpu/src/moderngpu/intrinsics.hxx \
		moderngpu/src/moderngpu/search.hxx \
		moderngpu/src/moderngpu/cta_search.hxx \
		moderngpu/src/moderngpu/memory.hxx \
		moderngpu/src/moderngpu/cta_mergesort.hxx \
		moderngpu/src/moderngpu/sort_networks.hxx \
		kernels.h \
		Kernels.cu
	/usr/local/cuda-7.5/bin/nvcc -c --use_fast_math -arch sm_30 -m64 -std=c++11 --expt-extended-lambda  -I/usr/local/cuda-7.5/include -I/home/shiy/KNN/cub -I/home/shiy/KNN/moderngpu/src  Kernels.cu -o release/cuda/Kernels_cuda.obj

compiler_rcc_make_all:
compiler_rcc_clean:
compiler_moc_header_make_all:
compiler_moc_header_clean:
compiler_moc_source_make_all:
compiler_moc_source_clean:
compiler_yacc_decl_make_all:
compiler_yacc_decl_clean:
compiler_yacc_impl_make_all:
compiler_yacc_impl_clean:
compiler_lex_make_all:
compiler_lex_clean:
compiler_clean: compiler_cuda_clean 

####### Compile

main.o: main.cpp /usr/local/cuda-7.5/include/cuda_runtime.h \
		/usr/local/cuda-7.5/include/host_config.h \
		/usr/local/cuda-7.5/include/builtin_types.h \
		/usr/local/cuda-7.5/include/device_types.h \
		/usr/local/cuda-7.5/include/host_defines.h \
		/usr/local/cuda-7.5/include/driver_types.h \
		/usr/local/cuda-7.5/include/surface_types.h \
		/usr/local/cuda-7.5/include/texture_types.h \
		/usr/local/cuda-7.5/include/vector_types.h \
		/usr/local/cuda-7.5/include/channel_descriptor.h \
		/usr/local/cuda-7.5/include/cuda_runtime_api.h \
		/usr/local/cuda-7.5/include/cuda_device_runtime_api.h \
		/usr/local/cuda-7.5/include/driver_functions.h \
		/usr/local/cuda-7.5/include/vector_functions.h \
		/usr/local/cuda-7.5/include/vector_functions.hpp \
		/usr/local/cuda-7.5/include/device_functions.h \
		/usr/local/cuda-7.5/include/device_functions.hpp \
		/usr/local/cuda-7.5/include/device_atomic_functions.h \
		/usr/local/cuda-7.5/include/device_atomic_functions.hpp \
		/usr/local/cuda-7.5/include/device_double_functions.h \
		/usr/local/cuda-7.5/include/device_double_functions.hpp \
		/usr/local/cuda-7.5/include/sm_20_atomic_functions.h \
		/usr/local/cuda-7.5/include/sm_20_atomic_functions.hpp \
		/usr/local/cuda-7.5/include/sm_32_atomic_functions.h \
		/usr/local/cuda-7.5/include/sm_32_atomic_functions.hpp \
		/usr/local/cuda-7.5/include/sm_35_atomic_functions.h \
		/usr/local/cuda-7.5/include/sm_20_intrinsics.h \
		/usr/local/cuda-7.5/include/sm_20_intrinsics.hpp \
		/usr/local/cuda-7.5/include/sm_30_intrinsics.h \
		/usr/local/cuda-7.5/include/sm_30_intrinsics.hpp \
		/usr/local/cuda-7.5/include/sm_32_intrinsics.h \
		/usr/local/cuda-7.5/include/sm_32_intrinsics.hpp \
		/usr/local/cuda-7.5/include/sm_35_intrinsics.h \
		/usr/local/cuda-7.5/include/surface_functions.h \
		/usr/local/cuda-7.5/include/cuda_surface_types.h \
		/usr/local/cuda-7.5/include/surface_functions.hpp \
		/usr/local/cuda-7.5/include/texture_fetch_functions.h \
		/usr/local/cuda-7.5/include/cuda_texture_types.h \
		/usr/local/cuda-7.5/include/texture_fetch_functions.hpp \
		/usr/local/cuda-7.5/include/texture_indirect_functions.h \
		/usr/local/cuda-7.5/include/texture_indirect_functions.hpp \
		/usr/local/cuda-7.5/include/surface_indirect_functions.h \
		/usr/local/cuda-7.5/include/surface_indirect_functions.hpp \
		/usr/local/cuda-7.5/include/common_functions.h \
		/usr/local/cuda-7.5/include/math_functions.h \
		/usr/local/cuda-7.5/include/math_constants.h \
		/usr/local/cuda-7.5/include/device_functions_decls.h \
		/usr/local/cuda-7.5/include/crt/func_macro.h \
		/usr/local/cuda-7.5/include/math_functions.hpp \
		/usr/local/cuda-7.5/include/math_functions_dbl_ptx3.h \
		/usr/local/cuda-7.5/include/math_functions_dbl_ptx3.hpp \
		/usr/local/cuda-7.5/include/device_launch_parameters.h \
		kernels.h
	$(CXX) -c $(CXXFLAGS) $(INCPATH) -o main.o main.cpp

####### Install

install:  FORCE

uninstall:  FORCE

FORCE:

