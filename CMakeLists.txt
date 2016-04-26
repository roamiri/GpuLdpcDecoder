set(PROJECTNAME GpuLdpcDecoder)
project(${PROJECTNAME})
cmake_minimum_required(VERSION 2.8)
find_package(CUDA QUIET REQUIRED)

include_directories(
${CUDA_INCLUDE_DIRS}
${CMAKE_SOURCE_DIR}
)

link_directories(
"/usr/local/cuda/lib64/"
)

# set(CMAKE_C_FLAGS ${CMAKE_C_FLAGS} "-Wall")
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-Wall  -fopenmp  -O3 -march=native -m64 -finline -funroll-loops")
# set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-Wall -fopenmp -O3 -march=native -mtune=cortex-a15 -marm -mfpu=neon -finline -funroll-loops -flax-vector-conversions")

# Pass options to NVCC
set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS}
    -arch=sm_21
#     -default-stream per-thread
    )

set(srcs
	mini.cpp
	awgn_channel/Chanel.cpp
	ber_analyzer/ErrorAnalyzer.cpp
	terminal/CTerminal.cpp
	frame/CFrame.cpp
	decoder/CGPUDecoder.cpp
	matrix/constantes_decoder.cpp
	utils/CTimerCpu.cpp
)

set(hdrs
	awgn_channel/Chanel.h 
	awgn_channel/Chanel_AWGN_SIMD.h
	
	ber_analyzer/ErrorAnalyzer.h
	custom_api/custom_cuda.h
	
	decoder/CGPU_Decoder_MS_SIMD.h
	decoder/cuda/CUDA_MS_SIMD.h  
	decoder/CGPUDecoder.h
	
	transpose/GPU_Transpose_uint8.h
	utils/simd_functions.h
	
	terminal/CTerminal.h
	frame/CFrame.h
	utils/CTimer.h
	utils/CTimerCpu.h
)

set(kernels
	awgn_channel/Chanel_AWGN_SIMD.cu
	custom_api/custom_cuda.cu
	
	decoder/CGPU_Decoder_MS_SIMD.cu
	decoder/cuda/CUDA_MS_SIMD.cu
	
	transpose/GPU_Transpose_uint8.cu
	
	utils/CTimer.cu
)
cuda_add_executable(${PROJECTNAME} ${srcs} ${hdrs} ${kernels})

target_link_libraries(${PROJECTNAME} -lcurand -lpthread)

install(TARGETS ${PROJECTNAME} RUNTIME DESTINATION bin)
