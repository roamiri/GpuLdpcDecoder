
#include <iostream>
#include "CGPU_Decoder_MS_SIMD.h"
#include "transpose/GPU_Transpose_uint8.h"

static const size_t BLOCK_SIZE = 128; 

CGPU_Decoder_MS_SIMD::CGPU_Decoder_MS_SIMD(size_t _nb_frames, size_t n, size_t k, size_t m):
CGPUDecoder(_nb_frames, n, k, m)
{
	struct cudaDeviceProp devProp;
  	cudaGetDeviceProperties(&devProp, 0);

	struct cudaFuncAttributes attr;    
	cudaFuncGetAttributes(&attr, LDPC_Sched_Stage_1_MS_SIMD); 

  	int nMP      = devProp.multiProcessorCount; // Number of STREAM PROCESSOR
  	int nWarp    = attr.maxThreadsPerBlock/32;  // PACKET threads EXECUTABLES PARALLEL
  	int nThreads = nWarp * 32;					// NUMBER DE THREAD MAXI PAR SP
  	int nDOF     = nb_frames;
  	int nBperMP  = 65536 / (attr.numRegs); 	// Nr of blocks per MP
  	int minB     = min(nBperMP*nThreads,1024);
  	int nBlocks  = max(minB/nThreads * nMP, nDOF/nThreads);  

}


CGPU_Decoder_MS_SIMD::~CGPU_Decoder_MS_SIMD()
{
	std::cout << "Destroy " << __FUNCTION__ << std::endl;
}

void CGPU_Decoder_MS_SIMD::decode(float Intrinsic_fix[_N], int Rprime_fix[_N], int number_iteration, bool stream)
{
    cudaError_t Status;
	cudaStream_t streams;
    if(stream)
    {
		cudaStreamCreate(&streams);
	}

    size_t nb_blocks = nb_frames / BLOCK_SIZE;
	if( nb_frames % BLOCK_SIZE != 0 )
	{
		printf("(%ld - %ld)  (%ld - %ld)\n", nb_frames, BLOCK_SIZE, nb_frames/BLOCK_SIZE, nb_frames%BLOCK_SIZE);
		exit( 0 );
	}

    Status = cudaMemcpy/*Async*/(d_MSG_C_2_V, Intrinsic_fix, sz_nodes * sizeof(float), cudaMemcpyHostToDevice);
    ERROR_CHECK(Status, __FILE__, __LINE__);
	{
		dim3 grid(1, nb_frames/32);
		dim3 threads(32, 32);
		if(stream)
			Interleaver_uint8<<<grid, threads,0,streams>>>((int*)d_MSG_C_2_V, (int*)device_V, _N, nb_frames);
		else
			Interleaver_uint8<<<grid, threads>>>((int*)d_MSG_C_2_V, (int*)device_V, _N, nb_frames);
	}

	{
		if(stream)
			LDPC_Sched_Stage_1_MS_SIMD<<<nb_blocks, BLOCK_SIZE,0, streams>>>((unsigned int*)device_V, (unsigned int*)d_MSG_C_2_V, d_transpose, number_iteration);
		else
			LDPC_Sched_Stage_1_MS_SIMD<<<nb_blocks, BLOCK_SIZE>>>((unsigned int*)device_V, (unsigned int*)d_MSG_C_2_V, d_transpose, number_iteration);
	}
	{
		dim3 grid(1, nb_frames/32);
		dim3 threads(32, 32);
		if(stream)
			InvInterleaver_uint8<<<grid, threads,0, streams>>>((int*)device_V, (int*)d_MSG_C_2_V, _N, nb_frames);
		else
			InvInterleaver_uint8<<<grid, threads>>>((int*)device_V, (int*)d_MSG_C_2_V, _N, nb_frames);
	}
	Status = cudaMemcpy(Rprime_fix, d_MSG_C_2_V, sz_nodes * sizeof(float), cudaMemcpyDeviceToHost);
	ERROR_CHECK(Status, __FILE__, __LINE__);
}
