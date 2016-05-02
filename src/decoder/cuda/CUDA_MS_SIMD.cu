
#include "matrix/constantes_gpu.h"
#include "utils/simd_functions.h"

#include <stdio.h>


union t_1x4
{
	unsigned int v;
    char c[4];
};

__global__ void LDPC_Sched_Stage_1_MS_SIMD(unsigned int var_nodes[_N], unsigned int var_mesgs[_M], unsigned int PosNoeudsVariable[_M], unsigned int loops) 
{

	const int i  = threadIdx.x + blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
	const int ii = blockDim.x  * blockDim.y * gridDim.x; // A VERIFIER

	__shared__ unsigned int iTable[DEG_1];

	///////////////////////////////////////////////////////////////////////////
	//
	//
	//
	{
		unsigned int *p_msg1w       = var_mesgs + i;
		unsigned int *p_indice_nod1 = PosNoeudsVariable;

		//
		// 
		//
		for (int z = 0; z < DEG_1_COMPUTATIONS; z++) 
		{
			unsigned int sign_du_check = 0x00000000;
			unsigned int min1 = 0x7F7F7F7F; // Initialize 1 A 127
			unsigned int min2 = 0x7F7F7F7F; // Initialize 2 A 127

			register unsigned int tab_vContr[DEG_1];

			//
			// 
			//
			__syncthreads();
			if( threadIdx.x < DEG_1){
				iTable[threadIdx.x] = p_indice_nod1[threadIdx.x];
			}
			__syncthreads();
			p_indice_nod1 += DEG_1;

			#pragma unroll
			for (int j = 0; j < DEG_1; j++) 
			{
				tab_vContr[j] = var_nodes[ iTable[j] * ii + i ];
				min2 = vminu4(min2, vmaxu4(vabs4(tab_vContr[j]), min1));
				min1 = vminu4(min1, vabs4(tab_vContr[j]));
				sign_du_check = sign_du_check ^ vcmpgts4(tab_vContr[j], 0x00000000);
			}

			const unsigned int cste_1 = vminu4(min2, 0x1F1F1F1F);
			const unsigned int cste_2 = vminu4(min1, 0x1F1F1F1F);

			#pragma unroll
			for (int j = 0; j < DEG_1; j++) 
			{
				unsigned int ab = vabs4(tab_vContr[j]);
				unsigned int m1 = vcmpeq4( ab, min1 );
				unsigned int m2 = vcmpne4( ab, min1 );
				unsigned int re = (m1 & cste_1) | (m2 & cste_2);
				unsigned int sign_msg = sign_du_check ^ vcmpgts4(tab_vContr[j], 0x00000000);
				unsigned int msg_sortant = (re & sign_msg) | (vneg4(re) & (sign_msg ^ 0xFFFFFFFF));
				*p_msg1w = msg_sortant;
				p_msg1w += ii;
				var_nodes[ iTable[j] * ii + i ] = vaddss4(tab_vContr[j], msg_sortant);
			}
		}

#if NB_DEGRES > 1
		for (int z = 0; z <DEG_2_COMPUTATIONS; z++) {
			unsigned int sign_du_check = 0x00000000;
			unsigned int min1 = 0x7F7F7F7F;
			unsigned int min2 = 0x7F7F7F7F;
			unsigned int tab_vContr [DEG_2];

			//
			//
			//
			__syncthreads();
			if( threadIdx.x < DEG_2){
				iTable[threadIdx.x] = p_indice_nod1[threadIdx.x];
			}
			__syncthreads();
			p_indice_nod1 += DEG_2;

			#pragma unroll
			for (int j = 0; j<DEG_2; j++) {
				tab_vContr[j] = var_nodes[ iTable[j] * ii + i ];
				min2 = vminu4(min2, vmaxu4(vabs4(tab_vContr[j]), min1));
				min1 = vminu4(min1, vabs4(tab_vContr[j]));
				sign_du_check = sign_du_check ^ vcmpgts4(tab_vContr[j], 0x00000000);
			}

			const unsigned int cste_1 = vminu4(min2, 0x1F1F1F1F);
			const unsigned int cste_2 = vminu4(min1, 0x1F1F1F1F);

			#pragma unroll
			for (int j = 0; j < DEG_2; j++) {
				unsigned int ab = vabs4(tab_vContr[j]);
				unsigned int m1 = vcmpeq4( ab, min1 );
				unsigned int m2 = vcmpne4( ab, min1 );
				unsigned int re = (m1 & cste_1) | (m2 & cste_2);
				unsigned int sign_msg = sign_du_check ^ vcmpgts4(tab_vContr[j], 0x00000000);
				unsigned int msg_sortant = (re & sign_msg) | (vneg4(re) & (sign_msg ^ 0xFFFFFFFF));
				*p_msg1w = msg_sortant;
				p_msg1w += ii;
				var_nodes[ iTable[j] * ii + i ] = vaddss4(tab_vContr[j], msg_sortant);
			}
		}
#endif
	}loops -= 1;

	////////////////////////////////////////////////////////////////////////////
	//
	//
	//
	while (loops--) {
		unsigned int *p_msg1r = var_mesgs + i;
		unsigned int *p_msg1w = var_mesgs + i;
		unsigned int *p_indice_nod1 = PosNoeudsVariable;
		//
		//
		//
		for (int z = 0; z < DEG_1_COMPUTATIONS; z++) {
			unsigned int sign_du_check = 0x00000000;
			unsigned int min1 = 0x7F7F7F7F; // ON INITIALISE LE MINIMUM 1 A 127
			unsigned int min2 = 0x7F7F7F7F; // ON INITIALISE LE MINIMUM 2 A 127
			unsigned int tab_vContr[DEG_1];

			//
			//
			//
			__syncthreads();
			if( threadIdx.x < DEG_1){
				iTable[threadIdx.x] = p_indice_nod1[threadIdx.x];
			}
			__syncthreads();
			p_indice_nod1 += DEG_1;

			#pragma unroll
			for (int j = 0; j < DEG_1; j++) {
				tab_vContr[j] = vsubss4(var_nodes[iTable[j] * ii + i], (*p_msg1r)); // 
				p_msg1r += ii;
				min2 = vminu4(min2, vmaxu4(vabs4(tab_vContr[j]), min1));
				min1 = vminu4(min1, vabs4(tab_vContr[j]));
				sign_du_check = sign_du_check ^ vcmpgts4(tab_vContr[j], 0x00000000);
			}

			const unsigned int cste_1 = vminu4(min2, 0x1F1F1F1F);
			const unsigned int cste_2 = vminu4(min1, 0x1F1F1F1F);

			#pragma unroll
			for (int j = 0; j < DEG_1; j++) {
				unsigned int ab          = vabs4(tab_vContr[j]);
				unsigned int m1          = vcmpeq4(ab, min1);
				unsigned int m2          = vcmpne4(ab, min1);
				unsigned int re          = (m1 & cste_1) | (m2 & cste_2);
				unsigned int sign_msg    = sign_du_check ^ vcmpgts4(tab_vContr[j], 0x00000000);
				unsigned int msg_sortant = (re & sign_msg) | (vneg4(re) & (sign_msg ^ 0xFFFFFFFF));
				*p_msg1w = msg_sortant;
				p_msg1w += ii;
				var_nodes[iTable[j] * ii + i] = vaddss4(tab_vContr[j], msg_sortant);
			}
		}

#if NB_DEGRES > 1
		for (int z = 0; z <DEG_2_COMPUTATIONS; z++) {

			unsigned int sign_du_check = 0x00000000;
			unsigned int min1 = 0x7F7F7F7F;
			unsigned int min2 = 0x7F7F7F7F;
			unsigned int tab_vContr [DEG_2];

			//
			//
			//
			__syncthreads();
			if( threadIdx.x < DEG_2){
				iTable[threadIdx.x] = p_indice_nod1[threadIdx.x];
			}
			__syncthreads();
			p_indice_nod1 += DEG_2;

			#pragma unroll
			for (int j = 0; j<DEG_2; j++) {
				tab_vContr[j] = vsubss4(var_nodes[iTable[j] * ii + i], (*p_msg1r));// 
				p_msg1r += ii;
				min2 = vminu4(min2, vmaxu4(vabs4(tab_vContr[j]), min1));
				min1 = vminu4(min1, vabs4(tab_vContr[j]));
				sign_du_check = sign_du_check ^ vcmpgts4(tab_vContr[j], 0x00000000);
			}

			const unsigned int cste_1 = vminu4(min2, 0x1F1F1F1F);
			const unsigned int cste_2 = vminu4(min1, 0x1F1F1F1F);

			#pragma unroll
			for (int j = 0; j < DEG_2; j++) {

				unsigned int ab = vabs4(tab_vContr[j]);
				unsigned int m1 = vcmpeq4( ab, min1 );
				unsigned int m2 = vcmpne4( ab, min1 );
				unsigned int re = (m1 & cste_1) | (m2 & cste_2);
				unsigned int sign_msg = sign_du_check ^ vcmpgts4(tab_vContr[j], 0x00000000);
				unsigned int msg_sortant = (re & sign_msg) | (vneg4(re) & (sign_msg ^ 0xFFFFFFFF));
				*p_msg1w = msg_sortant;
				p_msg1w += ii;
				var_nodes[ iTable[j] * ii + i ] = vaddss4(tab_vContr[j], msg_sortant);
			}
		}
#endif

	}
	__syncthreads();
}

