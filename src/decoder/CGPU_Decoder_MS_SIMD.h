
#include "CGPUDecoder.h"
#include "cuda/CUDA_MS_SIMD.h"

class CGPU_Decoder_MS_SIMD : public CGPUDecoder
{
public:
	CGPU_Decoder_MS_SIMD(size_t _nb_frames, size_t n, size_t k, size_t m );
    ~CGPU_Decoder_MS_SIMD();
	void decode(float var_nodes[1024], int Rprime_fix[1024], int number_iteration, bool stream);
};
