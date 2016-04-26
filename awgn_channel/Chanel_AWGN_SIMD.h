#ifndef CLASS_Chanel_AWGN_SIMD
#define CLASS_Chanel_AWGN_SIMD

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <curand.h>

#include "Chanel.h"
#include "custom_api/custom_cuda.h"

class Chanel_AWGN_SIMD : public Chanel
{
private:
//     double awgn(double amp);
    float *device_A;
    float *device_B;
    float *device_R;
	curandGenerator_t generator;

	unsigned int SEQ_LEVEL;

public:
	Chanel_AWGN_SIMD(CFrame *t, int _BITS_LLR, bool QPSK, bool Es_N0);
    ~Chanel_AWGN_SIMD();
    
    virtual void configure(double _Eb_N0);
    virtual void generate();
};

#endif

