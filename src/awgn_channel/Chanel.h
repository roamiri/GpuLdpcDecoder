#ifndef CLASS_Chanel
#define CLASS_Chanel

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "frame/CFrame.h"

#define small_pi  3.1415926536
#define _2pi  (2.0 * small_pi)

class Chanel
{
    
protected:
	size_t  _vars;
	size_t  _checks;
	size_t  _data;
    int  BITS_LLR;
//    int* data_in;
    int* data_out;
    bool qpsk;
    bool es_n0;
    size_t _frames;
    
    float*  t_noise_data;   // size (width)
    int*    t_coded_bits;   // size (width)
    
    double performance;
    double SigB;
    double Gauss;
    double Ph;
    double Qu;
    double Eb_N0;
    
public:
    Chanel(CFrame *t, int _BITS_LLR, bool QPSK, bool Es_N0);
    virtual ~Chanel();
    virtual void configure(double _Eb_N0) = 0;  
    virtual void generate() = 0;                
};

#endif

