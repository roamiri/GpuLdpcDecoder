#ifndef CLASS_FRAME
#define CLASS_FRAME

#include <stdio.h>
#include <stdlib.h>

class CFrame
{
    
protected:
    int  _width;
    int  _height;
    int  _frame;
    
    float*  t_noise_data;   // size (width)
    int*    t_fpoint_data;  // size (width/4)
    int*    t_decode_data;  // size (var)
    
public:
    CFrame(int width, int height);
    CFrame(int width, int height, int frame);
    ~CFrame();
    
    int nb_vars();
    int nb_checks();
    int nb_data();
    int nb_frames();

    float* get_t_noise_data();
    int*   get_t_fpoint_data();
    int*   get_t_decode_data();
};

#endif
