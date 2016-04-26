#include "CFrame.h"

#include "custom_api/custom_cuda.h"

CFrame::CFrame(int width, int height)
{
    _width        = width;
    _height       = height;
    _frame        = 1;

    CUDA_MALLOC_HOST(&t_noise_data, nb_data() + 1, __FILE__, __LINE__);
    CUDA_MALLOC_HOST(&t_fpoint_data, nb_data() + 1, __FILE__, __LINE__);
    CUDA_MALLOC_HOST(&t_decode_data, nb_data() + 1, __FILE__, __LINE__);
}

CFrame::CFrame(int width, int height, int frame)
{
    _width        = width;
    _height       = height;
    _frame        = frame;
	
    CUDA_MALLOC_HOST(&t_noise_data, nb_data()  * frame + 4, __FILE__, __LINE__);
    CUDA_MALLOC_HOST(&t_fpoint_data, nb_data() * frame + 4, __FILE__, __LINE__);
    CUDA_MALLOC_HOST(&t_decode_data, nb_data() * frame + 4, __FILE__, __LINE__);
}


CFrame::~CFrame()
{
	cudaFreeHost(t_noise_data);
	cudaFreeHost(t_fpoint_data);
	cudaFreeHost(t_decode_data);
}

int CFrame::nb_vars(){
    return  /*nb_frames() * */(nb_data()-nb_checks());
}

int CFrame::nb_frames(){
    return  _frame;
}

int CFrame::nb_checks(){
    return _height;
}

int CFrame::nb_data(){
    return _width;
}


float* CFrame::get_t_noise_data(){
    return t_noise_data;
}

int* CFrame::get_t_fpoint_data(){
    return t_fpoint_data;
}

int* CFrame::get_t_decode_data(){
    return t_decode_data;
}

