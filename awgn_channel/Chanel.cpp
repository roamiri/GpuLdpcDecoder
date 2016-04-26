
#include <iostream>
#include "Chanel.h"


Chanel::Chanel(CFrame *t, int _BITS_LLR, bool QPSK, bool ES_N0)
{
    _vars        = t->nb_vars();
    _data        = t->nb_data();
    _checks      = t->nb_checks();
    t_noise_data = t->get_t_noise_data();
    BITS_LLR     = _BITS_LLR;
    qpsk         = QPSK;
    es_n0        = ES_N0;
	_frames		 = t->nb_frames();
}

Chanel::~Chanel()
{
	std::cout << "calling " << __FUNCTION__ << std::endl;
}
