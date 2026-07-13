#ifndef __MULTIATTENTION_H__
#define __MULTIATTENTION_H__

#include<stdint.h>
void multiAttention(const double* __restrict W_Q,const double* __restrict W_K, 
    const double* __restrict W_V,const double* __restrict X, const uint32_t n, 
    const uint32_t d, const uint32_t q_h, const uint32_t kv_h, 
    const double* __restrict W_O, double* __restrict out, const bool masked);
#endif