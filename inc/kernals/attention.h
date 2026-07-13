#ifndef __ATTENTION_H__
#define __ATTENTION_H__
#include<stdint.h>
#include<stdbool.h>
void attention(const double* __restrict Q, const double* __restrict K, 
    const double* __restrict V,const uint32_t n, const uint32_t d, double*__restrict out, 
    const bool masked);
#endif