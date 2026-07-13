#ifndef __SOFTMAX_H__
#define __SOFTMAX_H__
#include<stdint.h>
void softmax(const double* __restrict x, double* __restrict y, const uint32_t n);
#endif