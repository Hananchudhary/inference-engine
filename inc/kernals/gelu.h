#ifndef GELU_H
#define GELU_H
#include<stdio.h>
#include<stdint.h>

void kernel_gelu_cpu_f32_forward(
    const float* __restrict x,
    float* __restrict y,
    size_t length
);


#endif