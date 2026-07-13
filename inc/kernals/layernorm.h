#ifndef LAYER_NORM_H
#define LAYER_NORM_H
#include<stdio.h>
#include<stdint.h>

void kernel_layernorm_cpu_f32_forward(
    const float* __restrict embed,
    const float* __restrict weight,
    const float* __restrict bias,
    float* __restrict y,
    const size_t seq_len,
    const size_t embed_dim,
    const float eps
);

#endif