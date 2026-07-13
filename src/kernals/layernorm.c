#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "blis.h"

void kernel_layernorm_cpu_f32_forward(
    const float* __restrict embed,
    const float* __restrict weight,
    const float* __restrict bias,
    float* __restrict y,
    const size_t seq_len,
    const size_t embed_dim,
    const float eps
) {

    const float inv_dim = 1.0f / (float)embed_dim;

    for (size_t i = 0; i < seq_len; i++) {
        const float* __restrict token = embed + i * embed_dim;
        float* __restrict output_token = y + i * embed_dim;

        float sum = 0.0f;
        for (size_t j = 0; j < embed_dim; j++) {
            sum += token[j];
        }
        float mean = sum * inv_dim;

        float var = 0.0f;
        for (size_t j = 0; j < embed_dim; j++) {
            float diff = token[j] - mean;
            var = fmaf(diff, diff, var); // fused multiplication additive (a*b + c)
        }
        float inv_std = 1.0f / sqrtf(fmaf(var, inv_dim, eps));

        for (size_t j = 0; j < embed_dim; j++) {
            float norm = (token[j] - mean) * inv_std;
            output_token[j] = fmaf(norm, weight[j], bias[j]);
        }
    }
}
