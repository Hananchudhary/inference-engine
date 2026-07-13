#include "../../inc/kernals/rmsnorm.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void kernel_rmsnorm_cpu_f32_forward(
    const float* __restrict embed,
    const float* __restrict weight,
    float* __restrict y,
    const size_t seq_len,
    const size_t embed_dim,
    const float eps
){

    const float inv_dim = 1.0f / (float)embed_dim;

    for (size_t i = 0; i < seq_len; i++) {
        const float* __restrict token = embed + i * embed_dim;
        float* __restrict output_token = y + i * embed_dim;

        float sum_sq = 0.0f;
        for (size_t j = 0; j < embed_dim; j++) {
            float val = token[j];
            sum_sq = fmaf(val, val, sum_sq); // fused multiplication additive (a*b + c)
        }

        // Compute inverse RMS: 1 / sqrt(mean(x²) + eps)
        float inv_rms = 1.0f / sqrtf(fmaf(sum_sq, inv_dim, eps));

        for (size_t j = 0; j < embed_dim; j++) {
            output_token[j] = token[j] * inv_rms * weight[j];
        }
    }
}
