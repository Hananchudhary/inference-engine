#include"../../inc/kernals/gelu.h"
#include <math.h>

void kernel_gelu_cpu_f32_forward(const float* __restrict x, float* __restrict y, const size_t length){
    const float half = 0.5f;
    const float root_two_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;

    for (size_t i = 0; i < length; i++) {
        float x1 = x[i];
        float x3 = x1 * x1 * x1;
        float inner = root_two_over_pi * (x1 + coeff * x3);
        y[i] = half * x1 * (1.0f + tanhf(inner));
    }
}