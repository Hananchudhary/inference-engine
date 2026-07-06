#include"../../inc/kernals/gelu.h"
#include <math.h>

void kernel_gelu_cpu_f32_forward(
    const float* __restrict input,
    float* __restrict output,
    const size_t length
) {
    const float half = 0.5f;
    const float root_two_over_pi = 0.7978845608028654f;
    const float coeff = 0.044715f;

    for (size_t i = 0; i < length; i++) {
        float x = input[i];
        float x3 = x * x * x;
        float inner = root_two_over_pi * (x + coeff * x3);
        output[i] = half * x * (1.0f + tanhf(inner));
    }
}