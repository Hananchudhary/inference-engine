#include"../../inc/kernals/gelu.h"
#include <math.h>

void kernel_gelu_cpu_f32_forward(
    const float* __restrict input,
    float* __restrict output,
    const size_t length
) {
    float half = 0.5f;
    float root_two_over_pi = 0.7978845608028654f; // sqrt(2/pi)
    for (size_t i = 0; i < length; i++) {
        output[i] = half * (1 + tanhf(root_two_over_pi * (input[i] + 0.044715f * input[i] * input[i] * input[i])));
    }

}