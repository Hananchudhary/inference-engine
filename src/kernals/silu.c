#include"../../inc/kernals/silu.h"
#include <math.h>

void SILU(const double* __restrict x, double* __restrict y, const uint32_t len){
    for (int i = 0;i < len;i++) {
        y[i] = x[i] / (1 + exp(-1 * x[i]));
    }
}