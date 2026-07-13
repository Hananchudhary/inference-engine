#include"../../inc/kernals/sigmoid.h"
#include <math.h>
void sigmoid(const double* __restrict x, double* __restrict y,const uint32_t len){
    for (int i = 0;i < len;i++) {
        y[i] = 1 / (1 + exp(-1 * x[i]));
    }
}