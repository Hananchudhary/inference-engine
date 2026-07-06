#include"../../inc/kernals/relu.h"

void RELU(double* x, uint32_t len) {
    uint64_t mask = 0x7FFFFFFFFFFFFFFF;
    for (int i = 0;i < len;i++) {
        uint64_t bits;
        memcpy(&bits, (x + i), sizeof(bits));
        uint64_t val = bits;
        val &= mask;
        double value;
        memcpy(&value, &val, sizeof(bits));
        x[i] += value;
        x[i] /= 2;
    }
}

// void RELU(double* x, uint32_t len) {
//     for (int i = 0;i < len;i++) {
//         fmaxf(0.0f, x[i]);
//     }
//     return;
// }
