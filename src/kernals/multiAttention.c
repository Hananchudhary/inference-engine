#include"blis.h"
#include"../../inc/kernals/multiAttention.h"
#include"../../inc/kernals/attention.h"
#include<stdio.h>
// expecting in row major but self attention works in column major
// remeber to use assert for exceptions
// dimensions of Ws are being calculated
void multiAttention(double* W_Q, double* W_K, double* W_V, double* X,
                            uint32_t n, uint32_t d, uint32_t q_h, uint32_t kv_h, 
                            double* W_O, double* out, bool masked){
     uint32_t Q_dim = n * d;
     uint32_t q_cols = d / q_h;
     uint32_t dims = q_cols * kv_h;
     uint32_t group_size = q_h / kv_h;
     uint32_t KV_dim = n * dims;
     double Q[Q_dim];
     double K[KV_dim];
     double V[KV_dim];
     double res[Q_dim];
     double alpha = 1.0;
     double beta = 0.0;
     
     bli_dgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, n, d, d, &alpha, X, d, 1, W_Q, 
          d,1, &beta, Q, 1, n);
     bli_dgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, n, dims, d, &alpha, X, d, 1, W_K, 
          dims,1, &beta, K, 1, n);
     bli_dgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, n, dims, d, &alpha, X, d, 1, W_V, 
          dims,1, &beta, V, 1, n);
     for(int i = 0;i < q_h;i++){
          uint32_t group_idx = i / group_size;
          attention((Q + (i * n * q_cols)), (K + (group_idx * n * q_cols)), 
          (V + (group_idx * n * q_cols)), n, q_cols, (res + (i * n * q_cols)), masked);
     }
     bli_dgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, n, d, d, &alpha, res, 1, n, W_O, 
     d, 1, &beta, out, d, 1);
}
// int main(void)
// {
//     const uint32_t n = 3;
//     const uint32_t d = 4;
//     const uint32_t q_h = 2;
//     const uint32_t kv_h = 2;

//     /* All matrices are COLUMN-MAJOR */

//     /* X =
//        [ 1  2  3  4 ]
//        [ 5  6  7  8 ]
//        [ 9 10 11 12 ]
//     */

//     double X[12] = {
//         1,2,3,4,5,6,7,8,9,10,11,12
//     };

//     /* W_Q = Identity */

//     double W_Q[16] = {
//         1,0,0,0,
//         0,1,0,0,
//         0,0,1,0,
//         0,0,0,1
//     };

//     /* W_K */

//     double W_K[16] = {
//         1,0,1,0,
//         0,1,0,1,
//         1,0,-1,0,
//         0,1,0,-1
//     };

//     /* W_V */

//     double W_V[16] = {
//          2,0,0,0,
//          0,2,0,0,
//          0,0,1,0,
//          0,0,0,1
//     };

//     /* W_O */

//     double W_O[16] = {
//         1,0,0,0,
//         0,1,0,0,
//         0,0,1,0,
//         0,0,0,1
//     };

//     double out[12];

//     multiAttention(
//         W_Q,
//         W_K,
//         W_V,
//         X,
//         n,
//         d,
//         q_h,
//         kv_h,
//         W_O,
//         out,
//         true
//     );

//     printf("Output:\n\n");

//     for (uint32_t i = 0; i < n; i++)
//     {
//         for (uint32_t j = 0; j < d; j++)
//         {
//             printf("%12.6f ", out[j + i*d]);
//         }
//         printf("\n");
//     }

//     return 0;
// }