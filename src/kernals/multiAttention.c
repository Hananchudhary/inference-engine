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
     uint32_t kv_cols = d / kv_h;
     uint32_t dims = q_cols * kv_h;
     uint32_t group_size = q_h / kv_h;
     uint32_t KV_dim = n * dims;
     double Q[Q_dim];
     double K[KV_dim];
     double V[KV_dim];
     double res[Q_dim];
     double alpha = 1.0;
     double beta = 0.0;
     
     bli_dgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, n, d, n, &alpha, W_Q, n, 1, X, 
          d,1, &beta, Q, 1, n);
     bli_dgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, n, dims, n, &alpha, W_K, n, 1, X, 
          d,1, &beta, K, 1, n);
     bli_dgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, n, dims, n, &alpha, W_V, n, 1, X, 
          d,1, &beta, V, 1, n);
     for(int i = 0;i < q_h;i++){
          uint32_t group_idx = i / group_size;
          attention((Q + (i * q_cols)), (K + (group_idx * kv_cols)), 
          (V + (group_idx * kv_cols)), n, q_cols, (res + (i * q_cols)), masked);
     }
     bli_dgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, n, d, n, &alpha, W_O, n, 1, res, 
     1, n, &beta, out, d, 1);
}
int main(){
    double X[4] =
     {
          1, 0,
          0, 1
     };
     double W_Q[4] = {
          1, 0,
          0, 1
     };
     double W_K[4] = {
          1, 0,
          0, 1
     };
     double W_V[4] = {
          1, 0,
          0, 1
     };
     double W_O[4] = {
          1,0,0,1
     };
     double out[4];

     uint32_t n = 2;
     uint32_t d = 2;
     multiAttention(W_Q, W_K, W_V, X, n, d, 1, 1, W_O, out, true);
     // for(int i = 0;i < n;i++){
     //      for(int j = 0;j < d;j++){
     //           printf("%f", out[i * d + j]);
     //           printf(" ");
     //      }
     //      printf("\n");
     // }
     return 0;
}