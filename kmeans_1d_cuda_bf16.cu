/* kmeans_1d_cuda_bf16.cu
   K-means 1D (CUDA C++), implementação para GPU com BF16 (BFloat16).
   - Usa __nv_bfloat16 para precisão de 16 bits e kernels CUDA.
   - Os cálculos internos (SSE, somas) são promovidos para float (FP32) para estabilidade.

   Compilar com o CUDA Toolkit:
   nvcc -O2 -arch=sm_89 kmeans_1d_cuda_bf16.cu -o kmeans_1d_cuda_bf16
   (-arch=sm_89 é para a RTX 4060, Compute Capability 8.9)

   Uso:      ./kmeans_1d_cuda_bf16 dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv]
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <cuda_bf16.h>

#define BF16_MAX (3.402823e38f)

// Macro para checagem de erros CUDA
#define CUDA_CHECK(err) { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "Erro CUDA em %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
}

/* ---------- Funções de I/O (executam na CPU) ---------- */
static int count_rows(const char *path){
    FILE *f = fopen(path, "r");
    if(!f){ fprintf(stderr,"Erro ao abrir %s\n", path); exit(1); }
    int rows=0; char line[8192];
    while(fgets(line,sizeof(line),f)){
        int only_ws=1;
        for(char *p=line; *p; p++){
            if(*p!=' ' && *p!='\t' && *p!='\n' && *p!='\r'){ only_ws=0; break; }
        }
        if(!only_ws) rows++;
    }
    fclose(f);
    return rows;
}

// Leitura adaptada para __nv_bfloat16
static __nv_bfloat16 *read_csv_1col(const char *path, int *n_out){
    int R = count_rows(path);
    if(R<=0){ fprintf(stderr,"Arquivo vazio: %s\n", path); exit(1); }
    __nv_bfloat16 *A = (__nv_bfloat16*)malloc((size_t)R * sizeof(__nv_bfloat16)); // TIPO MUDADO
    if(!A){ fprintf(stderr,"Sem memoria para %d linhas\n", R); exit(1); }
    FILE *f = fopen(path, "r");
    if(!f){ fprintf(stderr,"Erro ao abrir %s\n", path); free(A); exit(1); }
    char line[8192];
    int r=0;
    while(fgets(line,sizeof(line),f)){
        int only_ws=1;
        for(char *p=line; *p; p++){
            if(*p!=' ' && *p!='\t' && *p!='\n' && *p!='\r'){ only_ws=0; break; }
        }
        if(only_ws) continue;
        const char *delim = ",; \t";
        char *tok = strtok(line, delim);
        if(!tok){ fprintf(stderr,"Linha %d sem valor em %s\n", r+1, path); free(A); fclose(f); exit(1); }
        A[r] = __float2bfloat16(strtof(tok, NULL)); // CONVERSÃO MUDADA
        r++;
        if(r>R) break;
    }
    fclose(f);
    *n_out = R;
    return A;
}

static void write_assign_csv(const char *path, const int *assign, int N){
    if(!path) return;
    FILE *f = fopen(path, "w");
    if(!f){ fprintf(stderr,"Erro ao abrir %s para escrita\n", path); return; }
    for(int i=0;i<N;i++) fprintf(f, "%d\n", assign[i]);
    fclose(f);
}

// Escrita adaptada para __nv_bfloat16
static void write_centroids_csv(const char *path, const __nv_bfloat16 *C, int K){
    if(!path) return;
    FILE *f = fopen(path, "w");
    if(!f){ fprintf(stderr,"Erro ao abrir %s para escrita\n", path); return; }
    for(int c=0;c<K;c++) {
        float v = __bfloat162float(C[c]);
        fprintf(f, "%.6f\n", v); 
    }
    fclose(f);
}

/* ---------- Kernels CUDA (executam na GPU) ---------- */

__global__ void assignment_kernel(const __nv_bfloat16 *X, const __nv_bfloat16 *C, int *assign, float *sse_sum, int N, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int best = -1;
    float bestd_f = BF16_MAX;

    for (int c = 0; c < K; c++) {
        float cc = __bfloat162float(C[c]);
        float diff = __bfloat162float(X[i]) - cc;
        float d = diff * diff;

        if (d < bestd_f) {
            bestd_f = d;
            best = c;
        }
    }
    assign[i] = best;
    atomicAdd(sse_sum, bestd_f);
}

__global__ void zero_sums_kernel(__nv_bfloat16 *sum, int *cnt, int K) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= K) return;
    sum[c] = __float2bfloat16(0.0f);
    cnt[c] = 0;
}

__global__ void update_sums_kernel(const __nv_bfloat16 *X, const int *assign, __nv_bfloat16 *sum, int *cnt, int N) { // TIPO MUDADO
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int a = assign[i];
    atomicAdd(&sum[a], (X[i]));
    atomicAdd(&cnt[a], 1);
}

__global__ void update_centroids_kernel(const __nv_bfloat16 *X, __nv_bfloat16 *C, const __nv_bfloat16 *sum, const int *cnt, int K) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= K) return;
    if (cnt[c] > 0) {
        C[c] = sum[c] / __float2bfloat16((float)cnt[c]);
    } else {
        C[c] = X[0];
    }
}

/* ---------- Orquestrador (executa na CPU) ---------- */
static void kmeans_1d(const __nv_bfloat16 *h_X, __nv_bfloat16 *h_C, int *h_assign,
                      int N, int K, int max_iter, __nv_bfloat16 eps,
                      int *iters_out, float *sse_out)
{
    // Alocação de memória na GPU
    __nv_bfloat16 *d_X; 
    __nv_bfloat16 *d_C;
    __nv_bfloat16 *d_sum;
    int *d_assign, *d_cnt;
    float *d_sse_sum;
    CUDA_CHECK(cudaMalloc(&d_X, N * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_C, K * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_assign, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sum, K * sizeof(__nv_bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_cnt, K * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sse_sum, sizeof(float)));

    // Transferência de dados da CPU para a GPU
    CUDA_CHECK(cudaMemcpy(d_X, h_X, N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, K * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));

    int threads_per_block = 256;
    int blocks_N = (N + threads_per_block - 1) / threads_per_block;
    int blocks_K = (K + threads_per_block - 1) / threads_per_block;

    float sse = 0.0f, prev_sse = BF16_MAX;
    int it;
    for (it = 0; it < max_iter; it++) {
        // 1. Passo de Atribuição
        CUDA_CHECK(cudaMemset(d_sse_sum, 0, sizeof(float)));
        assignment_kernel<<<blocks_N, threads_per_block>>>(d_X, d_C, d_assign, d_sse_sum, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&sse, d_sse_sum, sizeof(float), cudaMemcpyDeviceToHost));

        // 2. Checagem de convergência (na CPU)
        float eps_f = __bfloat162float(eps);  
        float rel = fabsf(sse - prev_sse) / (prev_sse > 0.0f ? prev_sse : 1.0f);
        if (rel < eps_f) { it++; break; }
        prev_sse = sse;

        // 3. Passo de Atualização
        zero_sums_kernel<<<blocks_K, threads_per_block>>>(d_sum, d_cnt, K);
        update_sums_kernel<<<blocks_N, threads_per_block>>>(d_X, d_assign, d_sum, d_cnt, N);
        update_centroids_kernel<<<blocks_K, threads_per_block>>>(d_X, d_C, d_sum, d_cnt, K);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Copia os resultados de volta para a CPU
    CUDA_CHECK(cudaMemcpy(h_C, d_C, K * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_assign, d_assign, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Libera memória da GPU
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_assign));
    CUDA_CHECK(cudaFree(d_sum));
    CUDA_CHECK(cudaFree(d_cnt));
    CUDA_CHECK(cudaFree(d_sse_sum));

    *iters_out = it;
    *sse_out = sse;
}

/* ---------- main (executa na CPU) ---------- */
int main(int argc, char **argv){
    if(argc < 3){
        printf("Uso: %s dados.csv centroides_iniciais.csv [max_iter=50] [eps=1e-4] [assign.csv] [centroids.csv]\n", argv[0]);
        return 1;
    }
    const char *pathX = argv[1];
    const char *pathC = argv[2];
    int max_iter = (argc>3)? atoi(argv[3]) : 50;
    float eps   = (argc>4)? strtof(argv[4], NULL) : 1e-4f;
    const char *outAssign   = (argc>5)? argv[5] : NULL;
    const char *outCentroid = (argc>6)? argv[6] : NULL;

    if(max_iter <= 0 || __bfloat162float(eps) <= 0.0f){
        fprintf(stderr,"Parâmetros inválidos: max_iter>0 e eps>0\n");
        return 1;
    }

    int N=0, K=0;
    __nv_bfloat16 *X = read_csv_1col(pathX, &N);
    __nv_bfloat16 *C = read_csv_1col(pathC, &K);
    int *assign = (int*)malloc((size_t)N * sizeof(int));
    if(!assign){ fprintf(stderr,"Sem memoria para assign\n"); free(X); free(C); return 1; }

    clock_t t0 = clock();
    int iters = 0; float sse = 0.0f; 
    kmeans_1d(X, C, assign, N, K, max_iter, eps, &iters, &sse);
    clock_t t1 = clock();
    double ms = 1000.0 * (double)(t1 - t0) / (double)CLOCKS_PER_SEC;

    printf("K-means 1D (CUDA BF16)\n");
    printf("N=%d K=%d max_iter=%d eps=%g\n", N, K, max_iter, __bfloat162float(eps));
    printf("Iterações: %d | SSE final: %.6f | Tempo: %.1f ms\n", iters, sse, ms);

    write_assign_csv(outAssign, assign, N);
    write_centroids_csv(outCentroid, C, K);

    free(assign); free(X); free(C);
    return 0;
}