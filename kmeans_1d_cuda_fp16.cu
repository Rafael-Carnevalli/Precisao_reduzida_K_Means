/* kmeans_1d_cuda_fp16.cu
   K-means 1D (CUDA C++), implementação para GPU com FP16 .
   - Armazena dados em __half (FP16)
   - Compilar com: nvcc -O2 -arch=sm_89 kmeans_1d_cuda_fp16.cu -o kmeans_1d_cuda_fp16
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_fp16.h> // Inclui __half e funções de conversão

#define FLT_MAX 3.402823466e+38F
#define FP16_MAX (__float2half(65504.0f))

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

static __half *read_csv_1col(const char *path, int *n_out){
    int R = count_rows(path);
    if(R<=0){ fprintf(stderr,"Arquivo vazio: %s\n", path); exit(1); }
    __half *A = (__half*)malloc((size_t)R * sizeof(__half));
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
        //A[r] = __float2half(strtof(tok, NULL));
        A[r] = __double2half(atof(tok)); // normaliza para evitar overflow em FP16
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

static void write_centroids_csv(const char *path, const __half *C, int K){
    if(!path) return;
    FILE *f = fopen(path, "w");
    if(!f){ fprintf(stderr,"Erro ao abrir %s para escrita\n", path); return; }
    for(int c=0;c<K;c++) {
        float v = __half2float(C[c]);      // converte de __half para float
        fprintf(f, "%.8f\n", v);           // imprime como float
    }
    fclose(f);
}

/* ---------- Kernels CUDA (executam na GPU) ---------- */

// Passo de atribuição: cálculo de distância em FP32, SSE acumulada em float
__global__ void assignment_kernel(const __half *X, const __half *C,
                                  int *assign, float *sse_sum,
                                  int N, int K)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float xi = __half2float(X[i]);
    int best = -1;
    float bestd = FLT_MAX;

    for (int c = 0; c < K; c++) {
        float cc   = __half2float(C[c]);
        float diff = xi - cc;
        float d    = diff * diff; // distância ao quadrado em FP32
        if (d < bestd) {
            bestd = d;
            best  = c;
        }
    }

    assign[i] = best;
    atomicAdd(sse_sum, bestd);
}

// Zera somas e contadores: sum em half, cnt em int
__global__ void zero_sums_kernel(__half *sum, int *cnt, int K)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= K) return;
    sum[c] = __float2half(0.0f);
    cnt[c] = 0;
}

// Atualiza somas por cluster: atomicAdd em __half (sm_70+)
__global__ void update_sums_kernel(const __half *X, const int *assign,
                                   __half *sum, int *cnt, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int a = assign[i];
    // atomicAdd para __half é suportado em sm_70+ (sua sm_89 suporta)
    atomicAdd(&sum[a], X[i]);
    atomicAdd(&cnt[a], 1);
}

// Atualiza centróides: divisão em float (mínimo necessário), clamp e volta a half
__global__ void update_centroids_kernel(__half *C, const __half *sum,
                                        const int *cnt, int K)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= K) return;

    int n = cnt[c];
    if (n > 0) {
        // Converter apenas o necessário para divisão
        float s = __half2float(sum[c]);
        float nc = s / (float)n;

        // Clamp à faixa de FP16
        if (nc > 65504.0f)  nc = 65504.0f;
        if (nc < -65504.0f) nc = -65504.0f;

        C[c] = __float2half(nc);
    }
}

/* ---------- Orquestrador (executa na CPU) ---------- */
static void kmeans_1d(const __half *h_X, __half *h_C, int *h_assign,
                      int N, int K, int max_iter, __half eps,
                      int *iters_out, float *sse_out)
{
    __half *d_X, *d_C;
    __half *d_sum;
    int *d_assign, *d_cnt;
    float *d_sse_sum;

    CUDA_CHECK(cudaMalloc(&d_X, N * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_C, K * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_assign, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sum, K * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_cnt, K * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sse_sum, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_X, h_X, N * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, K * sizeof(__half), cudaMemcpyHostToDevice));

    int threads_per_block = 256;
    int blocks_N = (N + threads_per_block - 1) / threads_per_block;
    int blocks_K = (K + threads_per_block - 1) / threads_per_block;

    
    float sse = 0.0f, prev_sse = FLT_MAX;
    int it;
    for (it = 0; it < max_iter; it++) {
        // 1) Atribuição
        CUDA_CHECK(cudaMemset(d_sse_sum, 0, sizeof(float)));
        assignment_kernel<<<blocks_N, threads_per_block>>>(d_X, d_C, d_assign, d_sse_sum, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(&sse, d_sse_sum, sizeof(float), cudaMemcpyDeviceToHost));

        // 2) Convergência: comparar em float com eps convertido
        float eps_f = __half2float(eps);
        float rel = fabsf(sse - prev_sse) / (prev_sse > 0.0f ? prev_sse : 1.0f);
        if (rel < eps_f) { it++; break; }
        prev_sse = sse;

        // 3) Atualização
        zero_sums_kernel<<<blocks_K, threads_per_block>>>(d_sum, d_cnt, K);
        update_sums_kernel<<<blocks_N, threads_per_block>>>(d_X, d_assign, d_sum, d_cnt, N);
        update_centroids_kernel<<<blocks_K, threads_per_block>>>(d_C, d_sum, d_cnt, K);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaMemcpy(h_C, d_C, K * sizeof(__half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_assign, d_assign, N * sizeof(int), cudaMemcpyDeviceToHost));

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
    __half eps   = (argc>4)? __float2half(strtof(argv[4], NULL)) : __float2half(1e-4f);
    const char *outAssign   = (argc>5)? argv[5] : NULL;
    const char *outCentroid = (argc>6)? argv[6] : NULL;

    if(max_iter <= 0 || __half2float(eps) <= 0.0f){
        fprintf(stderr,"Parâmetros inválidos: max_iter>0 e eps>0\n");
        return 1;
    }

    int N=0, K=0;
    __half *X = read_csv_1col(pathX, &N);
    __half *C = read_csv_1col(pathC, &K);
    int *assign = (int*)malloc((size_t)N * sizeof(int));
    if(!assign){ fprintf(stderr,"Sem memoria para assign\n"); free(X); free(C); return 1; }

    clock_t t0 = clock();
    int iters = 0; float sse = 0.0f; 
    kmeans_1d(X, C, assign, N, K, max_iter, eps, &iters, &sse);
    clock_t t1 = clock();
    double ms = 1000.0 * (double)(t1 - t0) / (double)CLOCKS_PER_SEC;

    printf("K-means 1D (CUDA FP16 - MIXED)\n");
    printf("N=%d K=%d max_iter=%d eps=%g\n", N, K, max_iter, __half2float(eps));
    printf("Iterações: %d | SSE final: %.6g | Tempo: %.1f ms\n", iters, sse, ms);

    write_assign_csv(outAssign, assign, N);
    write_centroids_csv(outCentroid, C, K);

    free(assign); free(X); free(C);
    return 0;
}