#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Definição de tolerância (para evitar divisão por zero no MAPE)
#define EPSILON 1e-10

// Funções de I/O (count_rows e read_centroids) são as mesmas do exemplo anterior
// ... (mantenha count_rows e read_centroids exatamente como estavam)

// --- Funções de I/O (Mantidas do Código Anterior) ---

static int count_rows(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    int rows = 0;
    char line[8192];
    while (fgets(line, sizeof(line), f)) {
        int only_ws = 1;
        for (char *p = line; *p; p++) {
            if (*p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') { only_ws = 0; break; }
        }
        if (!only_ws) rows++;
    }
    fclose(f);
    return rows;
}

static double *read_centroids(const char *path, int *K_out) {
    int K = count_rows(path);
    if (K <= 0) {
        fprintf(stderr, "Erro: Arquivo vazio ou nao encontrado: %s\n", path);
        exit(1);
    }
    double *C = (double*)malloc((size_t)K * sizeof(double));
    if (!C) {
        fprintf(stderr, "Erro: Sem memoria para %d centroides\n", K);
        exit(1);
    }
    FILE *f = fopen(path, "r");
    char line[8192];
    int k = 0;
    while (fgets(line, sizeof(line), f)) {
        char *tok = strtok(line, ",; \t");
        if (tok) {
            C[k] = strtod(tok, NULL); 
            k++;
        }
    }
    fclose(f);
    *K_out = K;
    return C;
}


// --- Função para Calcular o Erro Quadrático Médio (MSE) ---
static double calculate_mse(const double *C_ref, const double *C_test, int K) {
    double total_squared_error = 0.0;
    for (int k = 0; k < K; k++) {
        double diff = C_ref[k] - C_test[k];
        total_squared_error += diff * diff;
    }
    return total_squared_error / K;
}

// --- Função para Calcular o Erro Percentual Absoluto Médio (MAPE) ---
static double calculate_mape(const double *C_ref, const double *C_test, int K) {
    double total_absolute_percentage_error = 0.0;
    int valid_points = 0;

    for (int k = 0; k < K; k++) {
        double ref = C_ref[k];
        double test = C_test[k];
        
        // Evitar divisão por zero: se a referência for muito próxima de zero,
        // a contribuição desse ponto é ignorada ou tratada separadamente, dependendo da sua métrica de validação.
        // Aqui, garantimos que ref não seja zero.
        if (fabs(ref) > EPSILON) {
            double absolute_error = fabs(ref - test);
            double percentage_error = absolute_error / fabs(ref);
            
            total_absolute_percentage_error += percentage_error;
            valid_points++;
        }
    }
    
    // Retorna a média apenas dos pontos válidos
    if (valid_points == 0) return 0.0;
    return (total_absolute_percentage_error / valid_points) * 100.0; // Multiplica por 100 para dar o resultado em %
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Uso: %s [arquivo_referencia_FP64.csv] [arquivo_teste_FPXX.csv]\n", argv[0]);
        return 1;
    }
    const char *path_ref = argv[1];
    const char *path_test = argv[2];

    int K_ref = 0, K_test = 0;
    double *C_ref = read_centroids(path_ref, &K_ref);
    double *C_test = read_centroids(path_test, &K_test);

    if (K_ref != K_test) {
        fprintf(stderr, "Erro: Os arquivos têm numeros diferentes de centroides (%d vs %d).\n", K_ref, K_test);
        free(C_ref); free(C_test);
        return 1;
    }

    double mse = calculate_mse(C_ref, C_test, K_ref);
    double mape = calculate_mape(C_ref, C_test, K_ref);

    printf("====================================================\n");
    printf(" Comparacao de Centroides\n");
    printf(" Referencia: %s\n", path_ref);
    printf(" Teste:      %s\n", path_test);
    printf(" Numero de Centroides (K): %d\n", K_ref);
    printf("----------------------------------------------------\n");
    printf(" Erro Quadratico Medio (MSE):      %.12e\n", mse);
    printf(" Erro Percentual Absoluto Medio (MAPE): %.4f%%\n", mape);
    printf("====================================================\n");

    free(C_ref);
    free(C_test);
    return 0;
}