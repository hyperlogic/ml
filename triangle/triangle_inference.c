#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <errno.h>
#include <string.h>

const char* PARAMS_FILENAME = "params.bin";

#define L0_SIZE 2
#define L1_SIZE 16
#define L2_SIZE 2

// structure of params.bin
typedef struct {
    float fc1_weight[L1_SIZE * L0_SIZE];  // 16 x 2
    float fc1_bias[L1_SIZE];
    float fc2_weight[L2_SIZE * L1_SIZE];  // 2 x 16
    float fc2_bias[L2_SIZE];
} Params;


Params* load_model(const char* params_filename) {
    // open params file
    FILE* fp = fopen(params_filename, "rb");
    if (!fp) {
        printf("Error opening file \"%s\", errno = %d\n", params_filename, errno);
        return NULL;
    }

    // read data
    Params* p_params = malloc(sizeof(Params));
    size_t bytes_read = fread(p_params, 1, sizeof(Params), fp);
    if (bytes_read != sizeof(Params)) {
        printf("Error reading from file \"%s\", bytes_read = %u, expected = %u\n", params_filename, (uint32_t)bytes_read, (uint32_t)sizeof(Params));
        free(p_params);
        return NULL;
    }
    fclose(fp);

    return p_params;
}

void print_matrix(const float* mat, int num_rows, int num_cols) {
    for (int r = 0; r < num_rows; r++) {
        for (int c = 0; c < num_cols; c++) {
            printf("%8.4f", mat[r * num_cols + c]);
        }
        printf("\n");
    }
}

void print_params(const Params* p_params)
{
    printf("fc1_weight =\n");
    print_matrix(p_params->fc1_weight, L1_SIZE, L0_SIZE);
    printf("fc1_bias =\n");
    print_matrix(p_params->fc1_bias, 1, L1_SIZE);
    printf("fc2_weight =\n");
    print_matrix(p_params->fc2_weight, L2_SIZE, L1_SIZE);
    printf("fc2_bias =\n");
    print_matrix(p_params->fc2_bias, 1, L2_SIZE);
}


int main(int argc, const char* argv[]) {
    Params* p_params = load_model(PARAMS_FILENAME);

    print_params(p_params);

    free(p_params);
    return 0;
}
