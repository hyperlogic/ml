#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <stdint.h>
#include <errno.h>
#include <string.h>

#include "dataset.h"

const char* PARAMS_FILENAME = "params.bin";
const char* TEST_IMAGES_FILENAME = "t10k-images-idx3-ubyte";
const char* TEST_LABELS_FILENAME = "t10k-labels-idx1-ubyte";

const char* image_7 = "\
\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\
\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\
\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\
\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\
\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\
\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\
\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\
\x00\x00\x00\x00\x00\x00\x54\xb9\x9f\x97\x3c\x24\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\
\x00\x00\x00\x00\x00\x00\xde\xfe\xfe\xfe\xfe\xf1\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xc6\xaa\x34\x00\x00\x00\x00\x00\x00\
\x00\x00\x00\x00\x00\x00\x43\x72\x48\x72\xa3\xe3\xfe\xe1\xfe\xfe\xfe\xfa\xe5\xfe\xfe\x8c\x00\x00\x00\x00\x00\x00\
\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x11\x42\x0e\x43\x43\x43\x3b\x15\xec\xfe\x6a\x00\x00\x00\x00\x00\x00\
\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x53\xfd\xd1\x12\x00\x00\x00\x00\x00\x00\
\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x16\xe9\xff\x53\x00\x00\x00\x00\x00\x00\x00\
\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x81\xfe\xee\x2c\x00\x00\x00\x00\x00\x00\x00\
\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x3b\xf9\xfe\x3e\x00\x00\x00\x00\x00\x00\x00\x00\
\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x85\xfe\xbb\x05\x00\x00\x00\x00\x00\x00\x00\x00\
\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x09\xcd\xf8\x3a\x00\x00\x00\x00\x00\x00\x00\x00\x00\
\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x7e\xfe\xb6\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\
\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x4b\xfb\xf0\x39\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\
\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x13\xdd\xfe\xa6\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\
\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x03\xcb\xfe\xdb\x23\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\
\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x26\xfe\xfe\x4d\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\
\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x1f\xe0\xfe\x73\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\
\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x85\xfe\xfe\x34\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\
\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x3d\xf2\xfe\xfe\x34\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\
\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x79\xfe\xfe\xdb\x28\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\
\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x79\xfe\xcf\x12\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\
\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\
";

#define L0_SIZE 784
#define L1_SIZE 512
#define L2_SIZE 10

// structure of params.bin
typedef struct {
    float fc1_weight[L1_SIZE * L0_SIZE];  // 512 x 784
    float fc1_bias[L1_SIZE];
    float fc2_weight[L2_SIZE * L1_SIZE];  // 10 x 512
    float fc2_bias[L2_SIZE];
} Params;


float dot_product(const float* w, const float* x, size_t count) {
    float accum = 0.0f;
    for (size_t i = 0; i < count; i++) {
        accum += w[i] * x[i];
    }
    return accum;
}

// w is matrix with rows x cols elements, x is a vector with cols elements.
// out is a vector with rows elements.
void mat_mul_vec(float* out, const float* w, const float* x, size_t rows, size_t cols) {
    for (size_t r = 0; r < rows; r++) {
        out[r] = dot_product(&w[r * cols], x, cols);
    }
}

float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

int argmax(const float* x, size_t count) {
    float max = -FLT_MAX;
    int max_i = 0;
    for (size_t i = 0; i < count; i++) {
        if (x[i] > max) {
            max = x[i];
            max_i = i;
        }
    }
    return max_i;
}

int eval_model(const Params* p_params, const uint8_t* image) {

    // convert image to flaot
    float fimage[L0_SIZE];
    for (size_t i = 0; i < L0_SIZE; i++) {
        fimage[i] = (float)image[i];
    }

    // layer 1
    float l1[L1_SIZE];
    mat_mul_vec(l1, p_params->fc1_weight, fimage, L1_SIZE, L0_SIZE);

    for (int i = 0; i < L1_SIZE; i++) {
        l1[i] = relu(l1[i] + p_params->fc1_bias[i]);
    }

    // layer 2
    float l2[L2_SIZE];
    mat_mul_vec(l2, p_params->fc2_weight, l1, L2_SIZE, L1_SIZE);
    for (int i = 0; i < L2_SIZE; i++) {
        l2[i] += p_params->fc2_bias[i];
    }

    return argmax(l2, L2_SIZE);
}

Params* load_model(const char* params_filename)
{
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

// be = big endian
uint32_t read_be_uint32(FILE* fp) {
    uint8_t bytes[4];
    size_t bytes_read = fread(&bytes, 1, 4, fp);
    if (bytes_read != 4) {
        printf("fread() error, bytes_read = %u expected = %u\n", (uint32_t)bytes_read, (uint32_t)4);
        return 0;
    }
    return (uint32_t)bytes[3] | (uint32_t)bytes[2] << 8 | (uint32_t)bytes[1] << 16 | (uint32_t)bytes[0] << 24;
}

int read_bytes(FILE* fp, uint8_t* bytes, size_t num_bytes) {
    uint8_t* p = bytes;
    size_t bytes_left = num_bytes;
    size_t bytes_read = 0;
    do {
        bytes_read = fread(p, 1, bytes_left, fp);
        if (bytes_read == 0) {
            printf("read_bytes: error 0 bytes read\n");
            return 0;
        }
        p += bytes_read;
        bytes_left -= bytes_read;
    } while (bytes_read < bytes_left);
    return 1;
}

DataSet* load_data_set(const char* test_images_filename, const char* test_labels_filename)
{
    FILE* i_fp = fopen(test_images_filename, "rb");
    if (!i_fp) {
        printf("Error opening file \"%s\", errno = %d\n", test_images_filename, errno);
        return NULL;
    }

    FILE* l_fp = fopen(test_labels_filename, "rb");
    if (!l_fp) {
        printf("Error opening file \"%s\", errno = %d\n", test_images_filename, errno);
        return NULL;
    }

    ImagesHeader i_header;
    i_header.magic = read_be_uint32(i_fp);
    const uint32_t IMAGE_MAGIC = 2051;
    if (i_header.magic != IMAGE_MAGIC) {
        printf("Error loading \"%s\", bad magic number %d, expected %d\n", test_images_filename, i_header.magic, IMAGE_MAGIC);
        return NULL;
    }
    i_header.num_images = read_be_uint32(i_fp);
    i_header.rows = read_be_uint32(i_fp);
    i_header.cols = read_be_uint32(i_fp);

    LabelsHeader l_header;
    l_header.magic = read_be_uint32(l_fp);
    const uint32_t LABEL_MAGIC = 2049;
    if (l_header.magic != LABEL_MAGIC) {
        printf("Error loading \"%s\", bad magic number %d, expected %d\n", test_labels_filename, l_header.magic, LABEL_MAGIC);
        return NULL;
    }
    l_header.num_labels = read_be_uint32(l_fp);
    if (i_header.num_images != l_header.num_labels) {
        printf("Error image and label count mismatch, num_images = %d, num_labels = %d\n", i_header.num_images, l_header.num_labels);
        return NULL;
    }

    DataSet* data_set = malloc_data_set(&i_header, &l_header);

    size_t i_size = i_header.num_images * i_header.rows * i_header.cols;
    if (!read_bytes(i_fp, data_set->images, i_size)) {
        fclose(i_fp);
        fclose(l_fp);
        free_data_set(data_set);
        return NULL;
    }
    fclose(i_fp);

    if (!read_bytes(l_fp, data_set->labels, l_header.num_labels)) {
        fclose(l_fp);
        free_data_set(data_set);
        return NULL;
    }
    fclose(l_fp);

    return data_set;
}

int main(int argc, const char* argv[]) {
    Params* p_params = load_model(PARAMS_FILENAME);
    DataSet* data_set = load_data_set(TEST_IMAGES_FILENAME, TEST_LABELS_FILENAME);

    // execute model
    int result = eval_model(p_params, (uint8_t*)image_7);
    printf("expected 7, result = %d\n", result);

    printf("testing\n");
    uint32_t i_size = data_set->i_header.rows * data_set->i_header.cols;
    uint32_t num_fails = 0;
    for (int i = 0; i < data_set->i_header.num_images; i++) {
        int result = eval_model(p_params, data_set->images + (i_size * i));
        if (data_set->labels[i] != result) {
            num_fails++;
        }
        if ((i % 100) == 0) {
            printf(".");
        }
    }
    printf("error rate = %.3f\n", 100.0f * (float)num_fails / (float)data_set->i_header.num_images);

    free(p_params);
    free_data_set(data_set);
    return 0;
}
