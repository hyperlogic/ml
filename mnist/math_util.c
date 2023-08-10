#include "math_util.h"

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
