#include "math_util.h"

#include <assert.h>

void tensor_mul(Tensor* result, const Tensor* lhs, const Tensor* rhs) {
    assert(lhs->shape.num_cols == rhs->shape.num_rows);
    assert(result->shape.num_rows == lhs->shape.num_rows && result->shape.num_cols == rhs->shape.num_cols);
    for (int r = 0; r < result->shape.num_rows; r++) {
        for (int c = 0; c < result->shape.num_cols; c++) {
            float accum = 0;
            for (int i = 0; i < lhs->shape.num_cols; i++) {
                accum += lhs->value[r * lhs->shape.num_cols + i] * rhs->value[i * rhs->shape.num_cols + c];
            }
            result->value[r * result->shape.num_cols + c] = accum;
        }
    }
}

void tensor_add(Tensor* result, const Tensor* lhs, const Tensor* rhs) {
    assert(lhs->shape.num_cols == rhs->shape.num_cols && lhs->shape.num_rows == rhs->shape.num_rows);
    const int num_values = lhs->shape.num_cols * lhs->shape.num_rows;
    for (int i = 0; i < num_values; i++) {
        result->value[i] = lhs->value[i] + rhs->value[i];
    }
}

void tensor_sub(Tensor* result, const Tensor* lhs, const Tensor* rhs) {
    assert(lhs->shape.num_cols == rhs->shape.num_cols && lhs->shape.num_rows == rhs->shape.num_rows);
    const int num_values = lhs->shape.num_cols * lhs->shape.num_rows;
    for (int i = 0; i < num_values; i++) {
        result->value[i] = lhs->value[i] - rhs->value[i];
    }
}

void tensor_transpose_xy(Tensor* result, const Tensor* lhs) {
    assert(result->shape.num_rows == lhs->shape.num_cols && result->shape.num_cols == lhs->shape.num_rows);
    for (int r = 0; r < result->shape.num_rows; r++) {
        for (int c = 0; c < result->shape.num_cols; c++) {
            result->value[r * result->shape.num_cols + c] = lhs->value[c * lhs->shape.num_cols + r];
        }
    }
}

void tensor_comp_func(Tensor* result, const Tensor* lhs, CompFunc func) {
    const int num_values = lhs->shape.num_cols * lhs->shape.num_rows;
    for (int i = 0; i < num_values; i++) {
        result->value[i] = func(lhs->value[i]);
    }
}

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
