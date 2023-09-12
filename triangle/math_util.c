#include "math_util.h"

#include <assert.h>
#include <stdio.h>

#define CHECK_SHAPE_SIZE(tensor) assert(tensor->shape.num_cols * tensor->shape.num_rows <= MAX_TENSOR_VALUES)

void tensor_copy(Tensor* result, const Tensor* t) {
    CHECK_SHAPE_SIZE(t);
    result->shape = t->shape;
    const int num_values = t->shape.num_cols * t->shape.num_rows;
    for (int i = 0; i < num_values; i++) {
        result->value[i] = t->value[i];
    }
}

void tensor_mul(Tensor* result, const Tensor* lhs, const Tensor* rhs) {
    CHECK_SHAPE_SIZE(lhs);
    CHECK_SHAPE_SIZE(rhs);

    assert(result->shape.num_cols * result->shape.num_rows <= MAX_TENSOR_VALUES);
    assert(lhs->shape.num_cols * lhs->shape.num_rows <= MAX_TENSOR_VALUES);
    assert(rhs->shape.num_cols * rhs->shape.num_rows <= MAX_TENSOR_VALUES);

    // ensure shapes make sense for matrix multiplication
    assert(lhs->shape.num_cols == rhs->shape.num_rows);

    result->shape.num_rows = lhs->shape.num_rows;
    result->shape.num_cols = rhs->shape.num_cols;

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
    CHECK_SHAPE_SIZE(lhs);
    CHECK_SHAPE_SIZE(rhs);

    assert(lhs->shape.num_cols == rhs->shape.num_cols && lhs->shape.num_rows == rhs->shape.num_rows);
    result->shape = lhs->shape;
    const int num_values = lhs->shape.num_cols * lhs->shape.num_rows;
    for (int i = 0; i < num_values; i++) {
        result->value[i] = lhs->value[i] + rhs->value[i];
    }
}

void tensor_sub(Tensor* result, const Tensor* lhs, const Tensor* rhs) {
    CHECK_SHAPE_SIZE(lhs);
    CHECK_SHAPE_SIZE(rhs);

    assert(lhs->shape.num_cols == rhs->shape.num_cols && lhs->shape.num_rows == rhs->shape.num_rows);
    result->shape = lhs->shape;
    const int num_values = lhs->shape.num_cols * lhs->shape.num_rows;
    for (int i = 0; i < num_values; i++) {
        result->value[i] = lhs->value[i] - rhs->value[i];
    }
}

void tensor_compmul(Tensor* result, const Tensor* lhs, const Tensor* rhs) {
    CHECK_SHAPE_SIZE(lhs);
    CHECK_SHAPE_SIZE(rhs);

    assert(lhs->shape.num_cols == rhs->shape.num_cols && lhs->shape.num_rows == rhs->shape.num_rows);
    result->shape = lhs->shape;
    const int num_values = lhs->shape.num_cols * lhs->shape.num_rows;
    for (int i = 0; i < num_values; i++) {
        result->value[i] = lhs->value[i] * rhs->value[i];
    }
}

void tensor_transpose_xy(Tensor* result, const Tensor* t) {
    CHECK_SHAPE_SIZE(t);

    result->shape.num_rows = t->shape.num_cols;
    result->shape.num_cols = t->shape.num_rows;
    for (int r = 0; r < result->shape.num_rows; r++) {
        for (int c = 0; c < result->shape.num_cols; c++) {
            result->value[r * result->shape.num_cols + c] = t->value[c * t->shape.num_cols + r];
        }
    }
}

void tensor_neg(Tensor* result, const Tensor* t) {
    CHECK_SHAPE_SIZE(result);
    CHECK_SHAPE_SIZE(t);

    const int num_values = t->shape.num_cols * t->shape.num_rows;
    for (int i = 0; i < num_values; i++) {
        result->value[i] = -t->value[i];
    }
}

void tensor_comp_func(Tensor* result, const Tensor* t, CompFunc func) {
    CHECK_SHAPE_SIZE(result);
    CHECK_SHAPE_SIZE(t);

    const int num_values = t->shape.num_cols * t->shape.num_rows;
    for (int i = 0; i < num_values; i++) {
        result->value[i] = func(t->value[i]);
    }
}

void tensor_print(const Tensor* t, int num_indent_spaces) {
    CHECK_SHAPE_SIZE(t);

    for (int r = 0; r < t->shape.num_rows; r++) {
        for (int i = 0; i < num_indent_spaces; i++) {
            printf(" ");
        }
        printf("|");

        for (int c = 0; c < t->shape.num_cols; c++) {
            printf("%9.4f", t->value[r * t->shape.num_cols + c]);
        }
        printf(" |\n");
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
