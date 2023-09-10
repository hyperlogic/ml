#include <stdlib.h>
#include <float.h>

#ifndef __MATH_UTIL__
#define __MATH_UTIL__

#define MAX_NUM_VALUES 32

// only 2d for now
typedef struct {
    int num_rows;
    int num_cols;
} Shape;

typedef struct {
    Shape shape;
    float value[MAX_NUM_VALUES];
} Tensor;

void tensor_mul(Tensor* result, const Tensor* lhs, const Tensor* rhs);
void tensor_add(Tensor* result, const Tensor* lhs, const Tensor* rhs);
void tensor_sub(Tensor* result, const Tensor* lhs, const Tensor* rhs);
void tensor_transpose_xy(Tensor* result, const Tensor* lhs);

typedef float (*CompFunc)(float);

void tensor_comp_func(Tensor* result, const Tensor* lhs, CompFunc func);

float dot_product(const float* w, const float* x, size_t count);

float relu(float x);

int argmax(const float* x, size_t count);

#endif
