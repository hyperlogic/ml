#include <stdlib.h>
#include <float.h>

#ifndef __MATH_UTIL__
#define __MATH_UTIL__

// only 2d for now
typedef struct {
    int num_rows;
    int num_cols;
} Shape;

#define MAX_TENSOR_VALUES 32

typedef struct {
    Shape shape;
    float value[MAX_TENSOR_VALUES];
} Tensor;

void tensor_copy(Tensor* result, const Tensor* t);
void tensor_mul(Tensor* result, const Tensor* lhs, const Tensor* rhs);
void tensor_add(Tensor* result, const Tensor* lhs, const Tensor* rhs);
void tensor_sub(Tensor* result, const Tensor* lhs, const Tensor* rhs);
void tensor_compmul(Tensor* result, const Tensor* lhs, const Tensor* rhs);
void tensor_transpose_xy(Tensor* result, const Tensor* t);
void tensor_neg(Tensor* result, const Tensor* t);

typedef float (*CompFunc)(float);

void tensor_comp_func(Tensor* result, const Tensor* t, CompFunc func);

void tensor_print(const Tensor* t, int num_indent_spaces);

float relu(float x);

int argmax(const float* x, size_t count);

#endif
