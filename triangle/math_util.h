#include <stdlib.h>
#include <float.h>

#ifndef __MATH_UTIL__
#define __MATH_UTIL__

float dot_product(const float* w, const float* x, size_t count);

// w is matrix with rows x cols elements, x is a vector with cols elements.
// out is a vector with rows elements.
void mat_mul_vec(float* out, const float* w, const float* x, size_t rows, size_t cols);

float relu(float x);

int argmax(const float* x, size_t count);

#endif
