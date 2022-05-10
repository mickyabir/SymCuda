#pragma once

#include "cuComplex.cuh"

__host__ __device__ int cuStrcmp(const char * p1, const char * p2);
__host__ __device__ size_t cuStrlen(const char * s);
__host__ __device__ void cuStrcpy(char * dest, const char * src);
__host__ __device__ void print(cuFloatComplex c);
