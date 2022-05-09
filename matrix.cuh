#pragma once

#include "symnode.cuh"

class Matrix {
public:
	__host__ __device__ Matrix();
  __host__ __device__ Matrix(int rows, int cols);
  __host__ __device__ Matrix(int rows, int cols, SymNode ** elements);
  __host__ __device__ Matrix(int rows, int cols, const float values[]);
  __host__ __device__ Matrix(int rows, int cols, const cuFloatComplex values[]);

  __host__ __device__ ~Matrix();

  __host__ __device__ Matrix operator*(Matrix & other);
  __host__ __device__ Matrix tensor(Matrix & other);
  __host__ __device__ SymNode * operator[](int i);
  __host__ __device__ void subst(const char ** names, const char ** new_names);
  __host__ __device__ cuFloatComplex * eval();

protected:
  int rows_, cols_;
  SymNode ** elements;
};
