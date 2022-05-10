#pragma once

#include "symnode.cuh"

class Matrix {
public:
	__host__ __device__ Matrix();
  __host__ __device__ Matrix(int rows, int cols);
  __host__ __device__ Matrix(int rows, int cols, SymNode ** elements);
  __host__ __device__ Matrix(int rows, int cols, const float values[]);
  __host__ __device__ Matrix(int rows, int cols, const cuFloatComplex values[]);
  __host__ __device__ void operator=(const Matrix & other) {
    if (NULL != this->elements) {
      for (int i = 0; i < rows_; i++) {
        for (int j = 0; j < cols_; j++) {
          if (NULL != this->elements[i * cols_ + j]) {
            this->elements[i * cols_ + j]->free();
            delete this->elements[i * cols_ + j];
          }
        }
      }
      delete[] this->elements;
      this->elements = NULL;
    }

    this->rows_ = other.rows_;
    this->cols_ = other.cols_;

    this->elements = new SymNode*[this->rows_ * this->cols_];

    for (int i = 0; i < this->rows_ * this->cols_; i++) {
      this->elements[i] = other.elements[i]->clone();
    }

  }

  __host__ __device__ ~Matrix();

  __host__ __device__ Matrix operator*(Matrix & other);
  __host__ __device__ Matrix tensor(Matrix & other);
  __host__ __device__ SymNode * operator[](int i);
  __host__ __device__ void subst(const char ** names, const char ** new_names);
  __host__ __device__ cuFloatComplex * eval();

  __host__ __device__ int getRows() { return rows_;  }
  __host__ __device__ int getCols() { return cols_;  }

protected:
  int rows_, cols_;
  SymNode ** elements;
};
