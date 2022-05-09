#include <assert.h>

#include "matrix.cuh"
#include "symbinop.cuh"

__host__ __device__
Matrix::Matrix()
{
  this->elements = NULL;
  this->rows_ = 0;
  this->cols_ = 0;
}

__host__ __device__
Matrix::Matrix(int rows, int cols)
{
  this->rows_ = rows;
  this->cols_ = cols;
  this->elements = new SymNode*[rows * cols];
}

__host__ __device__
Matrix::Matrix(int rows, int cols, SymNode ** elems)
{
  this->elements = elems;
  this->rows_ = rows;
  this->cols_ = cols;
}

__host__ __device__
Matrix::~Matrix()
{
  if (NULL != this->elements) {
    for (int i = 0; i < rows_; i++) {
      for (int j = 0; j < cols_; j++) {
        if (NULL != this->elements[i * cols_ + j]) {
          delete this->elements[i * cols_ + j];
        }
      }
    }
    delete[] this->elements;
    this->elements = NULL;
  }
}

__host__ __device__
SymNode * Matrix::operator[](int i)
{
  if (NULL != elements && i < rows_ * cols_) {
    return elements[i];
  }

  return NULL;
}

__host__ __device__
Matrix Matrix::operator*(Matrix & other)
{
  assert(cols_ == other.rows_);

  SymNode ** new_elements = new SymNode*[rows_ * other.cols_];

  for (int i = 0; i < rows_; i++) {
    for (int j = 0; j < other.cols_; j++) {
      SymNode * curr = new SymMul(this->elements[i * cols_], other.elements[j]);

      for (int k = 1; k < cols_; k++) {
        SymNode * mul = new SymMul(this->elements[i * cols_ + k], other.elements[k * other.rows_ + j]);
        SymNode * add = new SymAdd(curr, mul);
        curr = add;
      }

      new_elements[i * other.cols_ + j] = curr;
    }
  }

  Matrix m(rows_, other.cols_, new_elements);
  return m;
}

__host__ __device__
void Matrix::subst(const char ** names, const char ** new_names)
{
  if (NULL == elements) {
    return;
  }

  for (int i = 0; i < rows_ * cols_; i++) {
    if (NULL != elements[i]) {
      elements[i]->subst(names, new_names);
    }
  }
}

__host__ __device__
cuFloatComplex * Matrix::eval()
{
  if (NULL == elements) {
    return NULL;
  }

  cuFloatComplex * concrete = new cuFloatComplex[rows_ * cols_];

  for (int i = 0; i < rows_ * cols_; i++) {
    concrete[i] = elements[i]->eval();
  }

  return concrete;
}
