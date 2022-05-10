#include <assert.h>

#include "literals.cuh"
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
Matrix::Matrix(int rows, int cols, const float values[])
{
  this->rows_ = rows;
  this->cols_ = cols;

  SymNode ** elems = new SymNode*[rows * cols];

  for (int i = 0; i < rows * cols; i++) {
    elems[i] = new SymComplex(values[i], 0);
  }

  this->elements = elems;
}

__host__ __device__
Matrix::Matrix(int rows, int cols, const cuFloatComplex values[])
{
  this->rows_ = rows;
  this->cols_ = cols;

  SymNode ** elems = new SymNode*[rows * cols];

  for (int i = 0; i < rows * cols; i++) {
    elems[i] = new SymComplex(values[i]);
  }

  this->elements = elems;
}

__host__ __device__
Matrix::~Matrix()
{
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
Matrix Matrix::tensor(Matrix & other)
{
  int new_rows = rows_ * other.rows_;
  int new_cols = cols_ * other.cols_;

  SymNode ** new_elements = new SymNode*[new_rows * new_cols];

  for (int i = 0; i < rows_; i++) {
    for (int k = 0; k < other.rows_; k++) {
      for (int j = 0; j < cols_; j++) {
        for (int l = 0; l < other.cols_; l++) {
          int idx = i * cols_ * other.rows_ * other.cols_ + k * cols_ * other.cols_ + j * other.cols_ + l;
          SymNode * elem1 = this->elements[i * cols_ + j]->clone();
          SymNode * elem2 = other.elements[k * other.cols_ + l]->clone();
          new_elements[idx] = new SymMul(elem1, elem2);
        }
      }
    }
  }

  Matrix m(new_rows, new_cols, new_elements);
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
