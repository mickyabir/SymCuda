#pragma once

#include <iostream>
#include <string>

#include <complex.h>
#include <math.h>

#include "cuComplex.cuh"

#include "literals.cuh"
#include "symbinop.cuh"
#include "symnode.cuh"
#include "symop.cuh"
#include "util.cuh"

class Matrix {
public:
	Matrix() {}
  Matrix(int rows, int cols): rows_(rows), cols_(cols)  {}

  int rows_, cols_;
};


