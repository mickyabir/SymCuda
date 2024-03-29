#include <iostream>
#include <string>

#include <complex.h>
#include <math.h>

#include "symcuda.cuh"

__host__
Matrix generateSymbolicSquareMatrix(int n, const char ** symbol_names)
{
  SymNode ** elements = new SymNode*[n * n];

  for (int i = 0; i < n * n; i++) {
    elements[i] = new Symbol(symbol_names[i]);
  }

  return Matrix(n, n, elements);
}

__host__
Matrix generateIdentityMatrix(int n)
{
  float * elements = new float[n * n];

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (i == j) {
        elements[i * n + j] = 1;
      } else {
        elements[i * n + j] = 0;
      }
    }
  }

  Matrix m = Matrix(n, n, elements);
  delete[] elements;
  
  return m;
}

__host__
Matrix generateTof3()
{
  int n = 8;
  float * elements = new float[n * n];

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (i == j) {
        elements[i * n + j] = 1;
      } else {
        elements[i * n + j] = 0;
      }
    }
  }

  elements[n * n - 1] = 0;
  elements[n * n - 2] = 1;
  elements[(n - 1) * n - 1] = 1;
  elements[(n - 1) * n - 2] = 0;
  
  Matrix m = Matrix(n, n, elements);
  delete[] elements;

  return m;
}

__host__
Matrix generateRx(const char * theta_name)
{
  SymNode ** elements = new SymNode*[4];

  Symbol * theta0 = new Symbol(theta_name);
  SymComplex * two0 = new SymComplex(2, 0);
  SymDiv * div0 = new SymDiv(theta0, two0);
  SymCos * cos0 = new SymCos(div0);
  elements[0] = cos0;

  Symbol * theta1 = new Symbol(theta_name);
  SymComplex * two1 = new SymComplex(2, 0);
  SymDiv * div1 = new SymDiv(theta1, two1);
  SymSin * sin1 = new SymSin(div1);
  SymComplex * i1 = new SymComplex(0, -1);
  SymMul * mul1 = new SymMul(i1, sin1);
  elements[1] = mul1;

  Symbol * theta2 = new Symbol(theta_name);
  SymComplex * two2 = new SymComplex(2, 0);
  SymDiv * div2 = new SymDiv(theta2, two2);
  SymSin * sin2 = new SymSin(div2);
  SymComplex * i2 = new SymComplex(0, -1);
  SymMul * mul2 = new SymMul(i2, sin2);
  elements[2] = mul2;

  Symbol * theta3 = new Symbol(theta_name);
  SymComplex * two3 = new SymComplex(2, 0);
  SymDiv * div3 = new SymDiv(theta3, two3);
  SymCos * cos3 = new SymCos(div3);
  elements[3] = cos3;

  return Matrix(2, 2, elements);
}

__host__
Matrix generateRy(const char * theta_name)
{
  SymNode ** elements = new SymNode*[4];

  Symbol * theta0 = new Symbol(theta_name);
  SymComplex * two0 = new SymComplex(2, 0);
  SymDiv * div0 = new SymDiv(theta0, two0);
  SymCos * cos0 = new SymCos(div0);
  elements[0] = cos0;

  Symbol * theta1 = new Symbol(theta_name);
  SymComplex * two1 = new SymComplex(2, 0);
  SymDiv * div1 = new SymDiv(theta1, two1);
  SymSin * sin1 = new SymSin(div1);
  SymComplex * i1 = new SymComplex(-1, 0);
  SymMul * mul1 = new SymMul(i1, sin1);
  elements[1] = mul1;

  Symbol * theta2 = new Symbol(theta_name);
  SymComplex * two2 = new SymComplex(2, 0);
  SymDiv * div2 = new SymDiv(theta2, two2);
  SymSin * sin2 = new SymSin(div2);
  SymComplex * i2 = new SymComplex(-1, 0);
  SymMul * mul2 = new SymMul(i2, sin2);
  elements[2] = mul2;

  Symbol * theta3 = new Symbol(theta_name);
  SymComplex * two3 = new SymComplex(2, 0);
  SymDiv * div3 = new SymDiv(theta3, two3);
  SymCos * cos3 = new SymCos(div3);
  elements[3] = cos3;

  return Matrix(2, 2, elements);
}

__host__
Matrix generateRz(const char * theta_name)
{
  SymNode ** elements = new SymNode*[4];

  Symbol * theta0 = new Symbol(theta_name);
  SymComplex * two0 = new SymComplex(2, 0);
  SymDiv * div0 = new SymDiv(theta0, two0);
  SymComplex * i0 = new SymComplex(0, -1);
  SymMul * mul0 = new SymMul(i0, div0);
  SymExp * exp0 = new SymExp(mul0);
  elements[0] = exp0;

  elements[1] = new SymComplex(0, 0);
  elements[2] = new SymComplex(0, 0);

  Symbol * theta3 = new Symbol(theta_name);
  SymComplex * two3 = new SymComplex(2, 0);
  SymDiv * div3 = new SymDiv(theta3, two3);
  SymComplex * i3 = new SymComplex(0, 1);
  SymMul * mul3 = new SymMul(i3, div3);
  SymExp * exp3 = new SymExp(mul3);
  elements[3] = exp3;

  return Matrix(2, 2, elements);
}

__host__
Matrix generateU3(const char * theta_name, const char * phi_name, const char * lambda_name)
{
  SymNode ** elements = new SymNode*[4];

  Symbol * theta0 = new Symbol(theta_name);
  SymComplex * two0 = new SymComplex(2, 0);
  SymDiv * div0 = new SymDiv(theta0, two0);
  elements[0] = new SymCos(div0);

  SymComplex * i1 = new SymComplex(0, 1);
  Symbol * lambda1 = new Symbol(lambda_name);
  SymMul * mul10 = new SymMul(i1, lambda1);
  SymExp * exp1 = new SymExp(mul10);
  SymComplex * neg1 = new SymComplex(-1, 0);
  SymMul * mul11 = new SymMul(neg1, exp1);
  Symbol * theta1 = new Symbol(theta_name);
  SymComplex * two1 = new SymComplex(2, 0);
  SymDiv * div1 = new SymDiv(theta1, two1);
  SymSin * sin1 = new SymSin(div1);
  elements[1] = new SymMul(mul11, sin1);

  SymComplex * i2 = new SymComplex(0, 1);
  Symbol * phi2 = new Symbol(phi_name);
  SymMul * mul20 = new SymMul(i2, phi2);
  SymExp * exp2 = new SymExp(mul20);
  Symbol * theta2 = new Symbol(theta_name);
  SymComplex * two2 = new SymComplex(2, 0);
  SymDiv * div2 = new SymDiv(theta2, two2);
  SymSin * sin2 = new SymSin(div2);
  elements[2] = new SymMul(exp2, sin2);

  SymComplex * i3 = new SymComplex(0, 1);
  Symbol * phi3 = new Symbol(phi_name);
  Symbol * lambda3 = new Symbol(lambda_name);
  SymAdd * add3 = new SymAdd(phi3, lambda3);
  SymMul * mul30 = new SymMul(i3, add3);
  SymExp * exp3 = new SymExp(mul30);
  Symbol * theta3 = new Symbol(theta_name);
  SymComplex * two3 = new SymComplex(2, 0);
  SymDiv * div3 = new SymDiv(theta3, two3);
  SymCos * cos3 = new SymCos(div3);
  elements[3] = new SymMul(exp3, cos3);

  return Matrix(2, 2, elements);
}

__host__
Matrix generateInitialRotations(int n)
{
  Matrix m;

  for (int i = 0; i < n; i++) {
    int str_size = 20;
    char * theta = new char[str_size];
    char * phi = new char[str_size];
    char * lambda = new char[str_size];
    snprintf(theta, str_size, "initial_%d", i * 3);
    snprintf(phi, str_size, "initial_%d", i * 3 + 1);
    snprintf(lambda, str_size, "initial_%d", i * 3 + 2);
    
    Matrix u3 = generateU3(theta, phi, lambda);

    delete[] theta;
    delete[] phi;
    delete[] lambda;

    if (i == 0) {
      m = u3;
    } else {
      Matrix new_m = m.tensor(u3);
      m = new_m;
    }
  }
  
  return m;
}

int main(int argc, char const *argv[])
{
  Matrix initial = generateInitialRotations(2);

  for (int i = 0; i < initial.getRows(); i++) {
    for (int j = 0; j < initial.getCols(); j++) {
      initial[i * initial.getRows() + j]->print();
      printf(" | ");
    }
    printf("-------------------------------------------------------------\n");
  }

  cudaDeviceReset();

  return 0;
}
