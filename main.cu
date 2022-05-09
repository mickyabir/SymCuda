#include <iostream>
#include <string>

#include <complex.h>
#include <math.h>

#include "symcuda.cuh"

Matrix generateIdentityMatrix(int n)
{
  float elements[n * n];

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (i == j) {
        elements[i * n + j] = 1;
      } else {
        elements[i * n + j] = 0;
      }
    }
  }
  
  return Matrix(n, n, elements);
}

Matrix generateTof3()
{
  int n = 8;
  float elements[n * n];

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
  
  return Matrix(n, n, elements);
}

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

int main(int argc, char const *argv[])
{
  SymNode ** elemA = new SymNode*[1];
  SymNode ** elemB = new SymNode*[1];
  elemA[0] = new SymAdd(new SymComplex(2.0, 0.0), new SymComplex(1.0, 0.0));
  elemB[0] = new SymComplex(3.0, 0.0);

  Matrix a(1, 1, elemA);
  Matrix b(1, 1, elemB);

  Matrix m = a * b;

  m[0]->print();
  std::cout << std::endl;

  cuComplex * concrete = m.eval();

  print(concrete[0]);
  std::cout << std::endl;
  delete[] concrete;

  int n = 2;
  Matrix rz = generateRz("theta");

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      rz[i * n + j]->print();
      printf(" | ");
    }
    printf("-------------------------------------------------------------\n");
  }


  return 0;
}
