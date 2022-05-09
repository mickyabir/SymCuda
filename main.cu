#include <iostream>
#include <string>

#include <complex.h>
#include <math.h>

#include "symcuda.cuh"

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

  return 0;
}
