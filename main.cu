#include <iostream>
#include <string>

#include <complex.h>
#include <math.h>

#include "symcuda.cuh"

int main(int argc, char const *argv[])
{
//   Symbol x("x");
//   Symbol y("y");
//   SymAdd addNode(&x, &y);
//  	addNode.print();
//  	std::cout << std::endl;
//  
//  	const char *names[] = {"x", "y", NULL};
//  	const char *new_names[] = {"a", "b", NULL};
//  	addNode.subst(names, new_names);
//  	addNode.print();
//  	std::cout << std::endl;

 	SymComplex * a = new SymComplex(make_cuComplex(6.5, 0));
 	SymComplex * b = new SymComplex(make_cuComplex(0.5, 0));
 	SymAdd * addFloats = new SymAdd(a, b);
 
 	SymMul * mulFloats = new SymMul(a, b);

  SymNode ** elements = new SymNode*[1];
  SymNode * elem = new SymComplex(1.0, 0.0);

  elements[0] = elem;

  Matrix * m = new Matrix(1, 1, elements);

  (*m)[0]->print();

  std::cout << std::endl << "test" << std::endl;

  return 0;
}
