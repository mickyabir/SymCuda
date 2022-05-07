#include <iostream>
#include <string>

#include <complex.h>
#include <math.h>

#include "symcuda.cuh"

int main(int argc, char const *argv[])
{
  Symbol x("x");
  Symbol y("y");
  SymAdd addNode(&x, &y);
 	addNode.print();
 	std::cout << std::endl;
 
 	const char *names[] = {"x", "y", NULL};
 	const char *new_names[] = {"a", "b", NULL};
 	addNode.subst(names, new_names);
 	addNode.print();
 	std::cout << std::endl;

	SymComplex a(make_cuComplex(6.5, 0));
	SymComplex b(make_cuComplex(0.5, 0));
	SymAdd addFloats(&a, &b);
	addFloats.print();
	std::cout << std::endl;
	print(addFloats.eval());
	std::cout << std::endl;

	SymMul mulFloats(&a, &b);
	mulFloats.print();
	std::cout << std::endl;
	print(mulFloats.eval());
	std::cout << std::endl;


  return 0;
}
