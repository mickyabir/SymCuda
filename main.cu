#include <iostream>
#include <string>

#include <complex.h>
#include <math.h>

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

#include "symcuda.cuh"

__device__
float cost(float *thetas)
{
  return 0.0;
}

__global__
void kernel(Matrix *m)
{
	printf("%d\n", m->rows_);
}

int main(int argc, char const *argv[])
{
//  Matrix * m;
//  cudaMallocManaged(&m, sizeof(Matrix));
//
//	m->rows_ = 2;
//
//  kernel<<<1, 1>>>(m);
//
//	auto e = cudaGetLastError();
//	printf("%s\n", cudaGetErrorString(e));
//
//  cudaDeviceSynchronize();
//
//  cudaFree(m);

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

	SymFloat a(5.5);
	SymFloat b(1.5);
	SymAdd addFloats(&a, &b);
	addFloats.print();
	std::cout << std::endl << addFloats.eval() << std::endl;

  return 0;
}
