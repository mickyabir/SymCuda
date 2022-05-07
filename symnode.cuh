#pragma once

#include <stdio.h>

#include "cuComplex.cuh"

class SymNode {
public:
	__host__ __device__ virtual cuFloatComplex eval() = 0;
	__host__ __device__ virtual void subst(const char ** names, const char ** new_names) = 0;
	__host__ __device__ virtual void print() {
		printf("%s", name_);
	}

protected:
	const char * name_;
};

