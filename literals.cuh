#pragma once

#include "symnode.cuh"

class Symbol: public SymNode {
public:
	__host__ __device__ Symbol(const char * name) {
		name_ = name;
	}

	__host__ __device__ virtual cuFloatComplex eval() override {
		return make_cuFloatComplex(0, 0);
	}

	__host__ __device__ virtual void subst(const char ** names, const char ** new_names) override;
};

class SymImag: public SymNode {
public:
	__host__ __device__ SymImag() {
		name_ = "`I`";
	}

	__host__ __device__ virtual cuFloatComplex eval() override {
		return make_cuFloatComplex(0, 1);
	}

	__host__ __device__ virtual void subst(const char ** names, const char ** new_names) override;
};

class SymComplex: public SymNode {
public:
	__host__ __device__ SymComplex(cuFloatComplex data) {
		name_ = "complex";
		data_ = data;
	}

	__host__ __device__ virtual cuFloatComplex eval() override {
		return data_;
	}

	__host__ __device__ virtual void subst(const char ** names, const char ** new_names) override {}

	__host__ __device__ virtual void print() {
		printf("(%f + I * %f)", cuCrealf(data_), cuCimagf(data_));
	}

protected:
	cuFloatComplex data_;
};

