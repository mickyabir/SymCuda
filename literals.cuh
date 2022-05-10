#pragma once

#include "symnode.cuh"
#include "util.cuh"

class Symbol: public SymNode {
public:
	__host__ __device__ Symbol(const char * name) {
    name_ = "symbol";
    size_t length = cuStrlen(name);
    symbol_ = new char[length + 1];
    cuStrcpy(symbol_, name);
	}

	__host__ __device__ virtual cuFloatComplex eval() override {
		return make_cuFloatComplex(0, 0);
	}

	__host__ __device__ virtual void subst(const char ** names, const char ** new_names) override;

  __host__ __device__ virtual Symbol * clone() override {
    return new Symbol(name_);
  }

  __host__ __device__ virtual void print() override {
		printf("%s", symbol_);
  }

  __host__ __device__ ~Symbol() {
    delete[] symbol_;
  }

  __host__ __device__ virtual void free() override {
    delete[] symbol_;
  }

protected:
  char * symbol_;
};

class SymImag: public SymNode {
public:
	__host__ __device__ SymImag() {
		name_ = "`I`";
	}

	__host__ __device__ virtual cuFloatComplex eval() override {
		return make_cuFloatComplex(0, 1);
	}

  __host__ __device__ virtual SymImag * clone() override {
    return new SymImag();
  }

  __host__ __device__ virtual void free() override {}
};

class SymComplex: public SymNode {
public:
	__host__ __device__ SymComplex(cuFloatComplex data) {
		name_ = "complex";
		data_ = data;
	}

	__host__ __device__ SymComplex(float re, float im) {
		name_ = "complex";
		data_ = make_cuComplex(re, im);
	}

	__host__ __device__ virtual cuFloatComplex eval() override {
		return data_;
	}

	__host__ __device__ virtual void print() {
		printf("(%f + I * %f)", cuCrealf(data_), cuCimagf(data_));
	}

  __host__ __device__ virtual SymComplex * clone() override {
    return new SymComplex(this->data_);
  }

  __host__ __device__ virtual void free() override {}

protected:
	cuFloatComplex data_;
};

