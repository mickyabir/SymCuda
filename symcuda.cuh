#pragma once

#include <iostream>
#include <string>

#include <complex.h>
#include <math.h>

class SymNode {
public:
	__host__ __device__ virtual float eval() = 0;
	__host__ __device__ virtual void subst(const char ** names, const char ** new_names) = 0;
	__host__ __device__ virtual void print() {
		printf("%s", name_);
	}

protected:
	const char * name_;
};

class Symbol: public SymNode {
public:
	__host__ __device__ Symbol(const char * name) {
		name_ = name;
	}

	__host__ __device__ virtual float eval() override {
		return 0;
	}

	__host__ __device__ virtual void subst(const char ** names, const char ** new_names) override;
};

class SymFloat: public SymNode {
public:
	__host__ __device__ SymFloat(float data) {
		name_ = "float";
		data_ = data;
	}

	__host__ __device__ virtual float eval() override {
		return data_;
	}

	__host__ __device__ virtual void subst(const char ** names, const char ** new_names) override {}

	__host__ __device__ virtual void print() {
		printf("%f", data_);
	}

protected:
	float data_;
};


class SymOp: public SymNode {
public:
	__host__ __device__ virtual void subst(const char ** names, const char ** new_names) override {
		if (NULL != arg_) {
			arg_->subst(names, new_names);
		}
  }

  __host__ __device__ virtual void print() override {
    printf("%s(", name_);

		if (NULL != arg_) {
			arg_->print();
		}

    printf(")");
	}
  
protected:
  SymNode * arg_;
};

class SymBinOp: public SymNode {
public:
	__host__ __device__ virtual void subst(const char ** names, const char ** new_names) override {
		arg1_->subst(names, new_names);
		arg2_->subst(names, new_names);
  }

	__host__ __device__ virtual void print() override {
		if (NULL != arg1_) {
			arg1_->print();
		}

    printf(" %s ", name_);

		if (NULL != arg2_) {
			arg2_->print();
		}
	}
  
protected:
  SymNode * arg1_, * arg2_;
};

class SymAdd final: public SymBinOp {
public:
  __host__ __device__ SymAdd(SymNode * arg1, SymNode * arg2) {
		name_ = "+";
		arg1_ = arg1;
		arg2_ = arg2;
	}

	__host__ __device__ virtual float eval() override {
		if (NULL == arg1_ || NULL == arg2_) {
			return 0;
		}

		return arg1_->eval() + arg2_->eval();
	}

};

class Matrix {
public:
	Matrix() {}
  Matrix(int rows, int cols): rows_(rows), cols_(cols)  {}

  int rows_, cols_;
};


// UTILITY
__host__ __device__ int cuStrcmp(const char * p1, const char * p2);
