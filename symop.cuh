#pragma once

#include "symnode.cuh"
#include "cuComplexFunctions.cuh"

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

  __host__ __device__ virtual ~SymOp() {
    delete arg_;
  }
  
protected:
  SymNode * arg_;
};

class SymCos final: public SymOp {
public:
  __host__ __device__ SymCos(SymNode * arg) {
		name_ = "cos";
		arg_ = arg;
	}

	__host__ __device__ virtual cuFloatComplex eval() override {
		if (NULL == arg_) {
			return make_cuFloatComplex(0, 0);
		}

    return cuComplexCos(arg_->eval());
	}
};

class SymSin final: public SymOp {
public:
  __host__ __device__ SymSin(SymNode * arg) {
		name_ = "cos";
		arg_ = arg;
	}

	__host__ __device__ virtual cuFloatComplex eval() override {
		if (NULL == arg_) {
			return make_cuFloatComplex(0, 0);
		}

    return cuComplexSin(arg_->eval());
	}
};

class SymExp final: public SymOp {
public:
  __host__ __device__ SymExp(SymNode * arg) {
		name_ = "cos";
		arg_ = arg;
	}

	__host__ __device__ virtual cuFloatComplex eval() override {
		if (NULL == arg_) {
			return make_cuFloatComplex(0, 0);
		}

    return cuComplexExp(arg_->eval());
	}
};

