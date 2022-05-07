#pragma once

#include "symnode.cuh"

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

	__host__ __device__ virtual cuFloatComplex eval() override {
		if (NULL == arg1_ || NULL == arg2_) {
			return make_cuFloatComplex(0, 0);
		}

    return cuCaddf(arg1_->eval(), arg2_->eval());
	}
};

class SymMul final: public SymBinOp {
public:
  __host__ __device__ SymMul(SymNode * arg1, SymNode * arg2) {
		name_ = "*";
		arg1_ = arg1;
		arg2_ = arg2;
	}

	__host__ __device__ virtual cuFloatComplex eval() override {
		if (NULL == arg1_ || NULL == arg2_) {
			return make_cuFloatComplex(0, 0);
		}

    return cuCmulf(arg1_->eval(), arg2_->eval());
	}
};

