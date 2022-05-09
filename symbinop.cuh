#pragma once

#include "symnode.cuh"

class SymBinOp: public SymNode {
public:
  __host__ __device__ virtual ~SymBinOp() {
    delete arg1_;
    delete arg2_;
  }
  
	__host__ __device__ virtual void subst(const char ** names, const char ** new_names) override {
		arg1_->subst(names, new_names);
		arg2_->subst(names, new_names);
  }

	__host__ __device__ virtual void print() override {
    printf("(");

		if (NULL != arg1_) {
			arg1_->print();
		}

    printf(" %s ", name_);

		if (NULL != arg2_) {
			arg2_->print();
		}

    printf(")");
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

  __host__ __device__ virtual SymAdd * clone() override {
    return new SymAdd(this->arg1_, this->arg2_);
  }
};

class SymSub final: public SymBinOp {
public:
  __host__ __device__ SymSub(SymNode * arg1, SymNode * arg2) {
		name_ = "-";
		arg1_ = arg1;
		arg2_ = arg2;
	}

	__host__ __device__ virtual cuFloatComplex eval() override {
		if (NULL == arg1_ || NULL == arg2_) {
			return make_cuFloatComplex(0, 0);
		}

    return cuComplexDoubleToFloat(cuCsub(cuComplexFloatToDouble(arg1_->eval()), cuComplexFloatToDouble(arg2_->eval())));
	}

  __host__ __device__ virtual SymSub * clone() override {
    return new SymSub(this->arg1_, this->arg2_);
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

  __host__ __device__ virtual SymMul * clone() override {
    return new SymMul(this->arg1_, this->arg2_);
  }
};

class SymDiv final: public SymBinOp {
public:
  __host__ __device__ SymDiv(SymNode * arg1, SymNode * arg2) {
		name_ = "/";
		arg1_ = arg1;
		arg2_ = arg2;
	}

	__host__ __device__ virtual cuFloatComplex eval() override {
		if (NULL == arg1_ || NULL == arg2_) {
			return make_cuFloatComplex(0, 0);
		}

    return cuComplexDoubleToFloat(cuCdiv(cuComplexFloatToDouble(arg1_->eval()), cuComplexFloatToDouble(arg2_->eval())));
	}

  __host__ __device__ virtual SymDiv * clone() override {
    return new SymDiv(this->arg1_, this->arg2_);
  }
};
