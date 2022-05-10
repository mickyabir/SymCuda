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


  __host__ __device__ virtual void free() override {
    arg_->free();
    delete arg_;
  }

  __host__ __device__ virtual ~SymOp() {
    arg_->free();
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

  __host__ __device__ virtual SymCos * clone() override {
    return new SymCos(this->arg_->clone());
  }
};

class SymSin final: public SymOp {
public:
  __host__ __device__ SymSin(SymNode * arg) {
		name_ = "sin";
		arg_ = arg;
	}

	__host__ __device__ virtual cuFloatComplex eval() override {
		if (NULL == arg_) {
			return make_cuFloatComplex(0, 0);
		}

    return cuComplexSin(arg_->eval());
	}

  __host__ __device__ virtual SymSin * clone() override {
    return new SymSin(this->arg_->clone());
  }
};

class SymExp final: public SymOp {
public:
  __host__ __device__ SymExp(SymNode * arg) {
		name_ = "exp";
		arg_ = arg;
	}

	__host__ __device__ virtual cuFloatComplex eval() override {
		if (NULL == arg_) {
			return make_cuFloatComplex(0, 0);
		}

    return cuComplexExp(arg_->eval());
	}

  __host__ __device__ virtual SymExp * clone() override {
    return new SymExp(this->arg_->clone());
  }
};

