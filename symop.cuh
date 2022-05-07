#pragma once

#include "symnode.cuh"

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


