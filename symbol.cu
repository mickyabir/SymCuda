#include "symcuda.cuh"

// __host__ __device__
// void Symbol::print()
// {
// 	printf("%s", this->name_);
// }

__host__ __device__
void Symbol::subst(const char ** names, const char ** new_names)
{
  if (NULL == names || NULL == new_names) {
    return;
  }

  int i = 0;

  while (NULL != names[i] && NULL != new_names[i]) {
		if (0 == cuStrcmp(this->name_, names[i])) {
			this->name_ = new_names[i];
			return;
		}

		i++;
  }
}
