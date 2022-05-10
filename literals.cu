#include "literals.cuh"

#include "util.cuh"

__host__ __device__
void Symbol::subst(const char ** names, const char ** new_names)
{
  if (NULL == names || NULL == new_names) {
    return;
  }

  int i = 0;

  while (NULL != names[i] && NULL != new_names[i]) {
		if (0 == cuStrcmp(this->symbol_, names[i])) {
      delete[] this->symbol_;

      size_t length = cuStrlen(new_names[i]);
      this->symbol_ = new char[length + 1];
      cuStrcpy(symbol_, new_names[i]);

			return;
		}

		i++;
  }
}
