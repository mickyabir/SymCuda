#include "util.cuh"

#include <stdio.h>

__host__ __device__
int cuStrcmp(const char * p1, const char * p2)
{
	/* From glibc: https://github.com/zerovm/glibc/blob/master/string/strcmp.c  */
  const unsigned char *s1 = (const unsigned char *) p1;
  const unsigned char *s2 = (const unsigned char *) p2;
  unsigned char c1, c2;

  do {
      c1 = (unsigned char) *s1++;
      c2 = (unsigned char) *s2++;

      if (c1 == '\0') {
				return c1 - c2;
			}
	} while (c1 == c2);

  return c1 - c2;
}

__host__ __device__
size_t cuStrlen(const char * str)
{
  /* From glibc: https://github.com/lattera/glibc/blob/master/string/strlen.c */
  const char *char_ptr;
  const unsigned long int *longword_ptr;
  unsigned long int longword, himagic, lomagic;

  for (char_ptr = str; ((unsigned long int) char_ptr & (sizeof (longword) - 1)) != 0; ++char_ptr) {
    if (*char_ptr == '\0') {
      return char_ptr - str;
    }
  }

  longword_ptr = (unsigned long int *) char_ptr;

  himagic = 0x80808080L;
  lomagic = 0x01010101L;

  if (sizeof (longword) > 4) {
      himagic = ((himagic << 16) << 16) | himagic;
      lomagic = ((lomagic << 16) << 16) | lomagic;
  }

/* Temporary fix */
#ifndef __CUDA_ARCH__
  if (sizeof (longword) > 8)
    abort ();
#endif

  for (;;) {
      longword = *longword_ptr++;

    if (((longword - lomagic) & ~longword & himagic) != 0) {

      const char *cp = (const char *) (longword_ptr - 1);

      if (cp[0] == 0)
        return cp - str;
      if (cp[1] == 0)
        return cp - str + 1;
      if (cp[2] == 0)
        return cp - str + 2;
      if (cp[3] == 0)
        return cp - str + 3;
      if (sizeof (longword) > 4) {
        if (cp[4] == 0)
          return cp - str + 4;
        if (cp[5] == 0)
          return cp - str + 5;
        if (cp[6] == 0)
          return cp - str + 6;
        if (cp[7] == 0)
          return cp - str + 7;
      }
    }
  }
}

__host__ __device__
void cuStrcpy(char * dest, const char * src)
{
  memcpy(dest, src, cuStrlen (src) + 1);
}


__host__ __device__
void print(cuFloatComplex c)
{
  printf("%f + I * %f", cuCrealf(c), cuCimagf(c));
}
