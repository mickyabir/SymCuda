/*
 * Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  Users and possessors of this source code 
 * are hereby granted a nonexclusive, royalty-free license to use this code 
 * in individual and commercial software.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS,  WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION,  ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.   This source code is a "commercial item" as 
 * that term is defined at  48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer  software"  and "commercial computer software 
 * documentation" as such terms are  used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein. 
 *
 * Any use of this source code in individual and commercial software must 
 * include, in the user documentation and internal comments to the code,
 * the above Disclaimer and U.S. Government End Users Notice.
 */

#if !defined(CU_COMPLEX_H_)
#define CU_COMPLEX_H_

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

#include <math.h>       /* import fabs, sqrt */
#include "vector_types.h"


/* versions for hosts without native support for 'complex' */

#if (!defined(__CUDACC__) && defined(CU_USE_NATIVE_COMPLEX))
#include <complex.h>

/* wrapper functions around C99 native complex support. NOTE: Untested! */

/* -- Single Precision -- */
typedef complex cuFloatComplex;
__host__ __device__ static __inline__ float cuCrealf (cuFloatComplex x) { 
    return crealf(x); 
}

__host__ __device__ static __inline__ float cuCimagf (cuFloatComplex x) { 
    return cimagf(x); 
}

__host__ __device__ static __inline__ cuFloatComplex make_cuFloatComplex 
                                                             (float x, float y)
{ 
    return x + I * y; 
}

__host__ __device__ static __inline__ cuFloatComplex cuConjf (cuFloatComplex x)
{ 
    return conjf (x); 
}

__host__ __device__ static __inline__ cuFloatComplex cuCaddf (cuFloatComplex x,
                                                              cuFloatComplex y)
{ 
    return x + y; 
}

__host__ __device__ static __inline__ cuFloatComplex cuCsubf (cuFloatComplex x,
                                                              cuFloatComplex y)
{ 
    return x - y; 
}

__host__ __device__ static __inline__ cuFloatComplex cuCmulf (cuFloatComplex x,
                                                              cuFloatComplex y)
{ 
    return x * y; 
}

__host__ __device__ static __inline__ cuFloatComplex cuCdivf (cuFloatComplex x,
                                                              cuFloatComplex y)
{ 
    return x / y; 
}

__host__ __device__ static __inline__ float cuCabsf (cuFloatComplex x) 
{ 
    return cabsf (x); 
}

/* -- Double Precision -- */
typedef double complex cuDoubleComplex;
__host__ __device__ static __inline__ double cuCreal (cuDoubleComplex x) 
{ 
    return creal(x); 
}

__host__ __device__ static __inline__ double cuCimag (cuDoubleComplex x) 
{ 
    return cimag(x); 
}

__host__ __device__ static __inline__ cuDoubleComplex make_cuDoubleComplex 
                                                           (double x, double y)
{ 
    return x + I * y; 
}

__host__ __device__ static __inline__ cuDoubleComplex cuConj(cuDoubleComplex x)
{ 
    return conj (x); 
}

__host__ __device__ static __inline__ cuDoubleComplex cuCadd(cuDoubleComplex x,
                                                             cuDoubleComplex y)
{ 
    return x + y; 
}

__host__ __device__ static __inline__ cuDoubleComplex cuCsub(cuDoubleComplex x,
                                                             cuDoubleComplex y)
{ 
    return x - y; 
}

__host__ __device__ static __inline__ cuDoubleComplex cuCmul(cuDoubleComplex x,
                                                             cuDoubleComplex y)
{ 
    return x * y; 
}

__host__ __device__ static __inline__ cuDoubleComplex cuCdiv(cuDoubleComplex x,
                                                             cuDoubleComplex y)
{ 
    return x / y; 
}

__host__ __device__ static __inline__ double cuCabs (cuDoubleComplex x) 
{ 
    return cabs (x); 
}


	
/* versions for target or hosts without native support for 'complex' */

#else /* (defined(__CUDACC__) || (!(defined(CU_USE_NATIVE_COMPLEX)))) */

typedef float2 cuFloatComplex;

__host__ __device__ static __inline__ float cuCrealf (cuFloatComplex x) 
{ 
    return x.x; 
}

__host__ __device__ static __inline__ float cuCimagf (cuFloatComplex x) 
{ 
    return x.y; 
}

__host__ __device__ static __inline__ cuFloatComplex make_cuFloatComplex 
                                                             (float r, float i)
{
    cuFloatComplex res;
    res.x = r;
    res.y = i;
    return res;
}

__host__ __device__ static __inline__ cuFloatComplex cuConjf (cuFloatComplex x)
{
    return make_cuFloatComplex (cuCrealf(x), -cuCimagf(x));
}
__host__ __device__ static __inline__ cuFloatComplex cuCaddf (cuFloatComplex x,
                                                              cuFloatComplex y)
{
    return make_cuFloatComplex (cuCrealf(x) + cuCrealf(y), 
                                cuCimagf(x) + cuCimagf(y));
}

__host__ __device__ static __inline__ cuFloatComplex cuCsubf (cuFloatComplex x,
                                                              cuFloatComplex y)
{
        return make_cuFloatComplex (cuCrealf(x) - cuCrealf(y), 
                                    cuCimagf(x) - cuCimagf(y));
}

/* This implementation could suffer from intermediate overflow even though
 * the final result would be in range. However, various implementations do
 * not guard against this (presumably to avoid losing performance), so we 
 * don't do it either to stay competitive.
 */
__host__ __device__ static __inline__ cuFloatComplex cuCmulf (cuFloatComplex x,
                                                              cuFloatComplex y)
{
    cuFloatComplex prod;
    prod = make_cuFloatComplex  ((cuCrealf(x) * cuCrealf(y)) - 
                                 (cuCimagf(x) * cuCimagf(y)),
                                 (cuCrealf(x) * cuCimagf(y)) + 
                                 (cuCimagf(x) * cuCrealf(y)));
    return prod;
}

/* This implementation guards against intermediate underflow and overflow
 * by scaling. Such guarded implementations are usually the default for
 * complex library implementations, with some also offering an unguarded,
 * faster version.
 */
__host__ __device__ static __inline__ cuFloatComplex cuCdivf (cuFloatComplex x,
                                                              cuFloatComplex y)
{
    cuFloatComplex quot;
    float s = ((float)fabs((double)cuCrealf(y))) + 
              ((float)fabs((double)cuCimagf(y)));
    float oos = 1.0f / s;
    float ars = cuCrealf(x) * oos;
    float ais = cuCimagf(x) * oos;
    float brs = cuCrealf(y) * oos;
    float bis = cuCimagf(y) * oos;
    s = (brs * brs) + (bis * bis);
    oos = 1.0f / s;
    quot = make_cuFloatComplex (((ars * brs) + (ais * bis)) * oos,
                                ((ais * brs) - (ars * bis)) * oos);
    return quot;
}

/* 
 * We would like to call hypotf(), but it's not available on all platforms.
 * This discrete implementation guards against intermediate underflow and 
 * overflow by scaling. Otherwise we would lose half the exponent range. 
 * There are various ways of doing guarded computation. For now chose the 
 * simplest and fastest solution, however this may suffer from inaccuracies 
 * if sqrt and division are not IEEE compliant. 
 */
__host__ static __inline__ float cuCabsf (cuFloatComplex x)
{
    float a = cuCrealf(x);
    float b = cuCimagf(x);
    float v, w, t;
    a = (float)fabs(a);
    b = (float)fabs(b);
    if (a > b) {
        v = a;
        w = b; 
    } else {
        v = b;
        w = a;
    }
    t = w / v;
    t = 1.0f + t * t;
    t = v * (float)sqrt(t);
    if ((v == 0.0f) || (v > 3.402823466e38f) || (w > 3.402823466e38f)) {
        t = v + w;
    }
    return t;
}

/* Double precision */
typedef double2 cuDoubleComplex;

__host__ __device__ static __inline__ double cuCreal (cuDoubleComplex x) 
{ 
    return x.x; 
}

__host__ __device__ static __inline__ double cuCimag (cuDoubleComplex x) 
{ 
    return x.y; 
}

__host__ __device__ static __inline__ cuDoubleComplex make_cuDoubleComplex 
                                                           (double r, double i)
{
    cuDoubleComplex res;
    res.x = r;
    res.y = i;
    return res;
}

__host__ __device__ static __inline__ cuDoubleComplex cuConj(cuDoubleComplex x)
{
    return make_cuDoubleComplex (cuCreal(x), -cuCimag(x));
}

__host__ __device__ static __inline__ cuDoubleComplex cuCadd(cuDoubleComplex x,
                                                             cuDoubleComplex y)
{
    return make_cuDoubleComplex (cuCreal(x) + cuCreal(y), 
                                 cuCimag(x) + cuCimag(y));
}

__host__ __device__ static __inline__ cuDoubleComplex cuCsub(cuDoubleComplex x,
                                                             cuDoubleComplex y)
{
    return make_cuDoubleComplex (cuCreal(x) - cuCreal(y), 
                                 cuCimag(x) - cuCimag(y));
}

/* This implementation could suffer from intermediate overflow even though
 * the final result would be in range. However, various implementations do
 * not guard against this (presumably to avoid losing performance), so we 
 * don't do it either to stay competitive.
 */
__host__ __device__ static __inline__ cuDoubleComplex cuCmul(cuDoubleComplex x,
                                                             cuDoubleComplex y)
{
    cuDoubleComplex prod;
    prod = make_cuDoubleComplex ((cuCreal(x) * cuCreal(y)) - 
                                 (cuCimag(x) * cuCimag(y)),
                                 (cuCreal(x) * cuCimag(y)) + 
                                 (cuCimag(x) * cuCreal(y)));
    return prod;
}

/* This implementation guards against intermediate underflow and overflow
 * by scaling. Such guarded implementations are usually the default for
 * complex library implementations, with some also offering an unguarded,
 * faster version.
 */
__host__ __device__ static __inline__ cuDoubleComplex cuCdiv(cuDoubleComplex x,
                                                             cuDoubleComplex y)
{
    cuDoubleComplex quot;
    double s = (fabs(cuCreal(y))) + (fabs(cuCimag(y)));
    double oos = 1.0 / s;
    double ars = cuCreal(x) * oos;
    double ais = cuCimag(x) * oos;
    double brs = cuCreal(y) * oos;
    double bis = cuCimag(y) * oos;
    s = (brs * brs) + (bis * bis);
    oos = 1.0 / s;
    quot = make_cuDoubleComplex (((ars * brs) + (ais * bis)) * oos,
                                 ((ais * brs) - (ars * bis)) * oos);
    return quot;
}

/* This implementation guards against intermediate underflow and overflow
 * by scaling. Otherwise we would lose half the exponent range. There are
 * various ways of doing guarded computation. For now chose the simplest
 * and fastest solution, however this may suffer from inaccuracies if sqrt
 * and division are not IEEE compliant.
 */
__host__ __device__ static __inline__ double cuCabs (cuDoubleComplex x)
{
    double a = cuCreal(x);
    double b = cuCimag(x);
    double v, w, t;
    a = fabs(a);
    b = fabs(b);
    if (a > b) {
        v = a;
        w = b; 
    } else {
        v = b;
        w = a;
    }
    t = w / v;
    t = 1.0 + t * t;
    t = v * sqrt(t);
    if ((v == 0.0) || 
        (v > 1.79769313486231570e+308) || (w > 1.79769313486231570e+308)) {
        t = v + w;
    }
    return t;
}

#endif /* (!defined(__CUDACC__) && defined(CU_USE_NATIVE_COMPLEX))) */

#if defined(__cplusplus)
}
#endif /* __cplusplus */

/* aliases */
typedef cuFloatComplex cuComplex;
__host__ __device__ static __inline__ cuComplex make_cuComplex (float x, 
                                                                float y) 
{ 
    return make_cuFloatComplex (x, y); 
}

/* float-to-double promotion */
__host__ __device__ static __inline__ cuDoubleComplex cuComplexFloatToDouble
                                                      (cuFloatComplex c)
{
    return make_cuDoubleComplex ((double)cuCrealf(c), (double)cuCimagf(c));
}

__host__ __device__ static __inline__ cuFloatComplex cuComplexDoubleToFloat
(cuDoubleComplex c)
{
	return make_cuFloatComplex ((float)cuCreal(c), (float)cuCimag(c));
}



#endif /* !defined(CU_COMPLEX_H_) */
