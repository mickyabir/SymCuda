#pragma once

#include "cuComplex.cuh"

__host__ __device__
cuFloatComplex cuComplexExp(cuFloatComplex z)
{
	cuFloatComplex t = make_cuFloatComplex(exp(cuCrealf(z)), 0);

	// exp(x + y * i) = exp(x) * (cos(y) + sin(y) * i)
	// t = exp(x)
	// t1 = cos(y)
	// t2 = sin(y) * i
	// t3 = cos(y) + sin(y) * i

	cuFloatComplex t1 = make_cuFloatComplex(cos(cuCimagf(z)), 0);
	cuFloatComplex t2 = make_cuFloatComplex(0, sin(cuCimagf(z)));
	cuFloatComplex t3 = cuCaddf(t1, t2);

	return cuCmulf(t, t3);
}

__host__ __device__
cuFloatComplex cuComplexCos(cuFloatComplex z)
{
	// cos(z) = 0.5 * (exp(z * i) + exp(-z * i))
	// t1 = exp(z * i)
	// t2 = exp(-z * i)
	// t3 = exp(z * i) + ezp(-z * i)
	cuFloatComplex t1 = cuComplexExp(cuCmulf(z, make_cuFloatComplex(0, 1)));
	cuFloatComplex t2 = cuComplexExp(cuCmulf(make_cuFloatComplex(-cuCrealf(z), cuCimagf(z)), make_cuFloatComplex(0, 1)));
	cuFloatComplex t3 = cuCaddf(t1, t2);

	return cuCmulf(make_cuFloatComplex(0.5, 0), t3);
}

__host__ __device__
cuFloatComplex cuComplexSin(cuFloatComplex z)
{
	// sin(z) = (exp(z * i) - exp(-z * i)) / (2 * i)
	// t1 = exp(z * i)
	// t2 = exp(-z * i)
	// t3 = exp(z * i) - ezp(-z * i)
	// t4 = 2 * i
	cuFloatComplex t1 = cuComplexExp(cuCmulf(z, make_cuFloatComplex(0, 1)));
	cuFloatComplex t2 = cuComplexExp(cuCmulf(make_cuFloatComplex(-cuCrealf(z), cuCimagf(z)), make_cuFloatComplex(0, 1)));
	cuFloatComplex t3 = cuCsubf(t1, t2);
	cuFloatComplex t4 = make_cuFloatComplex(0, 2);

	return cuCdivf(t3, t4);
}

