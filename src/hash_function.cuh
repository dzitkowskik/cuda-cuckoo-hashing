/*
 * hash_function.cuh
 *
 *  Created on: 21-05-2015
 *      Author: Karol Dzitkowski
 */

#ifndef HASH_FUNCTION_CUH_
#define HASH_FUNCTION_CUH_

#define HASH_FUNC_PRIME_DIVISOR 4294967291u
#define HASH_FUNC_SALT 0xFAB011991

inline __device__ __host__
unsigned hashFunction(const unsigned constant, const int key, const size_t size)
{
	unsigned long long int value = (constant ^ key) + HASH_FUNC_SALT;
	return (value % HASH_FUNC_PRIME_DIVISOR) % size;
}

inline __device__ __host__
unsigned bucketHashFunction(
		const unsigned c0, const unsigned c1, const int key, const size_t size)
{
	unsigned long long int value = (c0 + c1*key) % HASH_FUNC_PRIME_DIVISOR;
	return value % size;
}

#endif /* HASH_FUNCTION_CUH_ */
