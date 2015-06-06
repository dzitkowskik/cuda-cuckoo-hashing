/*
 * hash_function.cuh
 *
 *  Created on: 21-05-2015
 *      Author: Karol Dzitkowski
 */

#ifndef HASH_FUNCTION_CUH_
#define HASH_FUNCTION_CUH_

#define HASH_FUNC_PRIME_DIVISOR 4294967291u
#define HASH_FUNC_PRIME_DIVISOR_2 1900813u
#define HASH_FUNC_SALT 0xFAB011991u

inline __device__ __host__
unsigned hashFunction(const unsigned constant, const int key, const size_t size)
{
	unsigned long long int val = constant ^ HASH_FUNC_SALT + constant * key;
	return (val % HASH_FUNC_PRIME_DIVISOR_2) % size;
}

inline __device__ __host__
unsigned bucketHashFunction(
		const unsigned c0, const unsigned c1, const int key, const size_t size)
{
	return ((c0 + c1*key) % HASH_FUNC_PRIME_DIVISOR) % size;
}

#endif /* HASH_FUNCTION_CUH_ */
