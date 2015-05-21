/*
 * hash_function.cuh
 *
 *  Created on: 21-05-2015
 *      Author: Karol Dzitkowski
 */

#ifndef HASH_FUNCTION_CUH_
#define HASH_FUNCTION_CUH_

inline __device__ __host__
unsigned hashFunction(const unsigned constant, const int key, const size_t size)
{
	unsigned long long int value = (constant ^ key) + HASH_FUNC_SALT;
	return (value % HASH_FUNC_PRIME_DIVISOR) % (size + 1);
}

#endif /* HASH_FUNCTION_CUH_ */
