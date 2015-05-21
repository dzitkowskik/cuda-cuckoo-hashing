/*
 * gpu_cuckoo.cu
 *
 *  Created on: 21-05-2015
 *      Author: Karol Dzitkowski
 */

#include "constants.h"
#include "hash_function.cuh"
#include "common_cuckoo_hash.hpp"

template<unsigned N>
__device__ int2 devRetrieveKey(
		const int2* hashMap,
		const int hashMap_size,
		const Constants<N> constants,
		const int stash_size,
		const int key
)
{
	unsigned idx = hashFunction(constants.values[0], key, hashMap_size);
	int2 entry = hashMap[idx];

	#pragma unroll
	for(unsigned i=1; i<N; ++i)
	{
		if(entry.x != key && entry.x != EMPTY_BUCKET_KEY)
		{
			idx = hashFunction(constants.values[i], key, hashMap_size);
			entry = hashMap[idx];
		}
	}

	if(stash_size && entry.x != key)
	{
		const int2* stash = hashMap + hashMap_size;
		idx = hashFunction(constants.values[0], key, stash_size);
		entry = stash[idx];
	}

	if(entry.x != key) entry = __EMPTY_BUCKET;
	return entry;
}

template<unsigned N>
__global__ void retrieve(
		const int* keys,
		const int count,
		const int2* hashMap,
		const int hashMap_size,
		const Constants<N> constants,
		const int stash_size,
		int2* result)
{
	unsigned long long int idx = threadIdx.x + blockIdx.x * blockDim.x +
	                          blockIdx.y * blockDim.x * gridDim.x;
	if(idx >= count) return;
	result[idx] = devRetrieveKey(hashMap, hashMap_size, constants, stash_size, keys[idx]);
}

















