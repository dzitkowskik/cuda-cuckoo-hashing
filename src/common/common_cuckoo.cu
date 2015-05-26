/*
 * gpu_cuckoo.cu
 *
 * Created on: 21-05-2015
 *      Author: Karol Dzitkowski
 *
 * This code was created as my implementation of CUDPP algorithm
 * of cuckoo hashing found on:  https://github.com/cudpp/cudpp
 * which I used as a model for this implementation
 */

#include "macros.h"
#include "constants.h"
#include "hash_function.cuh"
#include "cuckoo_hash.hpp"
#include "common_cuckoo_hash.cuh"

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

	if(entry.x != key)
	{
		entry.x = EMPTY_BUCKET_KEY;
		entry.y = EMPTY_BUCKET_KEY;
	}
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
	result[idx] = devRetrieveKey<N>(
			hashMap,
			hashMap_size,
			constants,
			stash_size,
			keys[idx]);
}

template<unsigned N>
__device__ unsigned next_loc_cuckoo(
		const Constants<N> constants,
		const int hashMap_size,
		const int key_value,
		const int last_loc)
{
	unsigned locations[N];
	#pragma unroll
	for (int i=0; i<N; ++i)
		locations[i] = hashFunction(constants.values[i], key_value, hashMap_size);


	unsigned next_location = locations[0];
	#pragma unroll
	for (int i=N-2; i>=0; --i)
	{
		next_location = (last_loc == locations[i] ? locations[i+1] : next_location);
	}
	return next_location;
}

union entry
{
	int2 value;
	unsigned long long hidden;
};

template<unsigned N>
__device__ bool devInsertElem(
		int2* hashMap,
		const int hashMap_size,
		const Constants<N> constants,
		const int stash_size,
		const int max_iters,
		int2 value)
{
	unsigned idx = hashFunction(constants.values[0], value.x, hashMap_size);
	entry e;
	e.value = value;

	unsigned long long int* slot;

	for(unsigned i = 1; i <= max_iters; i++)
	{
		slot = reinterpret_cast<unsigned long long int*>(hashMap + idx);
		e.hidden = atomicExch(slot, e.hidden);
		if(e.value.x == EMPTY_BUCKET_KEY) break;
		idx = next_loc_cuckoo(constants, hashMap_size, e.value.x, idx);
	}

	if (e.value.x != EMPTY_BUCKET_KEY)
	{
		idx = hashFunction(constants.values[0], e.value.x, stash_size);
		slot = (unsigned long long int*)(hashMap + (hashMap_size + idx));
		auto replaced = atomicCAS(slot, EMPTY_BUCKET, e.hidden);
		if (replaced != EMPTY_BUCKET) return false;
	}

	return true;
}

template<unsigned N>
__global__ void insert(
		const int2* keys,
		const int count,
		int2* hashMap,
		const int hashMap_size,
		const Constants<N> constants,
		const int stash_size,
		const int max_iters,
		bool* failure)
{
	unsigned long long int idx = threadIdx.x + blockIdx.x * blockDim.x +
		                          blockIdx.y * blockDim.x * gridDim.x;
	if(idx >= count) return;
	*failure = devInsertElem<N>(
			hashMap,
			hashMap_size,
			constants,
			stash_size,
			max_iters,
			keys[idx]);
}

template<unsigned N>
bool common_cuckooHash(
		int2* values,
		int in_size,
		int2* hashMap,
		int hashMap_size,
		Constants<N> constants,
		int stash_size)
{
	auto grid = CuckooHash<N>::GetGrid(in_size);
	bool* d_result;
	bool h_result;

	CUDA_CALL( cudaMalloc((void**)&d_result, sizeof(bool)) );
	int blockSize = CuckooHash<N>::DEFAULT_BLOCK_SIZE;
	int maxIters = MAX_RETRIES * N;

	insert<N><<<grid, blockSize>>>(
			values,
			in_size,
			hashMap,
			hashMap_size,
			constants,
			stash_size,
			maxIters,
			d_result);

	CUDA_CALL( cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost) );
	CUDA_CALL( cudaFree(d_result) );
	return h_result;
}

template<unsigned N>
int2* common_cuckooRetrieve(
		int* keys,
		int size,
		int2* hashMap,
		int hashMap_size,
		Constants<N> constants,
		int stash_size)
{
	auto grid = CuckooHash<N>::GetGrid(size);
	int2* d_result;
	CUDA_CALL( cudaMalloc((void**)&d_result, size*sizeof(int2)) );
	int blockSize = CuckooHash<N>::DEFAULT_BLOCK_SIZE;

	retrieve<N><<<grid, blockSize>>>(
			keys,
			size,
			hashMap,
			hashMap_size,
			constants,
			stash_size,
			d_result);

	return d_result;
}

template bool common_cuckooHash<2>(int2*, int, int2*, int, Constants<2>, int);
template bool common_cuckooHash<3>(int2*, int, int2*, int, Constants<3>, int);
template bool common_cuckooHash<4>(int2*, int, int2*, int, Constants<4>, int);
template bool common_cuckooHash<5>(int2*, int, int2*, int, Constants<5>, int);

template int2* common_cuckooRetrieve<2>(int*, int, int2*, int, Constants<2>, int);
template int2* common_cuckooRetrieve<3>(int*, int, int2*, int, Constants<3>, int);
template int2* common_cuckooRetrieve<4>(int*, int, int2*, int, Constants<4>, int);
template int2* common_cuckooRetrieve<5>(int*, int, int2*, int, Constants<5>, int);




