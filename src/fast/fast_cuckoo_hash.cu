/*
 *  fast_cuckoo_hash.hpp
 *
 *  Created on: 01-06-2015
 *      Author: Karol Dzitkowski
 *
 *  >>  Real-time Parallel Hashing on the GPU
 *
 *  Implementation of a fast cuckoo hashing method introduced in publication:
 *
 *  Dan A. Alcantara, Andrei Sharf, Fatemeh Abbasinejad, Shubhabrata Sengupta,
 *  Michael Mitzenmacher, John D. Owens, and Nina Amenta "Real-time Parallel
 *  Hashing on the GPU", ACM Transactions on Graphics
 *  (Proceedings of ACM SIGGRAPH Asia 2009)
 *
 *  which can be found here http://idav.ucdavis.edu/~dfalcant/research/hashing.php
 */


//
//Przeanalizowac problem dobierania funkcji hashujacej (wartosci losowej) pod wzgl«dem ewolucyjnym,
//zmieniamy pewne bity wartosci wylosowanej zamiast losowac nowˆ i sprawdzamy czy pomog¸o

#include "fast_cuckoo_hash.cuh"
#include "hash_function.cuh"
#include "helpers.h"
#include "macros.h"
#include <thrust/scan.h>
#include <cuda_runtime_api.h>
#include "helpers.h"

__global__ void divideKernel(
		const int2* values,
		const int size,
		const Constants<2> constants,
		unsigned int* counts,
		const int bucket_cnt,
		unsigned int* offsets,
		const unsigned int max_size,
		bool* failure)
{
	unsigned idx = threadIdx.x + blockIdx.x * blockDim.x +
			blockIdx.y * blockDim.x * gridDim.x;

	if(idx >= size) return;

	int key = values[idx].x;
	unsigned hash = bucketHashFunction(constants.values[0], constants.values[1], key, bucket_cnt);
	offsets[idx] = atomicAdd(&counts[hash], 1);
	if(offsets[idx] == max_size - 1) *failure = true;
}

__global__ void copyKernel(
		const int2* values,
		const int size,
		const Constants<2> constants,
		unsigned int* starts,
		const int bucket_cnt,
		unsigned int* offsets,
		int2* buffer)
{
	unsigned idx = threadIdx.x + blockIdx.x * blockDim.x +
			blockIdx.y * blockDim.x * gridDim.x;

	if(idx >= size) return;

	int key = values[idx].x;
	unsigned hash = bucketHashFunction(constants.values[0], constants.values[1], key, bucket_cnt);
	unsigned point = starts[hash] + offsets[idx];
	buffer[point] = values[idx];
}

bool splitToBuckets(
		const int2* values,
		const int size,
		const Constants<2> constants,
		const int bucket_cnt,
		const int block_size,
		unsigned int* starts,
		unsigned int* counts,
		int2* result)
{
	auto grid = CuckooHash<2>::GetGrid(size);
	int blockSize = CuckooHash<2>::DEFAULT_BLOCK_SIZE;

	bool h_failure;
	bool* d_failure;
	unsigned int* d_offsets;

	CUDA_CALL( cudaMalloc((void**)&d_offsets, size*sizeof(unsigned int)) );
	CUDA_CALL( cudaMemset(d_offsets, 0, size*sizeof(unsigned int)) );

	CUDA_CALL( cudaMalloc((void**)&d_failure, sizeof(bool)) );
	CUDA_CALL( cudaMemset(d_failure, 0, sizeof(bool)) );

	divideKernel<<<grid, blockSize>>>(
			values, size, constants, counts,
			bucket_cnt, d_offsets, block_size, d_failure);

	cudaDeviceSynchronize();
	CUDA_CALL( cudaMemcpy(&h_failure, d_failure, sizeof(bool), cudaMemcpyDeviceToHost) );
	CUDA_CALL( cudaFree(d_failure) );

	if(h_failure == false)
	{
		thrust::device_ptr<unsigned int> starts_ptr(starts);
		thrust::device_ptr<unsigned int> counts_ptr(counts);
		auto end = thrust::exclusive_scan(counts_ptr, counts_ptr+bucket_cnt, starts_ptr);

		copyKernel<<<grid, blockSize>>>(
				values, size, constants, starts, bucket_cnt, d_offsets, result);
		cudaDeviceSynchronize();
	}

	CUDA_CALL( cudaFree(d_offsets) );
	return !h_failure;
}

__global__ void insertKernel(
		const int2* valuesArray,
		const unsigned int* starts,
		const unsigned int* counts,
		const int arrId,
		int2* hashMap,
		int2* stash,
		int* stashCnt,
		Constants<3> constants,
		int* failures)
{
	unsigned i, hash, hash_idx;
	unsigned idx = threadIdx.x;
	unsigned idx2 = threadIdx.x + blockDim.x;
	__shared__ int2 s[PART_HASH_MAP_SIZE+1];

	// GET DATA
	const int2* values = valuesArray + starts[arrId];
	const int size = counts[arrId];
	const int part = PART_HASH_MAP_SIZE * arrId;
	int2* hashMap_part = hashMap + part;

	// COPY HASH MAP TO SHARED MEMORY
	s[idx] = hashMap_part[idx];
	if(idx2 < PART_HASH_MAP_SIZE)
		s[idx2] = hashMap_part[idx2];
	__syncthreads();

	int2 old_value = int2{EMPTY_BUCKET_KEY, EMPTY_BUCKET_KEY};
	bool working = idx < size;
	int2 value = working ? values[idx] : int2{EMPTY_BUCKET_KEY, EMPTY_BUCKET_KEY};

	#pragma unroll
	for(i = 0; i <= MAX_RETRIES; i++)
	{
		hash = hashFunction(constants.values[i%3], value.x, PIECE_SIZE);
		hash_idx = working ? hash + ((i%3) * PIECE_SIZE) : PART_HASH_MAP_SIZE;
		old_value = s[hash_idx];			// read old value
		__syncthreads();
		s[hash_idx] = value;				// write new value
		__syncthreads();
		if(value.x == s[hash_idx].x)		// check for success
		{
			if(value.y != s[hash_idx].y)
				s[hash_idx] = old_value;
			else if(old_value.x == EMPTY_BUCKET_KEY)
				working = false;
			else {
				value = old_value;
				old_value = int2{EMPTY_BUCKET_KEY, EMPTY_BUCKET_KEY};
			}
		}
		__syncthreads();
	}

	if(working) // try to add to stash
	{
		hash_idx = atomicAdd(stashCnt, 1);
		if(hash_idx < DEFAULT_STASH_SIZE)
			stash[hash_idx] = value;
		else
			atomicAdd(failures, 1);
	}

	// COPY SHARED MEMORY TO HASH MAP
	__syncthreads();
	if(idx2 < PART_HASH_MAP_SIZE)
		hashMap_part[idx2] = s[idx2];
	hashMap_part[idx] = s[idx];
}

bool fast_cuckooHash(
		const int2* values,
		const int in_size,
		int2* hashMap,
		const int bucket_cnt,
		Constants<2> bucket_constants,
		Constants<3> constants,
		int max_iters)
{
	const int block_size = FAST_CUCKOO_HASH_BLOCK_SIZE;
	unsigned int* starts;
	unsigned int* counts;
	int2* buckets;
	int* d_failure, * d_cnt;
	int h_failure, h_cnt;
	const int steam_no = bucket_cnt < MAX_STEAM_NO ? bucket_cnt : MAX_STEAM_NO;

	// CREATE STREAMS
	cudaStream_t* streams = new cudaStream_t[steam_no];
	for(int i=0; i<steam_no; i++)
		CUDA_CALL( cudaStreamCreate(&streams[i]) );

	// ALLOCATE MEMORY
	CUDA_CALL( cudaMalloc((void**)&starts, bucket_cnt*sizeof(unsigned int)) );
	CUDA_CALL( cudaMemset(starts, 0, bucket_cnt*sizeof(unsigned int)) );

	CUDA_CALL( cudaMalloc((void**)&counts, bucket_cnt*sizeof(unsigned int)) );
	CUDA_CALL( cudaMemset(counts, 0, bucket_cnt*sizeof(unsigned int)) );

	CUDA_CALL( cudaMalloc((void**)&buckets, in_size*sizeof(int2)) );
	CUDA_CALL( cudaMemset(buckets, 0xff, in_size*sizeof(int2)) );

	CUDA_CALL( cudaMalloc((void**)&d_failure, sizeof(int)) );
	CUDA_CALL( cudaMemset(d_failure, 0, sizeof(int)) );

	CUDA_CALL( cudaMalloc((void**)&d_cnt, sizeof(int)) );
	CUDA_CALL( cudaMemset(d_cnt, 0, sizeof(int)) );

	bool splitResult = splitToBuckets(
			values, in_size, bucket_constants, bucket_cnt,
			block_size, starts, counts, buckets);

//	printData(buckets, in_size, "Buckets: ");

	int2* stash = hashMap + (bucket_cnt * PART_HASH_MAP_SIZE);

	if(splitResult)
	{
		for(int i=0; i<bucket_cnt; i++)
		{
			insertKernel<<<1, block_size, 0, 0>>>(//streams[i%steam_no]>>>(
					buckets, starts, counts, i, hashMap, stash, d_cnt, constants, d_failure);
		}
		cudaDeviceSynchronize();
		CUDA_CALL( cudaMemcpy(&h_failure, d_failure, sizeof(int), cudaMemcpyDeviceToHost) );
		CUDA_CALL( cudaMemcpy(&h_cnt, d_cnt, sizeof(int), cudaMemcpyDeviceToHost) );
	} else return true;

	// FREE MEMORY
	CUDA_CALL( cudaFree(starts) );
	CUDA_CALL( cudaFree(counts) );
	CUDA_CALL( cudaFree(buckets) );
	CUDA_CALL( cudaFree(d_failure) );
	CUDA_CALL( cudaFree(d_cnt) );
	for(int i=0; i<steam_no; i++)
		CUDA_CALL( cudaStreamDestroy(streams[i]) );
	delete streams;

	printf("FAILURES NO: %d\n", h_failure);
	printf("STASH CNT: %d\n", h_cnt);

	return h_failure;
}

__global__ void toInt2Kernel(const int* keys, const int size, int2* out)
{
	unsigned idx = threadIdx.x + blockIdx.x * blockDim.x +
				blockIdx.y * blockDim.x * gridDim.x;

	if(idx >= size) return;
	out[idx].x = keys[idx];
	out[idx].y = EMPTY_BUCKET_KEY;
}

__global__ void retrieveKernel(
		int2* values,
		int2* hashMap,
		int2* stash,
		int size,
		int bucket_cnt,
		Constants<3> constants,
		Constants<2> bucket_constants, int2* debugData)
{
	unsigned idx = threadIdx.x + blockIdx.x * blockDim.x +
					blockIdx.y * blockDim.x * gridDim.x;
	if(idx >= size) return;

	int key = values[idx].x;

    unsigned hash = bucketHashFunction(
			bucket_constants.values[0], bucket_constants.values[1], key, bucket_cnt);
    const unsigned bucket_start = hash * PART_HASH_MAP_SIZE;
    int2 entry;
    unsigned hash_idx;

	#pragma unroll
	for(int i = 0; i < 3; i++)
	{
		if(entry.x != key)
		{
			hash = hashFunction(constants.values[i%3], key, PIECE_SIZE);
			hash_idx = hash + ((i%3) * PIECE_SIZE) + bucket_start;
			entry = hashMap[hash_idx];
			debugData[idx].x = entry.x;
			debugData[idx].y = hash_idx;
		}
	}

	if(entry.x != key) // check stash
	{
		for(int i=0; i < DEFAULT_STASH_SIZE; i++)
			if(stash[i].x == key)
			{
				entry = stash[i];
				break;
			}
	}

//	if(entry.x == key) values[idx] = entry;
//	else values[idx] = int2{-2,-2};
	values[idx] = entry;
	if(values[idx].x != entry.x || values[idx].y != entry.y)
		values[idx].x = -3;

}

int2* fast_cuckooRetrieve(
		const int* keys,
		const int size,
		int2* hashMap,
		const int bucket_cnt,
		const Constants<2> bucket_constants,
		const Constants<3> constants)
{
	auto grid = CuckooHash<2>::GetGrid(size);
	int blockSize = CuckooHash<2>::DEFAULT_BLOCK_SIZE;

	printHashMap(hashMap, (bucket_cnt + 1)*PART_HASH_MAP_SIZE, "Hash Map:");
	printHashMap(hashMap + bucket_cnt*PART_HASH_MAP_SIZE, DEFAULT_STASH_SIZE, "Stash:");

	// ALLOCATE MEMORY
	int2 *result;
	int2 *stash = hashMap + (bucket_cnt * PART_HASH_MAP_SIZE);
	CUDA_CALL( cudaMalloc((void**)&result, size*sizeof(int2)) );
	CUDA_CALL( cudaMemset(result, 0xff, size*sizeof(int2)) );

	int2* debugData;
	CUDA_CALL( cudaMalloc((void**)&debugData, size*sizeof(int2)));

	// SPLIT TO BUCKETS
	toInt2Kernel<<<grid, blockSize>>>(keys, size, result);
	cudaDeviceSynchronize();

	retrieveKernel<<<grid, blockSize>>>(
			result, hashMap, stash, size, bucket_cnt, constants, bucket_constants, debugData);
	cudaDeviceSynchronize();

	printData(debugData, size, "debug data");

	CUDA_CALL( cudaFree(debugData) );

	CudaCheckError();
	return result;
}
