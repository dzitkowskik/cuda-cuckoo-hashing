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

//	thrust::device_ptr<unsigned int> counts_ptr(counts);
//	PrintDevicePtr(counts_ptr, bucket_cnt, "Counts: ");

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

//	thrust::device_ptr<unsigned int> starts_ptr(starts);
//	PrintDevicePtr(starts_ptr, bucket_cnt, "Starts: ");
//	printData(result, size, "Split result: ");

	CUDA_CALL( cudaFree(d_offsets) );
	return !h_failure;
}

__global__ void insertKernel(
		const int2* valuesArray,
		const unsigned int* starts,
		const unsigned int* counts,
		const int arrId,
		int2* hashMap,
		const int bucket_size,
		Constants<3> constants,
		const unsigned max_iters,
		int* failures,
		int2* hashes)
{
	volatile unsigned i, hash;
	unsigned idx = threadIdx.x;
	unsigned idx2 = threadIdx.x + blockDim.x;
	__shared__ int2 s[PART_HASH_MAP_SIZE];

	// GET DATA
	const int2* values = valuesArray + starts[arrId];
	const int size = counts[arrId];
	const int part = PART_HASH_MAP_SIZE * arrId;

	int2* hashMap_part = hashMap + part;
	int2 old_value, value;
	bool working = idx < size;
	if(working) value = values[idx];

	// COPY HASH MAP TO SHARED MEMORY
	s[idx] = hashMap_part[idx];

	if(idx2 < PART_HASH_MAP_SIZE)
		s[idx2] = hashMap_part[idx2];
	__syncthreads();

	#pragma unroll
	for(i = 0; i <= MAX_RETRIES; i++)
	{
		hash = hashFunction(constants.values[i%3], value.x, PART_HASH_MAP_SIZE);
		old_value = s[hash];				// read old value
		__syncthreads();
		if(working) s[hash] = value;		// write new value
		__syncthreads();
		if(working && value.x == s[hash].x)		// check for success
		{
			if(value.y != s[hash].y)
				s[hash] = int2{EMPTY_BUCKET_KEY, EMPTY_BUCKET_KEY};
			else if(old_value.x == EMPTY_BUCKET_KEY)
			{
				working = false;
			}
			else
			{
				value = old_value;
			}
		}
	}
	if(working)
	{
		atomicAdd(failures, 1);
	}

	// COPY SHARED MEMORY TO HASH MAP
	__syncthreads();
	if(idx2 < PART_HASH_MAP_SIZE)
	{
		hashMap_part[idx2].x = s[idx2].x;
		hashMap_part[idx2].y = s[idx2].y;
	}
	__syncthreads();
	hashMap_part[idx].x = s[idx].x;
	hashMap_part[idx].y = s[idx].y;
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
//	printf("Constants %d %d %d\n", constants.values[0], constants.values[1], constants.values[2]);
//	printf("Bucket Constants %d %d\n", bucket_constants.values[0], bucket_constants.values[1]);

	const int block_size = FAST_CUCKOO_HASH_BLOCK_SIZE;
	unsigned int* starts;
	unsigned int* counts;
	int2* buckets;
	int* d_failure;
	int h_failure;

//	printf("bucket_cnt = %d\n", bucket_cnt);

	// CREATE STREAMS
//	cudaStream_t* streams = new cudaStream_t[bucket_cnt];
//	for(int i=0; i<bucket_cnt; i++)
//		CUDA_CALL( cudaStreamCreate(&streams[i]) );

	// ALLOCATE MEMORY
	int2* hashes;
	CUDA_CALL( cudaMalloc((void**)&hashes, in_size*sizeof(int2)) );
	CUDA_CALL( cudaMemset(hashes, 0, in_size*sizeof(int2)) );

	CUDA_CALL( cudaMalloc((void**)&starts, bucket_cnt*sizeof(unsigned int)) );
	CUDA_CALL( cudaMemset(starts, 0, bucket_cnt*sizeof(unsigned int)) );

	CUDA_CALL( cudaMalloc((void**)&counts, bucket_cnt*sizeof(unsigned int)) );
	CUDA_CALL( cudaMemset(counts, 0, bucket_cnt*sizeof(unsigned int)) );

	CUDA_CALL( cudaMalloc((void**)&buckets, in_size*sizeof(int2)) );
	CUDA_CALL( cudaMemset(buckets, 0xff, in_size*sizeof(int2)) );

	CUDA_CALL( cudaMalloc((void**)&d_failure, sizeof(int)) );
	CUDA_CALL( cudaMemset(d_failure, 0, sizeof(int)) );

	bool splitResult = splitToBuckets(
			values, in_size, bucket_constants, bucket_cnt,
			block_size, starts, counts, buckets);

	if(splitResult)
	{
//		printf("Split success!\n");
		const int shared_mem_size = PART_HASH_MAP_SIZE * sizeof(int2);
		for(int i=0; i<bucket_cnt; i++)
		{
			insertKernel<<<1, block_size>>>(
					buckets, starts, counts, i, hashMap,
					PART_HASH_MAP_SIZE, constants, max_iters, d_failure, hashes);
		}
		cudaDeviceSynchronize();
		CUDA_CALL( cudaMemcpy(&h_failure, d_failure, sizeof(int), cudaMemcpyDeviceToHost) );
	} else return true;

//	printData(hashes, in_size, "Insert Hashes: ");

	// FREE MEMORY
	CUDA_CALL( cudaFree(starts) );
	CUDA_CALL( cudaFree(counts) );
	CUDA_CALL( cudaFree(buckets) );
	CUDA_CALL( cudaFree(d_failure) );
	CUDA_CALL( cudaFree(hashes) );
//	for(int i=0; i<bucket_cnt; i++) CUDA_CALL( cudaStreamDestroy(streams[i]) );
//	delete [] streams;

//	printHashMap(hashMap, 2*576, "HASH MAP: ");
//	printf("NO FAILURES: %d\n", h_failure);

	return h_failure;
}

__global__ void toInt2Kernel(const int* keys, const int size, int2* out)
{
	unsigned idx = threadIdx.x + blockIdx.x * blockDim.x +
				blockIdx.y * blockDim.x * gridDim.x;

	if(idx >= size) return;
	out[idx].x = keys[idx];
	out[idx].y = -1; // SAVE OLD POSITION
}

__global__ void retrieveKernel(
		int2* values,
		int2* hashMap,
		int size,
		int bucket_cnt,
		Constants<3> constants,
		Constants<2> bucket_constants,
		int2* hashes)
{
	unsigned idx = threadIdx.x + blockIdx.x * blockDim.x +
					blockIdx.y * blockDim.x * gridDim.x;
	if(idx >= size) return;

	int key = values[idx].x;
	unsigned bucket = bucketHashFunction(
			bucket_constants.values[0], bucket_constants.values[1], key, bucket_cnt);
    volatile unsigned hash, hash_idx;
    int2 entry;

	for(int i = 0; i < 3; i++)
	{
		hash = hashFunction(constants.values[i%3], key, PART_HASH_MAP_SIZE);
		hash_idx = hash + bucket * PART_HASH_MAP_SIZE;
		entry = hashMap[hash_idx];
		if(entry.x == key)
		{
			break;
		}
	}

	if(entry.x == key)
		values[idx].y = entry.y;
	else values[idx] = int2{-1,-1};
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

//	printf("Constants %d %d %d\n", constants.values[0], constants.values[1], constants.values[2]);
//	printf("Bucket Constants %d %d\n", bucket_constants.values[0], bucket_constants.values[1]);
//	printHashMap(hashMap, bucket_cnt*576, "HASH MAP: ");

	// ALLOCATE MEMORY
	int2 *result, *hashes;
	CUDA_CALL( cudaMalloc((void**)&result, size*sizeof(int2)) );
	CUDA_CALL( cudaMemset(result, 0xff, size*sizeof(int2)) );
	CUDA_CALL( cudaMalloc((void**)&hashes, 3*size*sizeof(int2)) );
	CUDA_CALL( cudaMemset(hashes, 0, 3*size*sizeof(int2)) );

	// SPLIT TO BUCKETS
	toInt2Kernel<<<grid, blockSize>>>(keys, size, result);
	cudaDeviceSynchronize();

	retrieveKernel<<<grid, blockSize>>>(
			result, hashMap, size, bucket_cnt, constants, bucket_constants, hashes);
	cudaDeviceSynchronize();

//	printData(hashes, 3*size, "Hashes: ");

	// FREE MEMORY
	CUDA_CALL( cudaFree(hashes) );

	return result;
}
