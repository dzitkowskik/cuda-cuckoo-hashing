#include "fast_cuckoo_hash.cuh"
#include "hash_function.cuh"
#include "helpers.h"
#include "macros.h"
#include <thrust/scan.h>
#include <cuda_runtime_api.h>

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
	unsigned hash = bucketHashFunction(
			constants.values[0], constants.values[1], key, bucket_cnt);
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
	unsigned hash = bucketHashFunction(
			constants.values[0], constants.values[1], key, bucket_cnt);
	unsigned point = starts[hash] + offsets[idx];
	buffer[point] = values[idx];
}

bool splitToBuckets(
		int2* values,
		const int size,
		const Constants<2> constants,
		const int bucket_cnt,
		const int bucket_size,
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
	CUDA_CALL( cudaMalloc((void**)&d_failure, sizeof(bool)) );

	divideKernel<<<grid, blockSize>>>(
			values, size, constants, counts,
			bucket_cnt, d_offsets, bucket_size, d_failure);
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
		const int2* values,
		const int size,
		int2* hashMap_part,
		const int bucket_size,
		Constants<3> constants,
		const unsigned max_iters,
		bool* failure)
{
	unsigned i, hash;
	unsigned idx = threadIdx.x;
	unsigned idx2 = idx + blockDim.x;
	extern __shared__ int2 s[];

	// COPY HASH MAP TO SHARED MEMORY
	s[idx] = hashMap_part[idx];
	if(idx2 < bucket_size) s[idx2] = hashMap_part[idx2];
	__syncthreads();

	if(idx < size)
	{
		int2 value = values[idx];
		for(i = 1; i <= max_iters; i++)
		{
			hash = hashFunction(constants.values[i%3], value.x, bucket_size);
			int2 old_value = s[hash];	// read old value
			__syncthreads();
			s[hash] = value;			// write new value
			__syncthreads();
			if(value.x == s[hash].x)	// check for success
			{
				if(old_value.x == EMPTY_BUCKET_KEY) break;
				else value = old_value;
			}
		}
		if(i == max_iters) *failure = true;
	}

	// COPY SHARED MEMORY TO HASH MAP
	__syncthreads();
	if(idx2 < bucket_size) s[idx2] = hashMap_part[idx2];
	hashMap_part[idx] = s[idx];
}

bool fast_cuckooHash(
		int2* values,
		int in_size,
		int2* hashMap,
		int hashMap_size,
		Constants<3> constants,
		int stash_size)
{
	return false;
}

int2* fast_cuckooRetrieve(
		int* keys,
		int size,
		int2* hashMap,
		int hashMap_size,
		Constants<3> constants,
		int stash_size)
{
	return NULL;
}
