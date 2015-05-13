/*
 *  naive_gpu_cuckoo.cu
 *
 *  Created on: 03-05-2015
 *      Author: Karol Dzitkowski
 */

#include "cuckoo_hash.h"
#include "macros.h"
#include <random>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/gather.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <cuda_runtime_api.h>

#define CUCKOO_HASHING_BLOCK_SIZE 64
#define EMPTY_BUCKET_KEY 0xFFFFFFFF

// hashMap_size - number of hash maps

__device__ int hashFunctionDev(int value, int size, int num)
{
	return (0xFAB011991 ^ num + num * value) % (size+1);
}

__global__ void cuckooRefillStencilKernel(int2* values, int values_size, int2* hashMap, bool* stencil, int stencil_size, int seed)
{
	unsigned long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= values_size) return;
	int2 value = values[idx];
	int hash = hashFunctionDev(value.x, stencil_size, seed);
	stencil[hash] = hashMap[hash].x == EMPTY_BUCKET_KEY;
}

__global__ void cuckooFillKernel(int2* values, int values_size, int2* hashMap, int hashMap_size, int seed)
{
	unsigned long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= values_size) return;
	int2 value = values[idx];
	int hash = hashFunctionDev(value.x, hashMap_size, seed);
	hashMap[hash] = value;
}

__global__ void cuckooCheckKernel(int2* values, int values_size, int2* hashMap, int hashMap_size, bool* result, int seed)
{
	unsigned long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= values_size) return;
	int2 value = values[idx];
	int hash = hashFunctionDev(value.x, hashMap_size, seed);
	result[idx] = hashMap[hash].x != value.x;
}

struct is_true
{
  __host__ __device__
  bool operator()(const bool x)
  {
    return x;
  }
};

__host__ thrust::device_vector<int2>
cuckooFillHashMap(int2* values, int size, int2* hashMap, int hashMap_size, int seed)
{
	bool* stencil;
	int block_size = CUCKOO_HASHING_BLOCK_SIZE;
	int block_cnt = (size + block_size - 1) / block_size;
	int stencil_size = hashMap_size;
	thrust::device_vector<int2> result_vector;
	thrust::device_ptr<int2> hashMap_ptr(hashMap);

	// CREATE STENCIL CONTAINING 1 WHERE SOME ELEMENT WANTS TO BE PUT
	CUDA_CALL( cudaMalloc((void**)&stencil, stencil_size*sizeof(bool)) );
	CUDA_CALL( cudaMemset(stencil, 0, stencil_size*sizeof(bool)) );
	thrust::device_ptr<bool> stencil_ptr(stencil);
	cuckooRefillStencilKernel<<<block_size, block_cnt>>>(values, size, hashMap, stencil, stencil_size, seed);

	// COPY ELEMENTS INDICATED BY STENCIL TO RESULT VECTOR
	thrust::copy_if(hashMap_ptr, hashMap_ptr + hashMap_size, stencil_ptr, result_vector.begin(), is_true());
	CUDA_CALL( cudaFree(stencil) );

	// PUT ELEMENTS IN HASH MAP
	cuckooFillKernel<<<block_size, block_cnt>>>(values, size, hashMap, hashMap_size, seed);

	// CHECK IF MULTIPLE VALUES WERE NOT PUT IN SAME BUCKET AND CREATE A STENCIL OF THEM
	CUDA_CALL( cudaMalloc((void**)&stencil, size*sizeof(bool)) );
	CUDA_CALL( cudaMemset(stencil, 0, size*sizeof(bool)) );
	cudaDeviceSynchronize();
	cuckooCheckKernel<<<block_size, block_cnt>>>(values, size, hashMap, hashMap_size, stencil, seed);
	cudaDeviceSynchronize();

	// COPY ELEMENTS THAT DIDNT FIT TO HASH MAP TO RESULT VECTOR
	thrust::device_ptr<int2> values_ptr(values);
	stencil_ptr = thrust::device_pointer_cast(stencil);
	thrust::copy_if(values_ptr, values_ptr + size, stencil_ptr, result_vector.end(), is_true());
	CUDA_CALL( cudaFree(stencil) );

	return result_vector;
}

void naive_cuckooHash(int2* values, int in_size, int2* hashMap, int hashMap_size, int seeds[HASH_FUNC_NO])
{
	int i = 1;
	auto collisions = cuckooFillHashMap(values, in_size, hashMap, hashMap_size, seeds[i]);
	while(collisions.size())
	{
		collisions = cuckooFillHashMap(collisions.data().get(), collisions.size(), hashMap, hashMap_size, seeds[i]);
		i = (i+1)%HASH_FUNC_NO;
	}
}

__global__ void cuckooRetrieveKernel(int* keys, int size, int2* hashMap, int hashMap_size, int seeds[HASH_FUNC_NO], int2* out)
{
	unsigned long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= size) return;
	int key = keys[idx];
	int2 entry;
	#pragma unroll
	for(int i=0; i<HASH_FUNC_NO; i++)
	{
		int hash = hashFunctionDev(key, hashMap_size, seeds[i]);
		entry = hashMap[hash];
		if(entry.x == key) break;
	}

	if(entry.x != key) entry.x = EMPTY_BUCKET_KEY;
	out[idx] = entry;
}

int2* naive_cuckooRetrieve(int* keys, int size, int2* hashMap, int hashMap_size, int seeds[HASH_FUNC_NO])
{
	int block_size = CUCKOO_HASHING_BLOCK_SIZE;
	int block_cnt = (size + block_size - 1) / block_size;
	int2* result;
	CUDA_CALL( cudaMalloc((void**)&result, size*sizeof(int2)) );

	cuckooRetrieveKernel<<<block_size, block_cnt>>>(keys, size, hashMap, hashMap_size, seeds, result);
	cudaDeviceSynchronize();

	return result;
}

