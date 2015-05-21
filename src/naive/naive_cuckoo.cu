/*
 *  naive_cuckoo.cu
 *
 *  Created on: 03-05-2015
 *      Author: Karol Dzitkowski
 */

#include "cuckoo_hash.hpp"
#include "macros.h"
#include "constants.h"
#include "hash_function.cuh"
#include <random>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/gather.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include <cuda_runtime_api.h>

__global__ void cuckooRefillStencilKernel(int2* values, int values_size, int2* hashMap,
		int* stencil, int stencil_size, unsigned seed)
{
	unsigned long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= values_size) return;
	int2 value = values[idx];
	int hash = hashFunction(seed, value.x, stencil_size);
	stencil[hash] = hashMap[hash].x != EMPTY_BUCKET_KEY ? 1 : 0;
}

__global__ void cuckooFillKernel(int2* values, int values_size, int2* hashMap,
		int hashMap_size, unsigned seed)
{
	unsigned long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= values_size) return;
	int2 value = values[idx];
	int hash = hashFunction(seed, value.x, hashMap_size);
	hashMap[hash] = value;
}

__global__ void cuckooCheckKernel(int2* values, int values_size, int2* hashMap,
		int hashMap_size, int* result, unsigned seed)
{
	unsigned long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= values_size) return;
	int2 value = values[idx];
	int hash = hashFunction(seed, value.x, hashMap_size);
	result[idx] = hashMap[hash].x != value.x ? 1 : 0;
}

struct is_true
{
  __host__ __device__
  bool operator()(const int x)
  {
    return x == 1;
  }
};

__host__ thrust::device_vector<int2>
cuckooFillHashMap(int2* values, int size, int2* hashMap, int hashMap_size, unsigned seed)
{
	int block_size = CUCKOO_HASHING_BLOCK_SIZE;
	int block_cnt = (size + block_size - 1) / block_size;

	thrust::device_vector<int2> result_vector;
	thrust::device_ptr<int2> hashMap_ptr(hashMap);

	int* stencil;
	int stencil_size = hashMap_size;

	// CREATE STENCIL CONTAINING 1 WHERE SOME ELEMENT WANTS TO BE PUT
	CUDA_CALL( cudaMalloc((void**)&stencil, stencil_size*sizeof(int)) );
	CUDA_CALL( cudaMemset(stencil, 0, stencil_size*sizeof(int)) );
	cuckooRefillStencilKernel<<<block_size, block_cnt>>>(values, size, hashMap, stencil, stencil_size, seed);
	cudaDeviceSynchronize();

	thrust::device_ptr<int> stencil_ptr(stencil);
//	PrintStencil(stencil_ptr, hashMap_size, "First Stencil:");

	// resize result_vector to fit additional data pointed by stencil
	int cnt_1 = thrust::reduce(stencil_ptr, stencil_ptr + hashMap_size);
//	printf("Cnt 1 = %d\n", cnt_1);
	result_vector.resize(result_vector.size()+cnt_1);

	// COPY ELEMENTS INDICATED BY STENCIL TO RESULT VECTOR
	thrust::copy_if(hashMap_ptr, (hashMap_ptr+hashMap_size), stencil_ptr, result_vector.data(), is_true());
	cudaDeviceSynchronize();
	CUDA_CALL( cudaFree(stencil) );

//	PrintDeviceVector(result_vector, "Result Vector: ");

	// PUT ELEMENTS IN HASH MAP
	cuckooFillKernel<<<block_size, block_cnt>>>(values, size, hashMap, hashMap_size, seed);
	cudaDeviceSynchronize();

	// CHECK IF MULTIPLE VALUES WERE NOT PUT IN SAME BUCKET AND CREATE A STENCIL OF THEM
	CUDA_CALL( cudaMalloc((void**)&stencil, size*sizeof(int)) );
	CUDA_CALL( cudaMemset(stencil, 0, size*sizeof(int)) );
	cuckooCheckKernel<<<block_size, block_cnt>>>(values, size, hashMap, hashMap_size, stencil, seed);
	cudaDeviceSynchronize();
	stencil_ptr = thrust::device_pointer_cast(stencil);
//	PrintStencil(stencil_ptr, size, "Second Stencil:");

	// resize result_vector to fit additional data pointed by stencil
	int cnt_2 = thrust::reduce(stencil_ptr, stencil_ptr + size);
//	printf("Cnt 2 = %d\n", cnt_2);
	result_vector.resize(result_vector.size()+cnt_2);

	// COPY ELEMENTS THAT DIDNT FIT TO HASH MAP TO RESULT VECTOR
	thrust::device_ptr<int2> values_ptr(values);
	thrust::copy_if(values_ptr, values_ptr + size, stencil_ptr, result_vector.data()+cnt_1, is_true());
	cudaDeviceSynchronize();
	CUDA_CALL( cudaFree(stencil) );

//	PrintDeviceVector(result_vector, "Result Vector: ");
	result_vector.shrink_to_fit();
	return result_vector;
}

template<unsigned N>
bool naive_cuckooHash(int2* values, int in_size, int2* hashMap, int hashMap_size, Constants<N> constants)
{
	int i = 1, k = 0;
	auto collisions = cuckooFillHashMap(values, in_size, hashMap, hashMap_size, constants.values[i]);
	while(collisions.size() && k++ < MAX_RETRIES)
	{
		collisions = cuckooFillHashMap(collisions.data().get(), collisions.size(), hashMap, hashMap_size, constants.values[i]);
		i = (i+1)%N;
	}
	return collisions.size() == 0;
}

__constant__ unsigned const_seeds[MAX_HASH_FUNC_NO];

__global__ void cuckooRetrieveKernel(
		int* keys,
		int size,
		int2* hashMap,
		int hashMap_size,
		int2* out,
		unsigned N)
{
	unsigned long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= size) return;
	int key = keys[idx];
	int hash = hashFunction(const_seeds[0], key, hashMap_size);
	int2 entry = hashMap[hash];

	for(int i=1; i<N && entry.x != key; i++)
	{
		hash = hashFunction(const_seeds[i], key, hashMap_size);
		entry = hashMap[hash];
	}

	if(entry.x != key)
	{
		entry.x = EMPTY_BUCKET_KEY;
		entry.y = EMPTY_BUCKET_KEY;
	}
	out[idx] = entry;
}

template<unsigned N>
int2* naive_cuckooRetrieve(int* keys, int size, int2* hashMap, int hashMap_size, Constants<N> constants)
{
	int block_size = CUCKOO_HASHING_BLOCK_SIZE;
	int block_cnt = (size + block_size - 1) / block_size;
	int2* result;
	CUDA_CALL( cudaMalloc((void**)&result, size*sizeof(int2)) );
	cudaMemcpyToSymbol(const_seeds, &constants.values, N*sizeof(unsigned));
	cuckooRetrieveKernel<<<block_size, block_cnt>>>(keys, size, hashMap, hashMap_size, result, N);
	cudaDeviceSynchronize();

	return result;
}

template bool naive_cuckooHash<2>(int2*, int, int2*, int, Constants<2>);
template bool naive_cuckooHash<3>(int2*, int, int2*, int, Constants<3>);
template bool naive_cuckooHash<4>(int2*, int, int2*, int, Constants<4>);
template bool naive_cuckooHash<5>(int2*, int, int2*, int, Constants<5>);

template int2* naive_cuckooRetrieve<2>(int*, int, int2*, int, Constants<2>);
template int2* naive_cuckooRetrieve<3>(int*, int, int2*, int, Constants<3>);
template int2* naive_cuckooRetrieve<4>(int*, int, int2*, int, Constants<4>);
template int2* naive_cuckooRetrieve<5>(int*, int, int2*, int, Constants<5>);
