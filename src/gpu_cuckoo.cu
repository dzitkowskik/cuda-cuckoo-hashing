#include "gpu_cuckoo.cuh"
#include "macros.h"
#include <random>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/gather.h>
#include <thrust/scan.h>
#include <thrust/device_vector.h>

__device__ int hashFunctionDev(int value, int size, int num)
{
	return (0xFAB011991 ^ num + num * value) % (size+1);
}

__global__ void cuckooRefillStencilKernel(int2* values, int values_size, bool* stencil, int stencil_size, int hashNum)
{
	unsigned long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= values_size) return;
	int2 value = values[idx];
	int hash = hashFunctionDev(value.x, stencil_size, hashNum);
	stencil[hash] = true;
}

__global__ void cuckooFillKernel(int2* values, int values_size, int2* hashMap, int hashMap_size, int hashNum)
{
	unsigned long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= values_size) return;
	int2 value = values[idx];
	int hash = hashFunctionDev(value.x, hashMap_size, hashNum);
	hashMap[hash] = value;
}

__global__ void cuckooCheckKernel(int2* values, int values_size, int2* hashMap, int hashMap_size, int hashNum, bool* result)
{
	unsigned long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	if (idx >= values_size) return;
	int2 value = values[idx];
	int hash = hashFunctionDev(value.x, hashMap_size, hashNum);
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

__host__ thrust::device_vector<int2> cuckooFillHashMap(int2* values, int size, int2* hashMap, int hashMap_size, int hashNum)
{
	bool* stencil;

	thrust::device_vector<int2> result_vector;
	thrust::device_ptr<int2> hashMap_ptr(hashMap);

	int block_size = CUCKOO_HASHING_BLOCK_SIZE;
	int block_cnt = (size + block_size - 1) / block_size;

	int stencil_size = hashMap_size;
	CUDA_CALL( cudaMalloc((void**)&stencil, stencil_size*sizeof(bool)) );
	CUDA_CALL( cudaMemset(stencil, 0, stencil_size*sizeof(bool)) );
	thrust::device_ptr<bool> stencil_ptr(stencil);
	cuckooRefillStencilKernel<<<block_size, block_cnt>>>(values, size, stencil, stencil_size, hashNum);
	thrust::copy_if(hashMap_ptr, hashMap_ptr + hashMap_size, stencil_ptr, result_vector.begin(), is_true());
	CUDA_CALL( cudaFree(stencil) );

	cuckooFillKernel<<<block_size, block_cnt>>>(values, size, hashMap, hashMap_size, hashNum);
	CUDA_CALL( cudaMalloc((void**)&stencil, size*sizeof(bool)) );
	CUDA_CALL( cudaMemset(stencil, 0, size*sizeof(bool)) );
	cudaDeviceSynchronize();
	cuckooCheckKernel<<<block_size, block_cnt>>>(values, size, hashMap, hashMap_size, hashNum, stencil);
	cudaDeviceSynchronize();

	thrust::device_ptr<int2> values_ptr(values);
	stencil_ptr = thrust::device_pointer_cast(stencil);
	thrust::copy_if(values_ptr, values_ptr + size, stencil_ptr, result_vector.begin(), is_true());
	CUDA_CALL( cudaFree(stencil) );

	return result_vector;
}

__host__ int2 genSeeds()
{
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<> dis(1, 6);
	int seed_1 = dis(gen);
	int seed_2 = dis(gen);
	return make_int2(seed_1, seed_2);
}

int2** cuckooHash(int2* values, int in_size, int2& out_size, int2& out_seeds)
{
	out_seeds = genSeeds();

	int2** hashMaps = new int2*[NUM_HASHES];
	int hashMap_size = SLOTS_COEF * in_size;
	for(int i = 0; i < NUM_HASHES; i++)
	{
		CUDA_CALL( cudaMalloc((void**)&hashMaps[i], hashMap_size * sizeof(int2)) );
		CUDA_CALL( cudaMemset(hashMaps[i], 0xFF, hashMap_size * sizeof(int2)) ); // free slot has key and value equal 0xFFFFFFFF
	}

	auto collisions = cuckooFillHashMap(values, in_size, hashMaps[0], hashMap_size, out_seeds.x);
	int i = 1;
	while(collisions.size())
	{
		collisions = cuckooFillHashMap(collisions.data().get(), collisions.size(), hashMaps[i], hashMap_size, out_seeds.y);
		i = (i+1)%NUM_HASHES;
	}

	out_size.x = NUM_HASHES;
	out_size.y = hashMap_size;
	return hashMaps;
}

int2** cuckooHash(int2* values, int in_size, int2** hashMaps, int2& hashMap_size, int2 seeds)
{
	auto collisions = cuckooFillHashMap(values, in_size, hashMaps[0], hashMap_size.y, seeds.x);
	int i = 1;
	while(collisions.size())
	{
		collisions = cuckooFillHashMap(collisions.data().get(), collisions.size(), hashMaps[i], hashMap_size.y, i == 1 ? seeds.y : seeds.x);
		i = (i+1)%NUM_HASHES;
	}

	return hashMaps;
}

