/*
 * gpu_cuckoo_unittest.cpp
 *
 *  Created on: 01-05-2015
 *      Author: Karol Dzitkowski
 */

#include <gtest/gtest.h>
#include <cuda_runtime_api.h>
#include <random>
#include <chrono>
#include "macros.h"
#include <vector_functions.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include "cuckoo_hash.h"
#include <thrust/copy.h>

typedef std::chrono::high_resolution_clock myclock;

int2* GenerateRandomKeyValueData(const int N)
{
	auto time = std::chrono::system_clock::now();
	unsigned seed = time.time_since_epoch().count();
	std::minstd_rand0 generator(seed);
	std::uniform_int_distribution<int> distribution(1,20);
	auto h_data = new int2[N];

	int key = 0;
	for(int i=0; i<N; i++)
	{
		key += distribution(generator);
		h_data[i] = make_int2(key, distribution(generator));
	}

	int2* d_data;
	CUDA_CHECK_RETURN( cudaMalloc((void**)&d_data, N*sizeof(int2)) );
	CUDA_CHECK_RETURN( cudaMemcpy(d_data, h_data, N*sizeof(int2), cudaMemcpyHostToDevice) );

	delete [] h_data;
	return d_data;
}

void printData(int2* data, int N, const char* name)
{
	printf("%s\n", name);
	thrust::device_ptr<int2> data_ptr(data);
	for(int i=0; i<N; i++)
	{
		int2 temp = data_ptr[i];
		std::cout << "(" << temp.x << "," << temp.y << ")";
	}
	std::cout << std::endl;
}

bool compareData(int2* a, int2* b, int size)
{
	thrust::device_ptr<int2> a_ptr(a);
	thrust::device_ptr<int2> b_ptr(b);

	for(int i=0; i<size; i++)
	{
		int2 a_value = a_ptr[i];
		int2 b_value = b_ptr[i];
		if(a_value.x != b_value.x || a_value.y != b_value.y)
			return false;
	}
	return true;
}

int* getKeys(int2* key_values, int size)
{
	int* d_data;
	CUDA_CHECK_RETURN( cudaMalloc((void**)&d_data, size*sizeof(int)) );
	thrust::device_ptr<int> d_data_ptr(d_data);
	thrust::device_ptr<int2> key_values_ptr(key_values);
	for(int i=0; i<size; i++)
	{
		int2 key_value = key_values_ptr[i];
		d_data_ptr[i] = key_value.x;
	}
	return d_data;
}

TEST(GpuCuckooTest, cuckooHash_naive_noexception)
{
	int N = 1000;
	auto data = GenerateRandomKeyValueData(N);

	CuckooHash hash;
	hash.Init(N*100);

	hash.BuildTable(data, N);

	CUDA_CHECK_RETURN( cudaFree(data) );
}

TEST(GpuCuckooTest, copy_if_test)
{
	testCopy_If();
}

TEST(GpuCuckooTest, cuckooHash_naive_storeAndretrieve)
{
	int N = 100;
	auto data = GenerateRandomKeyValueData(N);
	auto keys = getKeys(data, N);

	CuckooHash hash;
	hash.Init(N*100);
	hash.BuildTable(data, N);
	auto result = hash.GetItems(keys, N);

	printData(data, N, "Actual:");
	printData(result, N, "Expected:");

	EXPECT_TRUE( compareData(data, result, N) );

	CUDA_CHECK_RETURN( cudaFree(data) );
	CUDA_CHECK_RETURN( cudaFree(keys) );
	CUDA_CHECK_RETURN( cudaFree(result) );
}

