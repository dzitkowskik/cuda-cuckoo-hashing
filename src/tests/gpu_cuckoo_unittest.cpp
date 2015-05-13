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

typedef std::chrono::high_resolution_clock myclock;

int2* GenerateRandomKeyValueData(const int N)
{
	auto time = std::chrono::system_clock::now();
	unsigned seed = time.time_since_epoch().count();
	std::minstd_rand0 generator(seed);
	std::uniform_int_distribution<int> distribution(0,20);
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

void printData(int2* data, int N)
{
	thrust::device_ptr<int2> data_ptr(data);
	for(int i=0; i<N; i++)
	{
		int2 temp = data_ptr[i];
		std::cout << temp.x << " " << temp.y << std::endl;
	}
}

TEST(GpuCuckooTest, cuckooHash_simpleHashmapCreate)
{
	int N = 1000;
	auto data = GenerateRandomKeyValueData(N);



	CUDA_CHECK_RETURN( cudaFree(data) );
}

