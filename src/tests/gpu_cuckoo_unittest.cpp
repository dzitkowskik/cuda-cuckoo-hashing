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

typedef std::chrono::high_resolution_clock myclock;

int2* GenerateRandomKeyValueData(const int N)
{
	std::default_random_engine generator(myclock::now());
	std::uniform_int_distribution<int> distribution(0,9);
	auto h_data = new int2[N];

	for(int i=0; i<N; i++)
		h_data[i] = make_int2(distribution(generator), distribution(generator));

	int2* d_data;
	CUDA_CHECK_RETURN( cudaMalloc((void**)d_data, N*sizeof(int2)) );
	CUDA_CHECK_RETURN( cudaMemcpy(d_data, h_data, N*sizeof(int2), cudaMemcpyHostToDevice) );

	delete [] h_data;
	return d_data;
}


TEST(GpuCuckooTest, cuckooHash_simpleHashmapCreate)
{
	auto data = GenerateRandomKeyValueData(1000);



	CUDA_CHECK_RETURN( cudaFree(data) );
}

