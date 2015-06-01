/*
 *  fast_cuckoo_hash_unittest.cpp
 *
 *  Created on: 01-05-2015
 *      Author: Karol Dzitkowski
 */

#include <gtest/gtest.h>
#include <cuda_runtime_api.h>
#include "macros.h"
#include <vector_functions.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include "fast/fast_cuckoo_hash.cuh"
#include "helpers.h"


TEST(GpuCuckooTest, cuckooHash_FAST_noexception)
{
	int N = 10000;
	auto data = GenerateRandomKeyValueData(N);

	FastCuckooHash hash;
	hash.Init(N*10);
	hash.BuildTable(data, N);

	CUDA_CHECK_RETURN( cudaFree(data) );
}

TEST(GpuCuckooTest, cuckooHash_FAST_storeSucceeded)
{
	int N = 10000;
	auto data = GenerateRandomKeyValueData(N);

	FastCuckooHash hash;
	hash.Init(N*5);

	EXPECT_TRUE( hash.BuildTable(data, N) );

	CUDA_CHECK_RETURN( cudaFree(data) );
}

TEST(GpuCuckooTest, cuckooHash_FAST_storeAndretrieve)
{
	int N = 10000;
	auto data = GenerateRandomKeyValueData(N);
	auto keys = getKeys(data, N);

	FastCuckooHash hash;
	hash.Init(N*5);
	hash.BuildTable(data, N);
	auto result = hash.GetItems(keys, N);

//	printData(data, N, "Actual:");
//	printData(result, N, "Expected:");

	EXPECT_TRUE( compareData(data, result, N) );

	CUDA_CHECK_RETURN( cudaFree(data) );
	CUDA_CHECK_RETURN( cudaFree(keys) );
	CUDA_CHECK_RETURN( cudaFree(result) );
}
