/*
 * gpu_cuckoo_unittest.cpp
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
#include "naive/naive_cuckoo_hash.hpp"
#include "helpers.h"

TEST(GpuCuckooTest, cuckooHash_naive_noexception)
{
	int N = 1000;
	auto data = GenerateRandomKeyValueData(N);

	NaiveCuckooHash<2> hash;
	hash.Init(N*100);
	hash.BuildTable(data, N);

	CUDA_CHECK_RETURN( cudaFree(data) );
}

TEST(GpuCuckooTest, cuckooHash_naive_storeSucceeded)
{
	int N = 1000;
	auto data = GenerateRandomKeyValueData(N);

	NaiveCuckooHash<2> hash;
	hash.Init(N*20);

	EXPECT_TRUE( hash.BuildTable(data, N) );

	CUDA_CHECK_RETURN( cudaFree(data) );
}

TEST(GpuCuckooTest, cuckooHash_naive_storeAndretrieve)
{
	int N = 1000;
	auto data = GenerateRandomKeyValueData(N);
	auto keys = getKeys(data, N);

	NaiveCuckooHash<2> hash;
	hash.Init(N*100);
	hash.BuildTable(data, N);
	auto result = hash.GetItems(keys, N);

//	printData(data, N, "Actual:");
//	printData(result, N, "Expected:");

	EXPECT_TRUE( compareData(data, result, N) );

	CUDA_CHECK_RETURN( cudaFree(data) );
	CUDA_CHECK_RETURN( cudaFree(keys) );
	CUDA_CHECK_RETURN( cudaFree(result) );
}

