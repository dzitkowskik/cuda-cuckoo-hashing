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
#include "hash_function.cuh"

TEST(GpuCuckooTest, cuckooHash_FAST_noexception)
{
	int N = 10000;
	auto data = GenerateRandomKeyValueData(N);

	FastCuckooHash hash;
	hash.Init(N*5);
	hash.BuildTable(data, N);

	CUDA_CHECK_RETURN( cudaFree(data) );
}

TEST(GpuCuckooTest, cuckooHash_FAST_storeSucceeded)
{
	int N = 1000;
	auto data = GenerateRandomKeyValueData(N);

	FastCuckooHash hash;
	hash.Init(N*5);

	EXPECT_TRUE( hash.BuildTable(data, N) );

	CUDA_CHECK_RETURN( cudaFree(data) );
}

TEST(GpuCuckooTest, cuckooHash_FAST_storeAndretrieve_Tiny)
{
	int N = 50;
	auto data = GenerateRandomKeyValueData(N);
	auto keys = getKeys(data, N);

	FastCuckooHash hash;
	hash.Init(N*20);
	hash.BuildTable(data, N);
	auto result = hash.GetItems(keys, N);

//	printData(data, N, "Expected:");
//	printData(result, N, "Actual:");

	EXPECT_TRUE( compareData(data, result, N) );

	CUDA_CHECK_RETURN( cudaFree(data) );
	CUDA_CHECK_RETURN( cudaFree(keys) );
	CUDA_CHECK_RETURN( cudaFree(result) );
}

TEST(GpuCuckooTest, cuckooHash_FAST_storeAndretrieve_Small)
{
	int N = 200;
	auto data = GenerateRandomKeyValueData(N);
	auto keys = getKeys(data, N);

	FastCuckooHash hash;
	hash.Init(N*16);
	hash.BuildTable(data, N);
	auto result = hash.GetItems(keys, N);

//	printData(data, N, "Expected:");
//	printData(result, N, "Actual:");

	EXPECT_TRUE( compareData(data, result, N) );

	CUDA_CHECK_RETURN( cudaFree(data) );
	CUDA_CHECK_RETURN( cudaFree(keys) );
	CUDA_CHECK_RETURN( cudaFree(result) );
}

TEST(GpuCuckooTest, cuckooHash_FAST_storeAndretrieve_Medium)
{
	int N = 1000;
	auto data = GenerateRandomKeyValueData(N);
	auto keys = getKeys(data, N);

	FastCuckooHash hash;
	hash.Init(N*5);
	EXPECT_TRUE( hash.BuildTable(data, N) );
	auto result = hash.GetItems(keys, N);

//	printData(data, N, "Expected:");
//	printData(result, N, "Actual:");

	EXPECT_TRUE( compareData(data, result, N) );

	CUDA_CHECK_RETURN( cudaFree(data) );
	CUDA_CHECK_RETURN( cudaFree(keys) );
	CUDA_CHECK_RETURN( cudaFree(result) );
}

TEST(GpuCuckooTest, cuckooHash_FAST_storeAndretrieve_Large)
{
	int N = 10000;
	auto data = GenerateRandomKeyValueData(N);
	auto keys = getKeys(data, N);

	FastCuckooHash hash;
	hash.Init(N*5);
	hash.BuildTable(data, N);
	auto result = hash.GetItems(keys, N);

//	printData(data, N, "Expected:");
//	printData(result, N, "Actual:");

	EXPECT_TRUE( compareData(data, result, N) );

	CUDA_CHECK_RETURN( cudaFree(data) );
	CUDA_CHECK_RETURN( cudaFree(keys) );
	CUDA_CHECK_RETURN( cudaFree(result) );
}
