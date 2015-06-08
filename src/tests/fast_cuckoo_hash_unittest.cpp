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
#include <string>
#include <iostream>
#include <fstream>

void SaveGE(int N, std::string name)
{
	auto d_data = GenerateRandomKeyValueData(N);
	int2* h_data = new int2[N];
	CUDA_CALL( cudaMemcpy(h_data, d_data, N*sizeof(int2), cudaMemcpyDeviceToHost) );
	std::ofstream fout(name);
	if(fout.is_open() && fout.good())
	{
		fout << N << std::endl;
		for(int i=0; i<N; i++)
			fout << h_data[i].x << " " << h_data[i].y << std::endl;
	}
	CUDA_CALL( cudaFree(d_data) );
	delete h_data;

	fout.close();
}

int2* LoadGE(int &N, std::string name)
{
	std::string line;
	std::ifstream infile(name);
	getline(infile, line);
	std::istringstream iss(line);
	if (!(iss >> N))
		throw std::runtime_error(string("Error while reading file: ") + name);
	int2* h_data = new int2[N];

	int i=0;
	while (getline(infile, line) && !line.empty() && i++ < N)
	{
		std::istringstream iss(line);
		int col, row;
		double val;
		if (!(iss >> h_data[i].x >> h_data[i].y))
			throw std::runtime_error(string("Error while reading file: ") + name);
	}

	return h_data;
}

//TEST(GpuCuckooTest, cuckooHash_FAST_noexception)
//{
//	int N = 10000;
//	auto data = GenerateRandomKeyValueData(N);
//
//	FastCuckooHash hash;
//	hash.Init(N*2);
//	hash.BuildTable(data, N);
//
//	CUDA_CHECK_RETURN( cudaFree(data) );
//}
//
//TEST(GpuCuckooTest, cuckooHash_FAST_storeSucceeded)
//{
//	int N = 1000;
//	auto data = GenerateRandomKeyValueData(N);
//
//	FastCuckooHash hash;
//	hash.Init(N*2);
//
//	EXPECT_TRUE( hash.BuildTable(data, N) );
//
//	CUDA_CHECK_RETURN( cudaFree(data) );
//}
//
//TEST(GpuCuckooTest, cuckooHash_FAST_storeAndretrieve_Tiny)
//{
//	int N = 200;
//	auto data = GenerateRandomKeyValueData(N);
//	auto keys = getKeys(data, N);
//
//	FastCuckooHash hash;
//	hash.Init(N*2);
//	hash.BuildTable(data, N);
//	auto result = hash.GetItems(keys, N);
//
////	printData(data, N, "Expected:");
////	printData(result, N, "Actual:");
//
//	EXPECT_TRUE( compareData(data, result, N) );
//
//	CUDA_CHECK_RETURN( cudaFree(data) );
//	CUDA_CHECK_RETURN( cudaFree(keys) );
//	CUDA_CHECK_RETURN( cudaFree(result) );
//}

TEST(GpuCuckooTest, cuckooHash_FAST_storeAndretrieve_FromFile)
{
	std::string name = "random.data";
	int N;
	int2* h_data = LoadGE(N, name);
	int2* d_data;
	CUDA_CALL( cudaMalloc((void**)&d_data, N*sizeof(int2)) );
	CUDA_CALL( cudaMemcpy(d_data, h_data, N*sizeof(int2), cudaMemcpyHostToDevice) );

	auto keys = getKeys(d_data, N);

	FastCuckooHash hash;
	hash.Init(N*2);
	hash.BuildTable(d_data, N);
	auto result = hash.GetItems(keys, N);

	printData(d_data, N, "Expected:");
	printData(result, N, "Actual:");

	EXPECT_TRUE( compareData(d_data, result, N) );

	CUDA_CHECK_RETURN( cudaFree(d_data) );
	CUDA_CHECK_RETURN( cudaFree(keys) );
	CUDA_CHECK_RETURN( cudaFree(result) );
}

//TEST(GpuCuckooTest, cuckooHash_FAST_storeAndretrieve_Small)
//{
//	int N = 1000;
//	auto data = GenerateRandomKeyValueData(N);
//	auto keys = getKeys(data, N);
//
//	FastCuckooHash hash;
//	hash.Init(N*2);
//	hash.BuildTable(data, N);
//	auto result = hash.GetItems(keys, N);
//
//	printData(data, N, "Expected:");
//	printData(result, N, "Actual:");
//
//	EXPECT_TRUE( compareData(data, result, N) );
//
//	CUDA_CHECK_RETURN( cudaFree(data) );
//	CUDA_CHECK_RETURN( cudaFree(keys) );
//	CUDA_CHECK_RETURN( cudaFree(result) );
//}

//TEST(GpuCuckooTest, cuckooHash_FAST_storeAndretrieve_Medium)
//{
//	int N = 8000;
//	auto data = GenerateRandomKeyValueData(N);
//	auto keys = getKeys(data, N);
//
//	FastCuckooHash hash;
//	hash.Init(N*2);
//	EXPECT_TRUE( hash.BuildTable(data, N) );
//	auto result = hash.GetItems(keys, N);
//
////	printData(data, N, "Expected:");
////	printData(result, N, "Actual:");
//
//	EXPECT_TRUE( compareData(data, result, N) );
//
//	CUDA_CHECK_RETURN( cudaFree(data) );
//	CUDA_CHECK_RETURN( cudaFree(keys) );
//	CUDA_CHECK_RETURN( cudaFree(result) );
//}
//
//TEST(GpuCuckooTest, cuckooHash_FAST_storeAndretrieve_Large)
//{
//	int N = 20000;
//	auto data = GenerateRandomKeyValueData(N);
//	auto keys = getKeys(data, N);
//
//	FastCuckooHash hash;
//	hash.Init(N*2);
//	EXPECT_TRUE( hash.BuildTable(data, N) );
//	auto result = hash.GetItems(keys, N);
//
////	printData(data, N, "Expected:");
////	printData(result, N, "Actual:");
//
//	EXPECT_TRUE( compareData(data, result, N) );
//
//	CUDA_CHECK_RETURN( cudaFree(data) );
//	CUDA_CHECK_RETURN( cudaFree(keys) );
//	CUDA_CHECK_RETURN( cudaFree(result) );
//}
