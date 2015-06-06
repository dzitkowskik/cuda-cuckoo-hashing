/*
 * gpu_cuckoo_benchmark.cpp
 *
 *  Created on: 01-05-2015
 *      Author: Karol Dzitkowski
 */

#include "helpers.h"
#include "macros.h"
#include "naive/naive_cuckoo_hash.cuh"
#include "common/common_cuckoo_hash.cuh"
#include "fast/fast_cuckoo_hash.cuh"
#include <benchmark/benchmark.h>

static void BM_NAIVE_BUILD_HASH_HNO_2(benchmark::State& state)
{
	int N = state.range_x();
	int2* data;
	int* keys;
	NaiveCuckooHash<2> hash;

	while (state.KeepRunning())
	{
		state.PauseTiming();
		hash.Init(N*100);
		data = GenerateRandomKeyValueData(N);
		state.ResumeTiming();

		// BUILD HASH
		hash.BuildTable(data, N);

		state.PauseTiming();
		CUDA_CHECK_RETURN( cudaFree(data) );
		hash.FreeMemory();
		state.ResumeTiming();
	}

	long long int it_processed = state.iterations() * state.range_x();
	state.SetItemsProcessed(it_processed);
	state.SetBytesProcessed(it_processed * sizeof(int2));
}
BENCHMARK(BM_NAIVE_BUILD_HASH_HNO_2)->Arg(1<<10)->Arg(1<<12);

static void BM_COMMON_BUILD_HASH_HNO_2(benchmark::State& state)
{
	int N = state.range_x();
	int2* data;
	int* keys;
	CommonCuckooHash<2> hash;

	while (state.KeepRunning())
	{
		state.PauseTiming();
		hash.Init(N*100);
		data = GenerateRandomKeyValueData(N);
		state.ResumeTiming();

		// BUILD HASH
		hash.BuildTable(data, N);

		state.PauseTiming();
		CUDA_CHECK_RETURN( cudaFree(data) );
		hash.FreeMemory();
		state.ResumeTiming();
	}

	long long int it_processed = state.iterations() * state.range_x();
	state.SetItemsProcessed(it_processed);
	state.SetBytesProcessed(it_processed * sizeof(int2));
}
BENCHMARK(BM_COMMON_BUILD_HASH_HNO_2)->Arg(1<<10)->Arg(1<<12)->Arg(1<<15);

static void BM_COMMON_BUILD_HASH_HNO_3(benchmark::State& state)
{
	int N = state.range_x();
	int2* data;
	int* keys;
	CommonCuckooHash<3> hash;

	while (state.KeepRunning())
	{
		state.PauseTiming();
		hash.Init(N*100);
		data = GenerateRandomKeyValueData(N);
		state.ResumeTiming();

		// BUILD HASH
		hash.BuildTable(data, N);

		state.PauseTiming();
		CUDA_CHECK_RETURN( cudaFree(data) );
		hash.FreeMemory();
		state.ResumeTiming();
	}

	long long int it_processed = state.iterations() * state.range_x();
	state.SetItemsProcessed(it_processed);
	state.SetBytesProcessed(it_processed * sizeof(int2));
}
BENCHMARK(BM_COMMON_BUILD_HASH_HNO_3)->Arg(1<<10)->Arg(1<<12)->Arg(1<<15);

static void BM_FAST_BUILD_HASH_HNO_3(benchmark::State& state)
{
	int N = state.range_x();
	int2* data;
	int* keys;
	FastCuckooHash hash;

	while (state.KeepRunning())
	{
		state.PauseTiming();
		hash.Init(N*100);
		data = GenerateRandomKeyValueData(N);
		state.ResumeTiming();

		// BUILD HASH
		hash.BuildTable(data, N);

		state.PauseTiming();
		CUDA_CHECK_RETURN( cudaFree(data) );
		hash.FreeMemory();
		state.ResumeTiming();
	}

	long long int it_processed = state.iterations() * state.range_x();
	state.SetItemsProcessed(it_processed);
	state.SetBytesProcessed(it_processed * sizeof(int2));
}
BENCHMARK(BM_FAST_BUILD_HASH_HNO_3)->Arg(1<<10)->Arg(1<<12)->Arg(1<<15);
