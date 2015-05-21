/*
 * gpu_cuckoo_benchmark.cpp
 *
 *  Created on: 01-05-2015
 *      Author: Karol Dzitkowski
 */

#include "helpers.h"
#include "macros.h"
#include "naive/naive_cuckoo_hash.hpp"
#include <benchmark/benchmark_api.h>

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
BENCHMARK(BM_NAIVE_BUILD_HASH_HNO_2)
	->ArgPair(1<<10, 10)->ArgPair(1<<10, 20)->ArgPair(1<<10, 40)->ArgPair(1<<15, 40);
