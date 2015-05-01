/*
 * main_benchmarks.cpp
 *
 *  Created on: 01-05-2015
 *      Author: ghash
 */

#include <benchmark/benchmark.h>

int main(int argc, char** argv)
{
    ::benchmark::Initialize(&argc, const_cast<const char**>(argv));
    ::benchmark::RunSpecifiedBenchmarks();
    return 0;
}



