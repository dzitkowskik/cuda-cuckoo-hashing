/*
 * cuckoo_hash.cpp
 *
 *  Created on: 03-05-2015
 *      Author: Karol Dzitkowski
 */

#include "cuckoo_hash.hpp"
#include "macros.h"
#include <cstdlib>
#include <time.h>

template<unsigned N>
dim3 CuckooHash<N>::getGrid(size_t size)
{
	auto block_cnt = (size + DEFAULT_BLOCK_SIZE-1) / DEFAULT_BLOCK_SIZE;
    dim3 grid( block_cnt );
    if (grid.x > MAX_GRID_DIM_SIZE)
    {
        grid.y = (grid.x + MAX_GRID_DIM_SIZE - 1) / MAX_GRID_DIM_SIZE;
        grid.x = MAX_GRID_DIM_SIZE;
    }
    return grid;
}

template<unsigned N>
void CuckooHash<N>::FreeMemory()
{
	CUDA_CALL( cudaFree(_data) );
    CUDA_CHECK_ERROR("Free memory failed!\n");
	_maxSize  = 0;
	_currentSize = 0;
    _data = NULL;
}

template<unsigned N>
void CuckooHash<N>::Init(const size_t maxSize)
{
	_maxSize = maxSize;

	// free slot has key and value equal 0xFFFFFFFF
	CUDA_CALL( cudaMalloc((void**)&_data, _maxSize * sizeof(int2)) );
	CUDA_CALL( cudaMemset(_data, 0xFF, _maxSize * sizeof(int2)) );

	srand (time(NULL));
	_hashConstants.initRandom();

	CUDA_CHECK_ERROR("Init failed!\n");
}

template dim3 CuckooHash<2>::getGrid(size_t size);
template dim3 CuckooHash<3>::getGrid(size_t size);
template dim3 CuckooHash<4>::getGrid(size_t size);
template dim3 CuckooHash<5>::getGrid(size_t size);

template void CuckooHash<2>::FreeMemory();
template void CuckooHash<3>::FreeMemory();
template void CuckooHash<4>::FreeMemory();
template void CuckooHash<5>::FreeMemory();

template void CuckooHash<2>::Init(const size_t maxSize);
template void CuckooHash<3>::Init(const size_t maxSize);
template void CuckooHash<4>::Init(const size_t maxSize);
template void CuckooHash<5>::Init(const size_t maxSize);
