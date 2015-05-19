/*
 * cuckoo_hash.cpp
 *
 *  Created on: 03-05-2015
 *      Author: Karol Dzitkowski
 */

#include "cuckoo_hash.h"
#include "macros.h"
#include <stdlib.h>
#include <time.h>

dim3 CuckooHash::getGrid(size_t size)
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

void CuckooHash::FreeMemory()
{
	CUDA_CALL( cudaFree(_data) );
    CUDA_CHECK_ERROR("Free memory failed!\n");

	_maxSize  = 0;
	_currentSize = 0;
    _data = NULL;
}

void CuckooHash::Init(const size_t maxSize)
{
	_maxSize = maxSize;

	// free slot has key and value equal 0xFFFFFFFF
	CUDA_CALL( cudaMalloc((void**)&_data, _maxSize * sizeof(int2)) );
	CUDA_CALL( cudaMemset(_data, 0xFF, _maxSize * sizeof(int2)) );

	srand (time(NULL));
	_hashConstants[0] = rand();
	_hashConstants[1] = rand();

	CUDA_CHECK_ERROR("Init failed!\n");
}

void CuckooHash::BuildTable(int2* values, size_t size)
{
	naive_cuckooHash(values, size, _data, _maxSize, _hashConstants);
}

int2* CuckooHash::GetItems(int* keys, size_t size)
{
	return naive_cuckooRetrieve(keys, size, _data, _maxSize, _hashConstants);
}

