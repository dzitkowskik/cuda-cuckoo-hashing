/*
 * cuckoo_hash.cpp
 *
 *  Created on: 03-05-2015
 *      Author: Karol Dzitkowski
 */

#include "cuckoo_hash.h"
#include "macros.h"

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
    CUDA_CALL( cudaFree(_hashConstants) );
    CUDA_CHECK_ERROR("Free memory failed!\n");
	_maxSize  = 0;
	_currentSize = 0;
    _data = NULL;
    _hashConstants = NULL;
}

void CuckooHash::Init(const size_t maxSize, const unsigned short hFuncNum)
{
	_maxSize = maxSize;
	_hFuncNum = hFuncNum;
	CUDA_CALL( cudaMalloc((void**)&_data, _maxSize*sizeof(int2)) );
	CUDA_CALL( cudaMalloc((void**)&_hashConstants, _hFuncNum*sizeof(int)) );
	CUDA_CHECK_ERROR("Init failed!\n");
}

void CuckooHash::BuildTable(int2* values, size_t size)
{

}

int2* CuckooHash::GetItems(int* keys, size_t size)
{
	return NULL;
}

