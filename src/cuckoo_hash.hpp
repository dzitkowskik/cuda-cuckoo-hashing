/*
 * cuckoo_hash.hpp
 *
 *  Created on: 03-05-2015
 *      Author: Karol Dzitkowski
 */

#ifndef CUCKOO_HASH_H_
#define CUCKOO_HASH_H_

#include <cstdlib>
#include <cuda_runtime_api.h>
#include <constants.h>

using namespace std;

template<unsigned hashFuncCount>
class CuckooHash
{
protected:
	static const unsigned MAX_RESTARTS = 7;
	static const unsigned DEFAULT_BLOCK_SIZE = 64;
	static const unsigned MAX_GRID_DIM_SIZE  = 16384;

	size_t _maxSize;
	size_t _currentSize;
	Constants<hashFuncCount> _hashConstants;
	int2* _data;

public:
	CuckooHash() : _maxSize(0), _currentSize(0), _data(NULL) {}
	virtual ~CuckooHash() { FreeMemory(); }

	void Init(const size_t maxSize);
	void FreeMemory();
	virtual bool BuildTable(int2* values, size_t size) = 0;
	virtual int2* GetItems(int* keys, size_t size) = 0;

	size_t getMaxSize() { return _maxSize; }
	size_t getCurrentSize() { return _currentSize; }

private:
	dim3 getGrid(size_t size);
};

#endif /* CUCKOO_HASH_H_ */