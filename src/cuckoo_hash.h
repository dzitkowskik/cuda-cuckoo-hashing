/*
 * cuckoo_hash.h
 *
 *  Created on: 03-05-2015
 *      Author: Karol Dzitkowski
 */

#ifndef CUCKOO_HASH_H_
#define CUCKOO_HASH_H_

#include <cstdlib>
#include <cuda_runtime_api.h>

using namespace std;

class CuckooHash
{
protected:
	static const unsigned MAX_RESTARTS = 7;
	static const unsigned DEFAULT_BLOCK_SIZE = 64;
	static const unsigned MAX_GRID_DIM_SIZE  = 16384;

private:
	size_t _maxSize;
	size_t _currentSize;
	unsigned short _hFuncNum;
	int* _hashConstants;
	int2* _data;

public:
	CuckooHash()
		: _maxSize(0), _currentSize(0), _hFuncNum(0), _data(NULL), _hashConstants(NULL)
	{}
	virtual ~CuckooHash() {FreeMemory();}

	virtual void Init(const size_t maxSize, const unsigned short hFuncNum = 2);
	virtual void FreeMemory();
	virtual void BuildTable(int2* values, size_t size);
	virtual int2* GetItems(int* keys, size_t size);

	// GETTERS
	dim3 getGrid(size_t size);
	size_t getMaxSize() { return _maxSize; }
	size_t getCurrentSize() { return _currentSize; }
	int2* getData() { return _data; }
	unsigned getIterationCount() { return MAX_RESTARTS; }
	unsigned short getHashFunctionsNumber() { return _hFuncNum; }
};

#endif /* CUCKOO_HASH_H_ */
