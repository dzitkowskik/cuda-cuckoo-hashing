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
#include "macros.h"

// DEFINES
#define CUCKOO_HASHING_BLOCK_SIZE 64	// BLOCK SIZE USED BY CUCKOO HASHING KERNELS
#define EMPTY_BUCKET_KEY 0xFFFFFFFF		// KEY USED FOR EMPTY BUCKETS
#define EMPTY_BUCKET 0xFFFFFFFFFFFFFFFF	// KEY USED FOR EMPTY BUCKETS
#define MAX_RETRIES 10					// HOW MANY TIMES WE SHOULD CHECK ALL BUCKETS
#define MAX_HASH_FUNC_NO 5				// MAX AMOUNT OF HASH FUNCTIONS

using namespace std;

template<unsigned hashFuncCount>
class CuckooHash
{
public:
	static const unsigned MAX_RESTARTS = 7;
	static const unsigned DEFAULT_BLOCK_SIZE = CUCKOO_HASHING_BLOCK_SIZE;
	static const unsigned MAX_GRID_DIM_SIZE  = 16384;

protected:
	int _maxSize;
	int _currentSize;
	Constants<hashFuncCount> _hashConstants;
	int2* _data;

public:
	CuckooHash() : _maxSize(0), _currentSize(0), _data(NULL) {}
	virtual ~CuckooHash() { FreeMemory(); }

	void Init(const size_t maxSize);
	void FreeMemory();
	virtual bool BuildTable(int2* values, size_t size) = 0;
	virtual int2* GetItems(int* keys, size_t size) = 0;

	int getMaxSize() { return _maxSize; }
	int getCurrentSize() { return _currentSize; }

	static dim3 GetGrid(size_t size);
};

#endif /* CUCKOO_HASH_H_ */
