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

#define HASH_FUNC_NO 2

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
	int _hashConstants[HASH_FUNC_NO];
	int2* _data;

public:
	CuckooHash()
		: _maxSize(0), _currentSize(0), _data(NULL)
	{}
	virtual ~CuckooHash() {FreeMemory();}

	virtual void Init(const size_t maxSize);
	virtual void FreeMemory();
	virtual void BuildTable(int2* values, size_t size);
	virtual int2* GetItems(int* keys, size_t size);

	// GETTERS

	dim3 getGrid(size_t size);
	size_t getMaxSize() { return _maxSize; }
	size_t getCurrentSize() { return _currentSize; }
	int2* getData() { return _data; }
	unsigned getIterationCount() { return MAX_RESTARTS; }

private:
	int genSeed();
};

void naive_cuckooHash(
		int2* values,
		int in_size,
		int2* hashMap,
		int hashMap_size,
		int seeds[HASH_FUNC_NO]);

int2* naive_cuckooRetrieve(
		int* keys,
		int size,
		int2* hashMap,
		int hashMap_size,
		int seeds[HASH_FUNC_NO]);

#endif /* CUCKOO_HASH_H_ */
