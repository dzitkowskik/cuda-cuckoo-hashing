/*
 * cuckoo_hash.h
 *
 *  Created on: 03-05-2015
 *      Author: Karol Dzitkowski
 */

#ifndef CUCKOO_HASH_H_
#define CUCKOO_HASH_H_

class CuckooHash
{
private:
	size_t _maxSize;
	size_t _currentSize;
	unsigned short _hFuncNum;
	int2** _data;

public:
	CuckooHash();
	virtual ~CuckooHash() {FreeMemory();}

	virtual void Init(const size_t max_size, const unsigned short hFuncNum = 2);
	virtual void FreeMemory();

	virtual void BuildTable(int2* values, size_t size);
	virtual int2* GetItems(int* keys, size_t size);

	// GETTERS
	size_t getMaxSize();
	size_t getCurrentSize();
	int2** getData();
	unsigned short getHashFunctionsNumber();
};

#endif /* CUCKOO_HASH_H_ */
