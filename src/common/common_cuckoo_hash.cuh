/*
 * common_cuckoo_hash.cuh
 *
 *  Created on: 21-05-2015
 *      Author: Karol Dzitkowski
 */

#ifndef COMMON_CUCKOO_HASH_HPP_
#define COMMON_CUCKOO_HASH_HPP_

#include "cuckoo_hash.hpp"

template<unsigned N>
bool common_cuckooHash(
		int2* values,
		int in_size,
		int2* hashMap,
		int hashMap_size,
		Constants<N> constants,
		int stash_size);

template<unsigned N>
int2* common_cuckooRetrieve(
		int* keys,
		int size,
		int2* hashMap,
		int hashMap_size,
		Constants<N> constants,
		int stash_size);

template<unsigned N>
class CommonCuckooHash : public CuckooHash<N>
{
private:
	static const unsigned DEFAULT_STASH_SIZE  = 16384;
	size_t _stashSize;
	size_t _hashSize;
public:
	CommonCuckooHash() : CuckooHash<N>()
	{
		_stashSize = DEFAULT_STASH_SIZE;
		_hashSize = this->_maxSize - _stashSize;
	}
	virtual ~CommonCuckooHash() {}
	virtual bool BuildTable(int2* values, size_t size)
	{
		int k = 0;

		while(!common_cuckooHash(
				values,
				size,
				this->_data,
				this->_hashSize,
				this->_hashConstants,
				this->_stashSize))
		{
			if(k == this->MAX_RESTARTS) return false;
			CUDA_CALL( cudaMemset(this->_data, 0xFF, this->_maxSize * sizeof(int2)) );
			this->_hashConstants.initRandom();
			k++;
		}
		return true;
	}

	virtual int2* GetItems(int* keys, size_t size)
	{
		return common_cuckooRetrieve(
				keys,
				size,
				this->_data,
				this->_hashSize,
				this->_hashConstants,
				this->_stashSize);
	}
};

#endif /* COMMON_CUCKOO_HASH_HPP_ */
