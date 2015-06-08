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
	int _stashSize;
public:
	CommonCuckooHash() : CuckooHash<N>()
	{
		this->_stashSize = DEFAULT_STASH_SIZE;
	}
	virtual ~CommonCuckooHash() {}
	virtual bool BuildTable(int2* values, size_t size)
	{
		int k = 0;
		int hashSize = this->_maxSize-this->_stashSize;
		while(common_cuckooHash(
				values,
				size,
				this->_data,
				hashSize,
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
		int hashSize = this->_maxSize-this->_stashSize;
		return common_cuckooRetrieve(
				keys,
				size,
				this->_data,
				hashSize,
				this->_hashConstants,
				this->_stashSize);
	}
};

#endif /* COMMON_CUCKOO_HASH_HPP_ */
