/*
 * naive_cuckoo_hash.hpp
 *
 *  Created on: 21-05-2015
 *      Author: Karol Dzitkowski
 */

#ifndef NAIVE_CUCKOO_HASH_HPP_
#define NAIVE_CUCKOO_HASH_HPP_

#include "cuckoo_hash.hpp"

template<unsigned N>
bool naive_cuckooHash(
		int2* values,
		int in_size,
		int2* hashMap,
		int hashMap_size,
		Constants<N> constants);

template<unsigned N>
int2* naive_cuckooRetrieve(
		int* keys,
		int size,
		int2* hashMap,
		int hashMap_size,
		Constants<N> constants);

template<unsigned N>
class NaiveCuckooHash : public CuckooHash<N>
{
public:
	virtual ~NaiveCuckooHash(){}
	virtual bool BuildTable(int2* values, size_t size)
	{
		int k = 0;

		while(!naive_cuckooHash(values, size, this->_data, this->_maxSize, this->_hashConstants))
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
		return naive_cuckooRetrieve(keys, size, this->_data, this->_maxSize, this->_hashConstants);
	}
};

#endif /* NAIVE_CUCKOO_HASH_HPP_ */
