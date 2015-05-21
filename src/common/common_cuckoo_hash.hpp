/*
 * common_cuckoo_hash.hpp
 *
 *  Created on: 21-05-2015
 *      Author: Karol Dzitkowski
 */

#ifndef COMMON_CUCKOO_HASH_HPP_
#define COMMON_CUCKOO_HASH_HPP_

#include "cuckoo_hash.hpp"

template<unsigned N>
class CommonCuckooHash : public CuckooHash<N>
{
public:
	virtual ~CommonCuckooHash(){}
	virtual bool BuildTable(int2* values, size_t size)
	{
		return false;
	}

	virtual int2* GetItems(int* keys, size_t size)
	{
		return NULL;
	}
};

#endif /* COMMON_CUCKOO_HASH_HPP_ */
