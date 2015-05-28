/*
 * fast_cuckoo_hash.cuh
 *
 *  Created on: 28-05-2015
 *      Author: Karol Dzitkowski
 */

#ifndef FAST_CUCKOO_HASH_CUH_
#define FAST_CUCKOO_HASH_CUH_

#define BUCKET_SIZE 512

#include "cuckoo_hash.hpp"

struct Bucket
{
	Constants<3> constants;
	int2* data;
};

class FastCuckooHash : public CuckooHash<3>
{
private:
	Constants<2> _bucketConstants;
	int _bucketCnt;
	Bucket* _buckets;

public:
	virtual bool BuildTable(int2* values, size_t size)
	{
//		int k = 0;
//		int hashSize = this->_maxSize-this->_stashSize;
//		while(!common_cuckooHash(
//				values,
//				size,
//				this->_data,
//				hashSize,
//				this->_hashConstants,
//				this->_stashSize))
//		{
//			if(k == this->MAX_RESTARTS) return false;
//			CUDA_CALL( cudaMemset(this->_data, 0xFF, this->_maxSize * sizeof(int2)) );
//			this->_hashConstants.initRandom();
//			k++;
//		}
		return true;
	}

		virtual int2* GetItems(int* keys, size_t size)
		{
//			int hashSize = this->_maxSize-this->_stashSize;
//			return common_cuckooRetrieve(
//					keys,
//					size,
//					this->_data,
//					hashSize,
//					this->_hashConstants,
//					this->_stashSize);
			return NULL;
		}
};

#endif /* FAST_CUCKOO_HASH_CUH_ */
