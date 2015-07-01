/*
 * fast_cuckoo_hash.cuh
 *
 *  Created on: 28-05-2015
 *      Author: Karol Dzitkowski
 */

#ifndef FAST_CUCKOO_HASH_CUH_
#define FAST_CUCKOO_HASH_CUH_

#define PART_HASH_MAP_SIZE 576
#define PIECE_SIZE 192
#define FAST_CUCKOO_HASH_BLOCK_SIZE 512
#define WANTED_BUCKET_CAPACITY 256
#define MAX_STREAM_NO 16

#include "cuckoo_hash.hpp"
#include <stdexcept>

bool fast_cuckooHash(
		const int2* values,
		const int in_size,
		int2* hashMap,
		const int bucket_cnt,
		Constants<2> bucket_constants,
		Constants<3> constants,
		int max_iters);

int2* fast_cuckooRetrieve(
		const int* keys,
		const int size,
		int2* hashMap,
		const int bucket_cnt,
		const Constants<2> bucket_constants,
		const Constants<3> constants);

class FastCuckooHash : public CuckooHash<3>
{
private:
	Constants<2> _bucketConstants;
	int _usedSize;
	int _bucketCnt;

public:
	virtual bool BuildTable(int2* values, size_t size)
	{
		this->_bucketCnt = (size / WANTED_BUCKET_CAPACITY) + 1;
		this->_usedSize = (this->_bucketCnt + 1) * PART_HASH_MAP_SIZE; // PART_HASH_MAP_SIZE <- bucket size
		this->_bucketConstants.initRandom();

		if(this->_usedSize > this->_maxSize)
		{
//			throw std::runtime_error("Hash map max size too small!");
			printf("Reallocate memory...\n");
			CUDA_CALL( cudaFree(this->_data) );
			this->_maxSize = this->_usedSize;
			CUDA_CALL( cudaMalloc((void**)&this->_data, this->_maxSize * sizeof(int2)) );
			CUDA_CALL( cudaMemset(this->_data, 0xFF, _maxSize * sizeof(int2)) );
		}

		int k = 0;
		while(fast_cuckooHash(
				values,
				size,
				this->_data,
				this->_bucketCnt,
				this->_bucketConstants,
				this->_hashConstants,
				MAX_RETRIES))
		{
			printf("AGAIN %d\n", k);
			if(k == this->MAX_RESTARTS) return false;
			CUDA_CALL( cudaMemset(this->_data, 0xFF, this->_maxSize * sizeof(int2)) );
			this->_hashConstants.initRandom();
			this->_bucketConstants.initRandom();
			k++;
		}

		return true;
	}

	virtual int2* GetItems(int* keys, size_t size)
	{
		return fast_cuckooRetrieve(
				keys,
				size,
				this->_data,
				this->_bucketCnt,
				this->_bucketConstants,
				this->_hashConstants);
	}
};

#endif /* FAST_CUCKOO_HASH_CUH_ */
