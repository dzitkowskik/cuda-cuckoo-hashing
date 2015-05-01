/*
 * gpu_cuckoo.cuh
 *
 *  Created on: 01-05-2015
 *      Author: Karol Dzitkowski
 */

#ifndef GPU_CUCKOO_CUH_
#define GPU_CUCKOO_CUH_

#include <cuda_runtime_api.h>

#define CUCKOO_HASHING_BLOCK_SIZE 64
#define NUM_HASHES 2
#define SLOTS_COEF 10

int2** cuckooHash(int2* values, int in_size, int2& out_size, int2& out_seeds);

#endif /* GPU_CUCKOO_CUH_ */
