/*
 * helpers.h
 *
 *  Created on: 19-05-2015
 *      Author: Karol Dzitkowski
 */

#ifndef HELPERS_H_
#define HELPERS_H_

#include <chrono>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>

typedef std::chrono::high_resolution_clock myclock;

int2* GenerateRandomKeyValueData(const int N);
void printData(int2* data, int N, const char* name);
bool compareData(int2* a, int2* b, int size);
int* getKeys(int2* key_values, int size);
void PrintStencil(thrust::device_ptr<int> stencil, int size, const char* name);
void PrintDeviceVector(thrust::device_vector<int2> data, const char* name);
void PrintIntVector(thrust::device_vector<int> data, const char* name);

#endif /* HELPERS_H_ */
