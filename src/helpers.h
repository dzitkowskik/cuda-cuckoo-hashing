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

typedef std::chrono::high_resolution_clock myclock;

int2* GenerateRandomKeyValueData(const int N);
void printData(int2* data, int N, const char* name);
bool compareData(int2* a, int2* b, int size);
int* getKeys(int2* key_values, int size);

#endif /* HELPERS_H_ */
