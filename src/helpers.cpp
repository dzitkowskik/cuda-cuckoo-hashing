#include "helpers.h"
#include <random>
#include <cuda_runtime_api.h>
#include "macros.h"
#include <vector_functions.h>

int2* GenerateRandomKeyValueData(const int N)
{
	auto time = std::chrono::system_clock::now();
	unsigned seed = time.time_since_epoch().count();
	std::minstd_rand0 generator(seed);
	std::uniform_int_distribution<int> distribution(1,20);
	auto h_data = new int2[N];

	int key = 0;
	for(int i=0; i<N; i++)
	{
		key += distribution(generator);
		h_data[i] = make_int2(key, distribution(generator));
	}

	int2* d_data;
	CUDA_CHECK_RETURN( cudaMalloc((void**)&d_data, N*sizeof(int2)) );
	CUDA_CHECK_RETURN( cudaMemcpy(d_data, h_data, N*sizeof(int2), cudaMemcpyHostToDevice) );

	delete [] h_data;
	return d_data;
}

void printData(int2* data, int N, const char* name)
{
	printf("%s\n", name);
	thrust::device_ptr<int2> data_ptr(data);
	for(int i=0; i<N; i++)
	{
		int2 temp = data_ptr[i];
		std::cout << "(" << temp.x << "," << temp.y << ")";
	}
	std::cout << std::endl;
}

void PrintStencil(thrust::device_ptr<int> stencil, int size, const char* name)
{
	std::cout << name << std::endl;
	thrust::copy(stencil, stencil+size, std::ostream_iterator<int>(std::cout, ""));
	std::cout << std::endl;
}

void PrintDeviceVector(thrust::device_vector<int2> data, const char* name)
{
	std::cout << name << std::endl;
    for(size_t i = 0; i < data.size(); i++)
    {
    	int2 value = data[i];
        std::cout << "(" << value.x << "," << value.y << ")";
    }
	std::cout << std::endl;
}

void PrintIntVector(thrust::device_vector<int> data, const char* name)
{
	std::cout << name << std::endl;
    for(size_t i = 0; i < data.size(); i++)
        std::cout << data[i] << " ";
	std::cout << std::endl;
}

bool compareData(int2* a, int2* b, int size)
{
	thrust::device_ptr<int2> a_ptr(a);
	thrust::device_ptr<int2> b_ptr(b);

	for(int i=0; i<size; i++)
	{
		int2 a_value = a_ptr[i];
		int2 b_value = b_ptr[i];
		if(a_value.x != b_value.x || a_value.y != b_value.y)
			return false;
	}
	return true;
}

int* getKeys(int2* key_values, int size)
{
	int* d_data;
	CUDA_CHECK_RETURN( cudaMalloc((void**)&d_data, size*sizeof(int)) );
	thrust::device_ptr<int> d_data_ptr(d_data);
	thrust::device_ptr<int2> key_values_ptr(key_values);
	for(int i=0; i<size; i++)
	{
		int2 key_value = key_values_ptr[i];
		d_data_ptr[i] = key_value.x;
	}
	return d_data;
}
