#include <iostream>
#include <cuda_runtime_api.h>

int main(int argc, char** argv)
{
	std::cout << sizeof(unsigned long long int) << std::endl;
	std::cout << sizeof(int2) << std::endl;

	return 0;
}
