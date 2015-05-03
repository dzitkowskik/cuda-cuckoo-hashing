#include <stdio.h>

#define CUDA_CHECK_RETURN(value) {                               \
    cudaError_t _m_cudaStat = value;                             \
    if (_m_cudaStat != cudaSuccess) {                            \
        fprintf(stderr, "Error %s at line %d in file %s\n",      \
            cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);\
        exit(1);                                                 \
    } }

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__); }} while(0)

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d:error=%d\n",__FILE__,__LINE__,x);}} while(0)

#define CUDA_CHECK_ERROR(errorMessage) {                                     \
    cudaError_t err = cudaGetLastError();                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);}}
