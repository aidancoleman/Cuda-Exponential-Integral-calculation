// File: cuda_implement.h
#ifndef CUDA_IMPLEMENT_H
#define CUDA_IMPLEMENT_H

#include <vector>
/* declaration of host-side -> entry point for CUDA implementation -> Memory allocation and transfers -> Launch CUDA kernels then get results and store them into the passed 2D vectors passed. I added block size argument. Using raw pointers instead of nested vectors now for 1D array flattening to be friendly to GPUs
*/
void computeGpuExponentialIntegrals(
    unsigned int n,
    unsigned int numberOfSamples,
    double a,
    double b,
    int maxIterations,
    float* resultsFloatGpu,
    double* resultsDoubleGpu,
    int blockSize
);
/* print GPU results*/
void outputResultsGpu(const float* resultsFloatGpu, const double* resultsDoubleGpu, unsigned int n, unsigned int numberOfSamples, double a, double b);

#endif // CUDA_IMPLEMENT_H
