#ifndef CUDA_IMPLEMENT_H
#define CUDA_IMPLEMENT_H

#include <vector>
/* declaration of host-side -> entry point for CUDA implementation -> Memory allocation and transfers -> Launch CUDA kernels then get results and store them into the passed 2D vectors passed
*/
void computeGpuExponentialIntegrals(
    unsigned int n,
    unsigned int numberOfSamples,
    double a,
    double b,
    int maxIterations,
    std::vector<std::vector<float>> &resultsFloatGpu,
    std::vector<std::vector<double>> &resultsDoubleGpu
);
/* print GPU results*/
void outputResultsGpu(
    const std::vector<std::vector<float>> &resultsFloatGpu,
    const std::vector<std::vector<double>> &resultsDoubleGpu,
    double a,
    double b
);

#endif // CUDA_IMPLEMENT_H
