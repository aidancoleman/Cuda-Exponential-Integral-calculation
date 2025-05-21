/* File: cuda_implement.cu */
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math_constants.h>
#include <vector>
#include <math.h>
#include <stdio.h>
#include "cuda_implement.h"

#define EPSILON 1e-30f
#define EULER_CONSTANT_F 0.5772156649015329f
#define EULER_CONSTANT_D 0.5772156649015329

// CUDA device function for float precision
__device__ float exponentialIntegralDeviceFloat(int n, float x, int maxIter) {
    const float eulerConstant = EULER_CONSTANT_F;
    float epsilon = EPSILON;
    float bigfloat = 1.0e+38f;
    int nm1 = n - 1;
    float a, b, c, d, del, fact, h, psi, ans = 0.0f;

    if (n < 0 || x < 0.0f || (x == 0.0f && (n == 0 || n == 1))) {
        return CUDART_INF_F;
    }

if (n == 0) {
        return expf(-x) / x;
    } else {
        if (x > 1.0f && n > 1) {
            b = x + n;
            c = bigfloat;
            d = 1.0f / b;
            h = d;
            for (int i = 1; i <= maxIter; i++) {
                a = -i * (nm1 + i);
                b += 2.0f;
                float denom = a * d + b;
                if (fabsf(denom) < epsilon) denom = epsilon;
                d = 1.0f / denom;

                if (fabsf(c) < epsilon) c = epsilon;
                c = b + a / c;

                del = c * d;
                h *= del;
                if (fabsf(del - 1.0f) <= epsilon) return h * expf(-x);
            }
            return h * expf(-x);
        } else {
            ans = (nm1 != 0 ? 1.0f / nm1 : -logf(x) - eulerConstant);
            fact = 1.0f;
            for (int i = 1; i <= maxIter; i++) {
                fact *= -x / i;
                if (i != nm1) {
                    del = -fact / (i - nm1);
                } else {
                    psi = -eulerConstant;
                    for (int ii = 1; ii <= nm1; ii++) psi += 1.0f / ii;
                    del = fact * (-logf(x) + psi);
                }
                ans += del;
                if (fabsf(del) < fabsf(ans) * epsilon) return ans;
            }
            return ans;
        }
    }
}

// CUDA device function for double precision
__device__ double exponentialIntegralDeviceDouble(int n, double x, int maxIter) {
    const double eulerConstant = EULER_CONSTANT_D;
    double epsilon = EPSILON;
double bigDouble = 1.0e+300;
    int nm1 = n - 1;
    double a, b, c, d, del, fact, h, psi, ans = 0.0;

    if (n < 0 || x < 0.0 || (x == 0.0 && (n == 0 || n == 1))) {
        return CUDART_INF;
    }

    if (n == 0) {
        return exp(-x) / x;
    } else {
        if (x > 1.0 && n > 1) {
            b = x + n;
            c = bigDouble;
            d = 1.0 / b;
            h = d;
            for (int i = 1; i <= maxIter; i++) {
                a = -i * (nm1 + i);
                b += 2.0;
                double denom = a * d + b;
                if (fabs(denom) < epsilon) denom = epsilon;
                d = 1.0 / denom;

                if (fabs(c) < epsilon) c = epsilon;
                c = b + a / c;

                del = c * d;
                h *= del;
                if (fabs(del - 1.0) <= epsilon) return h * exp(-x);
            }
            return h * exp(-x);
        } else {
            ans = (nm1 != 0 ? 1.0 / nm1 : -log(x) - eulerConstant);
            fact = 1.0;
            for (int i = 1; i <= maxIter; i++) {
                fact *= -x / i;
                if (i != nm1) {
                    del = -fact / (i - nm1);
                } else {
                    psi = -eulerConstant;
                    for (int ii = 1; ii <= nm1; ii++) psi += 1.0 / ii;
                    del = fact * (-log(x) + psi);
                }
                ans += del;
                if (fabs(del) < fabs(ans) * epsilon) return ans;
            }
            return ans;
        }
    }
}

// CUDA kernel for float
__global__ void computeKernelFloat(int n, int m, float a, float b, int maxIter, float* out) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && j < m) {
        float x = a + (j + 0.5f) * ((b - a) / m);
        out[i * m + j] = exponentialIntegralDeviceFloat(i + 1, x, maxIter);
    }
}

// CUDA kernel for double
__global__ void computeKernelDouble(int n, int m, double a, double b, int maxIter, double* out) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && j < m) {
        double x = a + (j + 0.5) * ((b - a) / m);
        out[i * m + j] = exponentialIntegralDeviceDouble(i + 1, x, maxIter);
    }
}

// Host function: runs CUDA kernels and collects results
void computeGpuExponentialIntegrals(unsigned int n, unsigned int m, double a, double b, int maxIter,
                                    std::vector<std::vector<float>> &resultsF,
                                    std::vector<std::vector<double>> &resultsD) {
    cudaEvent_t startF, endF, startD, endD;
    cudaEventCreate(&startF); cudaEventCreate(&endF);
    cudaEventCreate(&startD); cudaEventCreate(&endD);

    float *d_outF;
    double *d_outD;
    size_t sizeF = sizeof(float) * n * m;
    size_t sizeD = sizeof(double) * n * m;

    cudaMalloc(&d_outF, sizeF);
    cudaMalloc(&d_outD, sizeD);

    dim3 block(32, 32);
    dim3 grid((m + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    std::vector<float> tmpF(n * m);
    std::vector<double> tmpD(n * m);

    // Float version
cudaEventRecord(startF);
    computeKernelFloat<<<grid, block>>>(n, m, (float)a, (float)b, maxIter, d_outF);
    cudaDeviceSynchronize();
    cudaEventRecord(endF);
    cudaEventSynchronize(endF);
    cudaMemcpy(tmpF.data(), d_outF, sizeF, cudaMemcpyDeviceToHost);

    // Double version
    cudaEventRecord(startD);
    computeKernelDouble<<<grid, block>>>(n, m, a, b, maxIter, d_outD);
    cudaDeviceSynchronize();
    cudaEventRecord(endD);
    cudaEventSynchronize(endD);
    cudaMemcpy(tmpD.data(), d_outD, sizeD, cudaMemcpyDeviceToHost);

    float timeFms = 0.0f, timeDms = 0.0f;
    cudaEventElapsedTime(&timeFms, startF, endF);
    cudaEventElapsedTime(&timeDms, startD, endD);

    printf("GPU float time (incl. mem): %.6f sec\n", timeFms / 1000.0f);
    printf("GPU double time (incl. mem): %.6f sec\n", timeDms / 1000.0f);

    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int j = 0; j < m; ++j) {
            resultsF[i][j] = tmpF[i * m + j];
            resultsD[i][j] = tmpD[i * m + j];
        }
    }

    cudaEventDestroy(startF); cudaEventDestroy(endF);
    cudaEventDestroy(startD); cudaEventDestroy(endD);
    cudaFree(d_outF);
    cudaFree(d_outD);
}

// debugging
void outputResultsGpu(const std::vector<std::vector<float>> &resultsF,
                      const std::vector<std::vector<double>> &resultsD, double a, double b) {
    double division = (b - a) / (double)(resultsF[0].size());
    for (unsigned int i = 0; i < resultsF.size(); ++i) {
        for (unsigned int j = 0; j < resultsF[i].size(); ++j) {
            double x = a + (j + 0.5) * division;
            printf("GPU==> exponentialIntegralDouble (%u, %.6f)=%.8f , exponentialIntegralFloat (%u, %.6f)=%.8f\n",
                   i + 1, x, resultsD[i][j], i + 1, x, resultsF[i][j]);
        }
    }
}
