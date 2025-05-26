
/* File: cuda_implement.cu */
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math_constants.h>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <limits>
#include "cuda_implement.h"
/* macro to wrap CUDA runtime API call to check if it succeeds => prints an error message and exits the program if it fails*/
#define CHECK_CUDA(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (code %d), line %d\n", cudaGetErrorString(err), err, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Templated exponential integral computation (device + host)
template <typename real>
__host__ __device__ real exponentialIntegral(int n, real x, int maxIter) {
    const real eulerConstant = static_cast<real>(0.5772156649015329);
    real epsilon = static_cast<real>(1e-30);
    real bigReal = std::numeric_limits<real>::max();
    int nm1 = n - 1;
    real a, b, c, d, del, fact, h, psi, ans = 0.0;

    if (n < 0 || x < 0.0 || (x == 0.0 && (n == 0 || n == 1))) {
        return INFINITY;
    }

    if (n == 0) {
        return exp(-x) / x;
    }

    if (x > static_cast<real>(1.0) && n > 1) {
        b = x + n;
        c = bigReal;
        d = 1.0 / b;
        h = d;
        for (int i = 1; i <= maxIter; i++) {
            a = -i * (nm1 + i);
            b += 2.0;
            d = 1.0 / (a * d + b);
            c = b + a / c;
            del = c * d;
            h *= del;
            if (fabs(del - 1.0) <= epsilon) {
                return h * exp(-x);
            }
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
//templated version of kernel
template <typename real>
__global__ void exponentialIntegral_grid_GPU(real* results, int n, double a, double b, int maxIterations, int numberOfSamples) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < n && idy < numberOfSamples) {
        real x = static_cast<real>(a + (idy + 0.5) * ((b - a) / numberOfSamples));
        results[idx * numberOfSamples + idy] = exponentialIntegral<real>(idx + 1, x, maxIterations);
    }
}
//Now 1D memory for GPU access-friendliness
template <typename real>
void computeGpuExponentialIntegrals_Generic(unsigned int n, unsigned int m, double a, double b, int maxIter,
    real* results, int blockSize) {
    real* d_out;
    size_t size = sizeof(real) * n * m;
    CHECK_CUDA(cudaMalloc(&d_out, size));

    dim3 block(blockSize, blockSize);
    dim3 grid((m + block.x - 1) / block.x, (n + block.y - 1) / block.y);

    exponentialIntegral_grid_GPU<real><<<grid, block>>>(d_out, n, a, b, maxIter, m);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(results, d_out, size, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(d_out));
    }

    void computeGpuExponentialIntegrals(unsigned int n, unsigned int m, double a, double b, int maxIter,
    float* resultsF, double* resultsD, int blockSize) {
    cudaEvent_t startF, endF, startD, endD;
    CHECK_CUDA(cudaEventCreate(&startF));
    CHECK_CUDA(cudaEventCreate(&endF));
    CHECK_CUDA(cudaEventCreate(&startD));
    CHECK_CUDA(cudaEventCreate(&endD));

    CHECK_CUDA(cudaEventRecord(startF));
    computeGpuExponentialIntegrals_Generic<float>(n, m, a, b, maxIter, resultsF, blockSize);
    CHECK_CUDA(cudaEventRecord(endF));
    CHECK_CUDA(cudaEventSynchronize(endF));

    CHECK_CUDA(cudaEventRecord(startD));
    computeGpuExponentialIntegrals_Generic<double>(n, m, a, b, maxIter, resultsD, blockSize);
    CHECK_CUDA(cudaEventRecord(endD));
    CHECK_CUDA(cudaEventSynchronize(endD));

    float timeFms = 0.0f, timeDms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&timeFms, startF, endF));
    CHECK_CUDA(cudaEventElapsedTime(&timeDms, startD, endD));

    printf("GPU float time (incl. mem): %.6f sec\n", timeFms / 1000.0f);
    printf("GPU double time (incl. mem): %.6f sec\n", timeDms / 1000.0f);

    CHECK_CUDA(cudaEventDestroy(startF));
    CHECK_CUDA(cudaEventDestroy(endF));
    CHECK_CUDA(cudaEventDestroy(startD));
    CHECK_CUDA(cudaEventDestroy(endD));
}

void outputResultsGpu(const float* resultsF, const double* resultsD, unsigned int n, unsigned int m, double a, double b) {
    double dx = (b - a) / m;
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int j = 0; j < m; ++j) {
            double x = a + (j + 0.5) * dx;
            printf("GPU==> exponentialIntegralDouble (%u, %.6f)=%.8f , exponentialIntegralFloat (%u, %.6f)=%.8f\n",
                   i + 1, x, resultsD[i * m + j], i + 1, x, resultsF[i * m + j]);
        }
    }
}
