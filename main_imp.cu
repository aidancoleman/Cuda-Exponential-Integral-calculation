#include <iostream>
#include <vector>
#include <sys/time.h>
#include <cmath>
#include <cstdlib>
#include <unistd.h>
#include "cuda_implement.h"

extern float exponentialIntegralFloat(int n, float x);
extern double exponentialIntegralDouble(int n, double x);

bool verbose = false, timing = false, cpu = true, gpu = true, error = false, csv = false;
int maxIterations = 2000000000;
unsigned int n = 10, numberOfSamples = 10;
double a = 0.0, b = 10.0;
int blockSize = 32;

int parseArguments(int argc, char *argv[]);
void printUsage();

template <typename T>
T max_diff(const std::vector<std::vector<T>> &cpu, const T* gpu) {
    T max_diff = 0;
    for (size_t i = 0; i < cpu.size(); ++i) {
        for (size_t j = 0; j < cpu[i].size(); ++j) {
            T diff = fabs(cpu[i][j] - gpu[i * cpu[0].size() + j]);
            if (diff > max_diff) max_diff = diff;
        }
    }
    return max_diff;
}

int main(int argc, char *argv[]) {
    parseArguments(argc, argv);

    // Declare CPU results
    std::vector<std::vector<float>> resultsFloatCpu(n, std::vector<float>(numberOfSamples));
    std::vector<std::vector<double>> resultsDoubleCpu(n, std::vector<double>(numberOfSamples));

    // Flat GPU results
    float* resultsFloatGpu = new float[n * numberOfSamples];
    double* resultsDoubleGpu = new double[n * numberOfSamples];

    struct timeval start, end;
    double cpuTime = 0.0, gpuTime = 0.0;
    //cpu and gpu branches
    if (cpu) {
        gettimeofday(&start, NULL);
        double dx = (b - a) / numberOfSamples;
        for (unsigned int i = 0; i < n; ++i) {
            for (unsigned int j = 0; j < numberOfSamples; ++j) {
                double x = a + (j + 0.5) * dx;
                resultsFloatCpu[i][j] = exponentialIntegralFloat(i + 1, static_cast<float>(x));
                resultsDoubleCpu[i][j] = exponentialIntegralDouble(i + 1, x);
            }
        }
        gettimeofday(&end, NULL);
        cpuTime = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) * 1e-6;
    }

    if (gpu) {
        gettimeofday(&start, NULL);
        computeGpuExponentialIntegrals(n, numberOfSamples, a, b, maxIterations,
                                       resultsFloatGpu, resultsDoubleGpu, blockSize);
        gettimeofday(&end, NULL);
        gpuTime = (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) * 1e-6;
    }
    /* Print GPU results  */
    if (verbose && gpu) {
    outputResultsGpu(resultsFloatGpu, resultsDoubleGpu, n, numberOfSamples, a, b);
}

    if (timing) {
        if (cpu) std::cout << "CPU time: " << cpuTime << " sec\n";
        if (gpu) std::cout << "GPU time: " << gpuTime << " sec\n";
        if (cpu && gpu) std::cout << "Speedup: " << cpuTime / gpuTime << "x\n";
    }

    if (error && cpu && gpu){
    double threshold = 1e-5;
    for (unsigned int i = 0; i < n; ++i) {
        for (unsigned int j = 0; j < numberOfSamples; ++j) {
            double cpuF = resultsFloatCpu[i][j];
            double gpuF = resultsFloatGpu[i * numberOfSamples + j];
            double cpuD = resultsDoubleCpu[i][j];
            double gpuD = resultsDoubleGpu[i * numberOfSamples + j];

            if (fabs(cpuF - gpuF) > threshold || fabs(cpuD - gpuD) > threshold) {
                double x = a + (j + 0.5) * ((b - a) / numberOfSamples);
                std::cout << "Mismatch at (" << i << "," << j << ") x=" << x
                          << " | CPU float=" << cpuF << ", GPU float=" << gpuF
                          << " | CPU double=" << cpuD << ", GPU double=" << gpuD
                          << std::endl;
            }
        }
    }
}
    //for storing results in csv
    if (csv && cpu && gpu) {
    std::cout << n << "," << numberOfSamples << ","
              << cpuTime << "," << gpuTime << ","
              << max_diff(resultsFloatCpu, resultsFloatGpu) << ","
              << max_diff(resultsDoubleCpu, resultsDoubleGpu) << ","
              << cpuTime / gpuTime << std::endl;
}

    delete[] resultsFloatGpu;
    delete[] resultsDoubleGpu;
    return 0;
}
//parse the cmd line for flags
int parseArguments(int argc, char *argv[]) {
    int opt;
    while ((opt = getopt(argc, argv, "cghn:m:a:b:i:tvB:er")) != -1) {
        switch (opt) {
            case 'c': cpu = false; break;
            case 'e': error = true; break;
            case 'g': gpu = false; break;
            case 'h': printUsage(); exit(0); break;
            case 'n': n = atoi(optarg); break;
            case 'm': numberOfSamples = atoi(optarg); break;
            case 'a': a = atof(optarg); break;
            case 'b': b = atof(optarg); break;
            case 'i': maxIterations = atoi(optarg); break;
            case 'r': csv = true; break;
            case 't': timing = true; break;
            case 'v': verbose = true; break;
            case 'B': blockSize = atoi(optarg); break;
            default: printUsage(); exit(1);
        }
    }
    return 0;
}

void printUsage() {
    std::cout << "Usage: ./expIntegral.out [options]\n"
              << "  -a <val>  Set interval start (default 0.0)\n"
              << "  -b <val>  Set interval end (default 10.0)\n"
              << "  -c        Skip CPU test\n"
              << "  -g        Skip GPU test\n"
              << "  -h        Show help\n"
              << "  -i <val>  Set max iterations (default 2e9)\n"
              << "  -n <val>  Set max order (default 10)\n"
              << "  -m <val>  Set number of samples (default 10)\n"
              << "  -t        Show timing\n"
              << "  -v        Verbose output\n"
              << "  -B <val>  Set CUDA block size (default 32)\n";
}
