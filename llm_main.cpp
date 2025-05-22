#include <time.h>
#include <iostream>
#include <limits>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <sys/time.h>
#include <unistd.h>
#include "cuda_implement.h" //for our cuda implementation

using namespace std;

float exponentialIntegralFloat(const int n,const float x);
double exponentialIntegralDouble(const int n,const double x);
/*Assign the exponential integral computation to the GPU using CUDA.*/
void outputResultsCpu(const vector<vector<float>> &resultsFloatCpu, const vector<vector<double>> &resultsDoubleCpu);
void outputResultsGpu(const vector<vector<float>> &resultsFloatGpu, const vector<vector<double>> &resultsDoubleGpu, double a, double b); //verbose mode (-v) prints GPU-computed results, mirroring outputResultsCpu, for debugging and validation!
int parseArguments(int argc, char **argv);
void printUsage(void);

bool verbose, timing, cpu = true, gpu = true; //GPU code runs by default unless user disables it using -g on cmd line

int maxIterations;
unsigned int n, numberOfSamples;
double a, b;

/* I introduce template to print mismatches exceeding 0.00001 => validate numerical accuracy of GPU results versus CPU results, divergence is reported in stdout
*/
template<typename T>
bool compareResults(const vector<vector<T>> &cpu, const vector<vector<T>> &gpu, T threshold) {
    bool valid = true;
    for (unsigned int i = 0; i < cpu.size(); ++i) {
        for (unsigned int j = 0; j < cpu[i].size(); ++j) {
            if (fabs(cpu[i][j] - gpu[i][j]) > threshold) {
                printf("Mismatch at (%u,%u): CPU=%.8f, GPU=%.8f\n", i, j, (double)cpu[i][j], (double)gpu[i][j]);
                valid = false;
            }
        }
    }
    return valid;
}

int main(int argc, char *argv[]) {
    parseArguments(argc, argv);

    if(verbose) {
        printf("n=%u, m=%u, a=%f, b=%f\n", n, numberOfSamples, a, b);
    }

    // Vectors to store results
    vector<vector<float>> resultsFloatCpu(n, vector<float>(numberOfSamples));
    vector<vector<double>> resultsDoubleCpu(n, vector<double>(numberOfSamples));
    vector<vector<float>> resultsFloatGpu(n, vector<float>(numberOfSamples));
    vector<vector<double>> resultsDoubleGpu(n, vector<double>(numberOfSamples));

    // Timing variables
    double cpuStartTime, cpuEndTime, gpuStartTime, gpuEndTime;

    // CPU computation
    if (cpu) {
        struct timeval start, end;
        gettimeofday(&start, NULL);

        double delta = (b - a) / (double)numberOfSamples;
        for (unsigned int i = 1; i <= n; ++i) {
            for (unsigned int j = 0; j < numberOfSamples; ++j) {
                double x = a + (j + 0.5) * delta;
                if (x <= 0.0) x = 1e-7;  // Prevent underflow
                resultsFloatCpu[i - 1][j] = exponentialIntegralFloat(i, (float)x);
                resultsDoubleCpu[i - 1][j] = exponentialIntegralDouble(i, x);
            }
        }

        gettimeofday(&end, NULL);
        cpuStartTime = start.tv_sec + start.tv_usec * 1e-6;
        cpuEndTime = end.tv_sec + end.tv_usec * 1e-6;
    }

    // GPU computation
    if (gpu) {
        struct timeval start, end;
        gettimeofday(&start, NULL);

        computeGpuExponentialIntegrals(n, numberOfSamples, a, b, maxIterations, resultsFloatGpu, resultsDoubleGpu);

        gettimeofday(&end, NULL);
        gpuStartTime = start.tv_sec + start.tv_usec * 1e-6;
        gpuEndTime = end.tv_sec + end.tv_usec * 1e-6;
    }

    // Output timing results
    if (timing) {
        if (cpu) printf("CPU total time: %.6f sec\n", cpuEndTime - cpuStartTime);
        if (gpu) printf("GPU total time: %.6f sec\n", gpuEndTime - gpuStartTime);
        if (cpu && gpu) printf("Speedup: %.2fx\n", (cpuEndTime - cpuStartTime) / (gpuEndTime - gpuStartTime));
    }

    // Output results
    if (verbose) {
        if (cpu) outputResultsCpu(resultsFloatCpu, resultsDoubleCpu);
        if (gpu) outputResultsGpu(resultsFloatGpu, resultsDoubleGpu, a, b);
    }

    // Compare results
    if (cpu && gpu) {
        bool floatMatch = compareResults(resultsFloatCpu, resultsFloatGpu, 1e-5f);
        bool doubleMatch = compareResults(resultsDoubleCpu, resultsDoubleGpu, 1e-5);
        if (!floatMatch || !doubleMatch) {
            printf("Result mismatch detected!\n");
        } else {
            printf("Results match within threshold.\n");
        }
    }

    return 0;
}

