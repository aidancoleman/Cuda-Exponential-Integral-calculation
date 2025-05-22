#include <iostream>
#include <cmath>
#include <vector>
#include <limits>
#include <stdlib.h>
#include <unistd.h>

using namespace std;

extern bool verbose, timing, cpu;
extern int maxIterations;
extern unsigned int n, numberOfSamples;
extern double a, b;
extern bool gpu;   //  gpu flag
extern void printUsage();

float exponentialIntegralFloat(const int n, const float x) {
    static const float eulerConstant = 0.5772156649015329f;
    float epsilon = 1.E-30f;
    float bigfloat = std::numeric_limits<float>::max();
    int i, ii, nm1 = n - 1;
    float a, b, c, d, del, fact, h, psi, ans = 0.0f;

    if (n < 0 || x < 0.0f || (x == 0.0f && (n == 0 || n == 1))) {
        printf("Bad args in exponentialIntegralFloat: n=%d, x=%.12f\n", n, x);
        exit(1);
    }
    if (n == 0) {
        return expf(-x) / x;
    } else {
        if (x > 1.0f) {
            b = x + n;
            c = bigfloat;
            d = 1.0f / b;
            h = d;
            for (i = 1; i <= maxIterations; i++) {
                a = -i * (nm1 + i);
                b += 2.0f;
                d = 1.0f / (a * d + b);
                c = b + a / c;
                del = c * d;
                h *= del;
                if (fabsf(del - 1.0f) <= epsilon) return h * expf(-x);
            }
            return h * expf(-x);
        } else {
            ans = (nm1 != 0 ? 1.0f / nm1 : -logf(x) - eulerConstant);
            fact = 1.0f;
            for (i = 1; i <= maxIterations; i++) {
                fact *= -x / i;
                if (i != nm1) {
                    del = -fact / (i - nm1);
                } else {
                    psi = -eulerConstant;
                    for (ii = 1; ii <= nm1; ii++) psi += 1.0f / ii;
                    del = fact * (-logf(x) + psi);
                }
                ans += del;
                if (fabsf(del) < fabsf(ans) * epsilon) return ans;
            }
            return ans;
        }
    }
}

double exponentialIntegralDouble(const int n, const double x) {
    static const double eulerConstant = 0.5772156649015329;
    double epsilon = 1.E-30;
    double bigDouble = std::numeric_limits<double>::max();
    int i, ii, nm1 = n - 1;
    double a, b, c, d, del, fact, h, psi, ans = 0.0;

    if (n < 0 || x < 0.0 || (x == 0.0 && (n == 0 || n == 1))) {
        cout << "Bad arguments were passed to the exponentialIntegralDouble function call" << endl;
        exit(1);
    }
    if (n == 0) {
        return exp(-x) / x;
    } else {
        if (x > 1.0) {
            b = x + n;
            c = bigDouble;
            d = 1.0 / b;
            h = d;
            for (i = 1; i <= maxIterations; i++) {
                a = -i * (nm1 + i);
                b += 2.0;
                d = 1.0 / (a * d + b);
                c = b + a / c;
                del = c * d;
                h *= del;
                if (fabs(del - 1.0) <= epsilon) return h * exp(-x);
            }
            return h * exp(-x);
        } else {
            ans = (nm1 != 0 ? 1.0 / nm1 : -log(x) - eulerConstant);
            fact = 1.0;
            for (i = 1; i <= maxIterations; i++) {
                fact *= -x / i;
                if (i != nm1) {
                    del = -fact / (i - nm1);
                } else {
                    psi = -eulerConstant;
                    for (ii = 1; ii <= nm1; ii++) psi += 1.0 / ii;
                    del = fact * (-log(x) + psi);
                }
                ans += del;
                if (fabs(del) < fabs(ans) * epsilon) return ans;
            }
            return ans;
        }
    }
}

void outputResultsCpu(const std::vector<std::vector<float>> &resultsFloatCpu,
                      const std::vector<std::vector<double>> &resultsDoubleCpu) {
    double x, division = (b - a) / ((double)(numberOfSamples));
    for (unsigned int ui = 1; ui <= n; ui++) {
        for (unsigned int uj = 1; uj <= numberOfSamples; uj++) {
            x = a + (uj - 1 + 0.5) * division; //midpoint sampling
            std::cout << "CPU==> exponentialIntegralDouble (" << ui << "," << x << ")="
                      << resultsDoubleCpu[ui - 1][uj - 1] << " ,";
            std::cout << "exponentialIntegralFloat  (" << ui << "," << x << ")="
                      << resultsFloatCpu[ui - 1][uj - 1] << std::endl;
        }
    }
}

int parseArguments(int argc, char *argv[]) {
    int c;
    while ((c = getopt(argc, argv, "cghn:m:a:b:tv")) != -1) {
        switch (c) {
            case 'c': cpu = false; break;
            case 'g': gpu = false; break;
            case 'h': printUsage(); exit(0); break;
            case 'i': maxIterations = atoi(optarg); break;
            case 'n': n = atoi(optarg); break;
            case 'm': numberOfSamples = atoi(optarg); break;
            case 'a': a = atof(optarg); break;
            case 'b': b = atof(optarg); break;
            case 't': timing = true; break;
            case 'v': verbose = true; break;
            default:
                fprintf(stderr, "Invalid option given\n");
                printUsage();
                return -1;
        }
    }
    return 0;
}

void printUsage() {
    printf("exponentialIntegral program\n");
    printf("by: Jose Mauricio Refojo <refojoj@tcd.ie>\n");
    printf("This program will calculate a number of exponential integrals\n");
    printf("usage:\n");
    printf("exponentialIntegral.out [options]\n");
    printf("  -a value : set the a value of the interval (default: 0.0)\n");
    printf("  -b value : set the b value of the interval (default: 10.0)\n");
    printf("  -c       : skip the CPU test\n");
    printf("  -g       : skip the GPU test\n");
    printf("  -h       : show usage\n");
    printf("  -i size  : set max iterations (default: 2000000000)\n");
    printf("  -n size  : set the max order of E_n (default: 10)\n");
    printf("  -m size  : set the number of samples (default: 10)\n");
    printf("  -t       : show timing\n");
    printf("  -v       : verbose output\n");
}
