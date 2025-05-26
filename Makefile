# Compiler
NVCC = nvcc

# Executable name
TARGET = expIntegral.out

# Source files
SRCS = main_imp.cu cuda_implement.cu cpu_implement.cpp

# Compiler flags, optimise w O3
NVCCFLAGS = -O3 -std=c++11 -arch=sm_75 --use_fast_math --expt-relaxed-constexpr

# Debug flags
DEBUG = 
# DEBUG = -g -G

# Default target
all: $(TARGET)

# Compile/ link
$(TARGET): $(SRCS)
	$(NVCC) $(DEBUG) $(NVCCFLAGS) $(SRCS) -o $(TARGET)

# Clean up
clean:
	rm -f $(TARGET) *.o *.csv

.PHONY: all clean
