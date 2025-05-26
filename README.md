Computes the exp integral \( E_n(x) \) using both CPU and GPU (CUDA) code paths.

## Build Instructions

### Prerequisites

- NVIDIA GPU with CUDA capability
- CUDA Toolkit (e.g., `nvcc`)
- g++ or compatible C++ compiler
- GNU make

### Build

make

### Run

./exponentialIntegral.out

###CMD line args:
`-a` : Lower bound of the interval of integration 
`-b` : Upper bound
`-n` :Maximum order of E_n(x)
`-m` : Number of samples
`-i` : Maximum number of iterations
`-c` : Disable CPU comp
`-g` : Disable GPU comp
`-t` : Enable timing
`-v` : Verbose output (prints computed values)
`-h` : help

### Bash script for result collection 'res.sh'

###Make it executable with

chmod +x res.sh

###Run

./res.sh
