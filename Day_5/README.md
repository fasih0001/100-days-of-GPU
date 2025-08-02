# CUDA Matrix Sum Kernel Performance Analysis

Today we analyzes the performance of a CUDA matrix summation kernel using different block and grid configurations on an NVIDIA RTX A5000 Mobile Laptop GPU.

## Build and Profiling Commands

### Compilation
```bash
nvcc -arch=compute_86 -code=sm_86,compute_86 -o out sumMatrixOnGPU-2D-grid-2D-block.cu
```

### Profiling
```bash
ncu ./out
```

## Hardware Specifications

### GPU Information
| Specification | Value |
|---------------|-------|
| **GPU Name** | NVIDIA RTX A5000 Laptop GPU |
| **Chip Name** | GA104 |
| **Streaming Multiprocessors** | 48 |
| **L2 Cache Size** | 4.00 MiB |
| **Memory Size** | 15.61 GiB |
| **Memory Bandwidth** | 417.29 GiB/s |
| **Core Clock** | 1.57 GHz |
| **SM Frequency** | 900 MHz (from profiler output) |
| **DRAM Frequency** | 6.99 GHz (from profiler output) |
| **Bus Location** | 0000:01:00.0 |
| **UUID** | f966a2ad-3d32-d40f-f11d-a02d9d3e016e |
| **GSP Firmware Version** | 575.51.03 |
| **Video Accelerator Tracing** | Supported |

## Performance Results

### Kernel Execution Times by Block Configuration

| Grid Type | Block Dimensions | Duration (ms) | Performance Notes |
|-----------|------------------|---------------|-------------------|
| 2D-Grid   | 32 × 32          | 30.56         | Baseline configuration |
| 2D-Grid   | 32 × 16          | 29.17         | 4.5% improvement |
| 2D-Grid   | 16 × 16          | 14.89         | **51.3% improvement** |

## GPU Resource Limits (Compute Capability 8.6)

Your RTX A5000 Laptop GPU has the following per-SM limits:

| Resource | Maximum Value | Notes |
|----------|---------------|-------|
| **Maximum Warps per SM** | 48 | Confirmed by profiler output |
| **Maximum Threads per SM** | 1,536 | 48 warps × 32 threads/warp |
| **Maximum Thread Blocks per SM** | 16 | Hardware limit |
| **Threads per Warp** | 32 | Standard for all CUDA GPUs |
| **Register File Size** | 65,536 (64K) | 32-bit registers per SM |
| **Shared Memory per SM** | ~100 KB | Configurable between L1 and shared |

### Key Observations

- **Best Performance**: 16 × 16 block configuration achieved the fastest execution time
- **Performance Gain**: Using 16 × 16 blocks resulted in a 51.3% performance improvement compared to 32 × 32 blocks
- **Occupancy Achievement**: 16×16 blocks achieved 90.60% occupancy (43.49 out of 48 possible warps)
- **Resource Utilization**: The profiler shows shared memory is the limiting factor (8 blocks per SM)

## Detailed Performance Analysis

### Why 16×16 Block Configuration Performs Best

Based on the profiling data from the 16×16 configuration, several factors explain its superior performance:

#### 1. **Optimal Occupancy**
- **Achieved Occupancy**: 90.60% (very close to theoretical 100%)
- **Active Warps per SM**: 43.49 out of 48 theoretical warps
- **Block Limit**: Shared memory becomes the limiting factor (8 blocks per SM)

#### 2. **Memory Access Efficiency**
- **Memory Throughput**: 67.38% - indicates good memory utilization
- **DRAM Throughput**: 49.42% - reasonable bandwidth utilization
- **L1/TEX Cache Throughput**: 67.38% - efficient cache usage


#### 4. **Memory Access Pattern**
The kernel uses this indexing pattern:
```cuda
unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
unsigned int idx = iy + ix * ny;  // Row-major indexing
```

**Why 16×16 is optimal:**
- **Warp Alignment**: 16×16 = 256 threads = 8 warps per block, providing good granularity
- **Cache Line Utilization**: 16 threads in the x-dimension align well with memory coalescing
- **Shared Memory Constraint**: The profiler shows shared memory limits blocks to 8 per SM
- **Register Usage**: Only 16 registers per thread, leaving room for more blocks


