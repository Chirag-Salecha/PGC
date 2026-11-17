# Neural Network Layers in CUDA

> **Experiment 5 – Parallel and GPU Computing Laboratory**

A CUDA C++ implementation of a **two-layer feed-forward neural network** using custom GPU kernels. The project implements matrix multiplication with bias, ReLU, and sigmoid activation, validates CPU/GPU correctness, and benchmarks GPU acceleration.

##  Highlights

-  Custom CUDA kernels
-  Two-layer neural-network forward propagation
-  Individual known-answer kernel tests
-  CPU vs GPU benchmarking
-  Kernel-only and end-to-end timing analysis
-  Exact observed CPU/GPU validation match
-  **32.13× kernel-only speedup**

---

##  Neural Network Architecture

This project implements **forward propagation only**. It does not include training, backpropagation, loss calculation, or an optimizer.

### Mathematical flow

```text
Z₁ = X × W₁ + b₁
A₁ = ReLU(Z₁)

Z₂ = A₁ × W₂ + b₂
Output = Sigmoid(Z₂)
```

### Architecture diagram

```mermaid
flowchart LR
    X["Input X<br/>256 × 128"] --> L1["X × W1 + b1"]
    L1 --> R["ReLU"]
    R --> H["Hidden Layer<br/>256 × 64"]
    H --> L2["A1 × W2 + b2"]
    L2 --> S["Sigmoid"]
    S --> O["Output<br/>256 × 10"]
```

##  Configuration

| Parameter | Value |
|---|---:|
| Batch size | 256 |
| Input size | 128 |
| Hidden layer size | 64 |
| Output size | 10 |
| Timing repetitions | 20 |
| Matrix thread block | 16 × 16 |
| Activation threads per block | 256 |

### Tensor dimensions

```text
Input X       : 256 × 128
Weights W1    : 128 × 64
Bias b1       : 64
Hidden output : 256 × 64
Weights W2    : 64 × 10
Bias b2       : 10
Final output  : 256 × 10
```

Inputs, weights, and biases are generated deterministically using sine and cosine functions so that the CPU and GPU always process identical values.

---

#  Execution Workflow

```mermaid
flowchart TD
    A["Generate deterministic input and weights"] --> B["CPU forward pass"]
    A --> C["Allocate GPU memory"]
    C --> D["Host → Device transfer"]
    D --> E["GPU warm-up"]
    E --> F["Kernel-only timing"]
    F --> G["End-to-end timing"]
    B --> H["Compare CPU and GPU outputs"]
    G --> H
    H --> I{"Maximum difference < 1e-4?"}
    I -->|Yes| J["PASSED"]
    I -->|No| K["FAILED"]
```

### GPU pipeline

```mermaid
flowchart LR
    A["GPU Input"] --> B["Matrix Multiplication + Bias"]
    B --> C["ReLU Kernel"]
    C --> D["Matrix Multiplication + Bias"]
    D --> E["Sigmoid Kernel"]
    E --> F["Final Output"]
```

---

#  CUDA Implementation

The program uses three custom CUDA kernels:

| Kernel | Operation | Parallelization |
|---|---|---|
| `matrixMulBiasKernel` | Matrix multiplication + bias | One thread computes one output element |
| `reluKernel` | `max(0, x)` | One thread processes one element |
| `sigmoidKernel` | `1 / (1 + e^-x)` | One thread processes one element |

### Matrix multiplication

For:

```text
A: M × K
B: K × N
C: M × N
```

each CUDA thread computes:

```text
C[row][col] = Σ A[row][k] × B[k][col] + bias[col]
```

The matrix kernel uses **16 × 16 thread blocks**, while activation kernels use **256 threads per block**.

---

#  Performance Results

The benchmark below was obtained from a successful run on an **NVIDIA GeForce RTX 2050 (4 GB)**.

| Measurement | Result |
|---|---:|
| CPU average forward-pass time | **2.083975 ms** |
| GPU kernel-only time | **0.064870 ms** |
| GPU end-to-end time | **0.184795 ms** |
| Kernel-only speedup | **32.125205×** |
| End-to-end speedup | **11.277226×** |
| Maximum CPU/GPU difference | **0.000000** |
| Kernel tests | **All Passed** |
| Final validation | **PASSED** |

## Speedup calculation

```text
Speedup = CPU Execution Time / GPU Execution Time
```

Therefore:

```text
Kernel-only: 2.083975 / 0.064870 = 32.13×
End-to-end:  2.083975 / 0.184795 = 11.28×
```

---

#  Graphical Analysis

## 1. Execution time comparison

> Lower execution time is better.

```mermaid
xychart-beta
    title "CPU vs GPU Execution Time"
    x-axis ["CPU", "GPU Kernel", "GPU End-to-End"]
    y-axis "Time (ms)" 0 --> 2.2
    bar [2.083975, 0.064870, 0.184795]
```

### Interpretation

- The CPU takes approximately **2.08 ms** per forward pass.
- Actual GPU kernel computation takes approximately **0.065 ms**.
- Including memory transfers raises GPU execution to **0.185 ms**.
- The GPU remains substantially faster even when transfer overhead is included.

## 2. GPU speedup comparison

```mermaid
xychart-beta
    title "GPU Speedup over CPU"
    x-axis ["Kernel Only", "End-to-End"]
    y-axis "Speedup (×)" 0 --> 35
    bar [32.125205, 11.277226]
```

## 3. Relative execution time

Using CPU execution time as the baseline:

| Execution mode | Relative time |
|---|---:|
| CPU | 100.0% |
| GPU Kernel-only | ~3.1% |
| GPU End-to-End | ~8.9% |

### Key insight

The kernel-only benchmark is faster because data is already resident in GPU memory. End-to-end timing additionally includes memory movement and synchronization.

## 4. Memory-transfer overhead

```mermaid
flowchart LR
    A["CPU Memory"] -->|"Host → Device"| B["GPU Memory"]
    B -->|"CUDA Kernels"| C["GPU Computation"]
    C -->|"Device → Host"| D["CPU Output"]
```

The end-to-end timing includes:

- Host-to-device data transfer
- GPU kernel execution
- Device synchronization
- Device-to-host output transfer

For repeated inference or larger workloads, keeping data and model parameters on the GPU can reduce transfer overhead.

---

#  Validation

Every CUDA operation is tested before the full benchmark.

## Matrix multiplication + bias

```text
A × B = [22 28]
        [49 64]

Bias = [1, -1]

Expected = [23 27]
           [50 63]
```

**Result: PASSED**

## ReLU

```text
Input:    [-2, 0, 3, -4, 5]
Expected: [ 0, 0, 3,  0, 5]
```

**Result: PASSED**

## Sigmoid

```text
Input:    [-2, 0, 2]
Expected: approximately [0.119203, 0.500000, 0.880797]
```

**Result: PASSED**

## Final CPU vs GPU validation

```text
Maximum CPU-GPU difference: 0.000000
Result: PASSED
```

The acceptance threshold is:

```text
Maximum difference < 1 × 10⁻⁴
```

---

# System Configuration

| Component | Configuration |
|---|---|
| GPU | NVIDIA GeForce RTX 2050 |
| GPU Memory | 4 GB |
| Compute Capability | 8.6 |
| NVIDIA Driver | 592.27 |
| CUDA Toolkit Compiler | CUDA 13.4 |
| Operating System | Windows |
| C++ Compiler | Microsoft Visual C++ Build Tools |

The executable was compiled specifically for the RTX 2050 architecture using `-arch=sm_86`.

---

#  Build and Run

## Prerequisites

- NVIDIA CUDA-capable GPU and driver
- CUDA Toolkit with `nvcc`
- Microsoft Visual Studio Build Tools / MSVC
- Windows 10 or Windows 11

Verify your installation:

```cmd
nvcc --version
nvidia-smi
```

Compile:

```cmd
nvcc -O2 -arch=sm_86 neural_network.cu -o neural_network.exe
```

Run:

```cmd
neural_network.exe
```

> If you use a different NVIDIA GPU, use the appropriate CUDA architecture value for that GPU.

---

#  Repository Structure

```text
PGC-L5-NeuralNetwork-CUDA/
│
├── neural_network.cu      # Complete CUDA implementation
├── README.md              # Documentation and performance analysis
└── .gitignore             # Compiled artifact exclusions
```

---

#  Concepts Demonstrated

- CUDA kernel programming
- GPU thread and block organization
- 1D and 2D thread indexing
- GPU memory allocation
- Host ↔ Device memory transfer
- CUDA synchronization
- CUDA event-based timing
- CPU vs GPU performance comparison
- Numerical validation
- GPU architecture targeting

---

#  Conclusion

The two-layer neural network was successfully implemented and validated using custom CUDA kernels on an **NVIDIA GeForce RTX 2050**.

All individual kernels passed their validation tests, and the complete GPU forward pass matched the CPU implementation with a maximum observed difference of **0.000000**.

## Final Outcome

> **32.13× faster kernel execution**

> **11.28× faster end-to-end execution**

The experiment clearly demonstrates both the advantage of GPU parallelism for neural-network computations and the importance of memory-transfer overhead in real-world GPU performance.


