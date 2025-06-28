# CNN Accelerator on FPGA (PYNQ-Z1)

This project implements a hardware-accelerated 3-layer Fully Convolutional Neural Network (FCN) on the PYNQ-Z1 FPGA board. The accelerator is capable of performing inference on handwritten digits in real-time, using hardware modules for convolution, ReLU activation, pooling, and classification.

## Overview

The goal of this project was to design and deploy a complete CNN architecture directly onto an FPGA. The process included:

- Training the model in PyTorch to obtain quantized weights
- Implementing each layer (Conv, ReLU, MaxPooling) in SystemVerilog
- Designing the dataflow using the AXI4-Stream protocol
- Using the PYNQ platform for visualization and software-hardware interfacing
- Validating functionality in simulation with Verilator

## Architecture

- **3-layer FCN**: Conv → ReLU → Pooling (×3) + Fully Connected Layer
- **Custom FP16 Arithmetic Units**: Optimized for resource-constrained environments
- **DMA Integration**: Efficient transfer between PS (ARM processor) and PL (FPGA logic)
- **AXI4-Stream**: Used for internal data movement and external video output

## Features

- Real-time inference for MNIST-like datasets
- Python-PYNQ interface for control and visualization
- Hardware simulation using Verilator
- Modular design for easy hardware debugging and extension

## Repository Structure

```

├── notebooks/            # Jupyter notebooks (PYNQ interface)
├── hdl/                  # SystemVerilog source files
├── sim/                  # Verilator testbenches
├── images/               # Sample outputs and diagrams
├── scripts/              # Helper scripts (e.g., packaging, conversion)
└── README.md             # Project documentation

```

## Limitations

While the full architecture was successfully simulated in Verilator, we were unable to deploy the entire CNN to the PYNQ-Z1 board due to hardware constraints:

- The complete network exceeded available **DSPs** and **LUTs** on the PYNQ-Z1.
- Only individual layers could fit in isolation on the board.

## Future Work

If we were to revisit this project, we would consider two possible directions:

1. **Scaling Up**: Use a larger FPGA with more DSPs and logic resources to accommodate the full network in parallel.
2. **Dynamic Reuse via Software**: Shift more control to the software layer (e.g., Jupyter notebooks), dynamically loading kernel weights and layer configurations to reuse the same convolution hardware across layers. This would reduce hardware usage but introduce a significant performance bottleneck due to slower data transfer between the software (Zynq PS) and hardware (PL) domains.
