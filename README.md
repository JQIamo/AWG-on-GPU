# AWG_on_GPU
 Arbitrary waveform generation on a GPU using the additive synthesis framework for waveform synthesis

## Overview

This is a software achitecture for real-time arbitrary waveform generation based on a CUDA GPU and PCIe DAC module, allowing for a high throughput static waveform generation as well as a flexible and low-latecy computation of complex waveforms. 

Currently this project is implemented with two pathways of generating dynamic waveforms with chirping tones. 

## Hardware and Environment

The program only runs on a Linux system and requires a NVIDIA GPU and a PCIe interfaced DAC card both with GPUDirect RDMA support.

First install NVIDIA CUDA Tool  Kit, CUDA Driver, and Open GPU Kernel Modules following guide [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html), or following the guide provided by your DAC vendor. Remember to disable IOMMU to avoid RDMA errors; you should be able to do that in your BIOS setting.

The DAC we used is the[Spectrum Instrumentation M4i.6622-x8](https://spectrum-instrumentation.com/products/details/M4i6622-x8.php), whose driver includes are included in the folder `spcm_header`. The drivers and their installation guide could be found [here](https://spectrum-instrumentation.com/support/downloads.php). If you are using other DACs, you would need to configure the device handle `hCard` and the pinned buffer `pvDMABuffer_gpu` accordingly.

To compile the code, you may also need to [g++](https://gcc.gnu.org/) installed as the host compiler.

With the drivers and compilers verified, you should be able to compile with the code. You need to modify the `Makefile` to make sure: 1. The CUDA driver directory is set properly; 2. The source file name matches the code you would like to compile, which by default should be one of the `waveform_synthesis*.cu` files. Upon the successful compilation, an executable `waveform_synthesis` should appear in the code directory.


## Inference

Currently there is no GUI interface. The parameters of generated waveform could be editted in `parameters` files. For real-time interface, the server in `waveform_synthesis*.cu`, or other kinds of interruption needs to be implemented.

## Evaluation

The Nvidia NSight Profile of the cuda function in this program is placed in the `benchmark` folder; the code used for testing is located in the `benchmark\test_setting` folder.

## Citing

Please see the [2403.15582](https://arxiv.org/abs/2403.15582) correlated with this project. Please contanct Juntian Tu juntian"at"umd.edu for issues related to this repository.
