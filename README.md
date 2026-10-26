# AWG_on_GPU

**This is a mirror of the private `AWG-on-GPU-for-RbYb` repo used in RbYbTweezerLab organization for public view with private info removed.**

For the obsolete version of the code upon paper publication, check commit history.

Arbitrary waveform generation on a GPU using the additive synthesis framework for waveform synthesis.

<a name="top"></a>

## Table of Contents
- [AWG\_on\_GPU](#awg_on_gpu)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Hardware and Environment](#hardware-and-environment)
  - [Directory Overview](#directory-overview)
  - [Primer in Operation Theory](#primer-in-operation-theory)
    - [Initial Waveform Generation (Static Waveform)](#initial-waveform-generation-static-waveform)
    - [Dynamic Waveform Generation (Tone Chirping)](#dynamic-waveform-generation-tone-chirping)
    - [Tone Rearrangement](#tone-rearrangement)
  - [Acceptable TCP Commands](#acceptable-tcp-commands)
  - [Interface](#interface)
  - [Change Log and TODOs](#change-log-and-todos)
    - [TODOs](#todos)
  - [Citing](#citing)

## [Overview](#top)

This is a software architecture for real-time arbitrary waveform generation based on a CUDA GPU and PCIe DAC module, allowing for a high throughput static waveform generation as well as a flexible and low-latency computation of complex waveforms.  

Currently, this project is implemented with two pathways of generating dynamic waveforms with chirping tones, allowing the waveforms to be computed in advance or in parallel of streaming.  

The project is implemented with a primitive TCP based control logic that is subject to improvement or replacement according to users' need.  

## [Hardware and Environment](#top)

The program only runs on a Linux system and requires an NVIDIA GPU and a PCIe interfaced DAC card both with GPUDirect RDMA support.

First install NVIDIA CUDA Tool Kit, CUDA Driver, and Open GPU Kernel Modules following guide [here](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html), or following the guide provided by your DAC vendor. Remember to disable IOMMU to avoid RDMA errors; you should be able to do that in your BIOS setting.

The DAC we used is the [Spectrum Instrumentation M4i.6622-x8](https://spectrum-instrumentation.com/products/details/M4i6622-x8.php), whose driver includes are included in the folder `spcm_header` (with permission for sharing from Spectrum Instrumentation). The drivers and their installation guide could be found [here](https://spectrum-instrumentation.com/support/downloads.php). If you are using other DACs, you would need to configure the device handle `hCard` and the pinned buffer `pvDMABuffer_gpu` accordingly.

To compile the code, you may also need to have [g++](https://gcc.gnu.org/) installed as the host compiler together with NVCC.

With the drivers and compilers verified, you should be able to compile with the code. You need to modify the `Makefile` to make sure: 1. The CUDA driver directory is set properly; 2. The source file name matches the code you would like to compile, which by default should be one of the `waveform_synthesis*.cu` files. Upon the successful compilation, an executable `waveform_synthesis` should appear in the code directory.


## [Directory Overview](#top)

- **lib/**  
  Utility functions
  - **cuda_functions**:
    Cuda functions for static waveform generation
  - **cuda_functions_playback**:
    Cuda functions for dynamic waveform generation through playback pathway
  - **cuda_functions_streaming**:
    Cuda functions for dynamic waveform generation through streaming pathway
  - **parameters**:
    Global configurations
  - **server**: 
    Functions related to the TCP server and command interpretation.
  - **spectrum_analyzer**: 
    Functions probe the spectrum of generated waveform softwarely.

- **spcm_header/**  
  Headers for Spectrum Instrumentation API

- **test/**  
  Codes for testing and debugging
  - **dummy_spcm(_cu)**:
    Source code for a dummy spcm_linux library without/with a CUDA GPU support. Execute **test(_cu).sh** in shell for compilation.
  - **TCP_tester.ipynb**:
    A Jupyter notebook generating random waveform generation/rearrangement commands for testing purpose.

- **TCP_access.ipynb**:
  A Jupyter notebook includes example TCP commands to communicate with the program.

- **waveform_synthesis_playback(streaming).cu**: 
  Main code for the AWG program through playback/streaming pathway.

- **waveform_synthesis_sequence.cu**:
  Main code for generating a parametric heating trap.

## [Primer in Operation Theory](#top)

The program performs three kinds of behaviors: 1. initial waveform generation (static waveform), 2. tone chirping (dynamic waveform), 3. tone rearrangement.

### [Initial Waveform Generation (Static Waveform)](#top)

The program always starts operation with a static waveform generation.
Due to the purpose for tweezer arrays, the static waveform is synthesized by a sum of a set of tones (expected to have a uniformly spaced frequencies).
The generated waveform containing $N$ tones on each channel is expressed as:

$$V(t)=\sum_{n=0}^{N-1} \frac{A_n}{N_\textrm{max}} \sin(2\pi f_n t + \phi_n)$$

where $A_n$ is the amplitude factor for each tone (by default, $1$);
$N_\textrm{max}\ge N$ is the configurable maximum number of tones allowed on the channel (to avoid overpowering the AOD);
$f_n$ is the frequency of the tone;
$\phi_n$ is the [Schroeder phase](https://ieeexplore.ieee.org/document/1054411) of the tone to suppress the crest factor of the synthesized waveform.
Due to the main purpose of tweezer arrays, the Schroeder phase is computed as follows (assuming $f_n$ increases monotonically):

$$\phi_n = -\frac{\pi n^2}{N}$$

It is worth noting that the Schroeder phase is computed assuming all tones have the same amplitude (analogly, even if a digital amplitude prefactor is applied to homogenize the output).
Additionally, the crest factor decreases as the number of tones increases when $N\ge 2$.

### [Dynamic Waveform Generation (Tone Chirping)](#top)

When a frequency chirp is programmed, the phase of the dynamic tone is modulated to move the frequency in a minimum-jerk trajectory within a given time.
The dynamic waveform starting to chirp at $t=0$ and ending at $t=T$ can be expressed as:

$$V_n^\textrm{dynamic}(t)=\frac{A_n}{N_\textrm{max}}\sin\left[2\pi t\left(f_n^i+\left(f^f_n-f^i_n\right)\left[\frac{5}{2}\left(\frac{t}{T}\right)^3-3\left(\frac{t}{T}\right)^4+\left(\frac{t}{T}\right)^5\right]\right)+\phi_n\right]$$

where $f_n^i$ is the initial frequency of the dynamic tone;
$f_n^f$ is the final frequency of the dynamic tone.

After the chirping, the dynamic tone becomes a static tone at new frequency $f_n^f$:

$$V_n'(t)=\frac{A_n}{N_\textrm{max}}\sin[2\pi f_n^f t+\phi_n+\pi T(f_n^i-f_n^f)]$$

To preserve the phase continuity over the entire process, the Schroeder phase is no longer kept.
The additional phase shift is due to the minimum-jerk trajectory of the frequency chirp.

### [Tone Rearrangement](#top)

Each call of waveform generation program will generate a new list of tones.
To modify the existing spectrum pattern, the tone rearrangement process is used.
The rearrangement is a combination of three operations: 1. deleting the tones where no atoms are loaded, 2. chirping the tones to create the desired pattern, and 3. reindexing the tones for further rearrangement.

## [Acceptable TCP Commands](#top)

The available TCP commands are listed below. For simplicity, we use `m` to iterate over the number of channels (`m=0,1,2,3`), `n` to iterate over the number of tweezers, whose maximum depends on preceding commands. A number in the form of `Xmn` should be iterated over `n` times consecutively for each `m`. Commands for testing purposes are not listed.

- `"0,Sm...,Dm...,FSmn...,FDmn...,DIn...,go"`: Generate a new list of tones with digitally uniform amplitude and frequency chirps.
  * `Sm` is the total number of tones on Channel `m`;
  * `Dm` is the number of dynamic tones on Channel `m` (`Dm <= Sm` and can be `0`);
  * `FSmn` is the (initial) frequency of the tones on Channel `m` (`n = 0,1...,Sm-1`); 
  * `FDmn` is the final frequency of the dynamic tones on Channel `m` (`n = 0,1...,Dm-1`);
  * `DIn` is the index of the dynamic tone among tones across all channels (`n = 0,1...,D0+D1+D2+D3-1`).
  
  As an example, `"0,3,1,2,0,2,0,1,0,FS00,FS01,FS02,FS10,FS20,FS21,FD00,FD01,FD20,1,2,4,go"` creates 3 tones at frequencies `FS00,FS01,FS02` at channel `0`, 1 tone at frequency `FS10` at channel `1`, and 2 tones at frequencies `FS20,FS21` at channel `2`.
When triggered, the tones `1` and `2` in channel `0` will chirp from `FS01` and `FS02` to `FD00` and `FD01`, and tone `0` (`DI2 - S0 - S1 = 4-3-1 = 0`) in channel `2` will chirp from `FS20` to `FD20`.


- `"1,Sm...,Dm...,FSmn...,FDmn...,DIn...,Amn...,go"`: Generate a new list of tones with digital amplitude factors and frequency chirps.
  * `Sm` is the total number of tones on Channel `m`;
  * `Dm` is the number of dynamic tones on Channel `m` (`Dm <= Sm` and can be `0`);
  * `FSmn` is the (initial) frequency of the tones on Channel `m` (`n = 0,1...,Sm-1`);
  * `FDmn` is the final frequency of the dynamic tones on Channel `m` (`n = 0,1...,Dm-1`);
  * `DIn` is the index of the dynamic tone among tones across all channels (`n = 0,1...,D0+D1+D2+D3-1`, `DIn <= S0+S1+S2+S3-1`);
  * `Amn` is the amplitude factor of the tones on Channel `m` (`n = 0,1...,Sm-1`); the sum of `Amn` should be `Sm` for each `m`. 
  
  As an example, `"1,3,1,2,0,2,0,1,0,FS00,FS01,FS02,FS10,FS20,FS21,FD00,FD01,FD20,1,2,4,0.9,1,1.1,1,1,1,go"` performs the exact same behavior as the previous example, but with the 3 tones in channel `0` scaled at amplitude factors `0.9`, `1`, `1.1`. 


- `"Ch,S,D,SIn...,FDn...,DIn...,rearr"`: Perform a 1D rearrangement on Channel `Ch` (Can only be called AFTER a generation/rearrangement of waveforms).
    * `Ch` is the index of channel where rearrangement happens;
    * `S` is the number of tones kept (i.e., with atoms); `S` should be smaller than the number of tones on Channel `Ch` before rearrangement.
    * `D` is the number of dynamic tones (`D <= S`);
    * `SIn` is the indexes of the tones kept among previous tones on Channel `Ch` (`n = 0,1...,S-1`);
    * `FDn` is the final frequency of the dynamic tones on Channel `Ch` (`n = 0,1...,D-1`);
    * `DIn` is the index of the dynamic tone among the kept tones (`n = D-1`, `DIn <= S-1`);
  
  As an example, `"3,2,1,1,2,FD0,0,rearr"` turns off all other tones in channel `3` except the tones with indices `1` and `2` (2nd and 3rd tones), not changing other channels.
  Then moves tone `1` (i.e., the tone with new index `0` among the kept two tones) to frequency `FD0`.

  Note that the rearrangement will overwrite the tone indices so further rearrangement should follow select indices based on the previous rearrangement.


- `"Ch,AMP,amp_set"`: Change the overall DAC amplitude for channel `Ch`.
  * `Ch` is the index of channel where the DAC amplitude is changed;
  * `AMP` is the overall DAC amplitude (in Volt) for channel `Ch` (`80 < AMP < 2500`);


- `"T,time_set"`: Change the ramp time `T`.
  * `T` is the duration in seconds for any tone dynamics executed by the AWG;


- `"stop"`: Terminate the program.

The following commands for only compatible with `waveform_synthesis_sequence`:
- `"0,Sm...,FSmn...,SA,SF,shake"`: Generate a new list of static tones with digitally uniform amplitude. When triggered, the tones shake to heat the trapped atoms.
  * `Sm` is the total number of tones on Channel `m`;
  * `FSmn` is the frequency of the tones on Channel `m` (`n = 0,1...,Sm-1`);
  * `SA` is the oscillation amplitude (in Hz) in the frequency domain of each tone;
  * `SF` is the oscillation frequency (in Hz) of each tone;

  As an example, `"0,2,0,0,0,FS00,FS01,300000,10000,shake"` generates two tones at frequencies `FS00` and `FS01` at Channel `0`. When triggered, the frequency of each tone will shake with a pattern of `fn(t) = FS0n + 300000 Hz * sin(2π*10000 Hz*t)`.


- `"1,Sm...,FSmn...,Amn...,SA,SF,shake"`: Generate a new list of static tones with digital amplitude factors. When triggered, the tones shake to heat the trapped atoms.
  * `Sm` is the total number of tones on Channel `m`;
  * `FSmn` is the frequency of the tones on Channel `m` (`n = 0,1...,Sm-1`);
  * `Amn` is the amplitude factor of the tones on Channel `m` (`n = 0,1...,Sm-1`); the sum of `Amn` should be `Sm` for each `m`.
  * `SA` is the oscillation amplitude (in Hz) in the frequency domain of each tone;
  * `SF` is the oscillation frequency (in Hz) of each tone;

  As an example, `"1,2,0,0,0,FS00,FS01,1.1,0.9,300000,10000,shake"`performs the exact same behavior as the previous example, but with the tones scaled at amplitude factors `1.1,0.9`.


## [Interface](#top)

Currently, there is no available GUI interface. The parameters of generated waveform could be edited in `parameters` files. For real-time interface, the server in `waveform_synthesis*.cu`, or other kinds of interruption needs to be implemented.

An ongoing development of a GUI interface can be found at the [GPU-labscript-interface](https://github.com/RbYbTweezerLab/GPU-labscript-interface.git) repository.

## [Change Log and TODOs](#top)

### [TODOs](#top)

Current version still suffers from a "flattened" dynamic tone indexing `DIn` across all channels, making the command generation and parsing logic confusing.
This weird logic is a direct result of pointing to the specific tone in the GPU device memory, and is potentially optimizable with some scarification of the efficiency in memory access.

Additionally, the streaming pathway is not being updated for implementation of the new version due to lack of immediate necessity. 

## [Citing](#top)

Please see the [paper](https://doi.org/10.1109/TSP.2025.3574958) correlated with this project. Please contact Juntian Tu (juntian"at"umd.edu) for issues related to this repository.
