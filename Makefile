CUDA_PATH ?= /usr/local/cuda-12.1# Path to CUDA installation (with nvcc in bin)
SMS ?= 75  # SM architectures to compile

ifeq ($(GENCODE_FLAGS),)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif

################################################################################

all: playback

playback: make_obj waveform_synthesis_playback.bin

streaming: make_obj waveform_synthesis_streaming.bin

objects/waveform_synthesis_playback.o:waveform_synthesis_playback.cu
	$(CUDA_PATH)/bin/nvcc -ccbin g++ -dc -m64 -O3 $(GENCODE_FLAGS) -o $@ -c $<
objects/waveform_synthesis_streaming.o:waveform_synthesis_streaming.cu
	$(CUDA_PATH)/bin/nvcc -ccbin g++ -dc -m64 -O3 $(GENCODE_FLAGS) -o $@ -c $<
objects/cuda_functions_streaming.o:lib/cuda_functions_streaming.cu
	$(CUDA_PATH)/bin/nvcc -ccbin g++ -dc -m64 -O3 $(GENCODE_FLAGS) -o $@ -c $<
objects/cuda_functions_playback.o:lib/cuda_functions_playback.cu
	$(CUDA_PATH)/bin/nvcc -ccbin g++ -dc -m64 -O3 $(GENCODE_FLAGS) -o $@ -c $<
objects/cuda_functions.o:lib/cuda_functions.cu
	$(CUDA_PATH)/bin/nvcc -ccbin g++ -dc -m64 -O3 $(GENCODE_FLAGS) -o $@ -c $<
objects/parameters.o:lib/parameters.cu
	$(CUDA_PATH)/bin/nvcc -ccbin g++ -dc -m64 -O3 $(GENCODE_FLAGS) -o $@ -c $<
objects/amp_map.o:lib/amp_map.cu
	$(CUDA_PATH)/bin/nvcc -ccbin g++ -dc -m64 -O3 $(GENCODE_FLAGS) -o $@ -c $<
objects/spcm_cuda_common.o:spcm_header/spcm_cuda_common.cu
	$(CUDA_PATH)/bin/nvcc -ccbin g++ -dc -m64 -O3 $(GENCODE_FLAGS) -o $@ -c $<
objects/server.o:lib/server.cpp
	$(CUDA_PATH)/bin/nvcc -ccbin g++ -dc -m64 -O3 $(GENCODE_FLAGS) -o $@ -c $<
waveform_synthesis_playback.bin: objects/waveform_synthesis_playback.o objects/cuda_functions_playback.o objects/cuda_functions.o objects/parameters.o objects/spcm_cuda_common.o objects/server.o objects/amp_map.o
	$(CUDA_PATH)/bin/nvcc -ccbin g++ -m64 -O3 $(GENCODE_FLAGS) -o $@ $+ -lspcm_linux -lcuda -lculibos
waveform_synthesis_streaming.bin: objects/waveform_synthesis_streaming.o objects/cuda_functions_streaming.o objects/cuda_functions.o objects/parameters.o objects/spcm_cuda_common.o objects/server.o objects/amp_map.o
	$(CUDA_PATH)/bin/nvcc -ccbin g++ -m64 -O3 $(GENCODE_FLAGS) -o $@ $+ -lspcm_linux -lcuda -lculibos

run: playback
	./waveform_synthesis_playback

clean:
	rm -f -r *.o *_exe *.bin objects/

clear: clean

make_obj:
	mkdir -p objects