CUDA_PATH ?= /usr/local/cuda-12.1# Path to CUDA installation (with nvcc in bin)
SMS ?= 75  # SM architectures to compile for

ifeq ($(GENCODE_FLAGS),)
$(foreach sm,$(SMS),$(eval GENCODE_FLAGS += -gencode arch=compute_$(sm),code=sm_$(sm)))

HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
GENCODE_FLAGS += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif

################################################################################

all: build

build: waveform_synthesis.bin

# waveform_synthesis.o:waveform_synthesis_streaming.cu # Uncomment to build streaming version
waveform_synthesis.o:waveform_synthesis_playback.cu # Uncomment to build playback version
	$(CUDA_PATH)/bin/nvcc -ccbin g++ -dc -m64 -O3 $(GENCODE_FLAGS) -o $@ -c $< 
cuda_functions.o:lib/cuda_functions.cu
	$(CUDA_PATH)/bin/nvcc -ccbin g++ -dc -m64 -O3 $(GENCODE_FLAGS) -o $@ -c $< 
parameters.o:lib/parameters.cu
	$(CUDA_PATH)/bin/nvcc -ccbin g++ -dc -m64 -O3 $(GENCODE_FLAGS) -o $@ -c $< 
spcm_cuda_common.o:spcm_header/spcm_cuda_common.cu
	$(CUDA_PATH)/bin/nvcc -ccbin g++ -dc -m64 -O3 $(GENCODE_FLAGS) -o $@ -c $< 
server.o:lib/server.cpp
	$(CUDA_PATH)/bin/nvcc -ccbin g++ -dc -m64 -O3 $(GENCODE_FLAGS) -o $@ -c $< 
rearrangement1d.o: lib/rearrangement.cpp
	$(CUDA_PATH)/bin/nvcc -ccbin g++ -dc -m64 -O3 $(GENCODE_FLAGS) -o $@ -c $< 
waveform_synthesis_exe: waveform_synthesis.o cuda_functions.o parameters.o spcm_cuda_common.o  server.o
	$(CUDA_PATH)/bin/nvcc -ccbin g++ -m64 -O3 $(GENCODE_FLAGS) -o $@ $+ -lspcm_linux -lcuda -lculibos
test_exe: test.o cuda_functions.o parameters.o spcm_cuda_common.o server.o
	$(CUDA_PATH)/bin/nvcc -ccbin g++ -m64 -O3 $(GENCODE_FLAGS) -o $@ $+ -lspcm_linux -lcuda -lculibos
testst_exe: testst.o cuda_functions.o parameters.o spcm_cuda_common.o server.o
	$(CUDA_PATH)/bin/nvcc -ccbin g++ -m64 -O3 $(GENCODE_FLAGS) -o $@ $+ -lspcm_linux -lcuda -lculibos


run: build
	./waveform_synthesis_exe

test: test_exe

testrun: test_exe
	./test_exe

clean:
	rm -f -r *.o *_exe

clean_lib:
	rm -f -r *.o

clobber: clean

clear: clean

clear_lib: clean_lib