nvcc -arch=sm_75 -c dummy_spcm_cu.cu -o dummy.o -Xcompiler -fPIC
g++ -shared -o libspcm_linux.so dummy.o
rm dummy.o