LDFLAGS = -L /usr/local/cuda/lib -l cublas -l curand -lpython2.6 -lcuv
CCFLAGS = -O3 -arch=sm_12
CC      = /usr/local/cuda/bin/nvcc

%.o: %.cu
	$(CC) $(CCFLAGS) -c -o $@ $<

main: cuda_ndarray.o main.o
	$(CC) $(LDFLAGS) -o main $^ 
clean:
	rm -f *.o
	rm -f main
