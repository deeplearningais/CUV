LDFLAGS = -L /usr/local/cuda/lib -l cublas -l curand -lpython2.7 -lcuv
CCFLAGS = -O3 -arch=sm_12 -I /usr/include/python2.7
CC      = /usr/local/cuda/bin/nvcc

%.o: %.cu
	$(CC) $(CCFLAGS) -c -o $@ $<

main: cuda_ndarray.o main.o
	$(CC) $(LDFLAGS) -o main $^ 
clean:
	rm -f *.o
	rm -f main
