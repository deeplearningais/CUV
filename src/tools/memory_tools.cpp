/*
 * memory_tools.cpp
 *
 *  Created on: 28.04.2010
 *      Author: gerharda
 */

#include "memory_tools.h"

namespace cuv{

	/**
	 * @brief Returns the number of free bytes in global memory on the device with id dev_idx
	 *
	 * @param dev_idx	index of device to check
	 *
	 */
	int getFreeDeviceMemory(int dev_idx){

		// start with chunk 0,5 gigabyte
		int chunk_size= 1024*1024*1024;
		int memory_size = 0;
		float* memory;
		std::vector<float*> container;
		std::vector<float*>::iterator the_iterator;
		cudaError_t cuerr;
		cudaSetDevice(dev_idx);

		while( chunk_size >0) {
			//printf"Trying to allocate %i Bytes:", chunk_size);
			cuerr = cudaMalloc( (void**) &memory, chunk_size);
			if (cuerr != cudaSuccess) {
				//printf(" --> failed                                 \n");
				//std::cout << "Error:" << cuerr << std::endl;
				if(chunk_size >=512)
					chunk_size /= 2;
				else
					chunk_size--;
			} else {
				//printf(" --> success, Allocated %i Bytes\n", memory_size);
				cudaMemset( (void*) memory, rand(), chunk_size );
				memory_size += chunk_size;
				container.push_back(memory);
			}
		}

		// free mem
		the_iterator = container.begin();
		printf("releasing memory....\n");
		while( the_iterator != container.end() ) {
			cuerr = cudaFree(*the_iterator);
			if(! cuerr == cudaSuccess){
				std::cout << std::endl <<"Error while free mem: "<< cuerr << std::endl;
			}
		    ++the_iterator;
		}

		//std::cout << std::endl;
		return memory_size;

	}

	/**
	 * @brief Returns the size of the global memory in bytes on the device with id dev_idx
	 *
	 * @param dev_idx	index of device to check
	 *
	 */
	int getMaxDeviceMemory(int dev_idx){
		cudaDeviceProp deviceProp;
		memset( &deviceProp, 0, sizeof(deviceProp));
		if( cudaSuccess != cudaGetDeviceProperties(&deviceProp, dev_idx)){
			std::cout << std::endl << "Error while interrogating device"<< std::endl;
			return 0;
		}else{
			return deviceProp.totalGlobalMem;
		}
	}
}
