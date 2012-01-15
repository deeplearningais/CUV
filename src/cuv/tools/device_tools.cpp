/*
 * memory_tools.cpp
 *
 *  Created on: 28.04.2010
 *      Author: gerharda
 */

#include "device_tools.hpp"

namespace cuv{

	/**
	 * @brief Returns the number of free bytes in global memory on the current device 
	 *
	 *
	 */
	int getFreeDeviceMemory(){
        size_t free, total;
        cuvSafeCall(cuMemGetInfo(&free, &total));
        return free;
	}

	/**
	 * @brief Returns the size of the global memory in bytes on the current device
	 *
	 *
	 */
	int getMaxDeviceMemory(){
        size_t free, total;
        cuvSafeCall(cuMemGetInfo(&free, &total));
        return total;
	}

	int countDevices(){
		int nDevCount = 0;
		cuvSafeCall(cudaGetDeviceCount( &nDevCount ));
		return nDevCount;
	}
    int getCurrentDevice(){
            int dev;
            cuvSafeCall(cudaGetDevice(&dev));
            return dev;
    }


}
