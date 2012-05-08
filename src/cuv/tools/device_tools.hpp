/*
 * memory_tools.h
 *
 *  Created on: 28.04.2010
 *      Author: gerharda
 */

#ifndef MEMORY_TOOLS_H_
#define MEMORY_TOOLS_H_

#include <string>
#include <stdexcept>
#include <iostream>
#include <cstring>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include "cuv_general.hpp"

#include "exception_helper.hpp"

namespace cuv{
        /** 
         * @brief Get available memory on device
         * 
         * @return Bytes of free memory.
         *
         * @ingroup tools
         */
	int getFreeDeviceMemory();

        /** 
         * @brief Get device memory
         * 
         * @return Memory of GPU in bytes.
         *
         * @ingroup tools
         */
	int getMaxDeviceMemory();

        /** 
         * @brief Returns number of CUDA devices
         * 
         * @return Number of CUDA devices
         * @ingroup tools
         */
	int countDevices();
        /** 
         * @brief Get device id of current CUDA context
         * 
         * @return device id of current device
         * @ingroup tools
         */
        int getCurrentDevice();
}
#endif /* MEMORY_TOOLS_H_ */
