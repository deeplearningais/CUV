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
         * @param dev_idx Device id. Tries to set cuda context to this device.
         * 
         * @return Bytes of free memory.
         *
         * This function is highly discouraged and deprecated. It tries to fill the memory.
         * It should be replace by a call to the cuda API.
         */
	int getFreeDeviceMemory();

        /** 
         * @brief Get device memory
         * 
         * @param dev_idx Device id.
         * 
         * @return Memory of GPU in bytes.
         */
	int getMaxDeviceMemory();

        /** 
         * @brief Returns number of CUDA devices
         * 
         * @return Number of CUDA devices
         */
	int countDevices();
        /** 
         * @brief Get device id of current CUDA context
         * 
         * @return device id of current device
         */
        int getCurrentDevice();
}
#endif /* MEMORY_TOOLS_H_ */
