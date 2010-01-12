/*
 * gpu_locking.h
 *
 *  Created on: Dec 2, 2009
 *      Author: Alex Krizhevsky (akrizhevsky@gmail.com)
 */

#ifndef GPU_LOCKING_H_
#define GPU_LOCKING_H_

#define GPU_LOCK_NO_SCRIPT -2
#define GPU_LOCK_NO_BOARD  -1
#define GPU_LOCK_SCRIPT "/u/murray/bin/gpu_lock"

int get_board_lock();

#endif /* GPU_LOCKING_H_ */
