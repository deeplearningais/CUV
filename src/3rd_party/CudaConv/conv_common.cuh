/*
 * conv_common.cuh
 *
 *  Created on: Nov 24, 2009
 *      Author: Alex Krizhevsky (akrizhevsky@gmail.com)
 */

#ifndef CONV_COMMON_CUH_
#define CONV_COMMON_CUH_

#define MUL24 __mul24
#define MIN(x, y) ((x) > (y) ? (y) : (x))

#ifndef DIVUP
#define DIVUP(x, y) (((x) + (y) - 1) / (y))
#endif
#endif /* CONV_COMMON_CUH_ */
