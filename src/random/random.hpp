#ifndef __RANDOM_HPP__
#define __RANDOM_HPP__

#include <basics/dev_vector.hpp>

namespace cuv{

 /** @defgroup random Random functions for matrices and vectors
  * @{
  */

	/** 
	 * @brief Fill a matrix/vector with random numbers uniformly distributed between zero and one
	 * 
	 * @param dst Destionation matrix/vector
	 */
	template<class T> void fill_rnd_uniform(T& dst);  

	/** 
	 * @brief Binarize a matrix/vector to 1 or 0 with probability given by current values
	 * 
	 * @param dst	Destionation matrix/vector 
	 */
	template<class T> void rnd_binarize(T& dst);      

	/** 
	 * @brief Add random numbers (normally distributed, mean 0) to a matrix/vector
	 * 
	 * @param dst Destination matrix/vector
	 * @param std Standard deviation of normal distribution used
	 */
	template<class T> void add_rnd_normal(T& dst,const float& std=1.0f);  

	//template<class T> void fill_rnd_normal(T&);   ///< fill a matrix/vector with  numbers (normally distributed, mean 0, std 1)

	/** 
	 * @brief Initialize Mersenne twister to generate random numbers on GPU
	 * 
	 * @param seed Seed for initialization
	 *
	 * This function has to be called exactly _once_ before making use of any random functions.
	 */
	void initialize_mersenne_twister_seeds(unsigned int seed = 0); 

 /** @} */ // end of group random

} // cuv

#endif
