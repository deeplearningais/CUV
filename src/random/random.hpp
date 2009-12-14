#ifndef __RANDOM_HPP__
#define __RANDOM_HPP__

#include<dev_vector.hpp>

namespace cuv{

	void fill_rnd_uniform(dev_vector<float>&);
	void fill_rnd_normal(dev_vector<float>&);
	void initialize_mersenne_twister_seeds();

} // cuv

#endif
