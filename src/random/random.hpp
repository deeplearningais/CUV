#ifndef __RANDOM_HPP__
#define __RANDOM_HPP__

#include<dev_vector.hpp>

namespace cuv{

	template<class T> void fill_rnd_uniform(T&);
	template<class T> void rnd_binarize(T&);
	template<class T> void fill_rnd_normal(T&);
	void initialize_mersenne_twister_seeds();

} // cuv

#endif
