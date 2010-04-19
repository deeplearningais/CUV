//*LB*
// Copyright (c) 2010, Hannes Schulz, Andreas Mueller, Dominik Scherer
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 
//  * Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//  * Neither the name of the University of Bonn 
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this software without specific prior written
//    permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//*LE*




#include <string>
#include <boost/python.hpp>
#include <boost/python/extract.hpp>
#include <pyublas/numpy.hpp>
#include  <boost/type_traits/is_base_of.hpp>

#include <dense_matrix.hpp>
#include <matrix_ops/matrix_ops.hpp>
#include <convert.hpp>
#include <random/random.hpp>

//using namespace std;
using namespace boost::python;
using namespace cuv;
namespace ublas = boost::numeric::ublas;

template<class M>
void add_rnd_normal_matrix(M&m, const float& std){ add_rnd_normal(m.vec(),std); }
template<class M>
void rnd_binarize_matrix(M&m){ rnd_binarize(m.vec()); }
template<class M>
void fill_rnd_uniform_matrix(M&m){ fill_rnd_uniform(m.vec()); }

template <class T>
void export_functions() {
	def("add_rnd_normal",add_rnd_normal_matrix<T>,(arg("dst"),arg("std")=1));
	def("fill_rnd_uniform",fill_rnd_uniform_matrix<T>,(arg("dst")));
	def("rnd_binarize",rnd_binarize_matrix<T>,(arg("dst")));

	typedef typename T::vec_type V;
	def("add_rnd_normal",add_rnd_normal<V>,(arg("dst"),arg("std")=1));
	def("fill_rnd_uniform",fill_rnd_uniform<V>,(arg("dst")));
	def("rnd_binarize",rnd_binarize<V>,(arg("dst")));
}

void export_random(){
	typedef dense_matrix<float,column_major,dev_memory_space> fdev;
	typedef dense_matrix<float,column_major,host_memory_space> fhost;
	export_functions<fdev>();
	export_functions<fhost>();
	}
