#include <string>
#include <boost/python.hpp>
#include <boost/python/extract.hpp>
#include <pyublas/numpy.hpp>
#include  <boost/type_traits/is_base_of.hpp>

#include <dev_dense_matrix.hpp>
#include <host_dense_matrix.hpp>
#include <matrix_ops/matrix_ops.hpp>
#include <convert.hpp>
#include <random/random.hpp>

using namespace std;
using namespace boost::python;
using namespace cuv;
namespace ublas = boost::numeric::ublas;

template<class M>
void add_rnd_normal_matrix(M&m){ add_rnd_normal(m.vec()); }
template<class M>
void rnd_binarize_matrix(M&m){ rnd_binarize(m.vec()); }
template<class M>
void fill_rnd_uniform_matrix(M&m){ fill_rnd_uniform(m.vec()); }

template <class T>
void export_functions() {
	def("add_rnd_normal",add_rnd_normal_matrix<T>);
	def("fill_rnd_uniform",fill_rnd_uniform_matrix<T>);
	def("rnd_binarize",rnd_binarize_matrix<T>);

	typedef typename T::vec_type V;
	def("add_rnd_normal",add_rnd_normal<V>);
	def("fill_rnd_uniform",fill_rnd_uniform<V>);
	def("rnd_binarize",rnd_binarize<V>);
}

void export_random(){
	typedef dev_dense_matrix<float,column_major> fdev;
	typedef host_dense_matrix<float,column_major> fhost;
	export_functions<fdev>();
	export_functions<fhost>();
	}
