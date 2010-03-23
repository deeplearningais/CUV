#include <string>
#include <boost/python.hpp>
#include <boost/python/extract.hpp>
#include <pyublas/numpy.hpp>

#include <dev_dense_matrix.hpp>
#include <host_dense_matrix.hpp>

#include <image_ops/move.hpp>

//using namespace std;
using namespace boost::python;
using namespace cuv;
namespace ublas = boost::numeric::ublas;

template<class M, class N>
void export_move(){
	def("image_move",
			(void(*)(M&, const N&, const unsigned int&,const unsigned int&,const unsigned int&, const int&, const int&))
			image_move<M,N>, (arg("dst"),arg("src"),arg("image_w"),arg("image_h"),arg("num_maps"),arg("xshift"),arg("yshift")));
}

void export_image_ops(){
	export_move<dev_dense_matrix<float>,dev_dense_matrix<unsigned char> >();
	export_move<dev_dense_matrix<unsigned char>,dev_dense_matrix<unsigned char> >();
}
