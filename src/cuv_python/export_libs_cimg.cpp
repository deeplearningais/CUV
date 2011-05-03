#include <string>
#include <boost/python.hpp>
#include <boost/python/extract.hpp>
#include <pyublas/numpy.hpp>

#include <cuv/basics/tensor.hpp>
#include <cuv/libs/cimg/cuv_cimg.hpp>

//using namespace std;
using namespace boost::python;
using namespace cuv;
using namespace cuv::libs::cimg;
namespace ublas = boost::numeric::ublas;

template<class V, class M>
void export_cimg_loadsave(){
	def("cimg_load",load<V,M>,(arg("tensor"),arg("filename")));
	def("cimg_save",save<V,M>,(arg("tensor"),arg("filename")));
	def("cimg_show",show<V,M>,(arg("tensor"),arg("filename")));
}

void export_libs_cimg(){
	export_cimg_loadsave<float,row_major>();
	export_cimg_loadsave<unsigned char,row_major>();
}
