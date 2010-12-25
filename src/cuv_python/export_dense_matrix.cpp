//*LB*
// Copyright (c) 2010, University of Bonn, Institute for Computer Science VI
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
#include <vector.hpp>
#include <dense_matrix.hpp>
#include <matrix_ops/matrix_ops.hpp>
#include <convert.hpp>

using namespace std;
using namespace boost::python;
using namespace cuv;
namespace ublas = boost::numeric::ublas;
namespace bp    = boost::python;

/*
 * translate our storage type to the one of ublas
 */
template<class T>
struct matrix2ublas_traits                       { typedef ublas::row_major storage_type; };
template<>
struct matrix2ublas_traits<cuv::column_major>    { typedef ublas::column_major storage_type; };
template<>
struct matrix2ublas_traits<cuv::row_major>       { typedef ublas::row_major storage_type; };

template<class T>
struct ublas2matrix_traits                                       { typedef ublas::row_major storage_type; };
template<>
struct ublas2matrix_traits<boost::numeric::ublas::column_major > { typedef cuv::column_major storage_type; };
template<>
struct ublas2matrix_traits<boost::numeric::ublas::row_major >    { typedef cuv::row_major storage_type; };


/*
 * Create RM view of CM matrix
 */
template<class V, class T, class I>
dense_matrix<V,row_major,T,I>*
	make_rm_view(dense_matrix<V,column_major,T,I> &m,int h, int w){
	if (w==-1) {
		w=m.h();
		}
	if (h==-1) {
		h=m.w();
	}
	cuvAssert(w*h==m.h()*m.w());
	return new dense_matrix<V,row_major,T,I>(h,w,m.ptr(),true);
}

template<class V, class T, class I>
void export_rm_view(){
	def("make_rm_view",make_rm_view<V,T,I>,(bp::arg ("matrix"),bp::arg("height")=-1,bp::arg("width")=-1),return_value_policy<manage_new_object, with_custodian_and_ward_postcall<1, 0> >());
}

/*
 * Create CM view of RM matrix
 */
template<class V, class T,class I>
dense_matrix<V,column_major,T,I>*
	make_cm_view(dense_matrix<V,row_major,T,I> &m,int h, int w){
	if (w==-1) {
		w=m.h();
		}
	if (h==-1) {
		h=m.w();
	}
	cuvAssert(w*h==m.h()*m.w());
	return new dense_matrix<V,column_major,T,I>(h,w,m.ptr(),true);
}

template<class V, class T,class I>
void export_cm_view(){
	def("make_cm_view",make_cm_view<V,T,I>,(bp::arg ("matrix"),bp::arg("height")=-1,bp::arg("width")=-1),return_value_policy<manage_new_object, with_custodian_and_ward_postcall<1, 0> >());
}

/*
 * Create VIEWs at the same location in memory
 */
template<class T,class S>
cuv::vector<T,host_memory_space, S> *
vec_view(pyublas::numpy_vector<T> v){
	return new cuv::vector<T,host_memory_space>(v.size(),v.as_ublas().data().data(),true);
}
template<class T, class Mto, class Mfrom>
dense_matrix<T, Mto, host_memory_space>*
mat_view(pyublas::numpy_matrix<T, Mfrom> m){
	typedef typename dense_matrix<T,Mto,host_memory_space>::index_type index_type;
	cuv::vector<T,host_memory_space>* vec = new cuv::vector<T,host_memory_space>((index_type)m.size1()*m.size2(),m.as_ublas().data().data(),true);
	const bool same = boost::is_same<Mto, typename ublas2matrix_traits<Mfrom>::storage_type >::value;
	if(same) return new dense_matrix<T,Mto,host_memory_space>((index_type)m.size1(), (index_type)m.size2(), vec);
	else     return new dense_matrix<T,Mto,host_memory_space>((index_type)m.size2(), (index_type)m.size1(), vec);
}
/*
 * Create COPYs at another location in memory
 */
template<class T, class Mto, class Mfrom>
dense_matrix<T, Mto, host_memory_space>*
copy(pyublas::numpy_matrix<T, boost::numeric::ublas::column_major> m){
	typedef typename dense_matrix<T,Mto,host_memory_space>::index_type index_type;
	const bool same = boost::is_same<Mto, typename ublas2matrix_traits<Mfrom>::storage_type >::value;
	dense_matrix<T,Mto,host_memory_space>* mat;
	if(same) new dense_matrix<T,Mto,host_memory_space>((index_type)m.size1(), (index_type)m.size2());
	else     new dense_matrix<T,Mto,host_memory_space>((index_type)m.size2(), (index_type)m.size1());
    memcpy(mat->ptr(), m.as_ublas().data().data(), mat->n() * sizeof(T));
    return mat;
}

/*
 * Export the main matrix class
 *   this should work for _any_ class, namely, regardless of host/dev, value_type and memory_layout
 */
template<class T>
void
export_dense_matrix_common(std::string name){
	typedef T mat;
	typedef typename mat::value_type value_type;
	typedef typename mat::index_type index_type;
	typedef typename mat::memory_layout memlayout_type;
	typedef typename mat::memory_space_type memspace_type;
	typedef typename mat::vec_type   vec_type;

	class_<mat>(name.c_str(), init<typename mat::index_type, typename mat::index_type>())
		.def("__len__",&mat::n, "matrix number of elements")
		.def("alloc",  &mat::alloc, "allocate memory")
		.def("dealloc",&mat::dealloc, "deallocate memory")
		.def("set",    (void (mat::*)(const index_type&, const index_type&, const value_type&))(&mat::set),(bp::arg ("x"),bp::arg("y"),bp::arg("value")), "set a value in the matrix")
		.def("at",    (value_type (mat::*)(const index_type&,const index_type&))(&mat::operator()), "value at this position")
		.def("resize", (void (mat::*)(const index_type&, const index_type&)) (&mat::resize), "resize dimensions")
		.def(self += value_type())
		.def(self -= value_type())
		.def(self *= value_type())
		.def(self /= value_type())
		.def(self += self)
		.def(self -= self)
		.def(self *= self)
		.def(self /= self)
		.def("__add__", ( mat (*) (const mat&,const mat&))operator+<value_type,memlayout_type,memspace_type,index_type>)
		.def("__sub__", ( mat (*) (const mat&,const mat&))operator-<value_type,memlayout_type,memspace_type,index_type>)
		.def("__mul__", ( mat (*) (const mat&,const mat&))operator*<value_type,memlayout_type,memspace_type,index_type>)
		.def("__div__", ( mat (*) (const mat&,const mat&))operator/<value_type,memlayout_type,memspace_type,index_type>)
		.def("__add__", ( mat (*) (const mat&,const value_type&))operator+<value_type,memlayout_type,memspace_type,index_type>)
		.def("__sub__", ( mat (*) (const mat&,const value_type&))operator-<value_type,memlayout_type,memspace_type,index_type>)
		.def("__mul__", ( mat (*) (const mat&,const value_type&))operator*<value_type,memlayout_type,memspace_type,index_type>)
		.def("__div__", ( mat (*) (const mat&,const value_type&))operator/<value_type,memlayout_type,memspace_type,index_type>)
		.def("__neg__", ( mat (*) (const mat&))operator-<value_type,memlayout_type,memspace_type,index_type>)
		.add_property("h", &mat::h)
		.add_property("w", &mat::w)
		.add_property("n", &mat::n)
		.add_property("vec", make_function((vec_type* (mat::*)())(&mat::vec_ptr), return_internal_reference<>()))
		.add_property("memsize", &mat::memsize)
		;
}

/*
 * Export a dense matrix and corresponding conversion functions
 */
template <class T, class M, class M2>
void
export_dense_matrix(std::string typen){
	export_dense_matrix_common<dense_matrix<T, M, dev_memory_space> >  ( std::string( "dev_matrix_" ) + (typen));
	export_dense_matrix_common<dense_matrix<T, M2, host_memory_space> >( std::string( "host_matrix_" ) + (typen));
	def("convert", (void(*)(dense_matrix<T,M,dev_memory_space>&,const dense_matrix<T,M2,host_memory_space>&)) cuv::convert);
	def("convert", (void(*)(dense_matrix<T,M2,host_memory_space>&,const dense_matrix<T,M,dev_memory_space>&)) cuv::convert);
}

/*
 * Export a function to create a view of a numpy matrix. Only for HOST matrices, of course.
 *  A view is a matrix which resides at the same point in memory. 
 *  When the numpy matrix is destroyed, you should not operate on this matrix anymore!!!
 */
template <class T, class Mfrom, class Mto>
void
export_dense_matrix_view(const char* str){
	typedef dense_matrix<T,Mto,host_memory_space>             to_type;            // destination type
	typedef typename matrix2ublas_traits<Mfrom>::storage_type Mfrom_ublas_type;   // our column/row major type (derived from ublas)
	typedef pyublas::numpy_matrix<T,Mfrom_ublas_type>         from_type;          // source data type
	typedef to_type* (*func_type)(from_type)                  ;
	def(str,     (func_type) (mat_view<T,Mto,Mfrom_ublas_type>),return_value_policy<manage_new_object, with_custodian_and_ward_postcall<1, 0> >());
}

void export_view_simple() {
	def("simple_view",
			(dense_matrix<float,column_major,host_memory_space>*(*)
			 (pyublas::numpy_matrix<float,ublas::column_major>))
			mat_view<float,column_major,ublas::column_major>,return_value_policy<manage_new_object, with_custodian_and_ward_postcall<1, 0> >());
}
/*
 * Export dense matrix views for various type combinations
 */
template <class T>
void
export_dense_matrix_views(){
	export_dense_matrix_view<T,column_major,column_major>("view");
	export_dense_matrix_view<T,column_major,row_major>("view_rm");
	export_dense_matrix_view<T,row_major,column_major>("view_cm");
	export_dense_matrix_view<T,row_major,row_major>("view");
}

/*
 * convert a numpy matrix to a device matrix.
 *   TODO: slightly hackish, since space is allocated and deleted later on.
 *   Possible solution: A special constructor for matrix, which leaves it uninitialized.
 */
template<class T, class Mfrom, class Mto>
dense_matrix<T,Mto,dev_memory_space>*
numpy2dev_dense_mat(pyublas::numpy_matrix<T, Mfrom> m){
	dense_matrix<T,Mto,host_memory_space>* from = mat_view<T,Mto,Mfrom>(m);
	dense_matrix<T,Mto,dev_memory_space>* to = new dense_matrix<T,Mto,dev_memory_space>(from->h(),from->w());
	convert(*to,*from);
	delete from;
	return to;
}

/*
 * convert a numpy matrix to a host matrix.
 *   TODO: slightly hackish, since space is allocated and deleted later on.
 *   Possible solution: A special constructor for matrix, which leaves it uninitialized.
 */
template<class T, class Mfrom, class Mto>
dense_matrix<T,Mto,host_memory_space>*
numpy2host_dense_mat(pyublas::numpy_matrix<T, Mfrom> m){
	dense_matrix<T,Mto,host_memory_space>* from = mat_view<T,Mto,Mfrom>(m);
	dense_matrix<T,Mto,host_memory_space>* to = new dense_matrix<T,Mto,host_memory_space>(from->h(), from->w());
	cuv::copy(*to,*from);
	delete from;
	return to;
}

/*
 * convert a dense matrix on the device to a new numpy-matrix
 */
template<class T, class Mfrom, class Mto_ublas, class Mto_cuv>
pyublas::numpy_matrix<T,Mto_ublas>
dev_dense_mat2numpy(dense_matrix<T, Mfrom, dev_memory_space>& m){
	pyublas::numpy_matrix<T,Mto_ublas> to(m.h(),m.w());
	dense_matrix<T,Mto_cuv,host_memory_space>* to_view = mat_view<T,Mto_cuv,Mto_ublas>(to);
	convert(*to_view,m);
	delete to_view;
	return to;
}
/*
 * convert a dense matrix on the host to a new numpy-matrix
 */
template<class T, class Mfrom, class Mto_ublas, class Mto_cuv>
pyublas::numpy_matrix<T,Mto_ublas>
host_dense_mat2numpy(dense_matrix<T, Mfrom, host_memory_space>& m){
	pyublas::numpy_matrix<T,Mto_ublas> to(m.h(),m.w());
	dense_matrix<T,Mto_cuv,host_memory_space>* to_view = mat_view<T,Mto_cuv,Mto_ublas>(to);
	cuv::copy(*to_view,m);
	delete to_view;
	return to;
}

/*
 * export conversion of numpy matrix to device matrix (helper function)
 */
template<class T, class Mfrom, class Mto>
void export_numpy2dev_dense_mat(const char* c){
	typedef typename matrix2ublas_traits<Mfrom>::storage_type Mfrom_ublas_type;
	def(c, numpy2dev_dense_mat<T,Mfrom_ublas_type,Mto>, return_value_policy<manage_new_object>());
}

/*
 * export conversion of numpy matrix to host matrix (helper function)
 */
template<class T, class Mfrom, class Mto>
void export_numpy2host_dense_mat(const char* c){
	typedef typename matrix2ublas_traits<Mfrom>::storage_type Mfrom_ublas_type;
	def(c, numpy2host_dense_mat<T,Mfrom_ublas_type,Mto>, return_value_policy<manage_new_object>());
}

/*
 * export conversion of device matrix to numpy matrix (helper function)
 */
template<class T, class Mfrom, class Mto>
void export_dev_dense_mat2numpy(const char* c){
	typedef typename matrix2ublas_traits<Mto>::storage_type Mto_ublas_type;
	//def(c, dev_dense_mat2numpy<T,Mfrom,Mto_ublas_type,Mto>, return_value_policy<manage_new_object>());
	def(c, dev_dense_mat2numpy<T,Mfrom,Mto_ublas_type,Mto>);
}

/*
 * export conversion of host matrix to numpy matrix (helper function)
 */
template<class T, class Mfrom, class Mto>
void export_host_dense_mat2numpy(const char* c){
	typedef typename matrix2ublas_traits<Mto>::storage_type Mto_ublas_type;
	def(c, host_dense_mat2numpy<T,Mfrom,Mto_ublas_type,Mto>);
}

/*
 * export conversion of numpy matrices to device matrices for various types
 */
template<class T>
void
export_numpy2dev_dense_mats(){
	export_numpy2dev_dense_mat<T,column_major,column_major>("push");
	export_numpy2dev_dense_mat<T,column_major,row_major>("push_rm");
	export_numpy2dev_dense_mat<T,row_major,column_major>("push_cm");
	export_numpy2dev_dense_mat<T,row_major,row_major>("push");
}

/*
 * export conversion of numpy matrices to host matrices for various types
 */
template<class T>
void
export_numpy2host_dense_mats(){
	export_numpy2host_dense_mat<T,column_major,column_major>("push_host");
	export_numpy2host_dense_mat<T,column_major,row_major>("push_host_rm");
	export_numpy2host_dense_mat<T,row_major,column_major>("push_host_cm");
	export_numpy2host_dense_mat<T,row_major,row_major>("push_host");
}

/*
 * export conversion of device matrices to numpy matrices for various types
 */
template<class T>
void
export_dev_dense_mat2numpys(){
	export_dev_dense_mat2numpy<T,column_major,column_major>("pull");
	export_dev_dense_mat2numpy<T,column_major,row_major>("pull_rm");
	export_dev_dense_mat2numpy<T,row_major,column_major>("pull_cm");
	export_dev_dense_mat2numpy<T,row_major,row_major>("pull");
}

template<class T>
void
export_host_dense_mat2numpys(){
	export_host_dense_mat2numpy<T,column_major,column_major>("pull");
	//export_host_dense_mat2numpy<T,column_major,row_major>("pull_rm"); // the use of ``copy'' prohibits these 
	//export_host_dense_mat2numpy<T,row_major,column_major>("pull_cm");
	export_host_dense_mat2numpy<T,row_major,row_major>("pull");
}

/*
 * MAIN export function
 *   calls exporters for various value_types, column/row major combinations etc.
 */
void export_dense_matrix(){
	//export host and device matrices and conversions
	export_dense_matrix<float,column_major,column_major>("cmf");
	export_dense_matrix<float,row_major,row_major>("rmf");

	export_dense_matrix<signed char,column_major,column_major>("cmsc");
	export_dense_matrix<signed char,row_major,row_major>("rmsc");

	export_dense_matrix<unsigned char,column_major,column_major>("cmuc");
	export_dense_matrix<unsigned char,row_major,row_major>("rmuc");

	export_dense_matrix<int,column_major,column_major>("cmi");
	export_dense_matrix<int,row_major,row_major>("rmi");

	export_dense_matrix<unsigned int,column_major,column_major>("cmui");
	export_dense_matrix<unsigned int,row_major,row_major>("rmui");

	// numpy --> host matrix view
	export_dense_matrix_views<int>();
	export_dense_matrix_views<float>();
	export_dense_matrix_views<signed char>();
	export_dense_matrix_views<unsigned char>();
	export_dense_matrix_views<unsigned int>();

	// numpy --> dev matrix
	export_numpy2dev_dense_mats<int>();
	export_numpy2dev_dense_mats<float>();
	export_numpy2dev_dense_mats<signed char>();
	export_numpy2dev_dense_mats<unsigned char>();
	export_numpy2dev_dense_mats<unsigned int>();
	

	// numpy --> host matrix
	export_numpy2host_dense_mats<int>();
	export_numpy2host_dense_mats<float>();
	export_numpy2host_dense_mats<signed char>();
	export_numpy2host_dense_mats<unsigned char>();
	export_numpy2host_dense_mats<unsigned int>();

	// dev matrix --> numpy matrix
	export_dev_dense_mat2numpys<int>();
	export_dev_dense_mat2numpys<float>();
	export_dev_dense_mat2numpys<signed char>();
	export_dev_dense_mat2numpys<unsigned char>();
	export_dev_dense_mat2numpys<unsigned int>();
	
	// host matrix --> numpy matrix
	export_host_dense_mat2numpys<int>();
	export_host_dense_mat2numpys<float>();
	export_host_dense_mat2numpys<signed char>();
	export_host_dense_mat2numpys<unsigned char>();
	export_host_dense_mat2numpys<unsigned int>();

	// dev cm -> dev rm
	export_rm_view<int,dev_memory_space,unsigned int>();
	export_rm_view<float,dev_memory_space,unsigned int>();
	export_rm_view<signed char,dev_memory_space,unsigned int>();
	export_rm_view<unsigned char,dev_memory_space,unsigned int>();
	export_rm_view<unsigned int,dev_memory_space,unsigned int>();

	// dev rm -> dev cm
	export_cm_view<int,dev_memory_space,unsigned int>();
	export_cm_view<float,dev_memory_space,unsigned int>();
	export_cm_view<signed char,dev_memory_space,unsigned int>();
	export_cm_view<unsigned char,dev_memory_space,unsigned int>();
	export_cm_view<unsigned int,dev_memory_space,unsigned int>();
	export_view_simple();
}


