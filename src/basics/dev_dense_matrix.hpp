#ifndef __D_DENSE_MATRIX_H__
#define __D_DENSE_MATRIX_H__
#include <dense_matrix.hpp>
#include <dev_vector.hpp>

namespace cuv{
	template<class __value_type, class __mem_layout=cuv::column_major, class __index_type=unsigned int>
	class dev_dense_matrix
	:        public dense_matrix<__value_type, __mem_layout, __index_type>{
		public:
		  typedef __mem_layout        memory_layout;
		  typedef dense_matrix<__value_type, __mem_layout, __index_type>        base_type;
		  typedef typename matrix<__value_type,__index_type>::value_type        value_type;
		  typedef typename matrix<__value_type,__index_type>::index_type        index_type;
		  using matrix<__value_type, __index_type>::m_width;
		  using matrix<__value_type, __index_type>::m_height;
		  using base_type::m_vec;
		public:
		  template<class V, class I>
		  dev_dense_matrix(const matrix<V,I>* m)
		  :  base_type(m->h(),m->w())
		  { 
			  this->alloc(); 
		  }

		  dev_dense_matrix(const index_type& h, const index_type& w)
			:  base_type(h,w) { alloc(); }

		  dev_dense_matrix(const index_type& h, const index_type& w, vector<value_type,index_type>* p)
			:  base_type(h,w,p) { alloc(); }

		  dev_dense_matrix<value_type,memory_layout,index_type>& 
		  operator=(dev_dense_matrix<value_type,memory_layout,index_type>& o){
			  if(this==&o) return *this;
			  (dense_matrix<value_type,memory_layout,index_type>&) (*this)  = (dense_matrix<value_type,memory_layout,index_type>&) o; // copy width, height
			  return *this;
		  }
		  virtual void alloc(){   m_vec = new dev_vector<value_type,index_type>(this->n());}

		  inline const dev_vector<value_type,index_type>* vec()const { return (dev_vector<value_type,index_type>*)m_vec; }
		  inline       dev_vector<value_type,index_type>* vec()      { return (dev_vector<value_type,index_type>*)m_vec; }
	};
}

#endif

