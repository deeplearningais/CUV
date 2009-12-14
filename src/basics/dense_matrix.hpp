#ifndef __DENSE_MATRIX_HPP__
#define __DENSE_MATRIX_HPP__
#include <vector.hpp>
#include <matrix.hpp>
#include <cuv_general.hpp>

namespace cuv{
	struct memory_layout_tag{};
	struct column_major : public memory_layout_tag{};
	struct row_major    : public memory_layout_tag{};

	template<class __value_type, class __mem_layout, class __index_type>
	class dense_matrix 
	:        public matrix<__value_type, __index_type>{
	  public:
		  typedef __mem_layout                       memory_layout;
		  typedef matrix<__value_type, __index_type> base_type;
		  typedef typename matrix<__value_type, __index_type>::value_type value_type;
		  typedef typename matrix<__value_type, __index_type>::index_type index_type;
		protected:
		  vector<value_type,index_type>* m_vec;
		public:
			inline size_t memsize()       const { cuvAssert(m_vec); return m_vec->memsize(); }
			inline const value_type* ptr()const { cuvAssert(m_vec); return m_vec->ptr(); }
			inline       value_type* ptr()      { cuvAssert(m_vec); return m_vec->ptr(); }
			inline const vector<value_type,index_type>* vec()const { return m_vec; }
			inline       vector<value_type,index_type>* vec()      { return m_vec; }
			virtual void alloc(){};
			virtual void dealloc(){
				if(m_vec)
					delete m_vec;
				m_vec = NULL;
			};
			virtual ~dense_matrix(){ dealloc(); }
			dense_matrix(const index_type& h, const index_type& w)
				: base_type(h,w), m_vec(NULL){ alloc(); }
			dense_matrix(const index_type& h, const index_type& w, vector<value_type,index_type>* p)
				: base_type(h,w), m_vec(p){}

			dense_matrix<value_type, memory_layout, index_type>& 
				operator=(dense_matrix<value_type, memory_layout, index_type>& o){
					if(this == &o)
						return *this;
				  this->dealloc();
					(matrix<value_type,index_type>&) (*this)  = (matrix<value_type,index_type>&) o; // copy width, height
					m_vec   = o.vec();
					o.m_vec = NULL;                // transfer ownership of memory
					return *this;
				}
	};
}

#endif /* __DENSE_MATRIX_HPP__ */
