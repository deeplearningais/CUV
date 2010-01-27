#ifndef __HOST_VECTOR_HPP__
#define __HOST_VECTOR_HPP__

#include <tools/cuv_general.hpp>
#include "vector.hpp"

namespace cuv{

template<class __value_type, class __index_type=unsigned int>
class host_vector
:    public vector<__value_type, __index_type>
{
	public:
		typedef __value_type                       value_type;
		typedef __index_type                       index_type;
		typedef vector<__value_type, __index_type> base_type;
		typedef host_memory_space                  memspace_type;
		using base_type::m_ptr;
		using base_type::m_is_view;
	public:
		/*
		 * Member access
		 */
		inline value_type operator[](const index_type& idx)const{ return m_ptr[idx]; }
		//inline const value_type& operator[](const index_type& idx)const{ return m_ptr[idx]; }
		host_vector()
			:   base_type(0) {}
		host_vector(size_t s)
			:   base_type(s) { alloc(); }
		host_vector(size_t s, value_type* p, bool is_view)
			:   base_type(s,p,is_view) { } // do not alloc!
		~host_vector(){ dealloc(); }
		/*
		 * Memory Management
		 */
		virtual void alloc(){
			m_ptr = new value_type[this->size()];
		}
		virtual void dealloc(){
			if(m_ptr && !m_is_view){
				delete[] m_ptr;
			}
			m_ptr = NULL;
		}

		void set(const index_type& i, const value_type& val) { m_ptr[i] = val; }
};

template<class V, class I>
struct vector_traits<host_vector<V,I> >{
	typedef dev_memory_space memory_space_type;
};

} // cuv

#endif
