#ifndef __VECTOR_HPP__
#define __VECTOR_HPP__

namespace cuv{

	template<class T>
	struct vector_traits{
		typedef memory_space memory_space_type;
	};

template<class __value_type, class __index_type>
class vector{

	public:
	  typedef __value_type value_type;
	  typedef __index_type index_type;
	  
	protected:
	  value_type* m_ptr;
	  bool        m_is_view;
	  size_t      m_size;
	
	public:
	  /*
	   * Member Access
	   */
	  inline const value_type* ptr()const{ return m_ptr;  }
	  inline       value_type* ptr()     { return m_ptr;  }
	  inline size_t size() const         { return m_size; }
	  inline size_t memsize()       const{ return size() * sizeof(value_type); }
	  /*
	   * Construction
	   */
	  vector(size_t s):m_ptr(NULL),m_is_view(false),m_size(s) { alloc(); }
	  vector(size_t s,value_type* p, bool is_view):m_ptr(p),m_is_view(is_view),m_size(s) {}
	  ~vector(){ dealloc(); }
	  /*
	   * Memory Management
	   */
	  void alloc(){};
	  void dealloc(){};
};

};

#endif
