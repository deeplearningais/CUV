/** 
 * @file vector.hpp
 * @brief base class for vectors
 * @ingroup basics
 * @author Hannes Schulz
 * @date 2010-03-21
 */
#ifndef __VECTOR_HPP__
#define __VECTOR_HPP__

namespace cuv{

/** Parent struct that host/device traits inherit from
 */
	template<class T>
	struct vector_traits{
		typedef memory_space memory_space_type;
	};
	template<class V, class I>
	class host_vector;

/**
 * @brief Basic vector class
 *
 * This vector class is the parent of all other vector classes and has all the basic attributes that all matrices share.
 * This class is never actually instanciated.
 */
template<class __value_type, class __index_type>
class vector{
	public:
	  typedef __value_type value_type;	 ///< Type of the entries of matrix
	  typedef __index_type index_type;	 ///< Type of indices
	  template <class Archive, class V, class I> friend void serialize(Archive&, host_vector<V,I>&, unsigned int) ;
	  
	protected:
	  value_type* m_ptr;
	  bool        m_is_view;
	  size_t      m_size;
	
	public:
	  /*
	   * Member Access
	   */
	  /** 
	   * @brief Return pointer to matrix entries
	   */
	  inline const value_type* ptr()const{ return m_ptr;  }	
	  /** 
	   * @brief Return pointer to matrix entries
	   */
	  inline       value_type* ptr()     { return m_ptr;  }
	  /** 
	   * @brief Return length of vector
	   */
	  inline size_t size() const         { return m_size; }
	  /** 
	   * @brief Return size of vector in memory
	   */
	  inline size_t memsize()       const{ return size() * sizeof(value_type); } 
	  /*
	   * Construction
	   */
	  /** 
	   * @brief Empty constructor. Creates empty vector (allocates no memory)
	   */
	  vector():m_ptr(NULL),m_is_view(false),m_size(0) {} 
	  /** 
	   * @brief Creates vector of lenght s and allocates memory
	   * 
	   * @param s Length of vector
	   */
	  vector(size_t s):m_ptr(NULL),m_is_view(false),m_size(s) { alloc(); }
	  /** 
	   * @brief Creates vector from pointer to entries.
	   * 
	   * @param s Length of vector
	   * @param p Pointer to entries 
	   * @param is_view If true will not take responsibility of memory at p. Otherwise will dealloc p on destruction.
	   */
	  vector(size_t s,value_type* p, bool is_view):m_ptr(p),m_is_view(is_view),m_size(s) {
		  alloc();
	  }
	  /** 
	   * @brief Deallocate memory if is_view is false.
	   */
	  virtual ~vector(){ dealloc(); } 
	  /*
	   * Memory Management
	   */
	  /** 
	   * @brief Does nothing
	   */
	  virtual void alloc(){}; 
	  /** 
	   * @brief Does nothing
	   */
	  virtual void dealloc(){}; 

		/** 
		 * @brief Assignment operator. Assigns memory belonging to source to destination and sets source memory pointer to NULL (if source is not a view)
		 * 
		 * @param o Source matrix
		 * 
		 * @return Matrix of same size and type of o that now owns vector of entries of o.
		 *
		 * If source vector is a view, the returned vector is a view, too.
		 */
	  vector<value_type,index_type>& 
		  operator=(const vector<value_type,index_type>& o){
			  if(this==&o) return *this;
			  this->dealloc();
			  this->m_ptr = o.m_ptr;
			  this->m_is_view = o.m_is_view;
			  this->m_size = o.m_size;
			  if(!m_is_view){
				  // transfer ownership of memory (!)
				  (const_cast< vector<value_type,index_type> *>(&o))->m_ptr = NULL;
			  }
			  return *this;
		  }
};

};

#endif
