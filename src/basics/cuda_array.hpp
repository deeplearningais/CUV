#ifndef __CUDA_ARRAY_HPP__
#define __CUDA_ARRAY_HPP__

#include <basics/dense_matrix.hpp>

class cudaArray; //< forward declaration of cudaArray so we do not need to include cuda headers here
namespace cuv
{
	/** 
	 * @brief Wrapper for a 2D CUDAArray
	 */
	template<class __value_type, class __memory_space_type, class __index_type = unsigned int >
	class cuda_array 
	:        public matrix<__value_type, __index_type>{
		public:
		  typedef __memory_space_type								  memory_space_type;///< Indicates whether matrix resides on host or device
		  typedef matrix<__value_type, __index_type>				  base_type;		///< Basic matrix type
		  typedef typename base_type::value_type 					  value_type; 		///< Type of matrix entries
		  typedef typename base_type::index_type 					  index_type;		///< Type of indices
		  typedef cuda_array<value_type,memory_space_type,index_type>  my_type;	        ///< Type of this object
		  using base_type::m_width;
		  using base_type::m_height;

		private:
		  cudaArray*                  m_ptr;   //< data storage in cudaArray

		public:
		  /**
		   * Construct uninitialized memory with given width/height
		   */
		  cuda_array(const index_type& height, const index_type& width)
			  :base_type(height, width)
			  ,m_ptr(NULL)
		  {
			  alloc();
		  }
		  template<class S>
		  cuda_array(const dense_matrix<value_type,row_major,S,index_type>& src) //< construct by copying a dense matrix
		  : base_type(src.h(), src.w())
		  , m_ptr(NULL)
		  {
			  alloc();
			  assign(src);
		  }
		  ~cuda_array(){ //< when destroying, delete associated memory
			  dealloc();
		  }
		  inline index_type w()const{return m_width;}
		  inline index_type h()const{return m_height;}
		  inline index_type n()const{return m_width*m_height;}
		  inline       cudaArray* ptr()      {return m_ptr;}
		  inline const cudaArray* ptr() const{return m_ptr;}
		  void alloc();
		  void dealloc();
		  void assign(const dense_matrix<__value_type,row_major,dev_memory_space,__index_type>& src);
		  void assign(const dense_matrix<__value_type,row_major,host_memory_space,__index_type>& src);
		  __value_type operator()(const __index_type& i, const __index_type& j)const;
	};
}

#endif /* __CUDA_ARRAY_HPP__ */
