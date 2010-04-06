/** 
 * @file matrix.hpp
 * @brief general base class for 2D matrices
 * @ingroup basics
 * @author Hannes Schulz
 * @date 2010-03-21
 */
#ifndef __MATRIX_HPP__
#define __MATRIX_HPP__
namespace cuv{
/** 
 * @brief Trait that indicates whether device or host memory is used.
 */
	template<class T>
	struct matrix_traits{
		typedef memory_space memory_space_type;  ///< Trait for memory type (host/device)

	};
	template<class V, class I>
	class host_dia_matrix;

/**
 * @brief Basic matrix class
 *
 * This matrix class is the parent of all other matrix classes and has all the basic attributes that all matrices share.
 * This class is never actually instanciated.
 */
template<class __value_type, class __index_type>
class matrix
	{
	  public:
		  typedef __value_type value_type;	///< Type of the entries of matrix
		  typedef __index_type index_type;	///< Type of indices
		  template <class Archive, class V, class I> friend void serialize(Archive&, host_dia_matrix<V,I>&, unsigned int) ; ///< serialization function to save matrix
	  protected:
		  index_type m_width; ///< Width of matrix
		  index_type m_height; ///< Heigth of matrix
		public:
		  /** 
		   * @brief Basic constructor: set width and height of matrix but do not allocate any memory
		   * 
		   * @param h Height of matrix
		   * @param w Width of matrix
		   */
		  matrix(const index_type& h, const index_type& w) 
			: m_width(w), m_height(h)
			{
			}
		  /** 
		   * @brief Resizing matrix: changing width and height without changing memory layout
		   * 
		   * @param h New height of matrix 
		   * @param w New width of matrix
		   */
		  inline void resize(const index_type& h, const index_type& w) 
		  {
			  m_width=w;
			  m_height=h;
		  }
		  inline index_type w()const  { return m_width;                } ///< Return matrix width
		  inline index_type h()const  { return m_height;               } ///< Return matrix height
		  inline index_type n()const  { return w()*h();                } ///< Return number of entries in matrix
	};
}

#endif /* __MATRIX_HPP__ */
