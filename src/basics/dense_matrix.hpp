/** 
 * @file dense_matrix.hpp
 * @brief base class for dence matrices
 * @ingroup basics
 * @author Hannes Schulz
 * @date 2010-03-21
 */
#ifndef __DENSE_MATRIX_HPP__
#define __DENSE_MATRIX_HPP__
#include <basics/vector.hpp>
#include <basics/matrix.hpp>
#include <tools/cuv_general.hpp>

namespace cuv{
	struct memory_layout_tag{};
	struct column_major : public memory_layout_tag{}; ///< Trait for column major matrices
	struct row_major    : public memory_layout_tag{}; ///< Trait for row major matrices

	/** 
	 * @brief Parent class for dense matrices
	 */
	template<class __value_type, class __mem_layout, class __vec_type, class __index_type>
	class dense_matrix 
	:        public matrix<__value_type, __index_type>{
	  public:
		  typedef __mem_layout                       					  memory_layout; ///< Memory layout type: column_major or row_major
		  typedef matrix<__value_type, __index_type>					  base_type; ///< Basic matrix type
		  typedef typename base_type::value_type value_type; ///< Type of matrix entries
		  typedef typename base_type::index_type index_type; ///< Type of indices
		  typedef __vec_type       			          					  vec_type; ///< Basic vector type used
		  using base_type::m_width;
		  using base_type::m_height;
		  vec_type* m_vec;                      ///< stores the actual data 
		protected:
		public:
			virtual ~dense_matrix(){} ///< Destructor
			/** 
			 * @brief Constructor for host matrices that creates a new vector and allocates memory
			 * 
			 * @param h Height of matrix
			 * @param w Width of matrix
			 *
			 * Creates a new vector to store matrix entries.
			 */
			dense_matrix(const index_type& h, const index_type& w)
				: base_type(h,w),m_vec(NULL) {
					alloc();
				}
			/** 
			 * @brief Constructor for dense matrices that creates a matrix from a given host pointer
			 * 
			 * @param h Height of matrix
			 * @param w Weight of matrix 
			 * @param p Pointer to matrix entries
			 * @param is_view If true will not take responsibility of memory at p. Otherwise will dealloc p on destruction.
			 */
			dense_matrix(const index_type& h, const index_type& w, value_type* p, bool is_view = false)
				:	base_type(h,w) 
			{
				m_vec = new vec_type(h*w,p,is_view);
			}

			void dealloc() ///< Deallocate matrix entries. This calls deallocation of the vector storing entries.
			{
				if(m_vec)
					delete m_vec;
				m_vec = NULL;
			}

			void alloc() ///< Allocate matrix entries: Create vector to store entries.
			{
				cuvAssert(!m_vec);
				m_vec = new vec_type(m_width * m_height);
			}
	};
}

#endif /* __DENSE_MATRIX_HPP__ */
