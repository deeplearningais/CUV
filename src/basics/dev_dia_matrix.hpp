/** 
 * @file dev_dia_matrix.hpp
 * @brief sparse matrix in DIA format on device
 * @ingroup basics
 * @author Hannes Schulz
 * @date 2010-03-21
 */
#ifndef __DEV_SPARSE_MATRIX_HPP__
#define __DEV_SPARSE_MATRIX_HPP__

#include "dev_vector.hpp"
#include "sparse_matrix.hpp"

namespace cuv{
	/** 
	 * @brief Class for diagonal device (=GPU memory) matrices
	 */
	template<class __value_type, class __index_type=unsigned int>
	class dev_dia_matrix
	:	public dia_matrix<__value_type, __index_type, dev_vector<__value_type, __index_type>, dev_vector<int> >{
		public:
		  typedef dev_vector<__value_type, __index_type>          vec_type;					///< Basic vector type of matrix
		  typedef dev_vector<int>                                 intvec_type;				///< Type of offsets for diagonals
		  typedef dia_matrix<__value_type, __index_type, vec_type, intvec_type> base_type;	///< Basic matrix type
		  typedef dev_dia_matrix<value_type,index_type>            my_type;					///< Type of this matix
		  using typename base_type::value_type;
		  using typename base_type::index_type;
		  using base_type::m_row_fact;
		protected:
		public:
		  ~dev_dia_matrix(){} 	///< Calls destructor of parent class to deallocate memory
		  dev_dia_matrix(){}	///< Calls default constructor of parent class
			/** 
			 * @brief Creates diagonal matrix of given size, with given number of diagonals and stride.
			 * 
			 * @param h Height of matrix 
			 * @param w Width of matrix
			 * @param num_dia Number of diagonals in matrix
			 * @param stride Stride of matrix
			 * @param row_fact Steepness of diagonals. Only 1 is supported at the moment.
			 */
		  dev_dia_matrix(const index_type& h, const index_type& w, const int& num_dia, const int& stride, const int& row_fact=1)
			  :base_type(h,w,num_dia,stride,row_fact)
		  {
		  }
			/** 
			 * @brief Assignment operator. Assigns vector belonging to source to destination and sets source vector to NULL
			 * 
			 * @param o Source matrix
			 * 
			 * @return Matrix of same size and type of o that now owns vector of entries of o.
			 */
		  dev_dia_matrix<value_type,index_type>& 
			  operator=(const dev_dia_matrix<value_type,index_type>& o){
				  if(this==&o) return *this;
				  this->dealloc();
				  (base_type&) (*this)  = (base_type&) o; 
				   // transfer ownership of memory (!)
				  (const_cast< my_type *>(&o))->m_vec = NULL;
				  return *this;
			  }
	};
};

#include <iostream>
namespace std{
	template<class T, class I>
	/** 
	 * @brief Return stream containing matrix entries for debugging
	 * 
	 * @param o Output stream
	 * @param w2 Matrix to output
	 */
	ostream& 
	operator<<(ostream& o, const cuv::dev_dia_matrix<T,I>& w2){
		cout << "Device-Dia-Matrix: "<<endl;
		for(I i=0;i<w2.h();i++){
			for(I j=0;j<w2.w();j++){
				o << w2(i,j) << " ";
			}
			o << endl;
		}
		o << endl;
		return o;
	}
}

#endif

