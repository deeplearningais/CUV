#ifndef __DENSEDENSE_TO_SPARSE_HPP__
#define __DENSEDENSE_TO_SPARSE_HPP__


#include <dense_matrix.hpp>
#include <dia_matrix.hpp>

#define SPARSE_DIA_BLOCK_SIZE 16
#define SPARSE_DIA_BLOCK_SIZE_LEN (2*SPARSE_DIA_BLOCK_SIZE+2)

namespace cuv{

	/**
	 * Dummy Block descriptor on host.
	 * this class is needed ON DEVICE for 
	 *   DIA_Mat := Dense_Mat * Dense_Mat
	 * the dummy host descriptor does nothing and exists such that you can use
	 * the same interface for both device and host matrices.
	 */
	template<class __value_type, class __index_type=unsigned int>
	class host_block_descriptor{
		public:
			typedef __value_type value_type;  ///< type of matrix elements
			typedef __index_type index_type;  ///< type of matrix indices
			typedef dia_matrix<value_type,host_memory_space,index_type> diamat_type; ///< matrix-type associated with blockdescriptor
			/** constructor: does nothing
			 *  @param d unused
			 */
			host_block_descriptor(const diamat_type& d){}
	};

	/**
	 * Block descriptors on device
	 * this class is needed for DIA_Mat = Dense_Mat * Dense_Mat
	 * it stores all blocks of size SPARSE_DIA_BLOCK_SIZE x SPARSE_DIA_BLOCK_SIZE
	 * of a regluar grid where at least one diagonal crosses the block.
	 *
	 * Creating Block-Descriptors can take some time, but this pays off when calculating densedense_to_dia (see below)
	 */
	template<class __value_type, class __index_type=unsigned int>
	class dev_block_descriptor{
		public:
			typedef __value_type value_type; ///< type of matrix elements
			typedef __index_type index_type; ///< type of matrix indices
			typedef dia_matrix<value_type,dev_memory_space,index_type> diamat_type; ///< matrix-type associated with blockdescriptor
		protected:
			/**
			 * One block consists of the index of its upper-left corner and the offsets of all diagonals crossing this block, a Block has Size SPARSE_DIA_BLOCK_SIZE*SPARSE_DIA_BLOCK_SIZE.
			 */
			struct block{
				int              startx;  ///< upper left corner of block
				int				 starty;  ///< upper left corner of block
				int              diag[2*SPARSE_DIA_BLOCK_SIZE];  ///< the offsets of all diagonals crossing the block
			};
			struct block_array{  ///< memory for storing multiple blocks
				int*    ptr;     ///< data (on device)
				int     len;     ///< number of blocks stored in ptr
			} m_blocks;          ///< structure holding the actual data stored in the descriptor
		public:
			/** Create a block descriptor for a DIA matrix.
			 *
			 * @param  d a dia-matrix for which to create the block-descriptor.
			 */
			dev_block_descriptor(const diamat_type& d);
			/// destroy the block descriptor
			~dev_block_descriptor();

			/// @return the internal block structure
			const block_array& blocks()const{return m_blocks;}

			///  @return the number of blocks
			inline int len()const{ return m_blocks.len; }
	};

	/**
	 * DIA_Mat <- Dense_Mat * Dense_Mat_transposed.
	 *
	 * This is one special case for matrix multiplication, where the second
	 * matrix is transposed and only the elements on the diagonals of a DIA matrix
	 * must be computed. The function is needed for the backwards-pass of a
	 * neural network.
	 *
	 * @param C    target matrix
	 * @param Cbd  block descriptor (is not changed, so you can re-use the bd for all matrices with same layout)
	 * @param A    A as in C=A*B'
	 * @param B    B as in C=A*B'
	 * @param factAB the result of A*B is multiplied with this factAB and then added to factC*C
	 * @param factC the result of A*B is multiplied with this factAB and then added to factC*C
	 */
	template<class __dia_type, class __bd_type, class __dense_type >
	void densedense_to_dia(
		   __dia_type&           C,
		   const __bd_type&      Cbd,
		   const __dense_type&   A,
		   const __dense_type&   B,
		   const typename __dia_type::value_type& factAB=1.f,
		   const typename __dia_type::value_type& factC =0.f);

	
} // cuv


#endif
