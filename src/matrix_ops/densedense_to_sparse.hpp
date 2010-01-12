#ifndef __DENSEDENSE_TO_SPARSE_HPP__
#define __DENSEDENSE_TO_SPARSE_HPP__


#include <dev_dense_matrix.hpp>
#include <dev_dia_matrix.hpp>
#include <host_dia_matrix.hpp>

#define SPARSE_DIA_BLOCK_SIZE 16
#define SPARSE_DIA_BLOCK_SIZE_LEN (2*SPARSE_DIA_BLOCK_SIZE+2)

namespace cuv{

	/*
	 * Block descriptors on device
	 * this class is needed for DIA_Mat = Dense_Mat * Dense_Mat
	 * it stores all blocks of size SPARSE_DIA_BLOCK_SIZE x SPARSE_DIA_BLOCK_SIZE
	 * of a regluar grid where at least one diagonal crosses the block
	 */
	template<class __value_type, class __index_type=unsigned int>
	class host_block_descriptor{
		public:
			typedef __value_type value_type;
			typedef __index_type index_type;
			typedef host_dia_matrix<value_type,index_type> diamat_type;
			host_block_descriptor(const diamat_type&){}
	};
	template<class __value_type, class __index_type=unsigned int>
	class dev_block_descriptor{
		public:
			typedef __value_type value_type;
			typedef __index_type index_type;
			typedef dev_dia_matrix<value_type,index_type> diamat_type;
		protected:
			struct block{
				int              startx,starty;
				int              diag[2*SPARSE_DIA_BLOCK_SIZE];
			};
			struct block_array{ 
				int*    ptr; 
				int     len; 
			} m_blocks;
		public:
			dev_block_descriptor(const diamat_type&);
			~dev_block_descriptor();

			const block_array& blocks()const{return m_blocks;}
	};

	/*
	 * DIA_Mat <- Dense_Mat * Dense_Mat_transposed
	 */
	template<class __dia_type, class __bd_type, class __dense_type >
	void densedense_to_dia(
		   __dia_type&,
		   const __bd_type&,
		   const __dense_type&,
		   const __dense_type&);

	
} // cuv


#endif
