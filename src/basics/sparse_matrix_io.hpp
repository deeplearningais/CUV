/** 
 * @file sparse_matrix_io.hpp
 * @brief serialization of vectors and sparse matrices on host
 * @ingroup basics
 * @author Hannes Schulz
 * @date 2010-03-21
 */
#ifndef __SPARSE_MATRIX_IO_HPP__

#include <boost/serialization/array.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include "dia_matrix.hpp"

namespace cuv{

	/**
	 * Serialize/deserialize a host-vector to/from an archive.
	 *
	 * @param ar the archive
	 * @param v  the vector to serialize
	 * @param version not used
	 */
	template<class Archive, class value_type, class index_type>
		void serialize(Archive& ar, cuv::vector<value_type,host_memory_space,index_type>& v, const unsigned int version){
			ar & v.m_size;
			if(!v.ptr())
				v.alloc();
			ar & boost::serialization::make_array(v.ptr(),v.size());
		}

	/**
	 * Serialize/deserialize a host-dia-matrix to/from an archive.
	 *
	 * @param ar the archive
	 * @param m  the dia-matrix to serialize
	 * @param version not used
	 */
	template<class Archive, class value_type, class index_type>
		void serialize(Archive& ar, cuv::dia_matrix<value_type,host_memory_space,index_type>& m, const unsigned int version){
			ar & m.m_width;
			ar & m.m_height;
			ar & m.m_num_dia;
			ar & m.m_stride;
			ar & m.m_row_fact;
			ar & m.m_offsets;
			if(!m.vec_ptr()){
				m.alloc();
				m.post_update_offsets();
			}
			ar & m.vec();
		}
	/**
	 * explicit instantiation of serialization for dia-matrices in binary oarchives
	 */
	template
		void serialize(boost::archive::binary_oarchive&, dia_matrix<float, host_memory_space, unsigned int>&, unsigned int);
}

#endif

