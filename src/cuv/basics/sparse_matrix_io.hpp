//*LB*
// Copyright (c) 2010, University of Bonn, Institute for Computer Science VI
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 
//  * Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//  * Neither the name of the University of Bonn 
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this software without specific prior written
//    permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//*LE*





/** 
 * @file sparse_matrix_io.hpp
 * @brief serialization of vectors and sparse matrices on host
 * @ingroup basics
 * @author Hannes Schulz
 * @date 2010-03-21
 */
#ifndef __SPARSE_MATRIX_IO_HPP__

#include <boost/serialization/array.hpp>
#include <boost/serialization/vector.hpp> 
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <cuv/basics/dia_matrix.hpp>

namespace cuv{

	/**
	 * Serialize/deserialize a host-vector to/from an archive.
	 *
	 * @param ar the archive
	 * @param v  the vector to serialize
	 * @param version not used
	 */
	template<class Archive, class value_type>
		void serialize(Archive& ar, cuv::tensor<value_type,host_memory_space>& v, const unsigned int version){
			ar & v.m_shape;
			if(!v.ptr())
				v.allocate();
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

