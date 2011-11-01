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
 * @file io.hpp
 * @brief serialization for tensors
 * @ingroup basics
 * @author Hannes Schulz
 * @date 2011-06-27
 */

#include <boost/serialization/vector.hpp>
#include <boost/serialization/array.hpp>
#include <boost/serialization/split_free.hpp>
#include <cuv/basics/tensor.hpp>

/// cuv additions to the boost namespace
namespace boost
{
	/// serialization functions for cuv data structures
	namespace serialization
	{
		/**
		 * @addtogroup io input and output functions
		 * @{
		 */

		/****************************************************
		 * serialize linear memory
		 ****************************************************/
		/**
		 * save host linear memory
		 *
		 * @param ar an archive to save to
		 * @param m a memory to be stored
		 * @param version (unused) protocol version
		 */
		template<class Archive, class V, class TPtr, class I>
		void save(Archive& ar, const cuv::linear_memory<V,cuv::host_memory_space,TPtr,I>& m, const unsigned int version){
			unsigned int size = m.size();
			ar << size;
			if(size>0)
				ar << make_array(m.ptr(), m.size());
		}
		/**
		 * load host linear memory
		 *
		 * @param ar an archive to save to
		 * @param m a memory to be stored
		 * @param version (unused) protocol version
		 */
		template<class Archive, class V, class TPtr, class I>
		void load(Archive& ar, cuv::linear_memory<V,cuv::host_memory_space,TPtr, I>& m, const unsigned int version){
			unsigned int size;
			ar >> size;
			if(size>0){
				m = cuv::linear_memory<V,cuv::host_memory_space,TPtr, I>(size);
				ar >> make_array(m.ptr(), m.size());
			}else{
				m.dealloc();
			}
		}
		/**
		 * save device linear memory
		 *
		 * @param ar an archive to save to
		 * @param m a memory to be stored
		 * @param version (unused) protocol version
		 */
		template<class Archive, class V, class TPtr, class I>
		void save(Archive& ar, const cuv::linear_memory<V,cuv::dev_memory_space,TPtr, I>& m, const unsigned int version){
			unsigned int size = m.size();
			ar << size;
			if(size>0){
				cuv::linear_memory<V,cuv::host_memory_space,TPtr,I> mh(m);
				ar << make_array(mh.ptr(), mh.size());
			}
		}
		/**
		 * load device linear memory
		 *
		 * @param ar an archive to save to
		 * @param m a memory to be stored
		 * @param version (unused) protocol version
		 */
		template<class Archive, class V, class TPtr, class I>
		void load(Archive& ar, cuv::linear_memory<V,cuv::dev_memory_space,TPtr, I>& m, const unsigned int version){
			unsigned int size;
			ar >> size;
			if(size>0){
				cuv::linear_memory<V,cuv::host_memory_space,TPtr,I> mh(size);
				ar >> make_array(mh.ptr(), mh.size());
				m = mh;
			}else{
				m.dealloc();
			}
		}
		/**
		 * load/save linear memory (dispatch to load/save)
		 *
		 * @param ar an archive to save to
		 * @param m a memory to be stored
		 * @param version (unused) protocol version
		 */
		template<class Archive, class V, class MS, class TPtr, class I>
		void serialize(Archive& ar, cuv::linear_memory<V,MS,TPtr, I>& m, const unsigned int version){
			boost::serialization::split_free(ar, m, version);
		}


		/****************************************************
		 * serialize tensor
		 ****************************************************/
		/**
		 * serialize a tensor
		 *
		 * @param ar an archive to save to
		 * @param t a memory to be stored
		 * @param version (unused) protocol version
		 */
		template<class Archive, class V, class MS,class ML, class MC>
		void save(Archive& ar, const cuv::tensor<V,MS, ML, MC>& t, const unsigned int version){
			ar << t.shape();
			if(t.ndim()>0){
				unsigned int pitch = t.pitch();
				ar << pitch;
				ar << t.data();
			}
		}
		/**
		 * deserialize a tensor
		 *
		 * @param ar an archive to save to
		 * @param t a memory to be restored
		 * @param version (unused) protocol version
		 */
		template<class Archive, class V, class MS, class ML, class MC>
		void load(Archive& ar, cuv::tensor<V,MS, ML, MC>& t, const unsigned int version){
			std::vector<unsigned int> shape;
			ar >> shape;
			if(shape.size()>0){
				t = cuv::tensor<V,cuv::host_memory_space, ML, MC>(shape);
				unsigned int pitch;
				ar >> pitch;
				t.set_pitch(pitch);
				ar >> t.data();
			}else{
				t.dealloc();
			}
		}
		/**
		 * load/save tensor (dispatch to load/save)
		 *
		 * @param ar an archive to save to
		 * @param t a memory to be stored/restored
		 * @param version (unused) protocol version
		 */
		template<class Archive, class V, class MS, class ML, class MC>
		void serialize(Archive& ar, cuv::tensor<V,MS, ML, MC>& t, const unsigned int version){
			boost::serialization::split_free(ar, t, version);
		}

		/** @} */
	}
}
