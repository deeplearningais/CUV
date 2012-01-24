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

#include <boost/serialization/base_object.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/shared_ptr.hpp>
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
		 * @addtogroup io 
		 *
		 * \section sec_io Loading and saving of CUV data structures
		 *
		 * File and string I/O is implemented using boost::serialization.
		 * Nothing surprising here.
		 *
		 * Example usage:
		 * @code 
		 * tensor<...> m, n; // ...
		 *
		 * {
		 *   std::ofstream os("test.dat");
		 *   boost::archive::binary_oarchive oa(os);
		 *   oa << m;
		 * }
		 * {
		 *   std::ifstream is("test.dat");
		 *   boost::archive::binary_iarchive ia(is);
		 *   ia >> n;
		 * }
		 * @endcode
		 *
		 * @warning This probably will not work in .cu files which are processed by nvcc.
		 *
		 * @{
		 */

        /**
         * load a memory
         */
        template<class Archive, class V, class M>
            void load(Archive& ar, cuv::memory<V,M>& m, const unsigned int version ){
                typename cuv::memory<V,M>::size_type size;
                ar >> size;
                if(size){
                    V* tmp = new V[size];
                    ar >> make_array(tmp,size);
                    // copy to dev
                    cuv::allocator<V,unsigned int,M> a;
                    V* tmpo;
                    a.alloc(&tmpo, sizeof(V)*size);
                    a.copy(tmpo,tmp,size, cuv::host_memory_space());
                    m.reset(tmpo, size);
                    delete[] tmp;
                }
            }
        /**
         * save a memory
         */
        template<class Archive, class V, class M>
            void save(Archive& ar, const cuv::memory<V,M>& m, const unsigned int version ){
                // copy to host
                typename cuv::memory<V,M>::size_type size = m.size();
                ar << size;
                if(m.size()){
                    V* tmp = new V[size];
                    cuv::allocator<V,unsigned int,cuv::host_memory_space> a;
                    a.copy(tmp,m.ptr(),size,M());
                    ar << make_array(tmp, size);
                    delete[] tmp;
                }
            }
		/**
		 * load/save dev memory (dispatch to load/save)
		 *
		 * @param ar an archive to save to
		 * @param m a memory to be stored
		 * @param version (unused) protocol version
		 */
		template<class Archive, class V, class M>
		void serialize(Archive& ar, cuv::memory<V,M>& m, const unsigned int version){
			boost::serialization::split_free(ar, m, version);
		}

        /**
         * @overload
         */
        template<class Archive, class V>
            void load(Archive& ar, cuv::pitched_memory<V,cuv::dev_memory_space>& m_, const unsigned int version){
                ar >> boost::serialization::base_object<cuv::memory<V,cuv::dev_memory_space> >(m_);

                cuv::pitched_memory<V,cuv::host_memory_space> m;
                ar >>  m.m_rows
                    >> m.m_cols
                    >> m.m_pitch;
                m_ = m;
            }
        /**
         * @overload
         */
        template<class Archive, class V>
            void save(Archive& ar, const cuv::pitched_memory<V,cuv::dev_memory_space>& m_, const unsigned int version){
                //ar << boost::serialization::base_object<cuv::memory<V,cuv::dev_memory_space> >(m_);
                cuv::pitched_memory<V,cuv::host_memory_space> m(m_);
                ar  << m.m_rows
                    << m.m_cols
                    << m.m_pitch;
            }

		/**
		 * load/save dev pitched memory (dispatch to load/save)
		 *
		 * @param ar an archive to save to
		 * @param m a memory to be stored
		 * @param version (unused) protocol version
		 */
		template<class Archive, class V>
		void serialize(Archive& ar, cuv::pitched_memory<V,cuv::dev_memory_space>& m, const unsigned int version){
			boost::serialization::split_free(ar, m, version);
		}
        

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
		template<class Archive, class V>
		void save(Archive& ar, const cuv::linear_memory<V,cuv::host_memory_space>& m, const unsigned int version){
            ar << boost::serialization::base_object<cuv::memory<V,cuv::host_memory_space> >(m);
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
		template<class Archive, class V>
		void load(Archive& ar, cuv::linear_memory<V,cuv::host_memory_space>& m, const unsigned int version){
            ar >> boost::serialization::base_object<cuv::memory<V,cuv::host_memory_space> >(m);
			unsigned int size;
			ar >> size;
			if(size>0){
				m = cuv::linear_memory<V,cuv::host_memory_space>(size);
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
		template<class Archive, class V>
		void save(Archive& ar, const cuv::linear_memory<V,cuv::dev_memory_space>& m, const unsigned int version){
			unsigned int size = m.size();
			ar << size;
			if(size>0){
				cuv::linear_memory<V,cuv::host_memory_space> mh(m);
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
		template<class Archive, class V>
		void load(Archive& ar, cuv::linear_memory<V,cuv::dev_memory_space>& m, const unsigned int version){
			unsigned int size;
			ar >> size;
			if(size>0){
				cuv::linear_memory<V,cuv::host_memory_space> mh(size);
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
		template<class Archive, class V, class MS>
		void serialize(Archive& ar, cuv::linear_memory<V,MS>& m, const unsigned int version){
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
		template<class Archive, class V, class MS,class ML>
		void save(Archive& ar, const cuv::tensor<V,MS, ML>& t, const unsigned int version){
			ar <<  t.info().host_shape
                << t.info().host_stride;
            ar << t.mem();
			if(t.ndim()>0){
                long int i = (long int)t.ptr() - (long int)t.mem()->ptr();
                ar << i;
			}
		}
		/**
		 * deserialize a tensor
		 *
		 * @param ar an archive to save to
		 * @param t a memory to be restored
		 * @param version (unused) protocol version
		 */
		template<class Archive, class V, class MS, class ML>
		void load(Archive& ar, cuv::tensor<V,MS, ML>& t, const unsigned int version){
			ar >> t.info().host_shape;
			ar >> t.info().host_stride;
            ar >> t.mem();
			if(t.ndim()>0){
                long int i;
                ar >> i;
                t.set_ptr_offset(i);
			}
		}
		/**
		 * load/save tensor (dispatch to load/save)
		 *
		 * @param ar an archive to save to
		 * @param t a memory to be stored/restored
		 * @param version (unused) protocol version
		 */
		template<class Archive, class V, class MS, class ML>
		void serialize(Archive& ar, cuv::tensor<V,MS, ML>& t, const unsigned int version){
			boost::serialization::split_free(ar, t, version);
		}

		/** @} */
	}
}
