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
 * @file tensor.hpp
 * @brief general base class for n-dimensional matrices
 * @ingroup basics
 * @author Hannes Schulz
 * @date 2011-03-29
 */
#ifndef __TENSOR_HPP__
#define __TENSOR_HPP__

#include <vector>
#include <numeric>
#include <boost/multi_array/extent_gen.hpp>
#include <boost/multi_array/index_gen.hpp>
#include <cuv/tools/cuv_general.hpp>
#include <cuv/tools/meta_programming.hpp>
#include <cuv/basics/linear_memory.hpp>
#include <cuv/basics/reference.hpp>

namespace cuv
{
	/// Parent struct for row and column major tags
	struct memory_layout_tag{};
	/// Tag for column major matrices
	struct column_major : public memory_layout_tag{};
	/// Tag for row major matrices
	struct row_major    : public memory_layout_tag{};

	/// converts from column to row-major and vice versa
	template<class T>
	struct other_memory_layout{};
	/// specialisation: converts from column to row-major 
	template<>
	struct other_memory_layout<column_major>{ typedef row_major type; };
	/// specialisation: converts from row to column-major 
	template<>
	struct other_memory_layout<row_major>{ typedef column_major type; };

	/// converts from dev to host memory space and vice versa
	template<class T>
	struct other_memory_space{
	};
	/// specialisation: converts from dev_memory_space to host_memory_space
	template<>
	struct other_memory_space<dev_memory_space>{ typedef host_memory_space type; };
	/// specialisation: converts from host_memory_space to dev_memory_space
	template<>
	struct other_memory_space<host_memory_space>{ typedef dev_memory_space type; };


	using boost::detail::multi_array::extent_gen;
	using boost::detail::multi_array::index_gen;
	typedef boost::detail::multi_array::index_range<boost::detail::multi_array::index,boost::detail::multi_array::size_type> index_range;
#ifndef CUV_DONT_CREATE_EXTENTS_OBJ
	namespace{
		extent_gen<0> extents;
		index_gen<0,0> indices;
	}
#endif

	/**
	 * an n-dimensional tensor with only non-changing accessors
	 */
	template<class __value_type, class __memory_space_type, class __memory_layout_type = row_major, class Tptr=const __value_type*, class __mem_container=linear_memory_tag>
	class const_tensor{
		public:
			typedef unsigned int index_type;
			typedef typename unconst<__value_type>::type value_type;	///< Type of the entries of matrix
			typedef const value_type const_value_type;	///< Type of the entries of matrix
			typedef __memory_layout_type  memory_layout_type; ///< host or device
			typedef __memory_space_type memory_space_type; ///< C or Fortran storage
			typedef typename memory_traits<__value_type,__memory_space_type,Tptr,index_type,__mem_container>::type  memory_container_type;     ///< the thing that allocates our storage
			typedef typename memory_container_type::reference_type reference_type;       ///< the type of the references returned by access operator
			typedef typename memory_container_type::const_reference_type const_reference_type;
			typedef typename memory_container_type::pointer_type pointer_type;  ///< type of stored pointer, could be const or not-const value_type*
		protected:
			std::vector<index_type> m_shape;                                 ///< the shape of the tensor (size of dimensions for accessors)
			index_type              m_pitch;                                 ///< pitch of array in bytes
			memory_container_type m_data;   ///< the data of the tensor

			template<int D>
			index_type
			index_of(column_major,index_type* arr)const{
				index_type pos = 0;
				index_type dim = 1;
				for(unsigned int i=0; i<D; i++){
					index_type temp = arr[i];
					pos += temp * dim;
					dim *= m_shape[i];
				}
				return pos;
			}

			template<int D>
			index_type
			index_of(row_major,index_type* arr)const{
				index_type pos = 0;
				index_type dim = 1;
				for(int i=0; i<D; i++){
					index_type temp = arr[D-i-1];
					pos += temp * dim;
					dim *= m_shape[D-i-1];
				}
				return pos;
			}

			inline bool inner_is_last(){
				return IsSame<memory_layout_type,row_major>::Result::value;
			}
		public:

			/// default constructor
			const_tensor(){
			}
			/**
			 * construct tensor using extents object
			 */
			template<std::size_t D>
			explicit const_tensor(const extent_gen<D>& eg){
				m_shape.reserve(D);
				for(std::size_t i=0;i<D;i++)
					m_shape.push_back(eg.ranges_[i].finish());
				allocate();
			}

			/**
			 * construct tensor view using extents object and a pointer to the wrappable memory
			 *
			 * @deprecated you should use a constructor which knows about the spatial layout of the ptr
			 *
			 * @param eg   determines shape of new tensor
			 * @param ptr  determines start of data in memory
			 */
			template<int D, int E>
			explicit const_tensor(const index_gen<D,E>& eg, pointer_type ptr){
				m_shape.reserve(D);
				for(std::size_t i=0;i<D;i++)
					m_shape.push_back(eg.ranges_[i].finish());
				m_data.set_view(m_pitch,(index_type)0,m_shape, ptr, inner_is_last());
			}

			/**
			 * construct tensor view using extents object and a pointer to the wrappable memory
			 */
			explicit const_tensor(const std::vector<index_type> eg, pointer_type ptr)
				:m_shape(eg)
			{
				m_data.set_view(m_pitch,(index_type)0, m_shape, ptr, inner_is_last());
			}

			/**
			 * construct tensor view using int
			 */
			const_tensor(int _size, pointer_type ptr){
				m_shape.push_back(_size);
				m_data.set_view(m_pitch,(index_type)0, m_shape, ptr, inner_is_last());
			}

			/**
			 * construct tensor view using uint
			 */
			const_tensor(unsigned int _size, pointer_type ptr){
				m_shape.push_back(_size);
				m_data.set_view(m_pitch,(index_type)0, m_shape, ptr, inner_is_last());
			}

			/**
			 * construct tensor using uint
			 */
			const_tensor(unsigned int _size){
				m_shape.push_back(_size);
				allocate();
			}

			/**
			 * construct tensor using vector of sizes
			 */
			const_tensor(const std::vector<index_type>& _size)
                            : m_shape(_size)
                        {
				//m_shape.push_back(_size);
				allocate();
			}

			/**
			 * construct tensor using int
			 */
			explicit const_tensor(const int& _size){
				m_shape.push_back(_size);
				allocate();
			}

			/**
			 * Copy constructor
			 */
			const_tensor(const const_tensor& o)
			: m_shape(o.shape()),
			  m_data(o.data())
			{
			}

			/**
			 * Copy constructor
			 * also accepts assignment from other memoryspace type
			 * and convertible pointers
			 */
			template<class P, class OM, class OL, class OA>
			const_tensor(const const_tensor<__value_type,OM,OL,P,OA>& o)
			:m_shape(o.shape()),
			 m_data(o.data())
			{
				if(! IsSame<OL,__memory_layout_type>::Result::value)
					std::reverse(m_shape.begin(),m_shape.end());
			}
			template<int D, int E, class P, class OM, class OL, class OA>
			explicit const_tensor(const index_gen<D,E>& eg, const const_tensor<__value_type,OM,OL,P,OA>& o){
				m_shape.reserve(D);
				for(std::size_t i=0;i<D;i++){
					if(eg.ranges_[i].finish()-eg.ranges_[i].start()<=1)
						continue;
					m_shape.push_back(eg.ranges_[i].finish()-eg.ranges_[i].start());
				}

				index_type offset = o.index_of(eg);

				if(! IsSame<OL,__memory_layout_type>::Result::value)
					std::reverse(m_shape.begin(),m_shape.end());
				m_data.set_view(m_pitch,offset,m_shape,o.data(),inner_is_last());
			}

			/**
			 * Assignment operator
			 */
			const_tensor& operator=(const const_tensor& o){
				if(&o ==this)
					return *this;
				m_shape=o.m_shape;
				m_data.assign(m_pitch,o.shape(),o.data(), inner_is_last());
				return *this;
			}

			/**
			 * Assignment operator
			 * also accepts assignment from other memoryspace type
			 * and convertible pointers
			 */
			template<class P, class OM, class OL, class OA>
			const_tensor& operator=(const const_tensor<value_type,OM,OL,P,OA>& o){
				m_shape = o.shape();
				m_data.assign(m_pitch,o.shape(),o.data(),inner_is_last());
				if(! IsSame<OL,__memory_layout_type>::Result::value)
					std::reverse(m_shape.begin(),m_shape.end());
				return *this;
			}

			/**
			 * construct tensor using some collection
			 */
			//template<class Collection>
			//explicit const_tensor(const Collection& eg){
			//        m_shape.clear();
			//        for(typename Collection::iterator it=eg.begin();it!=eg.end();++it)
			//                m_shape.push_back(*it);
			//        allocate();
			//}

			/**
			 * returns the index in linear memory of start point of index-range
			 */
			template<int D, int E>
			index_type
			index_of(const index_gen<D,E>& eg)const{
				index_type point[D];
				for(unsigned int i=0;i<D;i++)
					point[i]=eg.ranges_[i].start();

				return index_of<D>(memory_layout_type(),&point[0]);
			}
			/**
			 * returns the index in linear memory of a point 
			 */
			template<std::size_t D>
			index_type
			index_of(const extent_gen<D>& eg)const{
				index_type point[D];
				for(unsigned int i=0;i<D;i++)
					point[i]=eg.ranges_[i].finish();

				return index_of<D>(memory_layout_type(),&point[0]);
			}

			/**
			 * return a reference to the value at this position in linear memory
			 */
			const_reference_type operator[](index_type d0)const{
				//index_type arr[1] = {d0};
				//index_type idx = index_of<1>(memory_layout_type(),arr);
				//return m_data[idx];
				return m_data[d0];
			}

			/**
			 * return a reference to the value at this position in linear memory
			 */
			const_reference_type operator()(index_type d0)const{
				//index_type arr[1] = {d0};
				//index_type idx = index_of<1>(memory_layout_type(),arr);
				//return m_data[idx];
				return m_data[d0];
			}
			/**
			 * return a reference to the value at this position in 2D memory
			 */
			const_reference_type operator()(index_type d0, index_type d1)const{
				index_type arr[2] = {d0,d1};
				index_type idx = index_of<2>(memory_layout_type(),arr);
				return m_data[idx];
			}

			/**
			 * return a reference to the value at this position in 3D memory
			 */
			const_reference_type operator()(index_type d0, index_type d1, index_type d2)const{
				index_type arr[3] = {d0,d1,d2};
				index_type idx = index_of<3>(memory_layout_type(),arr);
				return m_data[idx];
			}

			/**
			 * return a reference to the value at this position in 4D memory
			 */
			const_reference_type operator()(index_type d0, index_type d1, index_type d2, index_type d3)const{
				index_type arr[4] = {d0,d1,d2,d3};
				index_type idx = index_of<4>(memory_layout_type(),arr);
				return m_data[idx];
			}

			/**
			 * return reference to underlying memory object
			 */
			const memory_container_type& data()const{return m_data;}

			/**
			 * return the number of bytes in the innermost dimension
			 */
			index_type pitch()const{
				return m_pitch;
			}
			/**
			 * return the number of elements stored in this container
			 */
			index_type size()const{
				return std::accumulate(m_shape.begin(),m_shape.end(),(index_type)1,std::multiplies<index_type>());
			}

			/**
			 * return the number of bytes needed by this container
			 */
			size_t memsize()const{
				return m_data.memsize();
			}
			/**
			 * @ptr if ptr!=NULL, create a view on this pointer instead of allocating memory
			 */
			void allocate(){ 
				m_data.set_size(m_pitch,m_shape,inner_is_last()); 
			}
			/**
			 * delete the memory used by this container (calls dealloc of wrapped linear memory)
			 */
			void dealloc(){
				m_data.dealloc();
			}
			/**
			 * returns a vector whose values represent the shape of the tensor
			 */
			const std::vector<index_type>& shape()const{
				return m_shape;
			}
			/**
			 * returns a vector whose values represent the number of dimensions of the tensor
			 */
			index_type ndim()const{
				return m_shape.size();
			}
			/**
			 * Whether we do not own the memory. Delegates to contained linear memory
			 */
			bool is_view ()const{return m_data.is_view();}

			/**
			 * Returns the data pointer
			 */
			const pointer_type ptr()const{
				return m_data.ptr();
			}
			
		
	};
	
	/**
	 * Provides non-const accessors to const_tensor
	 */
	template<class __value_type, class __memory_space_type, class __memory_layout_type=row_major, class __mem_container=linear_memory_tag >
	class tensor
	: public const_tensor<__value_type, __memory_space_type,__memory_layout_type, __value_type*, __mem_container>
	{
		public:
			typedef const_tensor<__value_type, __memory_space_type, __memory_layout_type, __value_type*,__mem_container> super_type;
			typedef typename super_type::value_type                      value_type;
			typedef typename super_type::const_value_type          const_value_type;
			typedef typename super_type::memory_space_type        memory_space_type;
			typedef typename super_type::memory_layout_type      memory_layout_type;
			typedef typename super_type::index_type                      index_type;
			typedef typename super_type::reference_type              reference_type;
			typedef typename super_type::const_reference_type  const_reference_type;
			typedef typename super_type::pointer_type                  pointer_type;
			typedef typename super_type::memory_container_type memory_container_type;

			using super_type::m_data;
			using super_type::m_shape;
			using super_type::index_of;
			using super_type::size;
			using super_type::operator[];
			using super_type::operator();

		public:
			/// access pointer
			pointer_type ptr(){return m_data.ptr();}
			/// access pointer
			const pointer_type ptr()const{
				return m_data.ptr();
			}

			/// default constructor
			tensor(){
			}
			/**
			 * construct tensor using extents object
			 */
			template<std::size_t D>
			explicit tensor(const extent_gen<D>& eg)
			:super_type(eg)
			{
			}

			/**
			 * construct tensor view using extents object and a pointer to the wrappable memory
			 */
			template<int D, int E>
			explicit tensor(const index_gen<D,E>& eg, pointer_type ptr)
			:super_type(eg,ptr)
			{
			}
			/**
			 * construct a tensor view using index range and another tensor
			 */
			template<int D, int E, class OM, class OL, class OA>
			explicit tensor(const index_gen<D,E>& eg, const tensor<__value_type,OM,OL,OA>& o)
			:super_type(eg,o)
			{
			}
			/**
			 * construct tensor view using extents object and a pointer to the wrappable memory
			 */
			explicit tensor(const std::vector<index_type> eg, pointer_type ptr)
			:super_type(eg,ptr)
			{
			}

			/**
			 * construct tensor using only length
			 */
			explicit tensor(const unsigned int& len)
			:super_type(len)
			{
			}
			/**
			 * construct tensor using only height and with
			 */
			explicit tensor(const unsigned int& h, const unsigned int& w)
			:super_type(extents[h][w])
			{
			}

			/**
			 * construct tensor view using only length
			 */
			explicit tensor(const unsigned int& len, pointer_type ptr)
			:super_type(len,ptr)
			{
			}

			/**
			 * copy constructor
			 */
			tensor(const tensor& o)
				:super_type(o)
			{
			}

			/**
			 * copy constructor for other memory spaces
			 */
			template<class OM, class OA>
			tensor(const tensor<__value_type,OM,__memory_layout_type, OA>& o)
				:super_type(o)
			{
			}

			/**
			 * construct tensor using vector of sizes
			 */
			tensor(const std::vector<index_type>& _size)
                            : super_type(_size)
                        {
			}


			/**
			 * assignment operator
			 */
			tensor&
			operator=(const tensor& o){
				if(this == &o)
					return *this;
				super_type::operator=(o);
				return *this;
			}
			/**
			 * assignment operator for other memory spaces
			 */
			template<class OM, class OL, class OA>
			tensor&
			operator=(const tensor<__value_type, OM, OL, OA>& o){
				//if(this == &o)   // is different type, anyway.
				//        return *this;
				super_type::operator=(o);
				return *this;
			}

			/**
			 * assignment operator for scalars
			 */
                        tensor& operator=(const __value_type & f);

			/**
			 * returns a reference to this position in linear memory
			 */
			reference_type operator[](index_type d0){
				//index_type arr[1] = {d0};
				//index_type idx = super_type::template index_of<1>(memory_layout_type(),arr);
				//return m_data[idx];
				return m_data[d0];
			}

			/**
			 * returns a reference to this position in linear memory
			 */
			reference_type operator()(index_type d0){
				//index_type arr[1] = {d0};
				//index_type idx = super_type::template index_of<1>(memory_layout_type(),arr);
				//return m_data[idx];
				return m_data[d0];
			}
			/**
			 * returns a reference to this position in 2D memory
			 */
			reference_type operator()(index_type d0, index_type d1){
				index_type arr[2] = {d0,d1};
				index_type idx = super_type::template index_of<2>(memory_layout_type(),arr);
				return m_data[idx];
			}

			/**
			 * returns a reference to this position in 3D memory
			 */
			reference_type operator()(index_type d0, index_type d1, index_type d2){
				index_type arr[3] = {d0,d1,d2};
				index_type idx = super_type::template index_of<3>(memory_layout_type(),arr);
				return m_data[idx];
			}

			/**
			 * returns a reference to this position in 4D memory
			 */
			reference_type operator()(index_type d0, index_type d1, index_type d2, index_type d3){
				index_type arr[4] = {d0,d1,d2,d3};
				index_type idx = super_type::template index_of<4>(memory_layout_type(),arr);
				return m_data[idx];
			}
			
			/**
			 * change the shape of this tensor (product must be the same as before)
			 */
			template<std::size_t D>
			void reshape(const extent_gen<D>& eg){
				std::size_t new_size=1;	
				for(int i=0; i<D; i++)
					new_size *= eg.ranges_[i].finish();

				cuvAssert(size() == new_size);
				m_shape.clear();
				m_shape.reserve(D);
				for(std::size_t i=0;i<D;i++)
					m_shape.push_back(eg.ranges_[i].finish());
			}
			/**
			 * convenience overloading for matrices: change the shape of this tensor (product must be the same as before)
			 */
                        void reshape(index_type i, index_type j){
                                reshape(extents[i][j]);
                        }
			/**
			 * change the shape of this tensor (product must be the same as before)
			 */
                        void reshape(const std::vector<index_type>& new_shape){
				index_type new_size =  std::accumulate(new_shape.begin(),new_shape.end(),(index_type)1,std::multiplies<index_type>());
                                cuvAssert(new_size == size() );
                                m_shape = new_shape;
                        }
	};

      // forward declaration of fill to implement operator= for value_type	
      template<class __value_type, class __memory_space_type, class __memory_layout_type, class S>
      void fill(tensor<__value_type, __memory_space_type, __memory_layout_type>& v, const S& p);

      // forward declaration of fill to implement operator= for tensor type
      template<class __value_type, class __memory_space_type, class __memory_layout_type, class S>
      void copy(tensor<__value_type, __memory_space_type, __memory_layout_type>& v,
	 const  tensor<__value_type, __memory_space_type, __memory_layout_type>& w);

      template<class __value_type, class __memory_space_type, class __memory_layout_type, class A>
      tensor<__value_type, __memory_space_type, __memory_layout_type,A>& 
      tensor<__value_type, __memory_space_type, __memory_layout_type,A>::operator=(const __value_type & f){
          fill(*this,f);
          return *this;
      }
      
      /// create a tensor type with the same template parameters, but with switched value type
	template<class Mat, class NewVT>
		struct switch_value_type{
			typedef tensor<NewVT, typename Mat::memory_space_type, typename Mat::memory_layout_type> type;
		};
      /// create a tensor type with the same template parameters, but with switched memory_layout_type
	template<class Mat, class NewML>
		struct switch_memory_layout_type{
			typedef tensor<typename Mat::value_type, typename Mat::memory_space_type, NewML> type;
		};
      /// create a tensor type with the same template parameters, but with switched memory_space_type
	template<class Mat, class NewMS>
		struct switch_memory_space_type{
			typedef tensor<typename Mat::value_type, NewMS, typename Mat::memory_layout_type> type;
		};

	/**
	 * @brief returns true if both tensors have the same shape, regardless of their memory space and/or value type.
	 *
	 * @param s   the first tensor
	 * @param t   the second tensor
	 *
	 */
	template<class V, class M, class L,class P, class A, class V2, class M2, class P2, class A2>
	bool equal_shape(const const_tensor<V,M,L,P,A>& s, const const_tensor<V2,M2,L,P2,A2>& t){
		if(s.ndim()!=t.ndim())
			return false;
		typename std::vector<typename const_tensor<V ,M ,L,P >::index_type>::const_iterator it1=s.shape().begin(), end=s.shape().end();
		typename std::vector<typename const_tensor<V2,M2,L,P2>::index_type>::const_iterator it2=t.shape().begin();
		for(; it1!=end; it1++,it2++){
			if(*it1 != *it2)
				return false;
		}
		return true;
	}
}

#include <iostream>
namespace std{
	template<class V, class M, class T>
	/** 
	 * @brief Write matrix entries to stream
	 * 
	 * @param o Output stream
	 * @param w2 Matrix to output
	 */
	ostream& 
	operator<<(ostream& o, const cuv::tensor<V,M,T>& w2){
		cout << "shape=[";
                typedef typename cuv::tensor<V,M,T>::index_type I;
		typename std::vector<I>::const_iterator it=w2.shape().begin(), end=w2.shape().end();
		o <<*it;
		for(it++; it!=end; it++)
			o <<", "<<*it;
		o<<"]";
		return o;
	}
}

#endif /* __TENSOR_HPP__ */
