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

	using boost::detail::multi_array::extent_gen;
#ifndef CUV_DONT_CREATE_EXTENTS_OBJ
	namespace{
		extent_gen<0> extents;
	}
#endif

	template<class __value_type, class __memory_space_type, class __memory_layout = row_major, class Tptr=const __value_type*>
	class const_tensor{
		public:
			typedef typename unconst<__value_type>::type value_type;	///< Type of the entries of matrix
			typedef const value_type const_value_type;	///< Type of the entries of matrix
			typedef __memory_layout  memory_layout_type; ///< host or device
			typedef __memory_space_type memory_space_type; ///< C or Fortran storage
			typedef unsigned int index_type;       ///< the type of the tensor indices
			typedef linear_memory<value_type,memory_space_type,Tptr, index_type> linear_memory_type;  ///< the type of the underlying memory container
			typedef typename linear_memory_type::reference_type reference_type;       ///< the type of the references returned by access operator
			typedef typename linear_memory_type::const_reference_type const_reference_type;
			typedef typename linear_memory_type::pointer_type pointer_type;  ///< type of stored pointer, could be const or not-const value_type*
		protected:
			std::vector<index_type> m_shape;                                 ///< the shape of the tensor (size of dimensions)
			linear_memory_type m_data;   ///< the data of the tensor

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
		public:

			const_tensor(){
			}
			/**
			 * construct tensor using extents object
			 */
			template<unsigned long D>
			explicit const_tensor(const extent_gen<D>& eg){
				m_shape.clear();
				m_shape.reserve(D);
				for(unsigned long i=0;i<D;i++)
					m_shape.push_back(eg.ranges_[i].finish());
				allocate();
			}

			/**
			 * construct tensor view using extents object and a pointer to the wrappable memory
			 */
			template<unsigned long D>
			explicit const_tensor(const extent_gen<D>& eg, pointer_type ptr){
				m_shape.clear();
				m_shape.reserve(D);
				for(unsigned long i=0;i<D;i++)
					m_shape.push_back(eg.ranges_[i].finish());
				allocate(ptr);
			}

			/**
			 * construct tensor using uint
			 */
			explicit const_tensor(const unsigned int& _size){
				m_shape.clear();
				m_shape.push_back(_size);
				allocate();
			}

			/**
			 * construct tensor using int
			 */
			explicit const_tensor(const int& _size){
				m_shape.clear();
				m_shape.push_back(_size);
				allocate();
			}
			/**
			 * construct tensor using some collection
			 */
			template<class Collection>
			explicit const_tensor(const Collection& eg){
				m_shape.clear();
				for(typename Collection::iterator it=eg.begin();it!=eg.end();++it)
					m_shape.push_back(*it);
				allocate();
			}

			/**
			 * returns the index in linear memory of a point 
			 */
			template<unsigned long D>
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
				index_type arr[1] = {d0};
				index_type idx = index_of<1>(memory_layout_type(),arr);
				return m_data[idx];
			}

			const_reference_type operator()(index_type d0)const{
				index_type arr[1] = {d0};
				index_type idx = index_of<1>(memory_layout_type(),arr);
				return m_data[idx];
			}
			const_reference_type operator()(index_type d0, index_type d1)const{
				index_type arr[2] = {d0,d1};
				index_type idx = index_of<2>(memory_layout_type(),arr);
				return m_data[idx];
			}

			const_reference_type operator()(index_type d0, index_type d1, index_type d2)const{
				index_type arr[3] = {d0,d1};
				index_type idx = index_of<3>(memory_layout_type(),arr);
				return m_data[idx];
			}

			const_reference_type operator()(index_type d0, index_type d1, index_type d2, index_type d3)const{
				index_type arr[4] = {d0,d1,d2,d3};
				index_type idx = index_of<4>(memory_layout_type(),arr);
				return m_data[idx];
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
				return size()*sizeof(value_type);
			}
			/**
			 * @ptr if ptr!=NULL, create a view on this pointer instead of allocating memory
			 */
			void allocate(pointer_type ptr = NULL){ 
				if(ptr==NULL){
					m_data.set_size(size()); 
					return;
				}
				m_data = linear_memory_type(size(),ptr, true);
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
	
	template<class __value_type, class __memory_space_type, class __memory_layout=row_major>
	class tensor
	: public const_tensor<__value_type, __memory_space_type,__memory_layout, __value_type*>
	{
		public:
			typedef const_tensor<__value_type, __memory_space_type, __memory_layout, __value_type*> super_type;
			typedef typename super_type::value_type                      value_type;
			typedef typename super_type::const_value_type          const_value_type;
			typedef typename super_type::memory_space_type        memory_space_type;
			typedef typename super_type::memory_layout_type      memory_layout_type;
			typedef typename super_type::index_type                      index_type;
			typedef typename super_type::reference_type              reference_type;
			typedef typename super_type::const_reference_type  const_reference_type;
			typedef typename super_type::pointer_type                  pointer_type;
			typedef typename super_type::linear_memory_type      linear_memory_type;

			using super_type::m_data;
			using super_type::m_shape;
			using super_type::index_of;
			using super_type::size;
			using super_type::operator[];
			using super_type::operator();

		public:

			tensor(){
			}
			/**
			 * construct tensor using extents object
			 */
			template<unsigned long D>
			explicit tensor(const extent_gen<D>& eg)
			:super_type(eg)
			{
			}

			/**
			 * construct tensor view using extents object and a pointer to the wrappable memory
			 */
			template<unsigned long D>
			explicit tensor(const extent_gen<D>& eg, pointer_type ptr)
			:super_type(eg,ptr)
			{
			}

			/**
			 * construct tensor using some collection
			 */
			template<class Collection>
			explicit tensor(const Collection& eg)
			:super_type(eg)
			{
			}
			tensor&
			operator=(const tensor& o){
				if(this == &o)
					return *this;
				m_shape = o.m_shape;
				m_data  = o.m_data;
				return *this;
			}
                        tensor& operator=(const value_type & f);

			reference_type operator[](index_type d0){
				index_type arr[1] = {d0};
				index_type idx = super_type::template index_of<1>(memory_layout_type(),arr);
				return m_data[idx];
			}

			reference_type operator()(index_type d0){
				index_type arr[1] = {d0};
				index_type idx = super_type::template index_of<1>(memory_layout_type(),arr);
				return m_data[idx];
			}
			reference_type operator()(index_type d0, index_type d1){
				index_type arr[2] = {d0,d1};
				index_type idx = super_type::template index_of<2>(memory_layout_type(),arr);
				return m_data[idx];
			}

			reference_type operator()(index_type d0, index_type d1, index_type d2){
				index_type arr[3] = {d0,d1};
				index_type idx = super_type::template index_of<3>(memory_layout_type(),arr);
				return m_data[idx];
			}

			reference_type operator()(index_type d0, index_type d1, index_type d2, index_type d3){
				index_type arr[4] = {d0,d1,d2,d3};
				index_type idx = super_type::template index_of<4>(memory_layout_type(),arr);
				return m_data[idx];
			}
			
			template<unsigned long D>
			void reshape(const extent_gen<D>& eg){
				unsigned long new_size=1;	
				for(int i=0; i<D; i++)
					new_size *= eg.ranges_[i].finish();

				cuvAssert(size() == new_size);
				m_shape.clear();
				m_shape.reserve(D);
				for(unsigned long i=0;i<D;i++)
					m_shape.push_back(eg.ranges_[i].finish());
			}
	};

      // forward declaration of fill to implement operator= for value_type	
      template<class __value_type, class __memory_space_type, class __memory_layout_type, class S>
      void fill(tensor<__value_type, __memory_space_type, __memory_layout_type>& v, const S& p);

      template<class __value_type, class __memory_space_type, class __memory_layout_type>
      tensor<__value_type, __memory_space_type, __memory_layout_type>& tensor<__value_type, __memory_space_type, __memory_layout_type>::operator=(const typename tensor<__value_type, __memory_space_type, __memory_layout_type>::super_type::value_type & f){
          fill(*this,f);
          return *this;
      }
      
}


#endif /* __TENSOR_HPP__ */
