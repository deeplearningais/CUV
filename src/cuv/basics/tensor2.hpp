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

#ifndef __TENSOR2_HPP__
#     define __TENSOR2_HPP__

#include <iostream>
#include <vector>
#include <stdexcept>
#include <numeric>
#include <boost/shared_ptr.hpp>
#include <boost/multi_array/extent_gen.hpp>
#include <boost/multi_array/index_gen.hpp>
#include <cuv/tools/cuv_general.hpp>
#include <cuv/tools/meta_programming.hpp>
#include "reference.hpp"

namespace cuv
{
/**
 * @addtogroup basics Basic datastructures
 * @{
 */

	/// Parent struct for row and column major tags
	struct memory_layout_tag{};
	/// Tag for column major matrices
	struct column_major : public memory_layout_tag{};
	/// Tag for row major matrices
	struct row_major    : public memory_layout_tag{};

    /// Parent struct for linear/pitched/borrowed memory types
    struct memory_type_tag{};
    /// tag for linear memory
    struct linear_memory_tag : public memory_type_tag{};
    /// tag for pitched memory
    struct pitched_memory_tag : public memory_type_tag{};

	using boost::detail::multi_array::extent_gen;
	using boost::detail::multi_array::index_gen;
	/**
	 * defines an index range, stolen from boost::multi_array
	 *
	 * examples:
	 * @code
	 * index_range(1,3)
	 * index(1) <= index_range() < index(3)
	 * @endcode
	 */
	typedef boost::detail::multi_array::index_range<boost::detail::multi_array::index,boost::detail::multi_array::size_type> index_range;
	/**
	 * the index type used in index_range, useful for comparator syntax in @see index_range
	 */
	typedef index_range::index index;
#ifndef CUV_DONT_CREATE_EXTENTS_OBJ
	namespace{
		/**
		 * extents object, can be used to generate a multi-dimensional array conveniently.
		 *
		 * stolen from boost::multi_array.
		 *
		 * Example:
		 * @code
		 * tensor<...> v(extents[5][6][7]); // 3-dimensional tensor
		 * @endcode
		 */
		extent_gen<0> extents;
		/**
		 * indices object, can be used to generate multi-dimensional views conveniently.
		 *
		 * stolen form boost::multi_array.
		 *
		 * Example:
		 * @code
		 * tensor<...> v(indices[index_range(1,3)][index_range()], other_tensor);
		 * @endcode
		 */
		index_gen<0,0> indices;
	}
#endif


    template<class V, class M, class L> class tensor;
    template<class V, class M> class linear_memory;

    /**
     * abstract class providing interface for tensor to different memory
     * allocators
     */
    template<class V, class M>
        class memory{
            public:
                typedef typename unconst<V>::type value_type; ///< type of contained values
                typedef const V const_value_type;   ///< const version of value_type
                typedef M memory_space_type;        ///< host or dev memory_space
                typedef unsigned int size_type;               ///< type of shapes
                typedef int          index_type;              ///< how to index values
                typedef reference<V,M,index_type> reference_type; ///< type of reference you get using operator[]
                typedef const reference<V,M,index_type> const_reference_type; ///< type of reference you get using operator[]
                typedef value_type* pointer_type;
                typedef const_value_type* const_pointer_type;

            protected:
                pointer_type m_ptr;  ///< points to allocated memory
            public:
                /// @return pointer to allocated memory
                pointer_type ptr(){return m_ptr;}
                /// @return pointer to allocated memory (const)
                const_pointer_type ptr()const{return m_ptr;}

                /// default constructor (just sets ptr to NULL)
                memory():m_ptr(NULL){}

                /// set strides for this memory
                virtual void set_strides(
                        linear_memory<index_type,host_memory_space>& strides,
                        const linear_memory<size_type,host_memory_space>& shape, row_major)=0;
                virtual void set_strides(
                        linear_memory<index_type,host_memory_space>& strides,
                        const linear_memory<size_type,host_memory_space>& shape, column_major)=0;

        };

    /**
     * represents contiguous memory
     */
    template<class V, class M>
        class linear_memory
        : public memory<V,M> {
            private:
                typedef memory<V,M> super;
            public:
                typedef typename super::value_type       value_type; ///< type of contained values
                typedef typename super::const_value_type const_value_type;   ///< const version of value_type
                typedef typename super::memory_space_type memory_space_type; ///< host or dev memory_space
                typedef typename super::index_type index_type; ///< how to index values
                typedef typename super::size_type  size_type;       ///< type of shapes
                typedef typename super::reference_type reference_type; ///< type of reference you get using operator[]
                typedef typename super::const_reference_type const_reference_type; ///< type of reference you get using operator[]
            private:
                typedef linear_memory<V,M> my_type; ///< my own type
                allocator<value_type, size_type, memory_space_type> m_allocator; ///< how stored memory was allocated
                size_type m_size; ///< number of stored elements
                using super::m_ptr;
            public:


                /// @return number of stored elements
                size_type size()const{    return m_size; }

                /// @return number of stored bytes
                size_type memsize()const{ return m_size*sizeof(V); }

                /// default constructor: does nothing
                linear_memory():m_size(0){}

                /** constructor: reserves space for i elements
                 *  @param i number of elements
                 */
                linear_memory(size_type i):m_size(i){alloc();}

                /** sets the size (reallocates if necessary)
                 */
                void set_size(size_type s){
                    if(s!=size()){
                        dealloc();
                        m_size = s;
                        alloc();
                    }
                }

                /**
                 * allocate space according to size()   
                 */
                void alloc(){
                    assert(this->m_ptr == NULL);
                    m_allocator.alloc( &m_ptr,m_size);
                }

                /**
                 * dellocate space
                 */
                void dealloc(){
                    if (m_ptr)
                        m_allocator.dealloc(&this->m_ptr);
                    m_ptr=NULL;
                }

                /** 
                 * @brief Copy linear_memory.
                 * 
                 * @param o Source linear_memory
                 * 
                 * @return *this
                 *
                 */
                my_type& 
                    operator=(const my_type& o){
                        if(this == &o)
                            return *this;

                        if(this->size() != o.size()){
                            this->dealloc();
                            m_size = o.size();
                            this->alloc();
                        }
                        m_allocator.copy(this->m_ptr,o.ptr(),size(),memory_space_type());

                        return *this;
                    }

                /** 
                 * @overload
                 *
                 * @brief Copy linear_memory from other memory type.
                 * 
                 * @param o Source linear_memory
                 * 
                 * @return *this
                 *
                 */
                template<class OM>
                    my_type& 
                    operator=(const linear_memory<value_type, OM>& o){
                        if(this->size() != o.size()){
                            this->dealloc();
                            m_size = o.size();
                            this->alloc();
                        }
                        m_allocator.copy(m_ptr,o.ptr(),size(),OM());
                        return *this;
                    }

                /**
                 * construct from other linear memory
                 */
                linear_memory(const my_type& o){
                    operator=(o);
                }

                /**
                 * construct from other linear memory
                 */
                template<class OM>
                linear_memory(const linear_memory<V,OM>& o){
                    operator=(o);
                }

                /**
                 * @return a reference to memory at a position
                 * @param idx position
                 */
                reference_type
                    operator[](const index_type& idx)    
                    {
                        assert(idx>=0);
                        assert((size_type)idx<m_size);
                        return reference_type(this->m_ptr+idx);
                    }

                /**
                 * @overload
                 * 
                 * @return a reference to memory at a position
                 * @param idx position
                 */
                const_reference_type
                    operator[](const index_type& idx)const
                    {
                        assert(idx>=0);
                        assert((size_type)idx<m_size);
                        return const_reference_type(this->m_ptr+idx);
                    }
                
                /// deallocates memory
                ~linear_memory(){ dealloc(); }
                
                /// set strides for this memory
                virtual void set_strides(
                        linear_memory<index_type,host_memory_space>& strides,
                        const linear_memory<size_type,host_memory_space>& shape, row_major){
                    size_type size = 1;
                    for (int i = shape.size()-1; i >= 0; --i)
                    {
                        strides[i] = (shape[i] == 1) ? 0 : size;
                        size *= shape[i];
                    }
                }
                /// set strides for this memory
                virtual void set_strides(
                        linear_memory<index_type,host_memory_space>& strides,
                        const linear_memory<size_type,host_memory_space>& shape, column_major){
                    size_type size = 1;
                    for (int i = 0; i <  shape.size(); ++i)
                    {
                        strides[i] = (shape[i] == 1) ? 0 : size;
                        size *= shape[i];
                    }
                }
                /** reverse the array (for transposing etc)
                 * 
                 * currently only enabled for host memory space arrays
                 */
                typename boost::enable_if_c<IsSame<host_memory_space,memory_space_type>::Result::value>::type
                    reverse(){
                    value_type* __first = m_ptr, *__last = m_ptr + size();
                    while (true)
                        if (__first == __last || __first == --__last)
                            return;
                        else
                        {
                            std::iter_swap(__first, __last);
                            ++__first;
                        }
                }

        };

    namespace detail{

        /// true iff there are no "holes" in memory
        bool is_c_contiguous(row_major, const linear_memory<unsigned int,host_memory_space>& shape, const linear_memory<int,host_memory_space>& stride){
            bool c_contiguous = true;
            int size = 1;
            for (int i = shape.size()-1; (i >= 0) && c_contiguous; --i)
            {
                if (shape[i] == 1)
                    continue;
                if (stride[i] != size)
                    c_contiguous = false;
                size = size * shape[i];
            }
            return c_contiguous;
        }

        /// true iff there are no "holes" in memory
        bool is_c_contiguous(column_major, const linear_memory<unsigned int,host_memory_space>& shape, const linear_memory<int,host_memory_space>& stride){
            bool c_contiguous = true;
            int size = 1;
            for (int i = 0; i<shape.size() && c_contiguous; ++i)
            {
                if (shape[i] == 1)
                    continue;
                if (stride[i] != size)
                    c_contiguous = false;
                size = size * shape[i];
            }
            return c_contiguous;
        }

        /// returns true iff memory can be copied using copy2d
        bool is_2dcopyable(row_major, const linear_memory<unsigned int,host_memory_space>& shape, const linear_memory<int,host_memory_space>& stride){
            bool c_contiguous = shape.size()>1;
            const unsigned int pitched_dim = shape.size()-2;
            int size = 1;
            for (int i = shape.size()-1; (i >= 0) && c_contiguous; --i)
            {
                if(shape[i] == 1){
                    continue;
                }else if(i == pitched_dim){
                    size *= stride[i];
                }else if(stride[i] != size) {
                    c_contiguous = false;
                }else{
                    size *= shape[i];
                }
            }
            return c_contiguous;
        }

        /// returns true iff memory can be copied using copy2d
        bool is_2dcopyable(column_major, const linear_memory<unsigned int,host_memory_space>& shape, const linear_memory<int,host_memory_space>& stride){
            bool c_contiguous = shape.size()>1;
            const unsigned int pitched_dim = 1;
            int size = 1;
            for (int i = 0; (i <  shape.size()) && c_contiguous; ++i)
            {
                if(shape[i] == 1){
                    continue;
                }else if(i == pitched_dim){
                    size *= stride[i];
                }else if(stride[i] != size) {
                    c_contiguous = false;
                }else{
                    size *= shape[i];
                }
            }
            return c_contiguous;
        }

        /**
         * this is intended for copying pitched memory.
         *
         * given shape, stride and a memory layout, we can determine the number of
         * rows, columns and the pitch of a 
         */
        template<class index_type, class size_type>
        void get_pitched_params(size_type& rows, size_type& cols, size_type& pitch,
                const linear_memory<size_type,host_memory_space>& shape,
                const linear_memory<index_type,host_memory_space>& stride,
                row_major){
            // strided dimension is the LAST one
            rows  = std::accumulate(shape[0].ptr,
                    shape[0].ptr+shape.size()-1,
                    1, std::multiplies<index_type>());
            cols  = shape[shape.size()-1];
            pitch = stride[shape.size()-2];
        }
        /**
         * @overload
         */
        template<class index_type, class size_type>
        void get_pitched_params(size_type& rows, size_type& cols, size_type& pitch,
                const linear_memory<size_type,host_memory_space>& shape,
                const linear_memory<index_type,host_memory_space>& stride,
                column_major){
            // strided dimension is the FIRST one
            rows = std::accumulate(shape[0].ptr+1,
                    shape[0].ptr+shape.size(),
                    1, std::multiplies<index_type>());
            cols = shape[0];
            pitch = stride[1];
        }
    }


    /**
     * represents non-contiguous (pitched) memory
     */
    template<class V, class M>
        class pitched_memory 
        : public memory<V,M> {
            private:
                typedef memory<V,M> super;
            public:
                typedef typename super::value_type       value_type; ///< type of contained values
                typedef typename super::const_value_type const_value_type;   ///< const version of value_type
                typedef typename super::memory_space_type memory_space_type; ///< host or dev memory_space
                typedef typename super::index_type index_type; ///< how to index values
                typedef typename super::size_type  size_type;       ///< type of shapes
                typedef typename super::reference_type reference_type; ///< type of reference you get using operator[]
                typedef typename super::const_reference_type const_reference_type; ///< type of reference you get using operator[]
            private:
                typedef pitched_memory<V,M> my_type; ///< my own type
                allocator<value_type, size_type, memory_space_type> m_allocator; ///< how stored memory was allocated
                size_type m_rows;  ///< number of rows
                size_type m_cols;  ///< number of columns
                size_type m_pitch;  ///< pitch (multiples of sizeof(V))
                using super::m_ptr;
            public:

                /// @return the number of rows
                size_type rows()const{return m_rows;}
                
                /// @return the number of cols
                size_type cols()const{return m_cols;}
                
                /// @return the number of allocated cols
                size_type pitch()const{return m_pitch;}

                /// @return number of stored elements
                size_type size()const{    return m_rows*m_pitch; }

                /// @return number of stored bytes
                size_type memsize()const{ return size()*sizeof(V); }

                /// default constructor: does nothing
                pitched_memory():m_rows(0),m_cols(0),m_pitch(0){}

                /** constructor: reserves space for at least i*j elements
                 *  @param i number of rows
                 *  @param j minimum number of elements per row 
                 */
                pitched_memory(index_type i, index_type j)
                    :m_rows(i),m_cols(j),m_pitch(0){alloc();}

                /**
                 * allocate space according to size()   
                 */
                void alloc(){
                    assert(this->m_ptr == NULL);
                    m_allocator.alloc2d(&this->m_ptr,m_pitch,m_rows,m_cols);
                    assert(m_pitch%sizeof(value_type)==0);
                    m_pitch/=sizeof(value_type);
                }

                /** 
                 * @brief Deallocate memory 
                 */
                void dealloc(){
                    if (this->m_ptr)
                        m_allocator.dealloc(&this->m_ptr);
                    this->m_ptr=NULL;
                }

                /** 
                 * set the size (reallocating, if necessary)
                 * @param rows number of desired rows
                 * @param cols number of desired columns
                 */
                void set_size(size_type rows, size_type cols){
                    if(        cols>m_pitch
                            || rows>m_rows
                            ){
                                dealloc();
                                m_rows = rows;
                                m_cols = cols;
                                alloc();
                            }else{
                                m_rows = rows;
                                m_cols = cols;
                            }
                }


                /** 
                 * @brief Copy pitched_memory.
                 * 
                 * @param o Source pitched_memory
                 * 
                 * @return *this
                 *
                 */
                my_type& 
                    operator=(const my_type& o){
                        if(this==&o) return *this;

                        if(        m_pitch < o.m_cols
                                || m_rows  < o.m_rows
                                ){
                            this->dealloc();
                            m_cols = o.m_cols;
                            m_rows = o.m_rows;
                            this->alloc();
                        }
                        m_cols = o.m_cols;
                        m_rows = o.m_rows;
                        m_allocator.copy2d(this->m_ptr,o.ptr(),m_pitch*sizeof(value_type),o.m_pitch*sizeof(value_type),m_rows,m_cols,memory_space_type());

                        return *this;
                    }

                /** 
                 * @overload
                 *
                 * @brief Copy pitched_memory from other memory type.
                 * 
                 * @param o Source linear_memory
                 * 
                 * @return *this
                 *
                 */
                template<class OM>
                    my_type& 
                    operator=(const pitched_memory<value_type, OM>& o){
                        if(        m_pitch < o.m_cols
                                || m_rows  < o.m_rows
                                ){
                            this->dealloc();
                            m_cols = o.m_cols;
                            m_rows = o.m_rows;
                            this->alloc();
                        }
                        m_cols = o.m_cols;
                        m_rows = o.m_rows;
                        m_allocator.copy2d(this->m_ptr,o.ptr(),m_pitch*sizeof(value_type),o.m_pitch*sizeof(value_type),m_rows,m_cols,OM());
                        return *this;
                    }

                /**
                 * @return a reference to memory at a position as if this were linear memory
                 * @param idx position
                 */
                reference_type
                    operator[](const index_type& idx)    
                    {
                        assert(idx>=0);
                        index_type row = idx/m_cols;
                        index_type col = idx%m_cols;
                        assert((size_type)row < m_rows);
                        assert((size_type)col < m_cols);
                        return reference_type(this->m_ptr+row*m_pitch+col);
                    }

                /**
                 * @overload
                 * 
                 * @return a reference to memory at a position
                 * @param idx position
                 */
                const_reference_type
                    operator[](const index_type& idx)const
                    {
                        assert(idx>=0);
                        index_type row = idx/m_cols;
                        index_type col = idx%m_cols;
                        assert((size_type)row < m_rows);
                        assert((size_type)col < m_cols);
                        return const_reference_type(this->m_ptr+row*m_pitch+col);
                    }

                reference_type
                    operator()(const index_type& i, const index_type& j){
                        assert(i>=0);
                        assert(j>=0);
                        assert((size_type)i < m_rows);
                        assert((size_type)j < m_cols);
                        return reference_type(this->m_ptr+i*m_pitch+j);
                    }
                const_reference_type
                    operator()(const index_type& i, const index_type& j)const{
                        assert(i>=0);
                        assert(j>=0);
                        assert((size_type)i < m_rows);
                        assert((size_type)j < m_cols);
                        return const_reference_type(this->m_ptr+i*m_pitch+j);
                    }


                /// set strides for this memory
                virtual void set_strides(
                        linear_memory<index_type,host_memory_space>& strides,
                        const linear_memory<size_type,host_memory_space>& shape, row_major){
                    size_type size = 1;
                    assert(shape.size()>=2);
                    const size_type pitched_dim = shape.size()-2;
                    for (int i = shape.size()-1; i >= 0; --i)
                    {
                        if(shape[i] == 1){
                            strides[i] = 0;
                        }else if(i == pitched_dim){
                            strides[i] = pitch();
                            size *= pitch();
                        }else {
                            strides[i] = size;
                            size *= shape[i];
                        }
                    }
                }
                /// set strides for this memory
                virtual void set_strides(
                        linear_memory<index_type,host_memory_space>& strides,
                        const linear_memory<size_type,host_memory_space>& shape, column_major){
                    size_type size = 1;
                    assert(shape.size()>=2);
                    const size_type pitched_dim = 1;
                    for (int i = 0; i < shape.size(); ++i)
                    {
                        if(shape[i] == 1){
                            strides[i] = 0;
                        }else if(i == pitched_dim){
                            strides[i] = pitch();
                            size *= pitch();
                        }else {
                            strides[i] = size;
                            size *= shape[i];
                        }
                    }
                }
        };

    namespace detail
    {

        /**
         * allocate memory
         */
        template<class V, class M, class L>
            void allocate(tensor<V,M,L>& t,linear_memory_tag){
                linear_memory<V,M>* d  = NULL;
                if(t.m_memory.get()){
                    d = dynamic_cast<linear_memory<V,M>*>(t.m_memory.get());
                    if(d)
                        d->set_size(t.size()); // may get us arround reallocation
                }
                if(!d){ // did not succeed in reusing memory
                    d = new linear_memory<V,M>(t.size());
                    t.m_memory.reset(d);
                }
                d->set_strides(t.m_info.host_stride,t.m_info.host_shape, L());
                t.m_ptr = d->ptr();
            }

        /**
         * allocate memory (pitched)
         */
        template<class V, class M, class L>
            void allocate(tensor<V,M,L>& t,pitched_memory_tag){
                typename tensor<V,M,L>::size_type row,col,pitch;
                detail::get_pitched_params(row,col,pitch,t.m_info.host_shape, t.m_info.host_stride,L());
                pitched_memory<V,M>* d  = NULL;
                if(t.m_memory.get()){
                    d = dynamic_cast<pitched_memory<V,M>*>(t.m_memory.get());
                    if(d)
                        d->set_size(row,col); // may get us arround reallocation
                }
                if(!d){ // did not succeed in reusing memory
                    d = new pitched_memory<V,M>(row,col);
                    t.m_memory.reset(d);
                }
                d->set_strides(t.m_info.host_stride,t.m_info.host_shape, L());
                t.m_ptr = d->ptr();
            }
    }

    template<class M, class L>
    struct tensor_info{
        typedef unsigned int   size_type;        ///< type of shapes of the tensor
        typedef int            index_type;       ///< type of indices in tensor
        typedef M data_memory_space; ///< this is where the data lies
        /// shape stored in host memory
        linear_memory<size_type, host_memory_space> host_shape;
        /// strides stored in host memory
        linear_memory<index_type, host_memory_space> host_stride;

        /// shape stored in data memory
        linear_memory<size_type, data_memory_space> data_shape;
        /// strides stored in data memory
        linear_memory<index_type, data_memory_space> data_stride;

        /// default constructor: does nothing
        tensor_info(){}

        /// @return the size of the arrays (should all be the same)
        size_type size(){ return host_shape.size(); }

        /// construct with known shape
        tensor_info(size_type s){ resize(s); }

        /// resize all memories
        void resize(size_type s){
            host_shape.set_size(s);
            host_stride.set_size(s);
            //data_shape.set_size(s);
            //data_stride.set_size(s);
        }

        /// copy-constructor
        tensor_info(const tensor_info<M,L>& o)
            : host_shape(o.host_shape)
            , host_stride(o.host_stride)
            //, data_shape(o.data_shape)
            //, data_stride(o.data_stride)
        {}

        /// copy-construct from other memory space
        template<class OM>
        tensor_info(const tensor_info<OM,L>& o)
            : host_shape(o.host_shape)
            , host_stride(o.host_stride)
            //, data_shape(o.data_shape)
            //, data_stride(o.data_stride)
        {}

    } ;

    template<class V, class M, class L>
        class tensor;
    namespace detail {
        template<class V, class M0, class M1, class L0, class L1>
            void copy_memory(tensor<V,M0,L0>&, const tensor<V,M1,L1>&, linear_memory_tag);
        template<class V, class M0, class M1, class L0, class L1>
            void copy_memory(tensor<V,M0,L0>&, const tensor<V,M1,L1>&, pitched_memory_tag);
    }


    /**
     * represents an n-dimensional array on GPU or CPU.
     */
    template<class V, class M, class L=row_major>
    class tensor{
        public:
            typedef memory<V,M> memory_type; ///< type of stored memory
            typedef typename memory_type::reference_type reference_type; ///< values returned by operator() and []
            typedef typename memory_type::const_reference_type const_reference_type; ///< values returned by operator()
            typedef typename memory_type::memory_space_type memory_space_type; ///< dev/host
            typedef typename memory_type::value_type value_type; ///< type of stored values
            typedef typename memory_type::size_type size_type; ///< type shapes
            typedef typename memory_type::index_type index_type; ///< type strides
            typedef          L memory_layout_type; ///< column/row major

            typedef tensor_info<M,L> info_type; ///< type of shape info struct

            template<class _V, class M0, class M1, class L0, class L1>
                friend 
                void detail::copy_memory(tensor<_V,M0,L0>&, const tensor<_V,M1,L1>&, linear_memory_tag);
            template<class _V, class M0, class M1, class L0, class L1>
                friend 
                void detail::copy_memory(tensor<_V,M0,L0>&, const tensor<_V,M1,L1>&, pitched_memory_tag);
            template<class _V, class _M, class _L>
            friend void detail::allocate(tensor<_V,_M,_L>&, linear_memory_tag);
            template<class _V, class _M, class _L>
            friend void detail::allocate(tensor<_V,_M,_L>&, pitched_memory_tag);
        private:
            /// information about shape, strides
            info_type  m_info;  

            /// points to (possibly shared) memory
            boost::shared_ptr<memory_type> m_memory;
            
            /// points to start of actually referenced memory (within m_memory)
            V* m_ptr;

            /// determine linear index 
			size_type
			index_of(column_major, int D, index_type* arr)const{
				index_type pos = 0;
				index_type dim = 1;
				for(int i=0; i<D; i++){
					index_type temp = arr[i];
                    if(temp<0) temp = m_info.host_shape[i]+temp;
					pos += temp * dim * m_info.host_stride[i];
					dim *= m_info.host_stride[i];
				}
				return pos;
			}

			index_type
			index_of(row_major,int D, index_type* arr)const{
				index_type pos = 0;
				index_type dim = m_info.host_stride[D-1];
				for(int i=0; i<D; i++){
					index_type temp = arr[D-i-1];
                    if(temp<0) temp = m_info.host_shape[D-i-1]+temp;
					pos += temp * dim;
                    if(0<=D-i-2)
                        dim *= m_info.host_stride[D-i-2];
				}
				return pos;
			}

        public:
            /// return the number of dimensions
            index_type ndim()const{ return m_info.host_shape.size(); }

            /** return the size of the i-th dimension
             *  @param i the index of the queried dimension
             */
            index_type shape(const index_type& i)const{return m_info.host_shape[i];}

            /** return the stride of the i-th dimension
             *  @param i the index of the queried dimension
             */
            index_type stride(const index_type& i)const{return m_info.host_stride[i];}

            /** @return the number of stored elements
             */
            index_type size()const{
                return std::accumulate(m_info.host_shape[0].ptr, m_info.host_shape[0].ptr+m_info.host_shape.size(), 1, std::multiplies<index_type>());
            }

            /// return the shape of the tensor (as a vector for backward compatibility)
            std::vector<index_type> shape()const{
                return std::vector<index_type>(m_info.host_shape[0].ptr, m_info.host_shape[0].ptr+m_info.host_shape.size());
            }

            /// @return the tensor info struct (const)
            const info_type& info()const{return m_info;}

            /// true iff there are no "holes" in memory
            bool is_c_contiguous()const{
                return detail::is_c_contiguous(memory_layout_type(), m_info.host_shape, m_info.host_stride);
            }
            
            /// true iff it can be copied as a 2d array (only one dimension is pitched)
            bool is_2dcopyable()const{
                return detail::is_2dcopyable(memory_layout_type(), m_info.host_shape, m_info.host_stride);
            }

            const_reference_type operator[](index_type idx)const{
                return const_cast<tensor&>(*this)[idx];
            }
            /**
             * member access: "flat" access as if memory was linear
             */
            reference_type operator[](index_type idx){
                size_type pos = 0;
                if(IsSame<L,row_major>::Result::value){
                    // row major
                    int size = 1;
                    for(int i=m_info.host_shape.size()-1; i>=0; --i){
                        size_type s = idx % (m_info.host_shape[i] * m_info.host_stride[i]);
                        pos += s;
                        idx -= s * (m_info.host_shape[i] * m_info.host_stride[i]);
                    }
                }else{
                    // column major
                    for(size_type i=0; i<m_info.host_shape.size(); ++i){
                        size_type s = idx / (m_info.host_shape[i] * m_info.host_stride[i]);
                        pos += s;
                        idx -= s * (m_info.host_shape[i] * m_info.host_stride[i]);
                    }
                }
                return reference_type(m_ptr + pos);
            }

            const_reference_type operator()(index_type i0)const{ return const_cast<tensor&>(*this)(i0); }
            reference_type operator()(index_type i0){
#ifndef NDEBUG
                cuvAssert(ndim()==1);
                cuvAssert((i0>=0 && i0 < shape(0)) || (i0<0 && -i0<shape(0)+1) )
#endif
                if(i0>=0){
                        return reference_type(m_ptr + i0);
                }else{
                        return reference_type(m_ptr + shape(0) - i0);
                }
            }

            const_reference_type operator()(index_type i0, index_type i1)const{ return const_cast<tensor&>(*this)(i0,i1); }
            reference_type operator()(index_type i0, index_type i1){
#ifndef NDEBUG
                cuvAssert(ndim()==2);
                cuvAssert((i0>=0 && i0 < shape(0)) || (i0<0 && -i0<shape(0)+1) )
                cuvAssert((i1>=0 && i1 < shape(1)) || (i1<0 && -i1<shape(1)+1) )
#endif
                index_type arr[2] = {i0,i1};
                return reference_type(m_ptr + index_of(memory_layout_type(), 2,arr));
            }

            const_reference_type operator()(index_type i0, index_type i1, index_type i2)const{ return const_cast<tensor&>(*this)(i0,i1,i2); }
            reference_type operator()(index_type i0, index_type i1, index_type i2){
#ifndef NDEBUG
                cuvAssert(ndim()==3);
                cuvAssert((i0>=0 && i0 < shape(0)) || (i0<0 && -i0<shape(0)+1) )
                cuvAssert((i1>=0 && i1 < shape(1)) || (i1<0 && -i1<shape(1)+1) )
                cuvAssert((i2>=0 && i2 < shape(2)) || (i2<0 && -i2<shape(2)+1) )
#endif
                index_type arr[3] = {i0,i1,i2};
                return reference_type(m_ptr + index_of(memory_layout_type(), 3,arr));
            }

            const_reference_type operator()(index_type i0, index_type i1, index_type i2, index_type i3)const{ return const_cast<tensor&>(*this)(i0,i1,i2,i3); }
            reference_type operator()(index_type i0, index_type i1, index_type i2, index_type i3){
#ifndef NDEBUG
                cuvAssert(ndim()==4);
                cuvAssert((i0>=0 && i0 < shape(0)) || (i0<0 && -i0<shape(0)+1) )
                cuvAssert((i1>=0 && i1 < shape(1)) || (i1<0 && -i1<shape(1)+1) )
                cuvAssert((i2>=0 && i2 < shape(2)) || (i2<0 && -i2<shape(2)+1) )
                cuvAssert((i3>=0 && i3 < shape(3)) || (i3<0 && -i3<shape(3)+1) )
#endif
                index_type arr[4] = {i0,i1,i2,i3};
                return reference_type(m_ptr + index_of(memory_layout_type(), 4,arr));
            }

            const_reference_type operator()(index_type i0, index_type i1, index_type i2, index_type i3, index_type i4)const{ return const_cast<tensor&>(*this)(i0,i1,i2,i3,i4); }
            reference_type operator()(index_type i0, index_type i1, index_type i2, index_type i3, index_type i4){
#ifndef NDEBUG
                cuvAssert(ndim()==5);
                cuvAssert((i0>=0 && i0 < shape(0)) || (i0<0 && -i0<shape(0)+1) )
                cuvAssert((i1>=0 && i1 < shape(1)) || (i1<0 && -i1<shape(1)+1) )
                cuvAssert((i2>=0 && i2 < shape(2)) || (i2<0 && -i2<shape(2)+1) )
                cuvAssert((i3>=0 && i3 < shape(3)) || (i3<0 && -i3<shape(3)+1) )
                cuvAssert((i4>=0 && i4 < shape(4)) || (i4<0 && -i4<shape(4)+1) )
#endif
                index_type arr[5] = {i0,i1,i2,i3,i4};
                return reference_type(m_ptr + index_of(memory_layout_type(), 5,arr));
            }

            /**
             * default constructor (does nothing)
             */
            tensor(){}

            /////////////////////////////////////////////////////////////
            //        Constructing from other tensor
            /////////////////////////////////////////////////////////////

            /**
             * construct tensor from tensor of exact same type
             *
             * time O(1)
             */
            tensor(const tensor& o)
            : m_info(o.m_info)     // copy only shape
            , m_memory(o.m_memory) // increase ref counter
            , m_ptr(o.m_ptr){}     // same pointer in memory

            /**
             * construct tensor from tensor of other memory space
             * in (dense) /linear/ memory. Note: this /copies/ the memory!
             */
            template<class OM>
                tensor(const tensor<V,OM,L>& o)
                :m_info(o.m_info) // primarily to copy shape
                {
                    detail::copy_memory(*this, o, linear_memory_tag());
                    m_ptr = m_memory->ptr();
                }

            /**
             * construct tensor from tensor of same memory space
             * in  /pitched/ memory. Note: this /copies/ the memory!
             */
                tensor(const tensor& o, pitched_memory_tag)
                :m_info(o.m_info) // primarily to copy shape
                {
                    detail::copy_memory(*this, o, pitched_memory_tag());
                    m_ptr = m_memory->ptr();
                }

            /**
             * construct tensor from tensor of other memory space
             * in  /pitched/ memory. Note: this /copies/ the memory!
             */
            template<class OM>
                tensor(const tensor<V,OM,L>& o, pitched_memory_tag)
                :m_info(o.m_info) // primarily to copy shape
                {
                    detail::copy_memory(*this, o, pitched_memory_tag());
                    m_ptr = m_memory->ptr();
                }

            /**
             * construct tensor from tensor of same memory space
             * in (dense) /linear/ memory. Note: this /copies/ the memory!
             */
                tensor(const tensor& o, linear_memory_tag)
                :m_info(o.m_info) // primarily to copy shape
                {
                    detail::copy_memory(*this, o, linear_memory_tag());
                    m_ptr = m_memory->ptr();
                }
            /**
             * construct tensor from tensor of other memory space
             * in (dense) /linear/ memory. Note: this /copies/ the memory!
             */
            template<class OM>
                tensor(const tensor<V,OM,L>& o, linear_memory_tag)
                :m_info(o.m_info) // primarily to copy shape
                {
                    detail::copy_memory(*this, o, linear_memory_tag());
                    m_ptr = m_memory->ptr();
                }
            
            /////////////////////////////////////////////////////////////
            //        Constructing from SHAPE
            /////////////////////////////////////////////////////////////

            /**
             * construct tensor from a shape
             */
			template<std::size_t D>
			explicit tensor(const extent_gen<D>& eg){
				m_info.resize(D);
				for(std::size_t i=0;i<D;i++)
					m_info.host_shape[i] = eg.ranges_[i].finish();
                detail::allocate(*this,linear_memory_tag());
			}

            /**
             * construct tensor from a shape (pitched)
             */
			template<std::size_t D>
			explicit tensor(const extent_gen<D>& eg, pitched_memory_tag){
				m_info.resize(D);
				for(std::size_t i=0;i<D;i++)
					m_info.host_shape[i] = eg.ranges_[i].finish();
                detail::allocate(*this,pitched_memory_tag());
			}
    };

    namespace detail{

        /// copies between different memory spaces
        template<class V, class M0, class M1, class L0, class L1>
            void copy_memory(tensor<V,M0,L0>&dst, const tensor<V,M1,L1>&src, linear_memory_tag){
                typedef typename tensor<V,M0,L0>::size_type size_type;
                linear_memory<V,M0>* d  = NULL;
                if(dst.m_memory.get()){
                    d = dynamic_cast<linear_memory<V,M0>*>(dst.m_memory.get());
                    if(d)
                        d->set_size(src.size()); // may get us arround reallocation
                }
                if(!d){ // did not succeed in reusing memory
                    d = new linear_memory<V,M0>(src.size());
                    dst.m_memory.reset(d);
                }
                d->set_strides(dst.m_info.host_stride,dst.m_info.host_shape, L0());
                allocator<V, size_type, M0> a;
                if(src.is_c_contiguous()){ 
                    // easiest case: both linear, simply copy
                    a.copy(d->ptr(), src.m_ptr, src.size(), M1());
                }
                else if(src.is_2dcopyable()){
                    // other memory is probably a pitched memory or some view onto an array
                    size_type row,col,pitch;
                    detail::get_pitched_params(row,col,pitch,src.m_info.host_shape, src.m_info.host_stride,L1());
                    a.copy2d(d->ptr(), src.m_ptr, col,pitch,row,col,M1());
                }else{
                    throw std::runtime_error("copying arbitrarily strided memory not implemented");
                }
                if(!IsSame<L0,L1>::Result::value){
                    dst.m_info.host_stride.reverse();
                    dst.m_info.host_shape.reverse();
                }
            }

        /// copies between different memory spaces
        template<class V, class M0, class M1, class L0, class L1>
            void copy_memory(tensor<V,M0,L0>&dst, const tensor<V,M1,L1>&src, pitched_memory_tag){
                typedef typename tensor<V,M0,L0>::size_type size_type;
                assert(src.ndim()>=2);
                size_type row,col,pitch;
                detail::get_pitched_params(row,col,pitch,src.m_info.host_shape, src.m_info.host_stride,L1());
                pitched_memory<V,M0>* d = NULL;
                if(dst.m_memory.get()){
                    d = dynamic_cast<pitched_memory<V,M0>*>(dst.m_memory.get());
                    if(d)
                        d->set_size(row,col); // may get us arround reallocation
                }
                if(!d){
                    d = new pitched_memory<V,M0>(row,col);
                    dst.m_memory.reset(d);
                }
                d->set_strides(dst.m_info.host_stride,dst.m_info.host_shape, L0());
                allocator<V, size_type, M0> a;
                if(src.is_2dcopyable()){
                    // other memory is probably a pitched memory or some view onto an array
                    detail::get_pitched_params(row,col,pitch,src.m_info.host_shape, src.m_info.host_stride,L1());
                    a.copy2d(d->ptr(), src.m_ptr, d->pitch(),pitch,row,col,M1());
                }else{
                    throw std::runtime_error("copying arbitrarily strided memory not implemented");
                }

                if(!IsSame<L0,L1>::Result::value){
                    dst.m_info.host_stride.reverse();
                    dst.m_info.host_shape.reverse();
                }
            }
    }

    /** @} */

}
namespace std{
    /**
     * print a host linear memory to a stream
     * @param o the stream
     * @param t the tensor
     */
    template<class V>
    ostream& operator<<(ostream& o, const cuv::linear_memory<V, cuv::host_memory_space>& t){
        o << "[ ";
        for(unsigned int i=0;i<t.size();i++)
            o<< t[i]<<" ";
        o <<"]";
        return o;
    }
    /**
     * print a dev linear memory to a stream (copies first)
     * @param o the stream
     * @param t_ the tensor
     */
    template<class V>
    ostream& operator<<(ostream& o, const cuv::linear_memory<V, cuv::dev_memory_space>& t_){
        cuv::linear_memory<V, cuv::host_memory_space> t = t_; // pull
        o << "[ ";
        for(unsigned int i=0;i<t.size();i++)
            o<< t[i]<<" ";
        o <<"]";
        return o;
    }

    /**
     * print a host pitched memory to a stream
     * @param o the stream
     * @param t the tensor
     */
    template<class V>
    ostream& operator<<(ostream& o, const cuv::pitched_memory<V, cuv::host_memory_space>& t){
        o << "[ ";
        for(unsigned int i=0;i<t.rows();i++){
            for(unsigned int j=0;j<t.rows();j++){
                o<< t(i,j)<<" ";
            }
            if(i<t.rows()-1)
                o<< std::endl;
        }
        o <<"]";
        return o;
    }
    /**
     * print a dev pitched memory to a stream (copies first)
     * @param o the stream
     * @param t_ the tensor
     */
    template<class V>
    ostream& operator<<(ostream& o, const cuv::pitched_memory<V, cuv::dev_memory_space>& t_){
        cuv::pitched_memory<V, cuv::host_memory_space> t = t_; // pull
        o << "[ ";
        for(unsigned int i=0;i<t.rows();i++){
            for(unsigned int j=0;j<t.rows();j++){
                o<< t(i,j)<<" ";
            }
            if(i<t.rows()-1)
                o<< std::endl;
        }
        o <<"]";
        return o;
    }

    /**
     * print a host tensor to a stream
     *
     * @param o the stream
     * @param t the tensor
     */
    template<class V, class L>
    ostream& operator<<(ostream& o, const cuv::tensor<V, cuv::host_memory_space, L>& t){
        if(t.ndim()==0)
            return o << "[]";

        if(t.ndim()==1){
            o << "[ ";
            for(unsigned int i=0;i<t.shape(0);i++) o<< t[i]<<" ";
            return o <<"]";
        }
        if(t.ndim()==2){
            o << "[";
            for(unsigned int i=0;i<t.shape(0);++i){
                if(i>0)
                    o<<" ";
                o << "[ ";
                for(unsigned int j=0;j<t.shape(1);j++) o<< t(i,j)<<" ";
                o <<"]";
                if(i != t.shape(0)-1)
                    o <<std::endl;
            } 
            return o<<"]";
        }
        if(t.ndim()==3){
            o<<"["<<std::endl;
            for(unsigned int l=0;l<t.shape(0);l++){
                o << "[";
                for(unsigned int i=0;i<t.shape(1);++i){
                    if(i>0)
                        o<<" ";
                    o << "[ ";
                    for(unsigned int j=0;j<t.shape(2);j++) o<< t(l,i,j)<<" ";
                    o <<"]";
                    if(i != t.shape(1)-1)
                        o <<std::endl;
                } 
                o<<"]";
                if(l<t.shape(0)-1)
                    o<<std::endl;
            }
            return o<<"]";
        }
        throw std::runtime_error("printing of tensors with >3 dimensions not implemented");
    }
}
#endif /* __TENSOR2_HPP__ */
