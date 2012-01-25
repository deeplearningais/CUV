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
 * @brief an n-dimensional array on host or device
 * @author Hannes Schulz
 * @date 2012-01-25
 * @ingroup basics
 */

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

namespace boost { namespace serialization {
        class access; } }

namespace cuv
{
/**
 * @addtogroup basics Basic datastructures
 * @{
 */

    /**
     * @addtogroup tags
     * @{
     */
	/// Tag for column major matrices
	struct column_major {};
	/// Tag for row major matrices
	struct row_major    {};

    /// tag for linear memory
    struct linear_memory_tag{};
    /// tag for pitched memory
    struct pitched_memory_tag{};
    /** @} */ // tags
/** @} */

/**
 * @addtogroup MetaProgramming
 * @{
 */
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
	struct other_memory_space{};
	/// specialisation: converts from dev_memory_space to host_memory_space
	template<>
	struct other_memory_space<dev_memory_space>{ typedef host_memory_space type; };
	/// specialisation: converts from host_memory_space to dev_memory_space
	template<>
	struct other_memory_space<host_memory_space>{ typedef dev_memory_space type; };
/** @} */



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
		 * tensor_view<...> v(indices[index_range(1,3)][index_range()], other_tensor);
         * // or, equivalently
		 * other_tensor[indices[index_range(1,3)][index_range()]];
		 * @endcode
		 */
		index_gen<0,0> indices;
	}
#endif


    template<class V, class M, class L> class tensor;
    template<class V, class M> class linear_memory;

    // forward declaration of fill to implement operator= for value_type	
    template<class V, class M, class L, class S>
        void fill(tensor<V, M, L>& v, const V& p);

    /**
     * @addtogroup basics
     * @{
     */
    /**
     * simply keeps a pointer and deallocates it when destroyed
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

            private:
                friend class boost::serialization::access;
                allocator<value_type, size_type, memory_space_type> m_allocator; ///< how stored memory was allocated
                /// prohibit copying
                memory(const memory&){}
                /// prohibit copying
                memory& operator=(const memory& o){return *this;}
            protected:
                pointer_type m_ptr;  ///< points to allocated memory
                size_type    m_size; ///< size (for serialization)
            public:
                /// @return pointer to allocated memory
                pointer_type ptr(){return m_ptr;}
                /// @return pointer to allocated memory (const)
                const_pointer_type ptr()const{return m_ptr;}

                /// @return number of stored elements
                size_type size()const{    return m_size; }
                /// @return number of stored bytes
                size_type memsize()const{ return size()*sizeof(V); }

                /// reset information (use with care, for deserialization)
                void reset(pointer_type p, size_type s){ m_ptr = p; m_size = s; }


                /// default constructor (just sets ptr to NULL)
                memory():m_ptr(NULL),m_size(0){}
                
                /// construct with pointer (takes /ownership/ of this pointer and deletes it when destroyed!)
                memory(value_type* ptr, size_type size):m_ptr(ptr),m_size(size){}

                /// destructor (deallocates the memory)
                ~memory(){ dealloc(); }

                /**
                 * dellocate space
                 */
                void dealloc(){
                    if (m_ptr)
                        m_allocator.dealloc(&this->m_ptr);
                    m_ptr=NULL;
                }
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
                friend class boost::serialization::access;
                typedef linear_memory<V,M> my_type; ///< my own type
                allocator<value_type, size_type, memory_space_type> m_allocator; ///< how stored memory was allocated
                using super::m_size;
                using super::m_ptr;
            public:

                /// default constructor: does nothing
                linear_memory(){}

                /** constructor: reserves space for i elements
                 *  @param i number of elements
                 */
                linear_memory(size_type i){m_size = i; alloc();}

                /// releases ownership of pointer (for storage in memory class)
                value_type* release(){ value_type* ptr = m_ptr; m_ptr = NULL; return ptr; }

                /** sets the size (reallocates if necessary)
                 */
                void set_size(size_type s){
                    if(s!=this->size()){
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
                    if(m_size>0)
                        m_allocator.alloc( &m_ptr,m_size);
                }

                /**
                 * dellocate space
                 */
                void dealloc(){
                    if (m_ptr)
                        m_allocator.dealloc(&this->m_ptr);
                    this->m_ptr=NULL;
                    this->m_size=NULL;
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
                        m_allocator.copy(this->m_ptr,o.ptr(),this->size(),memory_space_type());

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
                        m_allocator.copy(m_ptr,o.ptr(),this->size(),OM());
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
                void set_strides(
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
                void set_strides(
                        linear_memory<index_type,host_memory_space>& strides,
                        const linear_memory<size_type,host_memory_space>& shape, column_major){
                    size_type size = 1;
                    for (unsigned int i = 0; i <  shape.size(); ++i)
                    {
                        strides[i] = (shape[i] == 1) ? 0 : size;
                        size *= shape[i];
                    }
                }

                /** reverse the array (for transposing etc)
                 * 
                 * currently only enabled for host memory space arrays
                 */
                void reverse(){
                    if(IsSame<dev_memory_space,memory_space_type>::Result::value)
                        throw std::runtime_error("reverse of dev linear memory not implemented");
                    value_type* __first = m_ptr, *__last = m_ptr + this->size();
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
    /** @} */ // basics

    namespace detail{

        /** 
         * true iff there are no "holes" in memory
         */
        inline bool is_c_contiguous(row_major, const linear_memory<unsigned int,host_memory_space>& shape, const linear_memory<int,host_memory_space>& stride){
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

        /**
         * @overload
         */
        inline bool is_c_contiguous(column_major, const linear_memory<unsigned int,host_memory_space>& shape, const linear_memory<int,host_memory_space>& stride){
            bool c_contiguous = true;
            int size = 1;
            for (unsigned int i = 0; i<shape.size() && c_contiguous; ++i)
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
        inline bool is_2dcopyable(row_major, const linear_memory<unsigned int,host_memory_space>& shape, const linear_memory<int,host_memory_space>& stride){
            bool c_contiguous = shape.size()>1;
            const int pitched_dim = shape.size()-1; // last dim
            int size = 1;
            for (int i = shape.size()-1; (i >= 0) && c_contiguous; --i)
            {
                if(shape[i] == 1){
                    continue;
                }else if(i == pitched_dim){
                    size *= stride[i-1];
                }else if(stride[i] != size) {
                    c_contiguous = false;
                }else{
                    size *= shape[i];
                }
            }
            return c_contiguous;
        }

        /// @overload
        inline bool is_2dcopyable(column_major, const linear_memory<unsigned int,host_memory_space>& shape, const linear_memory<int,host_memory_space>& stride){
            bool c_contiguous = shape.size()>1;
            const unsigned int pitched_dim = 1; 
            int size = 1;
            for (unsigned int i = 0; (i <  shape.size()) && c_contiguous; ++i)
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
     * @addtogroup basics
     * @{
     */

    /**
     * represents 2D non-contiguous ("pitched") memory
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
                friend class boost::serialization::access;
                typedef pitched_memory<V,M> my_type; ///< my own type
                allocator<value_type, size_type, memory_space_type> m_allocator; ///< how stored memory was allocated
                size_type m_rows;  ///< number of rows
                size_type m_cols;  ///< number of columns
                size_type m_pitch;  ///< pitch (multiples of sizeof(V))
                using super::m_ptr;
                using super::m_size;
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
                    m_size = m_rows*m_pitch; // in class memory
                }

                /** 
                 * @brief Deallocate memory 
                 */
                void dealloc(){
                    if (this->m_ptr)
                        m_allocator.dealloc(&this->m_ptr);
                    this->m_ptr=NULL;
                    this->m_size=NULL;
                }

                /// releases ownership of pointer (for storage in memory class)
                value_type* release(){ value_type* ptr = m_ptr; m_ptr = NULL; return ptr; }

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
                 * @param o Source pitched_memory
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
                 * @return a reference to memory at a position as if this were pitched memory
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
                        return const_cast<pitched_memory&>(*this)(idx);
                    }

                /**
                 * get a reference to a datum in memory
                 *
                 * @param i first (slow-changing) dimension index
                 * @param j second (fast-changing) dimension index
                 * @return reference to datum at index i,j
                 */
                reference_type
                    operator()(const index_type& i, const index_type& j){
                        assert(i>=0);
                        assert(j>=0);
                        assert((size_type)i < m_rows);
                        assert((size_type)j < m_cols);
                        return reference_type(this->m_ptr+i*m_pitch+j);
                    }
                /** @overload */
                const_reference_type
                    operator()(const index_type& i, const index_type& j)const{
                        return const_cast<pitched_memory&>(*this)(i,j);
                    }


                /**
                 * set strides for this memory
                 *
                 * determines the strides for a given shape, with special consideration to pitched dimension
                 *
                 * @param strides output vector
                 * @param shape   shape of the vector
                 *
                 * row major version
                 */
                void set_strides(
                        linear_memory<index_type,host_memory_space>& strides,
                        const linear_memory<size_type,host_memory_space>& shape, row_major){
                    size_type size = 1;
                    assert(shape.size()>=2);
                    const int pitched_dim = shape.size()-1;
                    for (int i = shape.size()-1; i >= 0; --i)
                    {
                        if(shape[i] == 1){
                            strides[i] = 0;
                        }else if(i == pitched_dim){
                            strides[i] = 1;
                            size *= pitch();
                        }else {
                            strides[i] = size;
                            size *= shape[i];
                        }
                    }
                }
                /**
                 * @overload
                 *
                 * column major version
                 */
                void set_strides(
                        linear_memory<index_type,host_memory_space>& strides,
                        const linear_memory<size_type,host_memory_space>& shape, column_major){
                    size_type size = 1;
                    assert(shape.size()>=2);
                    const size_type pitched_dim = 0;
                    for (unsigned int i = 0; i < shape.size(); ++i)
                    {
                        if(shape[i] == 1){
                            strides[i] = 0;
                        }else if(i == pitched_dim){
                            strides[i] = 1;
                            size *= pitch();
                        }else {
                            strides[i] = size;
                            size *= shape[i];
                        }
                    }
                }
        };

    /**
     * contains infos about shape and stride on host and in the tensor data space.
     */
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
    template<class V, class M, class L>
        class tensor_view;



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
            typedef typename memory_type::pointer_type pointer_type; ///< type of data pointers
            typedef typename memory_type::const_pointer_type const_pointer_type; ///< type of const data pointers
            typedef          L memory_layout_type; ///< column/row major

            typedef tensor_info<M,L> info_type; ///< type of shape info struct
            typedef tensor_view<V,M,L> view_type; ///< type of views on this tensor

        protected:
            /// information about shape, strides
            info_type  m_info;  

            /// points to (possibly shared) memory
            boost::shared_ptr<memory_type> m_memory;
            
            /// points to start of actually referenced memory (within m_memory)
            V* m_ptr;

            /// tensor views are our friends
            template <class _V, class _M, class _L>
            friend class tensor_view;

            /** 
             * determine linear index in memory of an index array
             *
             * this function takes strides etc. into account, so that indices
             * are interpreted as relative to the (strided) subtensor we're
             * referring to.
             *
             * @param D    size of index array
             * @param arr  index array
             * @return linear index in memory of index array
             *
             */
			size_type
			index_of(int D, index_type* arr)const{
				index_type pos = 0;
				for(int i=0; i<D; i++){
					index_type temp = arr[i];
                    if(temp<0) temp = m_info.host_shape[i]+temp;
					pos += temp * m_info.host_stride[i];
				}
				return pos;
			}
            /**
             * allocate linear memory (c-contiguous version)
             *
             * @param t tensor to allocate
             */
            void allocate(tensor& t,linear_memory_tag){
                    linear_memory<V,M> d(t.size());
                    d.set_strides(t.m_info.host_stride,t.m_info.host_shape, L());
                    t.m_ptr = d.ptr();
                    t.m_memory.reset(new memory<V,M>(d.release(), d.size()));
                }

            /**
             * @overload
             *
             * pitched version
             */
            void allocate(tensor& t,pitched_memory_tag){
                typename tensor<V,M,L>::size_type row,col,pitch;
                detail::get_pitched_params(row,col,pitch,t.m_info.host_shape, t.m_info.host_stride,L());
                pitched_memory<V,M> d(row,col);
                d.set_strides(t.m_info.host_stride,t.m_info.host_shape, L());
                t.m_ptr = d.ptr();
                t.m_memory.reset(new memory<V,M>(d.release(),d.size()));
            }


        public:
            /** 
             * determine linear index in memory of an index array
             *
             * this function takes strides etc. into account, so that indices
             * are interpreted as relative to the (strided) subtensor we're
             * referring to.
             *
             * @param D    size of index array
             * @param arr  index array
             * @return linear index in memory of index array
             *
             */
            template<std::size_t D>
			size_type
			index_of(const extent_gen<D>& eg)const{
				index_type pos = 0;
				for(int i=0; i<D; i++){
					index_type temp = eg.ranges_[i].finish();
                    if(temp<0) temp = m_info.host_shape[i]+temp;
					pos += temp * m_info.host_stride[i];
				}
				return pos;
			}
            /**
             * @name Accessors
             * @{
             */
            /// return the number of dimensions
            index_type ndim()const{ return m_info.host_shape.size(); }

            /** return the size of the i-th dimension
             *  @param i the index of the queried dimension
             */
            size_type shape(const index_type& i)const{return m_info.host_shape[i];}

            /** return the stride of the i-th dimension
             *  @param i the index of the queried dimension
             */
            index_type stride(const index_type& i)const{return m_info.host_stride[i];}

            /** @return the pointer to the referenced memory */
            pointer_type       ptr()       {return m_ptr;}

            /** 
             * @overload
             * @return the const pointer to the referenced memory 
             * */
            const_pointer_type ptr() const {return m_ptr;}

            /** set the pointer offset (used in deserialization) */
            void set_ptr_offset(long int i){ m_ptr = m_memory->ptr() + i; }

            /** * @return pointer to allocated memory */
            boost::shared_ptr<memory_type>& mem(){ return m_memory; }
            /** 
             * @overload
             * @return the const pointer to the allocated memory 
             * */
            const boost::shared_ptr<memory_type>& mem()const{ return m_memory; }


            /** @return the number of stored elements
             */
            size_type size()const{
                return std::accumulate(m_info.host_shape[0].ptr, m_info.host_shape[0].ptr+m_info.host_shape.size(), 1, std::multiplies<index_type>());
            }

            /**
             * determine size in bytes
             *
             * assumes that the memory is c_contiguous! 
             *
             * @return the size in bytes
             */
            size_type memsize()const{
#ifndef NDEBUG
                cuvAssert(is_c_contiguous());
#endif
                return std::accumulate(m_info.host_shape[0].ptr, m_info.host_shape[0].ptr+m_info.host_shape.size(), 1, std::multiplies<index_type>());
            }

            /// return the shape of the tensor (as a vector for backward compatibility)
            std::vector<size_type> shape()const{
                if(ndim()==0)
                    return std::vector<size_type>();
                return std::vector<size_type>(m_info.host_shape[0].ptr, m_info.host_shape[0].ptr+m_info.host_shape.size());
            }

            /// @return the tensor info struct (const)
            const info_type& info()const{return m_info;}

            /// @return the tensor info struct 
            info_type& info(){return m_info;}

            /// true iff there are no "holes" in memory
            bool is_c_contiguous()const{
                return detail::is_c_contiguous(memory_layout_type(), m_info.host_shape, m_info.host_stride);
            }
            
            /// true iff it can be copied as a 2d array (only one dimension is pitched)
            bool is_2dcopyable()const{
                return detail::is_2dcopyable(memory_layout_type(), m_info.host_shape, m_info.host_stride);
            }

            /** @} */ // accessors

            /**
             * @name accessing stored values
             * @{
             */

            /**
             * member access: "flat" access as if memory was linear
             */
            reference_type operator[](index_type idx){
                size_type ndim = m_info.host_shape.size();
                size_type* virtualstride = new size_type[ndim];
                size_type pos = 0;
                if(IsSame<L,row_major>::Result::value){
                    // row major
                    {   size_type virt_size = 1;
                        for(int i=ndim-1;i>=0;--i){
                            virtualstride[i] = virt_size;
                            virt_size *= m_info.host_shape[i];
                        }
                    }
                    for(size_type i=0; i<ndim; ++i){
                        pos += (idx/virtualstride[i])*m_info.host_stride[i];
                        idx -= (idx/virtualstride[i])*virtualstride[i];
                    }
                }else{
                    // column major
                    {   size_type virt_size = 1;
                        for(unsigned int i=0;i<ndim;++i){
                            virtualstride[i] = virt_size;
                            virt_size *= m_info.host_shape[i];
                        }
                    }
                    for(int i=ndim-1; i>=0; --i){
                        pos += (idx/virtualstride[i])*m_info.host_stride[i];
                        idx -= (idx/virtualstride[i])*virtualstride[i];
                    }
                }
                delete[] virtualstride;
                return reference_type(m_ptr + pos);
            }

            /** @overload */
            const_reference_type operator[](index_type idx)const{
                return const_cast<tensor&>(*this)[idx];
            }

            /**
             * get a reference to the datum at an index
             * @param i0 index for a 1-dimensional tensor
             * @return reference to datum at i0
             */
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
            /** @overload */
            const_reference_type operator()(index_type i0)const{ return const_cast<tensor&>(*this)(i0); }

            /** @overload */
            const_reference_type operator()(index_type i0, index_type i1)const{ return const_cast<tensor&>(*this)(i0,i1); }
            /** @overload */
            reference_type operator()(index_type i0, index_type i1){
#ifndef NDEBUG
                cuvAssert(ndim()==2);
                cuvAssert((i0>=0 && (size_type)i0 < shape(0)) || (i0<0 && (size_type)-i0<shape(0)+1) )
                cuvAssert((i1>=0 && (size_type)i1 < shape(1)) || (i1<0 && (size_type)-i1<shape(1)+1) )
#endif
                index_type arr[2] = {i0,i1};
                return reference_type(m_ptr + index_of( 2,arr));
            }

            /** @overload */
            const_reference_type operator()(index_type i0, index_type i1, index_type i2)const{ return const_cast<tensor&>(*this)(i0,i1,i2); }
            /** @overload */
            reference_type operator()(index_type i0, index_type i1, index_type i2){
#ifndef NDEBUG
                cuvAssert(ndim()==3);
                cuvAssert((i0>=0 && (size_type)i0 < shape(0)) || (i0<0 && (size_type)-i0<shape(0)+1) )
                cuvAssert((i1>=0 && (size_type)i1 < shape(1)) || (i1<0 && (size_type)-i1<shape(1)+1) )
                cuvAssert((i2>=0 && (size_type)i2 < shape(2)) || (i2<0 && (size_type)-i2<shape(2)+1) )
#endif
                index_type arr[3] = {i0,i1,i2};
                return reference_type(m_ptr + index_of( 3,arr));
            }

            /** @overload */
            const_reference_type operator()(index_type i0, index_type i1, index_type i2, index_type i3)const{ return const_cast<tensor&>(*this)(i0,i1,i2,i3); }
            /** @overload */
            reference_type operator()(index_type i0, index_type i1, index_type i2, index_type i3){
#ifndef NDEBUG
                cuvAssert(ndim()==4);
                cuvAssert((i0>=0 && (size_type)i0 < shape(0)) || (i0<0 && (size_type)-i0<shape(0)+1) )
                cuvAssert((i1>=0 && (size_type)i1 < shape(1)) || (i1<0 && (size_type)-i1<shape(1)+1) )
                cuvAssert((i2>=0 && (size_type)i2 < shape(2)) || (i2<0 && (size_type)-i2<shape(2)+1) )
                cuvAssert((i3>=0 && (size_type)i3 < shape(3)) || (i3<0 && (size_type)-i3<shape(3)+1) )
#endif
                index_type arr[4] = {i0,i1,i2,i3};
                return reference_type(m_ptr + index_of( 4,arr));
            }

            /** @overload */
            const_reference_type operator()(index_type i0, index_type i1, index_type i2, index_type i3, index_type i4)const{ return const_cast<tensor&>(*this)(i0,i1,i2,i3,i4); }
            /** @overload */
            reference_type operator()(index_type i0, index_type i1, index_type i2, index_type i3, index_type i4){
#ifndef NDEBUG
                cuvAssert(ndim()==5);
                cuvAssert((i0>=0 && (size_type)i0 < shape(0)) || (i0<0 && (size_type)-i0<shape(0)+1) )
                cuvAssert((i1>=0 && (size_type)i1 < shape(1)) || (i1<0 && (size_type)-i1<shape(1)+1) )
                cuvAssert((i2>=0 && (size_type)i2 < shape(2)) || (i2<0 && (size_type)-i2<shape(2)+1) )
                cuvAssert((i3>=0 && (size_type)i3 < shape(3)) || (i3<0 && (size_type)-i3<shape(3)+1) )
                cuvAssert((i4>=0 && (size_type)i4 < shape(4)) || (i4<0 && (size_type)-i4<shape(4)+1) )
#endif
                index_type arr[5] = {i0,i1,i2,i3,i4};
                return reference_type(m_ptr + index_of( 5,arr));
            }
            /** @} */ // accessing stored values

            /** @name constructors 
             * @{
             */
            /**
             * default constructor (does nothing)
             */
            tensor():m_ptr(NULL){}

            // ****************************************************************
            //        Constructing from other tensor
            // ****************************************************************

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
                explicit tensor(const tensor<value_type,OM,L>& o)
                :m_info(o.info()) // primarily to copy shape
                ,m_ptr(NULL)
                {
                    copy_memory(*this, o, linear_memory_tag());
                    m_ptr = m_memory->ptr();
                }

            /**
             * construct tensor from tensor of same memory space
             * in  /pitched/ memory. Note: this /copies/ the memory!
             */
                explicit tensor(const tensor& o, pitched_memory_tag)
                :m_info(o.m_info) // primarily to copy shape
                ,m_ptr(NULL)
                {
                    copy_memory(*this, o, pitched_memory_tag());
                    m_ptr = m_memory->ptr();
                }

            /**
             * construct tensor from tensor of other memory space
             * in  /pitched/ memory. Note: this /copies/ the memory!
             */
            template<class OM>
                explicit tensor(const tensor<value_type,OM,L>& o, pitched_memory_tag)
                :m_info(o.info()) // primarily to copy shape
                ,m_ptr(NULL)
                {
                    copy_memory(*this, o, pitched_memory_tag());
                    m_ptr = m_memory->ptr();
                }

            /**
             * construct tensor from tensor of same memory space
             * in (dense) /linear/ memory. Note: this /copies/ the memory!
             */
                explicit tensor(const tensor& o, linear_memory_tag)
                :m_info(o.m_info) // primarily to copy shape
                ,m_ptr(NULL)
                {
                    copy_memory(*this, o, linear_memory_tag());
                    m_ptr = m_memory->ptr();
                }
            /**
             * construct tensor from tensor of other memory space
             * in (dense) /linear/ memory. Note: this /copies/ the memory!
             */
            template<class OM>
                explicit tensor(const tensor<value_type,OM,L>& o, linear_memory_tag)
                :m_info(o.info()) // primarily to copy shape
                ,m_ptr(NULL)
                {
                    copy_memory(*this, o, linear_memory_tag());
                    m_ptr = m_memory->ptr();
                }
            /**
             * construct tensor from other memory layout
             *
             * this does not copy memory, but reverses dimensions and strides
             * (and therefore only takes O(1) time)
             */
            template<class OL>
                explicit tensor(const tensor<value_type,M,OL>& o)
            : m_memory(o.mem()) // increase ref counter
            , m_ptr(const_cast<pointer_type>( o.ptr() )) { // same pointer in memory
                m_info.host_shape = o.info().host_shape; 
                m_info.host_shape.reverse();
                m_info.host_stride = o.info().host_stride; 
                m_info.host_stride.reverse();
            }    
            
            // ****************************************************************
            //        Constructing from SHAPE
            // ****************************************************************

            /**
             * construct one-dimensional tensor
             */
			explicit tensor(const size_type i)
                :m_ptr(NULL)
            {
				m_info.resize(1);
                m_info.host_shape[0] = i;
                allocate(*this,linear_memory_tag());
			}
            /**
             * construct two-dimensional tensor
             */
			explicit tensor(const size_type i, const int j)
                :m_ptr(NULL)
            {
				m_info.resize(2);
                m_info.host_shape[0] = i;
                m_info.host_shape[1] = j;
                allocate(*this,linear_memory_tag());
			}
            /**
             * construct tensor from a shape
             */
			template<std::size_t D>
			explicit tensor(const extent_gen<D>& eg)
                :m_ptr(NULL)
            {
				m_info.resize(D);
				for(std::size_t i=0;i<D;i++)
					m_info.host_shape[i] = eg.ranges_[i].finish();
                allocate(*this,linear_memory_tag());
			}

            /**
             * construct tensor from a shape
             *
             * @deprecated
             */
			explicit tensor(const std::vector<size_type>& eg)
                :m_ptr(NULL)
            {
				m_info.resize(eg.size());
				for(std::size_t i=0;i<eg.size();i++)
					m_info.host_shape[i] = eg[i];
                allocate(*this,linear_memory_tag());
			}

            /**
             * construct tensor from a shape
             *
             * @deprecated
             */
			explicit tensor(const std::vector<size_type>& eg, pitched_memory_tag)
                :m_ptr(NULL)
            {
				m_info.resize(eg.size());
				for(std::size_t i=0;i<eg.size();i++)
					m_info.host_shape[i] = eg[i];
                allocate(*this,pitched_memory_tag());
			}

            /**
             * construct tensor from a shape (pitched)
             */
			template<std::size_t D>
			explicit tensor(const extent_gen<D>& eg, pitched_memory_tag)
                :m_ptr(NULL)
            {
				m_info.resize(D);
				for(std::size_t i=0;i<D;i++)
					m_info.host_shape[i] = eg.ranges_[i].finish();
                allocate(*this,pitched_memory_tag());
			}

            // ****************************************************************
            //        Constructing from shape and raw pointer
            // ****************************************************************

            /**
             * construct tensor from a shape and a pointer (does not copy memory)
             *
             * @warning You have to ensure that the memory lives as long as this object.
             */
            template<std::size_t D>
                explicit tensor(const extent_gen<D>& eg, value_type* ptr)
                :m_ptr(ptr)
                {
                    m_info.resize(D);
                    size_type size = 1;
                    if(IsSame<memory_layout_type,row_major>::Result::value)
                        for(int i=D-1;i>=0;i--){
                            m_info.host_shape[i] = eg.ranges_[i].finish();
                            m_info.host_stride[i] = size;
                            size *= eg.ranges_[i].finish();
                        }
                    else
                        for(std::size_t i=0;i<D;i++){
                            m_info.host_shape[i] = eg.ranges_[i].finish();
                            m_info.host_stride[i] = size;
                            size *= eg.ranges_[i].finish();
                        }
                }
            explicit tensor(const std::vector<size_type>& shape, value_type* ptr)
                :m_ptr(ptr)
            {
                unsigned int D = shape.size();
                m_info.resize(D);
                size_type size = 1;
                if(IsSame<memory_layout_type,row_major>::Result::value)
                    for(int i=D-1;i>=0;i--){
                        m_info.host_shape[i] = shape[i];
                        m_info.host_stride[i] = size;
                        size *= shape[i];
                    }
                else
                    for(std::size_t i=0;i<D;i++){
                        m_info.host_shape[i] = shape[i];
                        m_info.host_stride[i] = size;
                        size *= shape[i];
                    }
            }
            /**
             * construct tensor from a shape and a pointer (does not copy memory)
             *
             * @warning You have to ensure that the memory lives as long as this object.
             * @deprecated
             */
            template<int D, int E>
                explicit tensor(const index_gen<D,E>& idx, value_type* ptr)
                :m_ptr(ptr)
                {
                    m_info.resize(D);
                    size_type size = 1;
                    if(IsSame<memory_layout_type,row_major>::Result::value)
                        for(int i=D-1;i>=0;i--){
                            m_info.host_shape[i] = idx.ranges_[i].finish();
                            m_info.host_stride[i] = size;
                            size *= idx.ranges_[i].finish();
                        }
                    else
                        for(std::size_t i=0;i<D;i++){
                            m_info.host_shape[i] = idx.ranges_[i].finish();
                            m_info.host_stride[i] = size;
                            size *= idx.ranges_[i].finish();
                        }
                }
            // @} // constructors


            // ****************************************************************
            //   assignment operators (try not to reallocate if shapes match)
            // ****************************************************************

            /**
             * @name assigning other values to a tensor object
             * @{
             */

            /**
             * assign from tensor of same type 
             *
             * always an O(1) operation.
             */
            tensor& operator=(const tensor& o){
                if(this==&o) return *this; // check for self-assignment
                /*
                 *if(copy_memory(*this,o,false))
                 *    return *this;
                 */
                m_memory = o.mem();
                m_ptr = const_cast<pointer_type>(o.ptr());
                m_info = o.info();
                return *this;
            }

            /**
             * assign from value (sets all elements equal to one scalar)
             */
            template<class _V>
            typename boost::enable_if_c<boost::is_convertible<_V,value_type>::value, tensor&>::type
            operator=(const _V& scalar){
                fill(*this, scalar);
                return *this;
            }

            /**
             * assign from tensor of different memory space type.
             *
             * If shapes do not match, it defaults to linear memory.
             *
             * this copies memory (obviously) but tries to avoid reallocation
             */
            template<class OM>
            tensor& operator=(const tensor<value_type,OM,L>& o){
                if(!copy_memory(*this,o,false))
                    copy_memory(*this,o,linear_memory_tag());
                m_ptr = mem()->ptr();
                return *this;
            }

            /**
             * assign from tensor of different memory layout type.
             *
             * this does not copy memory, but reverses strides and shapes.
             */
            template<class OL>
            tensor& operator=(const tensor<value_type,M,OL>& o){
                m_memory = o.mem();
                m_ptr    = const_cast<V*>(o.ptr());
                m_info.host_shape   = o.info().host_shape;
                m_info.host_stride  = o.info().host_stride;
                m_info.host_stride.reverse();
                m_info.host_shape.reverse();
                return *this;
            }

            /** @} */ // assignment


            /**
             * copy memory using given allocator tag (linear/pitched)
             */
            template<class T>
            tensor copy(T tag=linear_memory_tag())const{
                    tensor t;
                    const tensor& o = *this;
                    t.m_info   = o.info();
                    copy_memory(t,o,tag);
                    t.m_ptr    = t.mem()->ptr();
                    return t;
                }

            /**
             * copy memory using linear memory
             */
            tensor copy()const{
                return copy(linear_memory_tag());
            }


            /**
             * create a subtensor of the current tensor 
             *
             * this works in O(1).
             */
            template<int D, int E>
                tensor_view<V,M,L>
                operator[](const index_gen<D,E>& idx)const
                {
                    tensor_view<V,M,L> t;
                    const tensor& o = *this;
                    t.m_memory = o.mem();
                    t.m_ptr    = const_cast<pointer_type>(o.ptr());

                    std::vector<int> shapes;
                    std::vector<int> strides;
                    shapes.reserve(D);
                    strides.reserve(D);
                    cuvAssert(o.ndim()==D);
                    for(std::size_t i=0;i<D;i++){
                        int start  = idx.ranges_[i].get_start(0);
                        int finish = idx.ranges_[i].get_finish(o.shape(i));
                        int stride = idx.ranges_[i].stride();
                        if (start <0) start  += o.shape(i);
                        if (finish<0) finish += o.shape(i);
#ifndef NDEBUG
                        cuvAssert(finish>start);
#endif
                        t.m_ptr += start*o.stride(i);
                        if(finish-start==1){
                            // skip dimension
                        }else{
                            shapes.push_back((finish-start)/stride);
                            strides.push_back(o.stride(i)*stride);
                        }
                    }
                    // store in m_info
                    t.m_info.resize(shapes.size());
                    std::copy(shapes.begin(),shapes.end(),t.m_info.host_shape[0].ptr);
                    std::copy(strides.begin(),strides.end(),t.m_info.host_stride[0].ptr);
                    return t; // should not copy mem, only m_info
                }

            /**
             * reshape the tensor (in place)
             *
             * works only for c_contiguous memory!
             *
             * @param eg new shape
             */
            template<std::size_t D>
            void reshape(const extent_gen<D>& eg){
                std::vector<size_type> shape(D);
                for(std::size_t i=0;i<D;i++) 
                    shape[i] = eg.ranges_[i].finish();
                reshape(shape);
            }
            /**
             * reshape the tensor (in place)
             *
             * works only for c_contiguous memory!
             *
             * @param eg new shape
             */
            void reshape(const std::vector<size_type>& shape){
                size_type new_size = std::accumulate(shape.begin(),shape.end(),1,std::multiplies<size_type>());
                if(!is_c_contiguous())
                    throw std::runtime_error("cannot reshape: tensor is not c_contiguous");
                if(size() != new_size)
                    throw std::runtime_error("cannot reshape: products do not match");
                m_info.resize(shape.size());
                size_type size = 1;
                if(IsSame<memory_layout_type,row_major>::Result::value)
                    for(int i=shape.size()-1;i>=0;i--){
                        m_info.host_shape[i] = shape[i];
                        m_info.host_stride[i] = size;
                        size *= shape[i];
                    }
                else
                    for(std::size_t i=0;i<shape.size();i++){
                        m_info.host_shape[i] = shape[i];
                        m_info.host_stride[i] = size;
                        size *= shape[i];
                    }
            }
            /**
             * convenience wrapper for reshape(extents[r][c])
             * @param r leading index of new shape
             * @param c following index of new shape
             */
            void reshape(size_type r, size_type c){
                reshape(extents[r][c]);
            }

            /**
             * force deallocation of memory if possible
             */
            void dealloc(){
                m_memory.reset();
                m_ptr = NULL;
                m_info.host_shape.set_size(0);
            }

    };

    /**
     * primarily used as result of tensor::operator[]
     */
    template<class V, class M, class L=row_major>
        class tensor_view
        : public tensor<V,M,L>
        {
            private:
                typedef tensor<V,M,L> super;
                using super::m_memory;
                using super::m_ptr;
                using super::m_info;
            public:
                /** default constructor does nothing */
                tensor_view(){}

                /**
                 * /always/ try to copy memory
                 */
                tensor_view& operator=(const tensor<V,M,L>& o){
                    if(!copy_memory(*this, o, false))
                        throw std::runtime_error("copying tensor to tensor_view did not succeed. Maybe a shape mismatch?");
                    return *this;
                }
                /**
                 * @overload for other memory space type
                 */
                template<class OM>
                tensor_view& operator=(const tensor<V,OM,L>& o){
                    if(!copy_memory(*this, o, false))
                        throw std::runtime_error("copying tensor to tensor_view did not succeed. Maybe a shape mismatch?");
                    return *this;
                }

                /**
                 * construct tensor_view
                 *
                 * @warning if a dimension has size 1, the resulting tensor has fewer dimensions than the original one.
                 *
                 * @warning most operations in CUV on tensors currently only work
                 *          if the subtensor is a connected area in memory.  Basically this
                 *          means that you can only slice in the first dimension which has
                 *          size>1.
                 *
                 * @param eg  the indices of the subtensor
                 * @param o   the original tensor
                 *
                 * Example:
                 * @code
                 * tensor<float,host_memory_space> v(extents[5][10]);
                 *
                 * // these are equivalent:
                 * tensor<float,host_memory_space> w0(v,indices[index_range(2,3)][index_range(0,10)]);
                 * tensor<float,host_memory_space> w0(v,indices[index_range(2,3)][index_range()]);
                 * tensor<float,host_memory_space> w0(v,indices[index_range(2,3)][index_range() < index(10)]);
                 * tensor<float,host_memory_space> w0(v,indices[index_range(2,3)][index(0) < index_range() < index(10)]);
                 *
                 * // yields a 1D-tensor corresponding to the 2nd slice in the 1st dimension:
                 * tensor<float,host_memory_space> w0(indices[1][index_range()]);
                 * @endcode
                 */
                template<int D, int E>
                    explicit tensor_view(const tensor<V,M,L>&o, const index_gen<D,E>& idx)
                    {
                        m_memory = o.mem();
                        m_ptr    = const_cast<V*>(o.ptr());
                        std::vector<int> shapes;
                        std::vector<int> strides;
                        shapes.reserve(D);
                        strides.reserve(D);
                        cuvAssert(o.ndim()==D);
                        for(std::size_t i=0;i<D;i++){
                            int start  = idx.ranges_[i].get_start(0);
                            int finish = idx.ranges_[i].get_finish(o.shape(i));
                            int stride = idx.ranges_[i].stride();
                            if (start <0) start  += o.shape(i);
                            if (finish<0) finish += o.shape(i);
#ifndef NDEBUG
                            cuvAssert(finish>start);
#endif
                            m_ptr += start*o.stride(i);
                            if(finish-start==1){
                                // skip dimension
                            }else{
                                shapes.push_back((finish-start)/stride);
                                strides.push_back(o.stride(i)*stride);
                            }
                        }
                        // store in m_info
                        m_info.resize(shapes.size());
                        std::copy(shapes.begin(),shapes.end(),m_info.host_shape[0].ptr);
                        std::copy(strides.begin(),strides.end(),m_info.host_stride[0].ptr);
                    }
                /**
                 * different order of arguments as above, all else being equal.
                 *
                 * @deprecated
                 * @param idx a set of index ranges into o
                 * @param o   other tensor
                 */
                template<int D, int E>
                    explicit tensor_view( const index_gen<D,E>& idx, const tensor<V,M,L>&o)
                    {
                        m_memory = o.mem();
                        m_ptr    = const_cast<V*>(o.ptr());
                        std::vector<int> shapes;
                        std::vector<int> strides;
                        shapes.reserve(D);
                        strides.reserve(D);
                        cuvAssert(o.ndim()==D);
                        for(std::size_t i=0;i<D;i++){
                            int start  = idx.ranges_[i].get_start(0);
                            int finish = idx.ranges_[i].get_finish(o.shape(i));
                            int stride = idx.ranges_[i].stride();
                            if (start <0) start  += o.shape(i);
                            if (finish<0) finish += o.shape(i);
#ifndef NDEBUG
                            cuvAssert(finish>start);
#endif
                            m_ptr += start*o.stride(i);
                            if(finish-start==1){
                                // skip dimension
                            }else{
                                shapes.push_back((finish-start)/stride);
                                strides.push_back(o.stride(i)*stride);
                            }
                        }
                        // store in m_info
                        m_info.resize(shapes.size());
                        std::copy(shapes.begin(),shapes.end(),m_info.host_shape[0].ptr);
                        std::copy(strides.begin(),strides.end(),m_info.host_stride[0].ptr);
                    }
        };

    //namespace detail{
        /// tries to copy memory, succeeds if shapes match AND both tensors are c_contiguous of 2dcopyable.
        template<class V, class M0, class M1, class L0, class L1>
            bool copy_memory(tensor<V,M0,L0>&dst, const tensor<V,M1,L1>&src, bool force_dst_contiguous){
                typedef typename tensor<V,M0,L0>::size_type size_type;
                allocator<V, size_type, M0> a;
                if(dst.shape() == src.shape() && dst.ptr()){
                    if(dst.is_c_contiguous() && src.is_c_contiguous()){
                        // can copy w/o bothering about m_memory
                        a.copy(dst.ptr(), src.ptr(), src.size(), M1());
                    }else if(dst.is_c_contiguous() && src.is_2dcopyable()){
                        size_type row,col,pitch;
                        detail::get_pitched_params(row,col,pitch,src.info().host_shape, src.info().host_stride,L1());
                        a.copy2d(dst.ptr(), src.ptr(), col*sizeof(V),pitch*sizeof(V),row,col,M1());
                    }else if(!force_dst_contiguous && dst.is_2dcopyable() && src.is_c_contiguous()){
                        size_type row,col,pitch;
                        detail::get_pitched_params(row,col,pitch,dst.info().host_shape, dst.info().host_stride,L0());
                        a.copy2d(dst.ptr(), src.ptr(), pitch*sizeof(V),col*sizeof(V),row,col,M1());
                    }else if(!force_dst_contiguous && dst.is_2dcopyable() && src.is_c_contiguous()){
                        size_type srow,scol,spitch;
                        size_type drow,dcol,dpitch;
                        detail::get_pitched_params(drow,dcol,dpitch,dst.info().host_shape, dst.info().host_stride,L0());
                        detail::get_pitched_params(srow,scol,spitch,src.info().host_shape, src.info().host_stride,L1());
                        cuvAssert(scol==srow);
                        cuvAssert(dcol==drow);
                        a.copy2d(dst.ptr(), src.ptr(), dpitch*sizeof(V),spitch*sizeof(V),srow,scol,M1());
                    }else{
                        throw std::runtime_error("copying of generic strides not implemented yet");
                    }
                    if(!IsSame<L0,L1>::Result::value){
                        dst.info().host_stride.reverse();
                        dst.info().host_shape.reverse();
                    }
                    return true;
                }
                return false;
            }

        /// copies between different memory spaces
        template<class V, class M0, class M1, class L0, class L1>
            void copy_memory(tensor<V,M0,L0>&dst, const tensor<V,M1,L1>&src, linear_memory_tag){
                typedef typename tensor<V,M0,L0>::size_type size_type;
                if(copy_memory(dst,src, true)) // destination must be contiguous
                    return;
                dst.info().resize(src.ndim());
                dst.info().host_shape = src.info().host_shape;
                linear_memory<V,M0> d(src.size());
                d.set_strides(dst.info().host_stride,dst.info().host_shape, L0());
                allocator<V, size_type, M0> a;
                if(src.is_c_contiguous()){ 
                    // easiest case: both linear, simply copy
                    a.copy(d.ptr(), src.ptr(), src.size(), M1());
                }
                else if(src.is_2dcopyable()){
                    // other memory is probably a pitched memory or some view onto an array
                    size_type row,col,pitch;
                    detail::get_pitched_params(row,col,pitch,src.info().host_shape, src.info().host_stride,L1());
                    a.copy2d(d.ptr(), src.ptr(), col*sizeof(V),pitch*sizeof(V),row,col,M1());
                }else{
                    throw std::runtime_error("copying arbitrarily strided memory not implemented");
                }
                dst.mem().reset(new memory<V,M0>(d.release(),d.size()));
                if(!IsSame<L0,L1>::Result::value){
                    dst.info().host_stride.reverse();
                    dst.info().host_shape.reverse();
                }
            }

        /// copies between different memory spaces
        template<class V, class M0, class M1, class L0, class L1>
            void copy_memory(tensor<V,M0,L0>&dst, const tensor<V,M1,L1>&src, pitched_memory_tag){
                typedef typename tensor<V,M0,L0>::size_type size_type;
                assert(src.ndim()>=2);
                if(copy_memory(dst,src,false)) // destination need not be contiguous
                    return;
                dst.info().resize(src.ndim());
                dst.info().host_shape = src.info().host_shape;
                size_type row,col,pitch;
                detail::get_pitched_params(row,col,pitch,src.info().host_shape, src.info().host_stride,L1());
                pitched_memory<V,M0> d(row,col);
                //dst.mem().reset(d);
                d->set_strides(dst.info().host_stride,dst.info().host_shape, L0());
                allocator<V, size_type, M0> a;
                if(src.is_2dcopyable()){
                    // other memory is probably a pitched memory or some view onto an array
                    detail::get_pitched_params(row,col,pitch,src.info().host_shape, src.info().host_stride,L1());
                    a.copy2d(d.ptr(), src.m_ptr, d.pitch()*sizeof(V),pitch*sizeof(V),row,col,M1());
                }else{
                    throw std::runtime_error("copying arbitrarily strided memory not implemented");
                }
                dst.mem().reset(new memory<V,M0>(d.release(),d.size()));

                if(!IsSame<L0,L1>::Result::value){
                    dst.info().host_stride.reverse();
                    dst.info().host_shape.reverse();
                }
            }
    //}

        /** @} */ // basics


        /**
         * test whether two tensors have the same shape
         * @ingroup tools
         */
    template<class V, class V2, class M, class M2, class L>
        bool equal_shape(const tensor<V,M,L>& a, const tensor<V2,M2,L>& b){
            return a.shape()==b.shape();
        }
    
    /**
     * @addtogroup MetaProgramming
     */
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

    /** @} */

}

/**
 * input and output operations
 *
 * @addtogroup io
 * @{
 */
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
     * print a dev tensor to a stream (copying to host first)
     *
     * @param o the stream
     * @param t the tensor
     */
    template<class V, class L>
    ostream& operator<<(ostream& o, const cuv::tensor<V, cuv::dev_memory_space, L>& t){
        return o << cuv::tensor<V,cuv::host_memory_space,L>(t);
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
                    //for(unsigned int j=0;j<t.shape(2);j++) o<< t(l,i,j)<<" ";
                    for(unsigned int j=0;j<t.shape(2);j++) o<< t[l*t.shape(1)*t.shape(2) + i*t.shape(2) + j]<<" ";
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
/** @} */ // io
#endif /* __TENSOR2_HPP__ */
