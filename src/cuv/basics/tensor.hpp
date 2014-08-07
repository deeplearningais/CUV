#ifndef __CUV_NDARRAY_HPP__
#define __CUV_NDARRAY_HPP__

#include <boost/multi_array/extent_gen.hpp>
#include <boost/multi_array/index_gen.hpp>
#include <boost/shared_ptr.hpp>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "allocators.hpp"
#include "memory.hpp"
#include <cuv/tools/meta_programming.hpp>
#include <cuv/tools/cuv_general.hpp>

namespace cuv {


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
typedef boost::detail::multi_array::index_range<boost::detail::multi_array::index, boost::detail::multi_array::size_type> index_range;

/**
 * the index type used in index_range, useful for comparator syntax in @see index_range
 */
typedef index_range::index index;

#ifndef CUV_DONT_CREATE_EXTENTS_OBJ

namespace {
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
 * tensor_view<...> v(indices[index_range(1,3)][index_range()], other_ndarray);
 * // or, equivalently
 * other_ndarray[indices[index_range(1,3)][index_range()]];
 * @endcode
 */
index_gen<0, 0> indices;
}
#endif

/**
 * @addtogroup data_structures Basic datastructures
 * @{
 */

template<class V, class M, class L> class tensor;
template<class V, class M, class L> class tensor_view;

/// used in implementation of tensor.operator= for value_type argument
template<class V, class M, class L, class S>
void fill(tensor<V, M, L>& v, const S& p);

namespace detail {

/**
 * this is intended for copying pitched memory.
 *
 * given shape, stride and a memory layout, we can determine the number of
 * rows, columns and the pitch of a
 */
template<class index_type, class size_type>
void get_pitched_params(size_type& rows, size_type& cols, size_type& pitch,
        const linear_memory<size_type, host_memory_space>& shape,
        const linear_memory<index_type, host_memory_space>& stride, row_major) {
    // strided dimension is the LAST one
    rows = std::accumulate(shape[0].ptr, shape[0].ptr + shape.size() - 1, 1, std::multiplies<index_type>());
    cols = shape[shape.size() - 1];
    pitch = stride[shape.size() - 2];
}

/**
 * @overload
 */
template<class index_type, class size_type>
void get_pitched_params(size_type& rows, size_type& cols, size_type& pitch,
        const linear_memory<size_type, host_memory_space>& shape,
        const linear_memory<index_type, host_memory_space>& stride, column_major) {
    // strided dimension is the FIRST one
    rows = std::accumulate(shape[0].ptr + 1, shape[0].ptr + shape.size(), 1, std::multiplies<index_type>());
    cols = shape[0];
    pitch = stride[1];
}

}

/**
 * contains infos about shape and stride on host and in the tensor data space.
 */
template<class M, class L>
class tensor_info {

public:

    typedef unsigned int size_type; ///< type of shapes of the tensor
    typedef int index_type; ///< type of indices in tensor
    typedef M data_memory_space; ///< this is where the data lies

    boost::shared_ptr<allocator> m_allocator;

    /// shape stored in host memory
    linear_memory<size_type, host_memory_space> host_shape;

    /// strides stored in host memory
    linear_memory<index_type, host_memory_space> host_stride;

    /// shape stored in data memory
    linear_memory<size_type, data_memory_space> data_shape;

    /// strides stored in data memory
    linear_memory<index_type, data_memory_space> data_stride;

    /// default constructor: does nothing
    tensor_info(const boost::shared_ptr<allocator>& _allocator) :
            m_allocator(_allocator), host_shape(_allocator), host_stride(_allocator),
                    data_shape(_allocator), data_stride(_allocator)
    {
    }

    /// @return the size of the arrays (should all be the same)
    size_type size() {
        return host_shape.size();
    }

    /// construct with known shape
    tensor_info(size_type s, const boost::shared_ptr<allocator>& _allocator) :
            m_allocator(_allocator), host_shape(_allocator), host_stride(_allocator),
                    data_shape(_allocator), data_stride(_allocator)
    {
        resize(s);
    }

    /// resize all memories
    void resize(size_type s) {
        host_shape.set_size(s);
        host_stride.set_size(s);
    }

    /// copy-constructor
    tensor_info(const tensor_info<M, L>& o) :
            m_allocator(o.m_allocator), host_shape(o.host_shape), host_stride(o.host_stride),
                    data_shape(m_allocator), data_stride(m_allocator)
    {
    }

    /// copy-construct from other memory space
    template<class OM>
    tensor_info(const tensor_info<OM, L>& o) :
            m_allocator(o.m_allocator), host_shape(o.host_shape), host_stride(o.host_stride),
                    data_shape(m_allocator), data_stride(m_allocator)
    {
    }

};

/**
 * represents an n-dimensional array on GPU or CPU.
 */
template<class V, class M, class L = row_major>
class tensor {

public:

    typedef memory<V, M> memory_type; ///< type of stored memory
    typedef typename memory_type::reference_type reference_type; ///< values returned by operator() and []
    typedef typename memory_type::const_reference_type const_reference_type; ///< values returned by operator()
    typedef typename memory_type::memory_space_type memory_space_type; ///< dev/host
    typedef typename memory_type::value_type value_type; ///< type of stored values
    typedef typename memory_type::size_type size_type; ///< type shapes
    typedef typename memory_type::index_type index_type; ///< type strides
    typedef L memory_layout_type; ///< column/row major

    typedef tensor_info<M, L> info_type; ///< type of shape info struct
    typedef tensor_view<V, M, L> view_type; ///< type of views on this tensor

public:
    boost::shared_ptr<allocator> m_allocator;

private:
    void check_size_limit(size_t size) const {
        if (size > static_cast<size_t>(std::numeric_limits<index_type>::max())) {
            throw std::runtime_error("maximum tensor size exceeded");
        }
    }

    /// tensor views are our friends
    template<class _V, class _M, class _L>
    friend class tensor_view;

protected:

    /// information about shape, strides
    info_type m_info;

    /// points to (possibly shared) memory
    boost::shared_ptr<memory_type> m_memory;

    /// points to start of actually referenced memory (within m_memory)
    V* m_ptr;

    /**
     * determine linear index in memory of an index array
     *
     * this function takes strides etc. into account, so that indices
     * are interpreted as relative to the (strided) sub-tensor we're
     * referring to.
     *
     * @param D    size of index array
     * @param arr  index array
     * @return linear index in memory of index array
     *
     */
    size_type index_of(int D, index_type* arr) const {
        index_type pos = 0;
        for (int i = 0; i < D; i++) {
            index_type temp = arr[i];
            if (temp < 0)
                temp = m_info.host_shape[i] + temp;
            pos += temp * m_info.host_stride[i];
        }
        return pos;
    }

    /**
     * allocate linear memory (c-contiguous version)
     *
     * @param t tensor to allocate
     */
    void allocate(tensor& t, linear_memory_tag) {
        linear_memory<V, M> mem(t.size(), t.m_allocator);
        mem.set_strides(t.m_info.host_stride, t.m_info.host_shape, L());
        t.m_ptr = mem.ptr();
        t.m_memory.reset(new memory<V, M>(mem.release(), mem.size(), t.m_allocator));
    }

    /**
     * @overload
     *
     * pitched version
     */
    void allocate(tensor& t, pitched_memory_tag) {
        typename tensor<V, M, L>::size_type row, col, pitch;
        detail::get_pitched_params(row, col, pitch, t.m_info.host_shape, t.m_info.host_stride, L());
        pitched_memory<V, M> d(row, col);
        d.set_strides(t.m_info.host_stride, t.m_info.host_shape, L());
        t.m_ptr = d.ptr();
        t.m_memory.reset(new memory<V, M>(d.release(), d.size(), t.m_allocator));
    }

public:

    /**
     * determine linear index in memory of an index array
     *
     * this function takes strides etc. into account, so that indices
     * are interpreted as relative to the (strided) sub-tensor we're
     * referring to.
     *
     * @tparam D    size of index array
     * @param eg  position in array
     * @return linear index in memory of index array
     *
     */
    template<size_t D>
    size_type index_of(const extent_gen<D>& eg) const {
        index_type pos = 0;
        for (size_t i = 0; i < D; i++) {
            index_type temp = eg.ranges_[i].finish();
            if (temp < 0)
                temp = m_info.host_shape[i] + temp;
            pos += temp * m_info.host_stride[i];
        }
        return pos;
    }

    /**
     * @name Accessors
     * @{
     */
    /// return the number of dimensions
    index_type ndim() const {
        return m_info.host_shape.size();
    }

    /** return the size of the i-th dimension
     *  @param i the index of the queried dimension
     */
    size_type shape(const size_t i) const {
        return m_info.host_shape[i];
    }

    /** return the stride of the i-th dimension
     *  @param i the index of the queried dimension
     */
    index_type stride(const size_t i) const {
        return m_info.host_stride[i];
    }

    /** @return the pointer to the referenced memory */
    V* ptr() {
        return m_ptr;
    }

    /**
     * @overload
     * @return the const pointer to the referenced memory
     * */
    const V* ptr() const {
        return m_ptr;
    }

    /** set the pointer offset (used in deserialization) */
    void set_ptr_offset(long int i) {
        m_ptr = m_memory->ptr() + i;
    }

    /** * @return pointer to allocated memory */
    boost::shared_ptr<memory_type>& mem() {
        return m_memory;
    }
    /**
     * @overload
     * @return the const pointer to the allocated memory
     * */
    const boost::shared_ptr<memory_type>& mem() const {
        return m_memory;
    }

    /** @return the number of stored elements
     */
    size_type size() const {
        size_t size = std::accumulate(m_info.host_shape[0].ptr, m_info.host_shape[0].ptr + m_info.host_shape.size(), 1,
                std::multiplies<size_t>());

        check_size_limit(size);

        return static_cast<size_type>(size);
    }

    /**
     * determine size in bytes
     *
     * assumes that the memory is c_contiguous!
     *
     * @return the size in bytes
     */
    size_type memsize() const {
#ifndef NDEBUG
        cuvAssert(is_c_contiguous());
#endif
        size_t size = std::accumulate(m_info.host_shape[0].ptr, m_info.host_shape[0].ptr + m_info.host_shape.size(), (size_t)sizeof(value_type),
                std::multiplies<size_type>());

        check_size_limit(size);

        return static_cast<size_type>(size);
    }

    /// return the shape of the tensor (as a vector for backward compatibility)
    std::vector<size_type> shape() const {
        if (ndim() == 0)
            return std::vector<size_type>();
        return std::vector<size_type>(m_info.host_shape[0].ptr, m_info.host_shape[0].ptr + m_info.host_shape.size());
    }

    /**
     * return the effective shape of the tensor (as a vector for backward compatibility)
     *
     * the effective shape removes all degenerate dimensions (i.e. shape(i)==1).
     */
    std::vector<size_type> effective_shape() const {
        std::vector<size_type> shape;
        shape.reserve(ndim());
        if (ndim() == 0)
            return shape;
        std::remove_copy_if(m_info.host_shape[0].ptr, m_info.host_shape[0].ptr + m_info.host_shape.size(),
                std::back_inserter(shape), std::bind2nd(std::equal_to<size_type>(), 1));
        return shape;
    }

    /// @return the tensor info struct (const)
    const info_type& info() const {
        return m_info;
    }

    /// @return the tensor info struct
    info_type& info() {
        return m_info;
    }

    /// true iff there are no "holes" in memory
    bool is_c_contiguous() const {
        return detail::is_c_contiguous(memory_layout_type(), m_info.host_shape, m_info.host_stride);
    }

    /// true iff it can be copied as a 2d array (only one dimension is pitched)
    bool is_2dcopyable() const {
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
    reference_type operator[](index_type idx) {
        size_type ndim = m_info.host_shape.size();
        size_type* virtualstride = new size_type[ndim];
        size_type pos = 0;
        if (IsSame<L, row_major>::Result::value) {
            // row major
            {
                size_type virt_size = 1;
                for (int i = ndim - 1; i >= 0; --i) {
                    virtualstride[i] = virt_size;
                    virt_size *= m_info.host_shape[i];
                }
            }
            for (size_type i = 0; i < ndim; ++i) {
                pos += (idx / virtualstride[i]) * m_info.host_stride[i];
                idx -= (idx / virtualstride[i]) * virtualstride[i];
            }
        } else {
            // column major
            {
                size_type virt_size = 1;
                for (unsigned int i = 0; i < ndim; ++i) {
                    virtualstride[i] = virt_size;
                    virt_size *= m_info.host_shape[i];
                }
            }
            for (int i = ndim - 1; i >= 0; --i) {
                pos += (idx / virtualstride[i]) * m_info.host_stride[i];
                idx -= (idx / virtualstride[i]) * virtualstride[i];
            }
        }
        delete[] virtualstride;
        return reference_type(m_ptr + pos);
    }

    /** @overload */
    const_reference_type operator[](index_type idx) const {
        return const_cast<tensor&>(*this)[idx];
    }

    /**
     * get a reference to the datum at an index
     * @param i0 index for a 1-dimensional tensor
     * @return reference to datum at i0
     */
    reference_type operator()(index_type i0) {
#ifndef NDEBUG
        cuvAssert(ndim()==1);
        cuvAssert((i0>=0 && (size_type)i0 < shape(0)) || (i0<0 && (size_type)(-i0)<shape(0)+1))
#endif
        if (i0 >= 0) {
            return reference_type(m_ptr + i0);
        } else {
            return reference_type(m_ptr + shape(0) - i0);
        }
    }

    /** @overload */
    const_reference_type operator()(index_type i0) const {
        return const_cast<tensor&>(*this)(i0);
    }

    /** @overload */
    const_reference_type operator()(index_type i0, index_type i1) const {
        return const_cast<tensor&>(*this)(i0, i1);
    }

    /** @overload */
    reference_type operator()(index_type i0, index_type i1) {
#ifndef NDEBUG
        cuvAssert(ndim()==2);
        cuvAssert((i0>=0 && (size_type)i0 < shape(0)) || (i0<0 && (size_type)(-i0)<shape(0)+1))
        cuvAssert((i1>=0 && (size_type)i1 < shape(1)) || (i1<0 && (size_type)(-i1)<shape(1)+1))
#endif
        index_type arr[2] = { i0, i1 };
        return reference_type(m_ptr + index_of(2, arr));
    }

    /** @overload */
    const_reference_type operator()(index_type i0, index_type i1, index_type i2) const {
        return const_cast<tensor&>(*this)(i0, i1, i2);
    }

    /** @overload */
    reference_type operator()(index_type i0, index_type i1, index_type i2) {
#ifndef NDEBUG
        cuvAssert(ndim()==3);
        cuvAssert((i0>=0 && (size_type)i0 < shape(0)) || (i0<0 && (size_type)-i0<shape(0)+1))
        cuvAssert((i1>=0 && (size_type)i1 < shape(1)) || (i1<0 && (size_type)-i1<shape(1)+1))
        cuvAssert((i2>=0 && (size_type)i2 < shape(2)) || (i2<0 && (size_type)-i2<shape(2)+1))
#endif
        index_type arr[3] = { i0, i1, i2 };
        return reference_type(m_ptr + index_of(3, arr));
    }

    /** @overload */
    const_reference_type operator()(index_type i0, index_type i1, index_type i2, index_type i3) const {
        return const_cast<tensor&>(*this)(i0, i1, i2, i3);
    }

    /** @overload */
    reference_type operator()(index_type i0, index_type i1, index_type i2, index_type i3) {
#ifndef NDEBUG
        cuvAssert(ndim()==4);
        cuvAssert((i0>=0 && (size_type)i0 < shape(0)) || (i0<0 && (size_type)-i0<shape(0)+1))
        cuvAssert((i1>=0 && (size_type)i1 < shape(1)) || (i1<0 && (size_type)-i1<shape(1)+1))
        cuvAssert((i2>=0 && (size_type)i2 < shape(2)) || (i2<0 && (size_type)-i2<shape(2)+1))
        cuvAssert((i3>=0 && (size_type)i3 < shape(3)) || (i3<0 && (size_type)-i3<shape(3)+1))
#endif
        index_type arr[4] = { i0, i1, i2, i3 };
        return reference_type(m_ptr + index_of(4, arr));
    }

    /** @overload */
    const_reference_type operator()(index_type i0, index_type i1, index_type i2, index_type i3,
            index_type i4) const {
        return const_cast<tensor&>(*this)(i0, i1, i2, i3, i4);
    }

    /** @overload */
    reference_type operator()(index_type i0, index_type i1, index_type i2, index_type i3, index_type i4) {
#ifndef NDEBUG
        cuvAssert(ndim()==5);
        cuvAssert((i0>=0 && (size_type)i0 < shape(0)) || (i0<0 && (size_type)-i0<shape(0)+1))
        cuvAssert((i1>=0 && (size_type)i1 < shape(1)) || (i1<0 && (size_type)-i1<shape(1)+1))
        cuvAssert((i2>=0 && (size_type)i2 < shape(2)) || (i2<0 && (size_type)-i2<shape(2)+1))
        cuvAssert((i3>=0 && (size_type)i3 < shape(3)) || (i3<0 && (size_type)-i3<shape(3)+1))
        cuvAssert((i4>=0 && (size_type)i4 < shape(4)) || (i4<0 && (size_type)-i4<shape(4)+1))
#endif
        index_type arr[5] = { i0, i1, i2, i3, i4 };
        return reference_type(m_ptr + index_of(5, arr));
    }

    /** @} */ // accessing stored values
    /** @name constructors
     * @{
     *
     */
    /**
     * default constructor (does nothing)
     */
    tensor(const boost::shared_ptr<allocator> _allocator = boost::make_shared<default_allocator>()) :
            m_allocator(_allocator), m_info(_allocator), m_ptr(NULL) {
    }

    // ****************************************************************
    //        Constructing from other tensor
    // ****************************************************************

    /**
     * construct tensor from tensor of exact same type
     *
     * time O(1)
     */
    tensor(const tensor& o) :
            m_allocator(o.m_allocator),
                    m_info(o.m_info), // copy only shape
                    m_memory(o.m_memory), // increase ref counter
                    m_ptr(o.m_ptr) {
    } // same pointer in memory

    /**
     * construct tensor from tensor of other memory space
     * in (dense) /linear/ memory. Note: this /copies/ the memory!
     */
    template<class OM>
    tensor(const tensor<value_type, OM, L>& o, cudaStream_t stream = 0) :
            m_allocator(o.m_allocator),
                    m_info(o.info()), // primarily to copy shape
                    m_ptr(NULL) {
        copy_memory(o, linear_memory_tag(), stream);
        m_ptr = m_memory->ptr();
    }

    /**
     * construct tensor from tensor of same memory space
     * in  /pitched/ memory. Note: this /copies/ the memory!
     */
    explicit tensor(const tensor& o, pitched_memory_tag, cudaStream_t stream = 0) :
            m_allocator(o.m_allocator),
                    m_info(o.m_info), // primarily to copy shape
                    m_ptr(NULL) {
        copy_memory(o, pitched_memory_tag(), stream);
        m_ptr = m_memory->ptr();
    }

    /**
     * construct tensor from tensor of other memory space
     * in  /pitched/ memory. Note: this /copies/ the memory!
     */
    template<class OM>
    explicit tensor(const tensor<value_type, OM, L>& o, pitched_memory_tag, cudaStream_t stream = 0) :
            m_allocator(o.m_allocator),
                    m_info(o.info()), // primarily to copy shape
                    m_ptr(NULL) {
        copy_memory(o, pitched_memory_tag(), stream);
        m_ptr = m_memory->ptr();
    }

    /**
     * construct tensor from tensor of same memory space
     * in (dense) /linear/ memory. Note: this /copies/ the memory!
     */
    explicit tensor(const tensor& o, linear_memory_tag, cudaStream_t stream = 0) :
            m_allocator(o.m_allocator),
                    m_info(o.m_info), // primarily to copy shape
                    m_ptr(NULL) {
        copy_memory(o, linear_memory_tag(), stream);
        m_ptr = m_memory->ptr();
    }

    /**
     * construct tensor from tensor of other memory space
     * in (dense) /linear/ memory. Note: this /copies/ the memory!
     */
    template<class OM>
    explicit tensor(const tensor<value_type, OM, L>& o, linear_memory_tag, cudaStream_t stream = 0) :
            m_allocator(o.m_allocator),
                    m_info(o.info()), // primarily to copy shape
                    m_ptr(NULL) {
        copy_memory(o, linear_memory_tag(), stream);
        m_ptr = m_memory->ptr();
    }

    /**
     * construct tensor from other memory layout
     *
     * this does not copy memory, but reverses dimensions and strides
     * (and therefore only takes O(1) time)
     */
    template<class OL>
    explicit tensor(const tensor<value_type, M, OL>& o) :
            m_allocator(o.m_allocator),
                    m_info(o.m_allocator),
                    m_memory(o.mem()), // increase ref counter
                    m_ptr(const_cast<V*>(o.ptr())) { // same pointer in memory
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
    explicit tensor(const size_type i,
            const boost::shared_ptr<allocator> _allocator = boost::make_shared<default_allocator>()) :
            m_allocator(_allocator),
                    m_info(_allocator),
                    m_ptr(NULL) {
        m_info.resize(1);
        m_info.host_shape[0] = i;
        allocate(*this, linear_memory_tag());
    }

    /**
     * construct two-dimensional tensor
     */
    explicit tensor(const size_type i, const int j, const boost::shared_ptr<allocator> _allocator =
            boost::make_shared<default_allocator>()) :
            m_allocator(_allocator),
                    m_info(_allocator),
                    m_ptr(NULL) {
        m_info.resize(2);
        m_info.host_shape[0] = i;
        m_info.host_shape[1] = j;
        allocate(*this, linear_memory_tag());
    }

    /**
     * construct tensor from a shape
     */
    template<size_t D>
    explicit tensor(const extent_gen<D>& eg,
            const boost::shared_ptr<allocator> _allocator = boost::make_shared<default_allocator>()) :
            m_allocator(_allocator),
                    m_info(_allocator),
                    m_ptr(NULL) {
        m_info.resize(D);
        for (size_t i = 0; i < D; i++)
            m_info.host_shape[i] = eg.ranges_[i].finish();
        allocate(*this, linear_memory_tag());
    }

    /**
     * construct tensor from a shape
     *
     * @deprecated
     */
    explicit tensor(const std::vector<size_type>& eg,
            const boost::shared_ptr<allocator> _allocator = boost::make_shared<default_allocator>()) :
            m_allocator(_allocator),
                    m_info(_allocator),
                    m_ptr(NULL) {
        m_info.resize(eg.size());
        for (size_t i = 0; i < eg.size(); i++)
            m_info.host_shape[i] = eg[i];
        allocate(*this, linear_memory_tag());
    }

    /**
     * construct tensor from a shape
     *
     * @deprecated
     */
    explicit tensor(const std::vector<size_type>& eg, pitched_memory_tag,
            const boost::shared_ptr<allocator> _allocator = boost::make_shared<default_allocator>()) :
            m_allocator(_allocator),
                    m_info(_allocator),
                    m_ptr(NULL) {
        m_info.resize(eg.size());
        for (size_t i = 0; i < eg.size(); i++)
            m_info.host_shape[i] = eg[i];
        allocate(*this, pitched_memory_tag());
    }

    /**
     * construct tensor from a shape (pitched)
     */
    template<size_t D>
    explicit tensor(const extent_gen<D>& eg, pitched_memory_tag, const boost::shared_ptr<allocator> _allocator =
            boost::make_shared<default_allocator>()) :
            m_allocator(_allocator),
                    m_info(_allocator),
                    m_ptr(NULL) {
        m_info.resize(D);
        for (size_t i = 0; i < D; i++)
            m_info.host_shape[i] = eg.ranges_[i].finish();
        allocate(*this, pitched_memory_tag());
    }

    // ****************************************************************
    //        Constructing from shape and raw pointer
    // ****************************************************************

    /**
     * construct tensor from a shape and a pointer (does not copy memory)
     *
     * @warning You have to ensure that the memory lives as long as this object.
     */
    template<size_t D>
    explicit tensor(const extent_gen<D>& eg, value_type* ptr, const boost::shared_ptr<allocator> _allocator =
            boost::make_shared<default_allocator>()) :
            m_allocator(_allocator),
                    m_info(_allocator),
                    m_ptr(ptr) {
        m_info.resize(D);
        size_t size = 1;
        if (IsSame<memory_layout_type, row_major>::Result::value) {
            for (int i = D - 1; i >= 0; i--) {
                m_info.host_shape[i] = eg.ranges_[i].finish();
                m_info.host_stride[i] = size;
                size *= eg.ranges_[i].finish();
            }
        } else {
            for (size_t i = 0; i < D; i++) {
                m_info.host_shape[i] = eg.ranges_[i].finish();
                m_info.host_stride[i] = size;
                size *= eg.ranges_[i].finish();
            }
        }
        m_memory.reset(new memory<V, M>(ptr, size, m_allocator, false));
    }

    explicit tensor(const std::vector<size_type>& shape, value_type* ptr,
            const boost::shared_ptr<allocator> _allocator = boost::make_shared<default_allocator>()) :
            m_allocator(_allocator),
                    m_info(_allocator),
                    m_ptr(ptr) {
        unsigned int D = shape.size();
        m_info.resize(D);
        size_type size = 1;
        if (IsSame<memory_layout_type, row_major>::Result::value)
            for (int i = D - 1; i >= 0; i--) {
                m_info.host_shape[i] = shape[i];
                m_info.host_stride[i] = size;
                size *= shape[i];
            }
        else
            for (size_t i = 0; i < D; i++) {
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
    explicit tensor(const index_gen<D, E>& idx, value_type* ptr, const boost::shared_ptr<allocator> _allocator =
            boost::make_shared<default_allocator>()) :
            m_allocator(_allocator),
                    m_info(_allocator),
                    m_ptr(ptr) {
        m_info.resize(D);
        size_type size = 1;
        if (IsSame<memory_layout_type, row_major>::Result::value)
            for (int i = D - 1; i >= 0; i--) {
                m_info.host_shape[i] = idx.ranges_[i].finish();
                m_info.host_stride[i] = size;
                size *= idx.ranges_[i].finish();
            }
        else
            for (size_t i = 0; i < D; i++) {
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
     * explicitly assign by copying memory
     */
    template<class _M, class _L>
    tensor& assign(const tensor<V, _M, _L>& o, cudaStream_t stream = 0) {
        if (!copy_memory(o, false, stream))
            throw std::runtime_error("copying tensor did not succeed. Maybe a shape mismatch?");
        return *this;
    }

    /**
     * assign from tensor of same type
     *
     * always an O(1) operation.
     */
    tensor& operator=(const tensor& o) {
        if (this == &o)
            return *this; // check for self-assignment

        // TODO make use of copy-and-swap idiom
        m_memory = o.mem();
        m_ptr = const_cast<V*>(o.ptr());
        m_info = o.info();
        return *this;
    }

    /**
     * assign from value (sets all elements equal to one scalar)
     */
    template<class _V>
    typename boost::enable_if_c<boost::is_convertible<_V, value_type>::value, tensor&>::type operator=(
            const _V& scalar) {
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
    tensor& assign(const tensor<value_type, OM, L>& o, cudaStream_t stream = 0) {
        if (!copy_memory(o, false, stream))
            copy_memory(o, linear_memory_tag(), stream);
        if (mem())
            // if mem() does not exist, we're just wrapping a pointer
            // of a std::vector or so -> simply keep it
            m_ptr = mem()->ptr();
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
    tensor& operator=(const tensor<value_type, OM, L>& o) {
        return assign(o);
    }

    /**
     * assign from tensor of different memory layout type.
     *
     * this does not copy memory, but reverses strides and shapes.
     */
    template<class OL>
    tensor& operator=(const tensor<value_type, M, OL>& o) {
        return assign(o);
    }

    /** @} */ // assignment
    /**
     * copy memory using given allocator tag (linear/pitched)
     */
    template<class T>
    tensor copy(T tag = linear_memory_tag(), cudaStream_t stream = 0) const {
        tensor t(m_allocator);
        const tensor& o = *this;
        t.m_info = o.info();
        t.copy_memory(o, tag, stream);
        t.m_ptr = t.mem()->ptr();
        return t;
    }

    /**
     * copy memory using linear memory
     */
    tensor copy() const {
        return copy(linear_memory_tag());
    }

    /**
     * create a sub-tensor of the current tensor
     *
     * this works in O(1).
     */
    template<int D, int E>
    tensor_view<V, M, L> operator[](const index_gen<D, E>& idx) const {

        tensor_view<V, M, L> t(m_allocator);
        const tensor& o = *this;
        t.m_memory = o.mem();
        t.m_ptr = const_cast<V*>(o.ptr());

        std::vector<int> shapes;
        std::vector<int> strides;
        shapes.reserve(D);
        strides.reserve(D);
        //cuvAssert(o.ndim()==D);

        for (size_t i = 0; i < D; i++) {
            int start = idx.ranges_[i].get_start(0);
            int finish = idx.ranges_[i].get_finish(o.shape(i));
            int stride = idx.ranges_[i].stride();
            if (start < 0)
                start += o.shape(i);
            if (finish < 0)
                finish += o.shape(i);
#ifndef NDEBUG
            cuvAssert(finish>start);
            cuvAssert(finish <= (int) o.shape(i));
#endif
            t.m_ptr += start * o.stride(i);
            if (idx.ranges_[i].is_degenerate()) {
                // skip dimension
            } else {
                shapes.push_back((finish - start) / stride);
                strides.push_back(o.stride(i) * stride);
            }
        }

        // adds missing shapes
        for(int i = D; i < o.ndim();i++){
            shapes.push_back(o.shape(i));
            strides.push_back(o.stride(i));
        }

        // store in m_info
        t.m_info.resize(shapes.size());

        std::copy(shapes.begin(), shapes.end(), t.m_info.host_shape[0].ptr);
        std::copy(strides.begin(), strides.end(), t.m_info.host_stride[0].ptr);
        return t; // should not copy mem, only m_info
    }

    /**
     * reshape the tensor (in place)
     *
     * works only for c_contiguous memory!
     *
     * @param eg new shape
     */
    template<size_t D>
    void reshape(const extent_gen<D>& eg) {
        std::vector<size_type> shape(D);
        for (size_t i = 0; i < D; i++)
            shape[i] = eg.ranges_[i].finish();
        reshape(shape);
    }
    /**
     * reshape the tensor (in place)
     *
     * works only for c_contiguous memory!
     *
     * @param shape new shape
     */
    void reshape(const std::vector<size_type>& shape) {
        size_type new_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_type>());
        if (!is_c_contiguous())
            throw std::runtime_error("cannot reshape: tensor is not c_contiguous");
        if (size() != new_size)
            throw std::runtime_error("cannot reshape: products do not match");
        m_info.resize(shape.size());
        size_type size = 1;
        if (IsSame<memory_layout_type, row_major>::Result::value)
            for (int i = shape.size() - 1; i >= 0; i--) {
                m_info.host_shape[i] = shape[i];
                m_info.host_stride[i] = size;
                size *= shape[i];
            }
        else
            for (size_t i = 0; i < shape.size(); i++) {
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
    void reshape(size_type r, size_type c) {
        reshape(extents[r][c]);
    }

    /**
     * resize the tensor (deallocates memory if product changes, otherwise equivalent to reshape)
     *
     * @param shape new shape
     */
    void resize(const std::vector<size_type>& shape) {
        if (ndim() != 0) {
            size_type new_size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<size_type>());
            if (is_c_contiguous() && size() == new_size) {
                reshape(shape);
                return;
            }
        }

        // free memory before we allocate new memory (important if pooling is active)
        m_memory.reset(new memory<V, M>(0, 0, m_allocator));
        *this = tensor(shape, m_allocator);
    }
    /**
     * resize the tensor (deallocates memory if product changes, otherwise equivalent to reshape)
     *
     * @overload
     *
     * @param eg new shape
     */
    template<size_t D>
    void resize(const extent_gen<D>& eg) {
        std::vector<size_type> shape(D);
        for (size_t i = 0; i < D; i++)
            shape[i] = eg.ranges_[i].finish();
        resize(shape);
    }

    /**
     * convenience wrapper for resize(extents[size])
     * @param size size of the new shape
     */
    void resize(size_type size) {
        resize(extents[size]);
    }

    /**
     * convenience wrapper for resize(extents[r][c])
     * @param r leading index of new shape
     * @param c following index of new shape
     */
    void resize(size_type r, size_type c) {
        resize(extents[r][c]);
    }

    /**
     * force deallocation of memory if possible
     */
    void dealloc() {
        m_memory.reset();
        m_ptr = NULL;
        m_info.host_shape.set_size(0);
    }

    /** Tries to copy memory w/o reallocation.
     *
     * Succeeds if shapes match AND both ndarrays are c_contiguous or
     * 2d-copyable.
     */
    template<class OM, class OL>
    bool copy_memory(const tensor<V, OM, OL>& src, bool force_dst_contiguous, cudaStream_t stream) {
        if (effective_shape() != src.effective_shape() || !ptr()) {
            return false;
        }

        assert(m_memory.get());
        //assert(m_ptr == m_memory->ptr());

        // TODO: this could be probably implemented in the memory classes as well

        if (is_c_contiguous() && src.is_c_contiguous()) {
            // can copy w/o bothering about m_memory
            //m_memory->copy_from(src.ptr(), src.size(), OM(), stream);
            detail::copy(m_ptr, src.ptr(), src.size(), memory_space_type(), OM(), stream);
        } else if (is_c_contiguous() && src.is_2dcopyable()) {
            size_type row, col, pitch;
            detail::get_pitched_params(row, col, pitch, src.info().host_shape, src.info().host_stride, OL());
            //m_memory->copy2d_from(src.ptr(), col, pitch, row, col, OM(), stream);
            detail::copy2d(m_ptr, src.ptr(), col, pitch,
                    row, col, memory_space_type(), OM(), stream);
        } else if (!force_dst_contiguous && is_2dcopyable() && src.is_c_contiguous()) {
            size_type row, col, pitch;
            detail::get_pitched_params(row, col, pitch, info().host_shape, info().host_stride, L());
            //m_memory->copy2d_from(src.ptr(), pitch, col, row, col, OM(), stream);
            detail::copy2d(m_ptr, src.ptr(), pitch, col, row, col, memory_space_type(), OM(), stream);
        } else if (!force_dst_contiguous && is_2dcopyable() && src.is_2dcopyable()) {
            size_type srow, scol, spitch;
            size_type drow, dcol, dpitch;
            detail::get_pitched_params(drow, dcol, dpitch, info().host_shape, info().host_stride, L());
            detail::get_pitched_params(srow, scol, spitch, src.info().host_shape, src.info().host_stride, OL());
            cuvAssert(scol==dcol);
            cuvAssert(srow==drow);
            //m_memory->copy2d_from(src.ptr(), dpitch, spitch, srow, scol, OM(), stream);
            detail::copy2d(m_ptr, src.ptr(), dpitch, spitch, srow, scol, memory_space_type(), OM(), stream);
        } else {
            throw std::runtime_error("copying of generic strides not implemented yet");
        }

        if (!IsSame<L, OL>::Result::value) {
            info().host_stride.reverse();
            info().host_shape.reverse();
        }
        return true;
    }

    /**
     * Copies between different memory spaces, without pitching destination.
     *
     * Reallocates the destination if its shape does not match.
     *
     */
    template<class OM, class OL>
    void copy_memory(const tensor<V, OM, OL>& src, linear_memory_tag, cudaStream_t stream) {
        if (copy_memory(src, true, stream)) // destination must be contiguous
            return;
        info().resize(src.ndim());
        info().host_shape = src.info().host_shape;

        // free old memory
        m_memory.reset(new memory<V, M>(m_allocator));

        linear_memory<V, M> d(src.size(), m_allocator);
        d.set_strides(info().host_stride, info().host_shape, L());
        if (src.is_c_contiguous()) {
            // easiest case: both linear, simply copy
            d.copy_from(src.ptr(), src.size(), OM(), stream);
        } else if (src.is_2dcopyable()) {
            // other memory is probably a pitched memory or some view onto an array
            size_type row, col, pitch;
            detail::get_pitched_params(row, col, pitch, src.info().host_shape, src.info().host_stride, OL());
            d.copy2d_from(src.ptr(), col, pitch, row, col, OM(), stream);
        } else {
            throw std::runtime_error("copying arbitrarily strided memory not implemented");
        }
        mem().reset(new memory<V, M>(d.release(), d.size(), m_allocator));
        m_ptr = mem()->ptr();
        if (!IsSame<L, OL>::Result::value) {
            info().host_stride.reverse();
            info().host_shape.reverse();
        }
    }

    /**
     * Copies between different memory spaces, with pitched destination.
     *
     * Reallocates the destination if its shape does not match.
     */
    template<class OM, class OL>
    void copy_memory(const tensor<V, OM, OL>& src, pitched_memory_tag, cudaStream_t stream) {
        assert(src.ndim()>=2);
        if (copy_memory(src, false, stream)) // destination need not be contiguous
            return;
        info().resize(src.ndim());
        info().host_shape = src.info().host_shape;
        size_type row, col, pitch;
        detail::get_pitched_params(row, col, pitch, src.info().host_shape, src.info().host_stride, OL());
        pitched_memory<V, M> d(row, col);
        //dst.mem().reset(d);
        d->set_strides(info().host_stride, info().host_shape, L());
        if (src.is_2dcopyable()) {
            // other memory is probably a pitched memory or some view onto an array
            detail::get_pitched_params(row, col, pitch, src.info().host_shape, src.info().host_stride, OL());
            d.copy2d_from(src, stream);
        } else {
            throw std::runtime_error("copying arbitrarily strided memory not implemented");
        }
        mem().reset(new memory<V, M>(d.release(), d.size(), m_allocator));

        m_ptr = mem()->ptr();
        if (!IsSame<L, OL>::Result::value) {
            info().host_stride.reverse();
            info().host_shape.reverse();
        }
    }

};

/**
 * primarily used as result of tensor::operator[]
 */
template<class V, class M, class L = row_major>
class tensor_view: public tensor<V, M, L>
{
private:
    typedef tensor<V, M, L> super;
    using super::m_memory;
    using super::m_ptr;
    using super::m_info;

    template<class _V, class _M, class _L>
    friend class tensor;

public:

    /** default constructor does nothing */
    tensor_view(const boost::shared_ptr<allocator>& allocator) :
            tensor<V, M, L>(allocator) {
    }

    /**
     * /always/ try to copy memory
     */
    tensor_view& assign(const tensor<V, M, L>& o, cudaStream_t stream = 0) {
        if (!this->copy_memory(o, false, stream))
            throw std::runtime_error("copying tensor to tensor_view did not succeed. Maybe a shape mismatch?");
        return *this;
    }

    /**
     * /always/ try to copy memory
     */
    tensor_view& assign(const tensor_view<V, M, L>& o, cudaStream_t stream = 0) {
        if (!this->copy_memory(o, false, stream))
            throw std::runtime_error("copying tensor to tensor_view did not succeed. Maybe a shape mismatch?");
        return *this;
    }

    /**
     * assignment operator for other memory space type
     *
     * @param o a tensor of another memory space type
     */
    template<class OM>
    tensor_view& assign(const tensor<V, OM, L>& o, cudaStream_t stream = 0) {
        if (!this->copy_memory(o, false, stream))
            throw std::runtime_error("copying tensor to tensor_view did not succeed. Maybe a shape mismatch?");
        return *this;
    }

    /**
     * assignment operator for views in other memory space types
     *
     * @param o a tensor_view of another memory space type
     */
    template<class OM>
    tensor_view& assign(const tensor_view<V, OM, L>& o, cudaStream_t stream = 0) {
        if (!this->copy_memory(o, false, stream))
            throw std::runtime_error("copying tensor to tensor_view did not succeed. Maybe a shape mismatch?");
        return *this;
    }

    /**
     * /always/ try to copy memory
     */
    tensor_view& operator=(const tensor<V, M, L>& o) {
        return assign(o);
    }

    /**
     * /always/ try to copy memory
     */
    tensor_view& operator=(const tensor_view<V, M, L>& o) {
        return assign(o);
    }

    /**
     * assign from value (sets all elements equal to one scalar)
     *
     * @param scalar value which should be assigned to all elements
     */
    template<class _V>
    typename boost::enable_if_c<boost::is_convertible<_V, V>::value, tensor_view&>::type operator=(
            const _V& scalar) {
        super::operator=(scalar);
        return *this;
    }

    /**
     * assignment operator for other memory space type
     *
     * @param o a tensor of another memory space type
     */
    template<class OM>
    tensor_view& operator=(const tensor<V, OM, L>& o) {
        return assign(o);
    }

    /**
     * assignment operator for views in other memory space types
     *
     * @param o a tensor_view of another memory space type
     */
    template<class OM>
    tensor_view& operator=(const tensor_view<V, OM, L>& o) {
        return assign(o);
    }

    /**
     * construct tensor_view
     *
     * @warning if a dimension has size 1, the resulting tensor has fewer dimensions than the original one.
     *
     * @warning most operations in CUV on ndarrays currently only work
     *          if the sub-tensor is a connected area in memory.  Basically this
     *          means that you can only slice in the first dimension which has
     *          size>1.
     *
     * @param idx  the indices of the sub-tensor
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
    explicit tensor_view(const tensor<V, M, L>& o, const index_gen<D, E>& idx) :
            tensor<V, M, L>(o.m_allocator)
    {
        m_memory = o.mem();
        m_ptr = const_cast<V*>(o.ptr());
        std::vector<int> shapes;
        std::vector<int> strides;
        shapes.reserve(D);
        strides.reserve(D);
        //cuvAssert(o.ndim()==D);
        for (size_t i = 0; i < D; i++) {
            int start = idx.ranges_[i].get_start(0);
            int finish = idx.ranges_[i].get_finish(o.shape(i));
            int stride = idx.ranges_[i].stride();
            if (start < 0)
                start += o.shape(i);
            if (finish < 0)
                finish += o.shape(i);
#ifndef NDEBUG
            cuvAssert(finish>start);
            cuvAssert(finish <= (int) o.shape(i));
#endif
            m_ptr += start * o.stride(i);
            if (idx.ranges_[i].is_degenerate()) {
                // skip dimension
            } else {
                shapes.push_back((finish - start) / stride);
                strides.push_back(o.stride(i) * stride);
            }
        }
        // adds missing shapes
        for(int i = D; i < o.ndim();i++){
            shapes.push_back(o.shape(i));
            strides.push_back(o.stride(i));
        }
        // store in m_info
        m_info.resize(shapes.size());
        std::copy(shapes.begin(), shapes.end(), m_info.host_shape[0].ptr);
        std::copy(strides.begin(), strides.end(), m_info.host_stride[0].ptr);
    }

    /**
     * different order of arguments as above, all else being equal.
     *
     * @deprecated
     * @param idx a set of index ranges into o
     * @param o   other tensor
     */
    template<int D, int E>
    explicit tensor_view(const index_gen<D, E>& idx, const tensor<V, M, L>& o) :
            tensor<V, M, L>(o.m_allocator)
    {
        m_memory = o.mem();
        m_ptr = const_cast<V*>(o.ptr());
        std::vector<int> shapes;
        std::vector<int> strides;
        shapes.reserve(D);
        strides.reserve(D);
        //cuvAssert(o.ndim()==D);
        for (size_t i = 0; i < D; i++) {
            int start = idx.ranges_[i].get_start(0);
            int finish = idx.ranges_[i].get_finish(o.shape(i));
            int stride = idx.ranges_[i].stride();
            if (start < 0)
                start += o.shape(i);
            if (finish < 0)
                finish += o.shape(i);
#ifndef NDEBUG
            cuvAssert(finish>start);
            cuvAssert(finish <= (int) o.shape(i));
#endif
            m_ptr += start * o.stride(i);
            if (idx.ranges_[i].is_degenerate()) {
                // skip dimension
            } else {
                shapes.push_back((finish - start) / stride);
                strides.push_back(o.stride(i) * stride);
            }
        }
        // adds missing shapes
        for(int i = D; i < o.ndim();i++){
            shapes.push_back(o.shape(i));
            strides.push_back(o.stride(i));
        }
        // store in m_info
        m_info.resize(shapes.size());
        std::copy(shapes.begin(), shapes.end(), m_info.host_shape[0].ptr);
        std::copy(strides.begin(), strides.end(), m_info.host_stride[0].ptr);
    }
};

/** @} */ // data_structures
/**
 * test whether two ndarrays have the same shape
 * @ingroup tools
 * @param a first tensor
 * @param a second tensor
 */
template<class V, class V2, class M, class M2, class L>
bool equal_shape(const tensor<V, M, L>& a, const tensor<V2, M2, L>& b) {
    return a.effective_shape() == b.effective_shape();
}

/**
 * @addtogroup MetaProgramming
 */
/// create a tensor type with the same template parameters, but with switched value type
template<class Mat, class NewVT>
struct switch_value_type {
    typedef tensor<NewVT, typename Mat::memory_space_type, typename Mat::memory_layout_type> type; ///< new tensor type after switch
};
/// create a tensor type with the same template parameters, but with switched memory_layout_type
template<class Mat, class NewML>
struct switch_memory_layout_type {
    typedef tensor<typename Mat::value_type, typename Mat::memory_space_type, NewML> type; ///< new tensor type after switch
};
/// create a tensor type with the same template parameters, but with switched memory_space_type
template<class Mat, class NewMS>
struct switch_memory_space_type {
    typedef tensor<typename Mat::value_type, NewMS, typename Mat::memory_layout_type> type; ///< new tensor type after switch
};

/** @} */

}

/**
 * input and output operations
 *
 * @addtogroup io
 * @{
 */
namespace std {

/**
 * print a host linear memory to a stream
 * @param o the stream
 * @param t the tensor
 */
template<class V>
ostream& operator<<(ostream& o, const cuv::linear_memory<V, cuv::host_memory_space>& t) {
    o << "[ ";
    for (unsigned int i = 0; i < t.size(); i++)
        o << t[i] << " ";
    o << "]";
    return o;
}

/**
 * print a dev linear memory to a stream (copies first)
 * @param o the stream
 * @param t_ the tensor
 */
template<class V>
ostream& operator<<(ostream& o, const cuv::linear_memory<V, cuv::dev_memory_space>& t_) {
    cuv::linear_memory<V, cuv::host_memory_space> t = t_; // pull
    o << "[ ";
    for (unsigned int i = 0; i < t.size(); i++)
        o << t[i] << " ";
    o << "]";
    return o;
}

/**
 * print a host pitched memory to a stream
 * @param o the stream
 * @param t the tensor
 */
template<class V>
ostream& operator<<(ostream& o, const cuv::pitched_memory<V, cuv::host_memory_space>& t) {
    o << "[ ";
    for (unsigned int i = 0; i < t.rows(); i++) {
        for (unsigned int j = 0; j < t.rows(); j++) {
            o << t(i, j) << " ";
        }
        if (i < t.rows() - 1)
            o << std::endl;
    }
    o << "]";
    return o;
}

/**
 * print a dev pitched memory to a stream (copies first)
 * @param o the stream
 * @param t_ the tensor
 */
template<class V>
ostream& operator<<(ostream& o, const cuv::pitched_memory<V, cuv::dev_memory_space>& t_) {
    cuv::pitched_memory<V, cuv::host_memory_space> t = t_; // pull
    o << "[ ";
    for (unsigned int i = 0; i < t.rows(); i++) {
        for (unsigned int j = 0; j < t.rows(); j++) {
            o << t(i, j) << " ";
        }
        if (i < t.rows() - 1)
            o << std::endl;
    }
    o << "]";
    return o;
}

/**
 * print a dev tensor to a stream (copying to host first)
 *
 * @param o the stream
 * @param t the tensor
 */
template<class V, class L>
ostream& operator<<(ostream& o, const cuv::tensor<V, cuv::dev_memory_space, L>& t) {
    return o << cuv::tensor<V, cuv::host_memory_space, L>(t);
}

/**
 * print a host tensor to a stream
 *
 * @param o the stream
 * @param t the tensor
 */
template<class V, class L>
ostream& operator<<(ostream& o, const cuv::tensor<V, cuv::host_memory_space, L>& t) {
    if (t.ndim() == 0)
        return o << "[]";

    if (t.ndim() == 1) {
        o << "[ ";
        for (unsigned int i = 0; i < t.shape(0); i++)
            o << t[i] << " ";
        return o << "]";
    }
    if (t.ndim() == 2) {
        o << "[";
        for (unsigned int i = 0; i < t.shape(0); ++i) {
            if (i > 0)
                o << " ";
            o << "[ ";
            for (unsigned int j = 0; j < t.shape(1); j++)
                o << t(i, j) << " ";
            o << "]";
            if (i != t.shape(0) - 1)
                o << std::endl;
        }
        return o << "]";
    }
    if (t.ndim() == 3) {
        o << "[" << std::endl;
        for (unsigned int l = 0; l < t.shape(0); l++) {
            o << "[";
            for (unsigned int i = 0; i < t.shape(1); ++i) {
                if (i > 0)
                    o << " ";
                o << "[ ";
                //for(unsigned int j=0;j<t.shape(2);j++) o<< t(l,i,j)<<" ";
                for (unsigned int j = 0; j < t.shape(2); j++)
                    o << t[l * t.shape(1) * t.shape(2) + i * t.shape(2) + j] << " ";
                o << "]";
                if (i != t.shape(1) - 1)
                    o << std::endl;
            }
            o << "]";
            if (l < t.shape(0) - 1)
                o << std::endl;
        }
        return o << "]";
    }
    throw std::runtime_error("printing of ndarrays with >3 dimensions not implemented");
}
}
/** @} */ // io
#endif
