#ifndef __CUV_MEMORY_HPP__
#define __CUV_MEMORY_HPP__

#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>
#include <cuda_runtime_api.h>
#include <limits>
#include <stdexcept>

#include "allocators.hpp"
#include "reference.hpp"

namespace boost {
namespace serialization {
class access;
}
}

namespace cuv {

/**
 * @addtogroup data_structures
 * @{
 */

/**
 * @addtogroup tags
 * @{
 */
/// Tag for column major matrices
struct column_major {
};
/// Tag for row major matrices
struct row_major {
};

/// tag for linear memory
struct linear_memory_tag {
};

/// tag for pitched memory
struct pitched_memory_tag {
};

/** @} */ // tags
namespace detail {

/// copy from host to host
template<class value_type>
void copy(value_type* dst, const value_type* src, size_t size, host_memory_space, host_memory_space, cudaStream_t);

/// copy from device to host
template<class value_type>
void copy(value_type* dst, const value_type* src, size_t size, host_memory_space, dev_memory_space, cudaStream_t);

/// copy from host to host
template<class value_type, class value_type2>
void copy(value_type* dst, const value_type2* src, size_t size, host_memory_space, host_memory_space, cudaStream_t);

/// copy from device to host
template<class value_type, class value_type2>
void copy(value_type* dst, const value_type2* src, size_t size, host_memory_space, dev_memory_space, cudaStream_t);

/// copy from host to device
template<class value_type>
void copy(value_type* dst, const value_type* src, size_t size, dev_memory_space, host_memory_space, cudaStream_t);

/// copy from device to device
template<class value_type>
void copy(value_type* dst, const value_type* src, size_t size, dev_memory_space, dev_memory_space, cudaStream_t);

/// copy from host to device
template<class value_type, class value_type2>
void copy(value_type* dst, const value_type2* src, size_t size, dev_memory_space, host_memory_space, cudaStream_t);

/// copy from device to device
template<class value_type, class value_type2>
void copy(value_type* dst, const value_type2* src, size_t size, dev_memory_space, dev_memory_space, cudaStream_t);

/// copy from host to host
template<class value_type, class value_type2>
void copy2d(value_type* dst, const value_type2* src, size_t dpitch, size_t spitch, size_t h, size_t w,
        host_memory_space, host_memory_space, cudaStream_t);

/// copy from device to host
template<class value_type, class value_type2>
void copy2d(value_type* dst, const value_type2* src, size_t dpitch, size_t spitch, size_t h, size_t w,
        host_memory_space, dev_memory_space, cudaStream_t);

/// copy from host to device
template<class value_type, class value_type2>
void copy2d(value_type* dst, const value_type2* src, size_t dpitch, size_t spitch, size_t h, size_t w,
        dev_memory_space, host_memory_space, cudaStream_t);

/// copy from device to device
template<class value_type, class value_type2>
void copy2d(value_type* dst, const value_type2* src, size_t dpitch, size_t spitch, size_t h, size_t w,
        dev_memory_space, dev_memory_space, cudaStream_t);
}

/**
 * simply keeps a pointer and deallocates it when destroyed
 */
template<class V, class M>
class memory {

public:
    typedef typename unconst<V>::type value_type; ///< type of contained values
    typedef const V const_value_type; ///< const version of value_type
    typedef M memory_space_type; ///< host or dev memory_space
    typedef unsigned int size_type; ///< type of shapes
    typedef int index_type; ///< how to index values
    typedef reference<V, M> reference_type; ///< type of reference you get using operator[]
    typedef const reference<V, M> const_reference_type; ///< type of reference you get using operator[]

private:
    friend class boost::serialization::access;

    /// prohibit copying
    memory(const memory&);

    /// prohibit copying
    memory& operator=(const memory& o);

protected:
    V* m_ptr; ///< points to allocated memory
    size_type m_size; ///< size (for serialization)
    boost::shared_ptr<allocator> m_allocator; ///< how stored memory was allocated
    bool m_owned; ///< flag is this instance owns the memory (m_ptr) and is responsibly for destroying

    void check_size_limit(size_t size) const {
        if (size > static_cast<size_t>(std::numeric_limits<index_type>::max())) {
            throw std::runtime_error("maximum memory size exceeded");
        }
    }

public:

    /// @return pointer to allocated memory
    V* ptr() {
        return m_ptr;
    }

    /// @return pointer to allocated memory (const)
    const V* ptr() const {
        return m_ptr;
    }

    /// @return number of stored elements
    size_type size() const {
        return m_size;
    }

    /// @return number of stored bytes
    size_type memsize() const {
        return size() * sizeof(V);
    }

    /// reset information (use with care, for deserialization)
    void reset(V* p, size_type s) {
        m_ptr = p;
        m_size = s;
    }

    /// default constructor (just sets ptr to NULL)
    explicit memory(const boost::shared_ptr<allocator>& _allocator) :
            m_ptr(NULL), m_size(0), m_allocator(_allocator), m_owned(true) {
    }

    /// construct with pointer (takes /ownership/ of this pointer and deletes it when destroyed!)
    explicit memory(value_type* ptr, size_type size, const boost::shared_ptr<allocator>& _allocator, bool owned = true) :
            m_ptr(ptr), m_size(size), m_allocator(_allocator), m_owned(owned) {
    }

    /// destructor (deallocates the memory)
    ~memory() {
        dealloc();
    }

    /// dellocate space
    void dealloc() {
        if (m_ptr && m_owned) {
            m_allocator->dealloc(reinterpret_cast<void**>(&this->m_ptr), memory_space_type());
        }
        m_ptr = NULL;
        m_size = 0;
    }

    template<class value_type2, class memory_space>
    void copy_from(const value_type2* src, size_t size, memory_space m, cudaStream_t stream) {
        detail::copy(m_ptr, src, size, M(), m, stream);
    }

    template<class value_type2, class memory_space>
    void copy2d_from(const value_type2* src, size_t dpitch, size_t spitch, size_t h, size_t w,
            memory_space m, cudaStream_t stream) {
        detail::copy2d(m_ptr, src, dpitch, spitch, h, w, M(), m, stream);
    }

};

/**
 * represents contiguous memory
 */
template<class V, class M>
class linear_memory: public memory<V, M> {
private:
    typedef memory<V, M> super;
    public:
    typedef typename super::value_type value_type; ///< type of contained values
    typedef typename super::const_value_type const_value_type; ///< const version of value_type
    typedef typename super::memory_space_type memory_space_type; ///< host or dev memory_space
    typedef typename super::index_type index_type; ///< how to index values
    typedef typename super::size_type size_type; ///< type of shapes
    typedef typename super::reference_type reference_type; ///< type of reference you get using operator[]
    typedef typename super::const_reference_type const_reference_type; ///< type of reference you get using operator[]

private:

    friend class boost::serialization::access;
    typedef linear_memory<V, M> my_type; ///< my own type
    using super::m_ptr;
    using super::m_size;
    using super::m_allocator;

public:

    /// default constructor: does nothing
    explicit linear_memory(const boost::shared_ptr<allocator> _allocator = boost::make_shared<default_allocator>()) :
            memory<V, M>(_allocator) {
    }

    /** constructor: reserves space for i elements
     *  @param i number of elements
     */
    explicit linear_memory(size_type i, const boost::shared_ptr<allocator> _allocator =
            boost::make_shared<default_allocator>()) :
            memory<V, M>(_allocator) {
        m_size = i;
        alloc();
    }

    /// releases ownership of pointer (for storage in memory class)
    value_type* release() {
        value_type* ptr = m_ptr;
        m_ptr = NULL;
        return ptr;
    }

    /// sets the size (reallocates if necessary)
    void set_size(size_type s) {
        if (s != this->size()) {
            this->dealloc();
            m_size = s;
            alloc();
        }
    }

    /// allocate space according to size()
    void alloc() {
        assert(this->m_ptr == NULL);
        if (m_size > 0)
            m_allocator->alloc(reinterpret_cast<void**>(&m_ptr), m_size, sizeof(V), memory_space_type());
    }

    /**
     * @brief Copy linear_memory.
     *
     * @param o Source linear_memory
     *
     * @return *this
     *
     */
    my_type& operator=(const my_type& o) {
        if (this == &o)
            return *this;

        if (this->size() != o.size()) {
            this->dealloc();
            m_size = o.size();
            this->alloc();
        }

        // TODO async copy
        cudaStream_t stream = 0;
        this->copy_from(o, stream);

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
    my_type& operator=(const linear_memory<value_type, OM>& o) {
        if (this->size() != o.size()) {
            this->dealloc();
            m_size = o.size();
            this->alloc();
        }

        // TODO async copy
        cudaStream_t stream = 0;
        this->copy_from(o, stream);
        return *this;
    }

    /**
     * construct from other linear memory
     */
    explicit linear_memory(const my_type& o) :
            memory<V, M>(o.m_allocator) {
        operator=(o);
    }

    /**
     * construct from other linear memory
     */
    template<class OM>
    explicit linear_memory(const linear_memory<V, OM>& o) :
            memory<V, M>(o.m_allocator) {
        operator=(o);
    }

    /**
     * @return a reference to memory at a position
     * @param idx position
     */
    reference_type operator[](const index_type& idx) {
        assert(idx >= 0);
        assert((size_type) idx < m_size);
        return reference_type(this->m_ptr + idx);
    }

    /**
     * @overload
     *
     * @return a reference to memory at a position
     * @param idx position
     */
    const_reference_type operator[](const index_type& idx) const {
        assert(idx >= 0);
        assert((size_type) idx < m_size);
        return const_reference_type(this->m_ptr + idx);
    }

    /// deallocates memory
    ~linear_memory() {
        this->dealloc();
    }

    /// set strides for this memory
    void set_strides(linear_memory<index_type, cuv::host_memory_space>& strides,
            const linear_memory<size_type, cuv::host_memory_space>& shape, row_major) {
        size_t size = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            strides[i] = (shape[i] == 1) ? 0 : size;
            size *= shape[i];
        }
        this->check_size_limit(size);
    }

    /// set strides for this memory
    void set_strides(linear_memory<index_type, cuv::host_memory_space>& strides,
            const linear_memory<size_type, cuv::host_memory_space>& shape, column_major) {
        size_t size = 1;
        for (size_t i = 0; i < shape.size(); ++i) {
            strides[i] = (shape[i] == 1) ? 0 : size;
            size *= shape[i];
        }
        this->check_size_limit(size);
    }

    /** reverse the array (for transposing etc)
     *
     * currently only enabled for host memory space arrays
     */
    void reverse() {
        if (IsSame<dev_memory_space, memory_space_type>::Result::value)
            throw std::runtime_error("reverse of dev linear memory not implemented");
        value_type* __first = m_ptr, *__last = m_ptr + this->size();
        while (true)
            if (__first == __last || __first == --__last)
                return;
            else {
                std::iter_swap(__first, __last);
                ++__first;
            }
    }

    template<class value_type2, class memory_space>
    void copy_from(const value_type2* src, size_t size, memory_space m, cudaStream_t stream) {
        memory<V, M>::copy_from(src, size, m, stream);
    }

    template<class V2, class OM>
    void copy_from(const linear_memory<V2, OM>& src, cudaStream_t stream) const {
        detail::copy(m_ptr, src.ptr(), src.size(), M(), OM(), stream);
    }

};

/**
 * represents 2D non-contiguous ("pitched") memory
 */
template<class V, class M>
class pitched_memory: public memory<V, M> {

private:
    typedef memory<V, M> super;

public:

    typedef typename super::value_type value_type; ///< type of contained values
    typedef typename super::const_value_type const_value_type; ///< const version of value_type
    typedef typename super::memory_space_type memory_space_type; ///< host or dev memory_space
    typedef typename super::index_type index_type; ///< how to index values
    typedef typename super::size_type size_type; ///< type of shapes
    typedef typename super::reference_type reference_type; ///< type of reference you get using operator[]
    typedef typename super::const_reference_type const_reference_type; ///< type of reference you get using operator[]

private:
    friend class boost::serialization::access;
    typedef pitched_memory<V, M> my_type; ///< my own type
    size_type m_rows; ///< number of rows
    size_type m_cols; ///< number of columns
    size_type m_pitch; ///< pitch (multiples of sizeof(V))
    using super::m_ptr;
    using super::m_size;
    using super::m_allocator;
    public:

    /// @return the number of rows
    size_type rows() const {
        return m_rows;
    }

    /// @return the number of cols
    size_type cols() const {
        return m_cols;
    }

    /// @return the number of allocated cols
    size_type pitch() const {
        return m_pitch;
    }

    /// @return number of stored elements
    size_type size() const {
        return m_rows * m_pitch;
    }

    /// @return number of stored bytes
    size_type memsize() const {
        return size() * sizeof(V);
    }

    /// default constructor: does nothing
    explicit pitched_memory(const boost::shared_ptr<allocator> _allocator = boost::make_shared<default_allocator>()) :
            memory<V, M>(_allocator), m_rows(0), m_cols(0), m_pitch(0) {
    }

    /** constructor: reserves space for at least i*j elements
     *  @param i number of rows
     *  @param j minimum number of elements per row
     */
    explicit pitched_memory(index_type i, index_type j, const boost::shared_ptr<allocator> _allocator =
            boost::make_shared<default_allocator>()) :
            memory<V, M>(_allocator), m_rows(i), m_cols(j), m_pitch(0) {
        alloc();
    }

    /**
     * allocate space according to size()
     */
    void alloc() {
        assert(this->m_ptr == NULL);
        size_t pitch;
        m_allocator->alloc2d(reinterpret_cast<void**>(&this->m_ptr), pitch, m_rows, m_cols, sizeof(V),
                memory_space_type());
        assert(this->m_ptr != NULL);
        m_pitch = pitch;
        assert(m_pitch % sizeof(value_type) == 0);
        m_pitch /= sizeof(value_type);
        m_size = m_rows * m_pitch; // in class memory
    }

    /// releases ownership of pointer (for storage in memory class)
    value_type* release() {
        value_type* ptr = m_ptr;
        m_ptr = NULL;
        return ptr;
    }

    /**
     * set the size (reallocating, if necessary)
     * @param rows number of desired rows
     * @param cols number of desired columns
     */
    void set_size(size_type rows, size_type cols) {
        if (cols > m_pitch || rows > m_rows) {
            this->dealloc();
            m_rows = rows;
            m_cols = cols;
            this->alloc();
        } else {
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
    my_type& operator=(const my_type& o) {
        if (this == &o)
            return *this;

        if (m_pitch < o.m_cols || m_rows < o.m_rows) {
            this->dealloc();
            m_cols = o.m_cols;
            m_rows = o.m_rows;
            this->alloc();
        }
        m_cols = o.m_cols;
        m_rows = o.m_rows;
        this->copy_from(o);
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
    operator=(const pitched_memory<value_type, OM>& o) {
        if (m_pitch < o.m_cols || m_rows < o.m_rows) {
            this->dealloc();
            m_cols = o.m_cols;
            m_rows = o.m_rows;
            this->alloc();
        }
        m_cols = o.m_cols;
        m_rows = o.m_rows;
        this->copy_from(o);
        return *this;
    }

    /**
     * @return a reference to memory at a position as if this were pitched memory
     * @param idx position
     */
    reference_type operator[](const index_type& idx) {
        assert(idx >= 0);
        index_type row = idx / m_cols;
        index_type col = idx % m_cols;
        assert((size_type) row < m_rows);
        assert((size_type) col < m_cols);
        return reference_type(this->m_ptr + row * m_pitch + col);
    }

    /**
     * @overload
     *
     * @return a reference to memory at a position
     * @param idx position
     */
    const_reference_type operator[](const index_type& idx) const {
        return const_cast<pitched_memory&>(*this)(idx);
    }

    /**
     * get a reference to a datum in memory
     *
     * @param i first (slow-changing) dimension index
     * @param j second (fast-changing) dimension index
     * @return reference to datum at index i,j
     */
    reference_type operator()(const index_type& i, const index_type& j) {
        assert(i >= 0);
        assert(j >= 0);
        assert((size_type) i < m_rows);
        assert((size_type) j < m_cols);
        return reference_type(this->m_ptr + i * m_pitch + j);
    }
    /** @overload */
    const_reference_type operator()(const index_type& i, const index_type& j) const {
        return const_cast<pitched_memory&>(*this)(i, j);
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
    void set_strides(linear_memory<index_type, cuv::host_memory_space>& strides,
            const linear_memory<size_type, cuv::host_memory_space>& shape, row_major) {
        size_type size = 1;
        assert(shape.size() >= 2);
        const int pitched_dim = shape.size() - 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            if (shape[i] == 1) {
                strides[i] = 0;
            } else if (i == pitched_dim) {
                strides[i] = 1;
                size *= pitch();
            } else {
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
    void set_strides(linear_memory<index_type, cuv::host_memory_space>& strides,
            const linear_memory<size_type, cuv::host_memory_space>& shape, column_major) {
        size_type size = 1;
        assert(shape.size() >= 2);
        const size_type pitched_dim = 0;
        for (unsigned int i = 0; i < shape.size(); ++i) {
            if (shape[i] == 1) {
                strides[i] = 0;
            } else if (i == pitched_dim) {
                strides[i] = 1;
                size *= pitch();
            } else {
                strides[i] = size;
                size *= shape[i];
            }
        }
    }

    template<class V2, class OM>
    void copy2d_from(const memory<V2, OM> src, cudaStream_t stream) const {
        memory<V, M>::copy2d_from(m_ptr, src.ptr(), m_pitch / sizeof(value_type), src.m_pitch / sizeof(V2),
                m_rows, m_cols, M(), OM(), stream);
    }

    template<class V2, class OM>
    void copy_from(const pitched_memory<V2, OM>& src, cudaStream_t stream) const {
        detail::copy(m_ptr, src.ptr(), src.size(), M(), OM(), stream);
    }

};

/** @} */ // data_structures
namespace detail {

/**
 * true iff there are no "holes" in memory
 */
inline bool is_c_contiguous(row_major, const linear_memory<unsigned int, cuv::host_memory_space>& shape,
        const linear_memory<int, cuv::host_memory_space>& stride) {
    bool c_contiguous = true;
    int size = 1;
    for (int i = shape.size() - 1; (i >= 0) && c_contiguous; --i) {
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
inline bool is_c_contiguous(column_major, const linear_memory<unsigned int, cuv::host_memory_space>& shape,
        const linear_memory<int, cuv::host_memory_space>& stride) {
    bool c_contiguous = true;
    int size = 1;
    for (unsigned int i = 0; i < shape.size() && c_contiguous; ++i) {
        if (shape[i] == 1)
            continue;
        if (stride[i] != size)
            c_contiguous = false;
        size = size * shape[i];
    }
    return c_contiguous;
}

/// returns true iff memory can be copied using copy2d
inline bool is_2dcopyable(row_major, const linear_memory<unsigned int, cuv::host_memory_space>& shape,
        const linear_memory<int, cuv::host_memory_space>& stride) {
    bool c_contiguous = shape.size() > 1;
    int pitched_dim = shape.size() - 1; // last dim
    while (shape[pitched_dim] == 1 && stride[pitched_dim] == 1)
        pitched_dim--;
    int size = 1;
    for (int i = shape.size() - 1; (i >= 0) && c_contiguous; --i) {
        if (shape[i] == 1) {
            continue;
        } else if (i == pitched_dim) {
            size *= stride[i - 1];
        } else if (stride[i] != size) {
            c_contiguous = false;
        } else {
            size *= shape[i];
        }
    }
    return c_contiguous;
}

/// @overload
inline bool is_2dcopyable(column_major, const linear_memory<unsigned int, cuv::host_memory_space>& shape,
        const linear_memory<int, cuv::host_memory_space>& stride) {
    bool c_contiguous = shape.size() > 1;
    unsigned int pitched_dim = 0;
    while (shape[pitched_dim] == 1 && stride[pitched_dim] == 1)
        pitched_dim++;
    int size = 1;
    for (unsigned int i = 0; (i < shape.size()) && c_contiguous; ++i) {
        if (shape[i] == 1) {
            continue;
        } else if (i == pitched_dim) {
            size *= stride[i];
        } else if (stride[i] != size) {
            c_contiguous = false;
        } else {
            size *= shape[i];
        }
    }
    return c_contiguous;
}

}
}

#endif
