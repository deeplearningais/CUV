#ifndef __CUV_REFERENCE_HPP__
#define __CUV_REFERENCE_HPP__

#include <boost/type_traits/is_convertible.hpp>
#include <boost/utility/enable_if.hpp>
#include <iostream>

#include <cuv/tools/meta_programming.hpp>
#include "tags.hpp"

namespace cuv {

namespace detail {

/**
 * @brief Setting entry of host linear_memory at ptr at index idx to value val
 *
 * @param ptr Address of array in memory
 * @param idx Index of value to set
 * @param val Value to set linear_memory entry to
 *
 */
template<class value_type>
void entry_set(value_type* ptr, size_t idx, value_type val, host_memory_space);

/**
 * @brief Getting entry of host linear_memory at ptr at index idx
 *
 * @param ptr Address of array in memory
 * @param idx Index of value to get
 *
 * @return
 */
template<class value_type>
value_type entry_get(const value_type* ptr, size_t idx, host_memory_space);

template<class value_type>
void entry_set(value_type* ptr, size_t idx, value_type val, dev_memory_space);

/**
 * Set the value at *(ptr+idx) to val, when ptr is in dev_memory_space.
 */
template<class value_type>
value_type entry_get(const value_type* ptr, size_t idx, dev_memory_space);

}

/**
 * This objects acts like a reference to the object stored at the wrapped pointer.
 * \ingroup data_structures
 */
template<class T, class M>
class reference
{

public:

    typedef typename unconst<T>::type value_type; ///< the type of the pointer
    typedef M memory_space_type; ///< the memory space of the pointer
    typedef reference<T, M> my_type; ///< the type of this reference

    value_type* ptr; ///< the wrapped pointer

    /// convert to the stored value
    operator value_type() const {
        return detail::entry_get(ptr, 0, memory_space_type());
    }

    /// assign a new value
    void operator=(const value_type& v) {
        detail::entry_set(ptr, 0, v, memory_space_type());
    }

    /// assign a value of a different (but convertible) value type
    template<class _T>
    typename boost::enable_if_c<boost::is_convertible<_T, value_type>::value>::type operator=(const _T& v) {
        detail::entry_set(ptr, 0, (value_type) v, memory_space_type());
    }

    /// assignment from reference of same type
    reference& operator=(const reference& o) {
        if (&o == &(*this)) // operator & is overloaded and returns value_type*
            return *this;
        (*this) = (value_type) o;
        return *this;
    }

    /// assignment from reference of other memory type
    template<class OM>
    reference& operator=(const reference<T, OM>& o) {
        (*this) = static_cast<T>(o);
        return *this;
    }

    /// get the wrapped pointer
    const value_type* operator&() const {
        return ptr;
    }

    /// get the wrapped pointer
    value_type* operator&() {
        return ptr;
    }

    /// construct using a pointer
    reference(const T* p) :
            ptr(p) {
    }

    /// construct using a pointer
    reference(T* p) :
            ptr(p) {
    }

    /// implicit construction using value
    reference(value_type& p) :
            ptr(&p) {
    }

    /// implicit construction using value
    reference(const value_type& p) :
            ptr(&p) {
    }

    /// add to the value stored at ptr
    my_type& operator+=(const value_type& v) {
        *this = (value_type) (*this) + v;
        return *this;
    }

    /// subtract from the value stored at ptr
    my_type& operator-=(const value_type& v) {
        *this = (value_type) (*this) - v;
        return *this;
    }

    /// multiply with the value stored at ptr
    my_type& operator*=(const value_type& v) {
        *this = (value_type) (*this) * v;
        return *this;
    }

    /// divide by the value stored at ptr
    my_type& operator/=(const value_type& v) {
        *this = (value_type) (*this) / v;
        return *this;
    }

    /// increment value at ptr
    value_type operator++(int) {
        value_type v = *this;
        *this = v + 1;
        return v;
    }

    /// decrement value at ptr
    value_type operator--(int) {
        value_type v = *this;
        *this = v - 1;
        return v;
    }

    /// increment value at ptr
    value_type operator++() {
        value_type v = *this;
        *this = v + 1;
        return v + 1;
    }

    /// decrement value at ptr
    value_type operator--() {
        value_type v = *this;
        *this = v - 1;
        return v - 1;
    }

    /// compare value at ptr with another
    bool operator==(const value_type& v) {
        return ((value_type) *this) == v;
    }

    /// compare value at ptr with another
    bool operator<=(const value_type& v) {
        return ((value_type) *this) <= v;
    }

    /// compare value at ptr with another
    bool operator<(const value_type& v) {
        return ((value_type) *this) < v;
    }

    /// compare value at ptr with another
    bool operator>=(const value_type& v) {
        return ((value_type) *this) >= v;
    }

    /// compare value at ptr with another
    bool operator>(const value_type& v) {
        return ((value_type) *this) > v;
    }
};

}

template<class T, class M>
std::ostream& operator<<(std::ostream& os, const cuv::reference<T, M>& reference);

#endif
