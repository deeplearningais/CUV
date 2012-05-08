#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/export.hpp>
#include <cuv/basics/tensor.hpp>
#include "io.hpp"


/** get first param */
/*
 *template<typename T> struct gfp;
 *template<typename R, typename P1> struct gfp<R(P1)> {
 *  typedef P1 type;
 *};
 *
 *#undef BOOST_CLASS_EXPORT_IMPLEMENT
 *#undef BOOST_CLASS_EXPORT_KEY2
 *#define BOOST_CLASS_EXPORT_IMPLEMENT(T)                      \
 *    namespace boost {                                        \
 *    namespace archive {                                      \
 *    namespace detail {                                       \
 *    namespace {                                              \
 *    template<>                                               \
 *    struct init_guid< gfp<void T>::type > {                                  \
 *        static guid_initializer< gfp<void T>::type > const & g;              \
 *    };                                                       \
 *    guid_initializer< gfp<void T>::type > const & init_guid< gfp<void T>::type >::g =        \
 *        ::boost::serialization::singleton<                   \
 *            guid_initializer< gfp<void T>::type >                            \
 *        >::get_mutable_instance().export_guid();             \
 *    }}}}                                                     \
 *[><]
 *
 *#define BOOST_CLASS_EXPORT_KEY2(T, K)          \
 *namespace boost {                              \
 *namespace serialization {                      \
 *template<>                                     \
 *struct guid_defined<gfp<void T>::type> : boost::mpl::true_ {}; \
 *template<>                                     \
 *inline const char * guid< gfp<void T>::type >(){                 \
 *    return K;                                  \
 *}                                              \
 *} [> serialization <]                          \
 *} [> boost <]                                  \
 *[><]
 *
 *BOOST_CLASS_EXPORT((cuv::linear_memory<float,cuv::host_memory_space>));
 *BOOST_CLASS_EXPORT((cuv::linear_memory<float,cuv::dev_memory_space>));
 *BOOST_CLASS_EXPORT((cuv::linear_memory<unsigned char,cuv::host_memory_space>));
 *BOOST_CLASS_EXPORT((cuv::linear_memory<unsigned char,cuv::dev_memory_space>));
 *BOOST_CLASS_EXPORT((cuv::linear_memory<signed char,cuv::host_memory_space>));
 *BOOST_CLASS_EXPORT((cuv::linear_memory<signed char,cuv::dev_memory_space>));
 *BOOST_CLASS_EXPORT((cuv::linear_memory<char,cuv::host_memory_space>));
 *BOOST_CLASS_EXPORT((cuv::linear_memory<char,cuv::dev_memory_space>));
 */
