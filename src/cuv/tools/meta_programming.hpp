#ifndef __META_PROGRAMMING_HPP__
#define __META_PROGRAMMING_HPP__
namespace cuv{
// is same template metaprogramming
// checks whether two types are the same
// usage: IsSame<FirstClass,SecondClass>::Result::value

struct FalseType { enum { value = false }; };
struct TrueType { enum { value = true }; };


/** 
 * @brief Checks whether two types are equal
 */
template <typename T1, typename T2>
struct IsSame
{
	typedef FalseType Result;
};


template <typename T>
struct IsSame<T,T>
{
	typedef TrueType Result;
};

/** 
 * @brief Remove "const" from a type
 */
template <typename T>
struct unconst{
	typedef T type;
};

template <typename T>
struct unconst<const T>{
	typedef T type;
};

/**
 * @brief Switch result depending on Condition
 */
template <bool Condition, class Then, class Else>
struct If{
	typedef Then result;
};
template<class Then, class Else>
struct If<false,Then,Else>{
	typedef Else result;
};
};
#endif /* __META_PROGRAMMING_HPP__ */
