#ifndef __META_PROGRAMMING_HPP__
#define __META_PROGRAMMING_HPP__
namespace cuv{
// is same template metaprogramming
// checks whether two types are the same
// usage: IsSame<FirstClass,SecondClass>::Result::value

/**
 * @defgroup MetaProgramming
 * @{
 */

	/// defines "False"
struct FalseType { enum { value = false }; };
	/// defines "True"
struct TrueType { enum { value = true }; };


/** 
 * @brief Checks whether two types are equal
 */
template <typename T1, typename T2>
struct IsSame
{
	typedef FalseType Result;
};


/** 
 * @see IsSame
 */
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

/**
 * @see unconst
 */
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
/**
 * @see If
 */
template<class Then, class Else>
struct If<false,Then,Else>{
	typedef Else result;
};


/**
 * @}
 */
};
#endif /* __META_PROGRAMMING_HPP__ */
