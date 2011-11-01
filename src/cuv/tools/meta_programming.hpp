#ifndef __META_PROGRAMMING_HPP__
#define __META_PROGRAMMING_HPP__
namespace cuv{

/**
 * @addtogroup MetaProgramming
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
	/// is true only if T1==T2
	typedef FalseType Result;
};


/** 
 * @see IsSame
 */
template <typename T>
struct IsSame<T,T>
{
	/// T==T, therefore Result==TrueType
	typedef TrueType Result;
};

/** 
 * @brief Checks whether two types are different
 */
template <typename T1, typename T2>
struct IsDifferent
{
	/// is true only if T1!=T2
	typedef TrueType Result;
};


/** 
 * @see IsDifferent
 */
template <typename T>
struct IsDifferent<T,T>
{
	/// T==T, therefore Result==FalseType
	typedef FalseType Result;
};

/** 
 * @brief Remove "const" from a type
 */
template <typename T>
struct unconst{
	/// no change
	typedef T type;
};

/**
 * @see unconst
 */
template <typename T>
struct unconst<const T>{
	/// T without the const
	typedef T type;
};

/**
 * @brief Switch result depending on Condition
 */
template <bool Condition, class Then, class Else>
struct If{
	/// assume condition is true
	typedef Then result;
};
/**
 * @see If
 */
template<class Then, class Else>
struct If<false,Then,Else>{
	/// condition is false
	typedef Else result;
};

/**
 * @brief enable-if controlled creation of SFINAE conditions
 */
template <bool B, class T = void>
struct EnableIfC {
  typedef T type; /// enabling succeeded :-)
};

/// @see EnableIfC
template <class T>
struct EnableIfC<false, T> {};

/// @see EnableIfC
template <class Cond, class T = void>
struct EnableIf : public EnableIfC<Cond::value, T> {};

/// @see EnableIfC
template <class Cond, class T = void>
struct DisableIf : public EnableIfC<!Cond::value, T> {};

/**
 * @}
 */
};
#endif /* __META_PROGRAMMING_HPP__ */
