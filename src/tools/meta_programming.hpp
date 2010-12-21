// is same template metaprogramming
// checks whether two types are the same
// usage: IsSame<FirstClass,SecondClass>::Result::value

struct FalseType { enum { value = false }; };
struct TrueType { enum { value = true }; };


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
