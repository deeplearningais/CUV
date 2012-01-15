#include <boost/static_assert.hpp>
namespace cuv
{
	/**
	 * @addtogroup typetraits
	 * @{
	 */


	template<class T>
	struct vector_type_traits{
		BOOST_STATIC_ASSERT(sizeof(T)==0);
	};

	template<>
	struct vector_type_traits<float4>{
		typedef float base_type;
		static const int dim = 4;
		static const bool is_base_type = false;
	};

	template<>
	struct vector_type_traits<float3>{
		typedef float base_type;
		static const int dim = 3;
		static const bool is_base_type = false;
	};

	template<>
	struct vector_type_traits<float2>{
		typedef float base_type;
		static const int dim = 2;
		static const bool is_base_type = false;
	};
	template<>
	struct vector_type_traits<float1>{
		typedef float base_type;
		static const int dim = 1;
		static const bool is_base_type = false;
	};

	template<>
	struct vector_type_traits<float>{
		typedef float base_type;
		static const int dim = 1;
		static const bool is_base_type = true;
		/// get high-dim datatypes of a base type
		template<int I> struct vector{  };
	};
	template<>     struct vector_type_traits<float>::template vector<1>{ typedef float  type; };
	template<>     struct vector_type_traits<float>::template vector<2>{ typedef float2 type; };
	template<>     struct vector_type_traits<float>::template vector<3>{ typedef float3 type; };
	template<>     struct vector_type_traits<float>::template vector<4>{ typedef float4 type; };



	/**
	 * @}
	 */

}
