#include <cmath>
#include <cuv_general.hpp>

#define sgn(a) (copysign(1.f,a))
struct binary_functor_tag{};
struct quadrary_functor_tag{};

namespace cuv {
struct binary_functor{
	typedef binary_functor_tag functor_type;
};

struct quadrary_functor{
	typedef quadrary_functor_tag functor_type;
};

template<class T>
struct uf_exp{  __host__ __device__         T operator()(const T& t)const{ return expf(t);    } };
template<class T>
struct uf_exact_exp{  __device__ __host__   T operator()(const T& t)const{ return expf(t);    } };
template<class T>
struct uf_log{  __device__ __host__         T operator()(const T& t)      const{ return logf(t);    } };
template<class T>
struct uf_sign{  __device__ __host__        T operator()(const T& t)      const{ return sgn((float)t);    } };
template<class T>
struct uf_abs{  __device__ __host__        T operator()(const T& t)      const{ return t<0 ? -t : t;    } };
template<>
struct uf_abs<unsigned char>{  __device__ __host__        unsigned char operator()(const unsigned char& t)      const{ return t;    } };
template<class T>
struct uf_exact_sigm{  __device__  __host__ T operator()(const T& t)      const{ return ((T)1)/(((T)1)+expf(-t));    } };
template<class T>
struct uf_dsigm{  __device__ __host__       T operator()(const T& t)      const{ return t * (((T)1)-t); } };
template<class T>
struct uf_tanh{  __device__  __host__       T operator()(const T& t)      const{ return tanhf(t); } };
template<class T>
struct uf_dtanh{  __device__  __host__      T operator()(const T& t)      const{ return ((T)1) - (t*t); } };
template<class T>
struct uf_square{  __device__  __host__     T operator()(const T& t)      const{ return t*t; } };
template<class T>
struct uf_sublin{  __device__  __host__     T operator()(const T& t)      const{ return ((T)1)-t; } };
template<class T>
struct uf_energ{  __device__  __host__      T operator()(const T& t)      const{ return -logf(t); } };
template<class T>
struct uf_inv{  __device__  __host__        T operator()(const T& t)      const{ return ((T)1)/(t+((T)0.00000001)); } };
template<class T>
struct uf_sqrt{  __device__  __host__       T operator()(const T& t)      const{ return sqrtf(t); } };
template<class T>
struct uf_smax{  __device__  __host__      T operator()(const T& t)      const{ return (((T)1)/t - (T) 1) * t; } };

template<class T>
struct uf_is_nan{                 __device__  __host__   bool operator()(const T& t)             const{ return (t!=t) ; } };
template<>
struct uf_is_nan<int>{            __device__  __host__   bool operator()(const int& t)           const{ return false ; } };
template<>
struct uf_is_nan<unsigned char>{  __device__  __host__   bool operator()(const unsigned char& t) const{ return false ; } };
template<>
struct uf_is_nan<signed char>{    __device__  __host__   bool operator()(const signed char& t)   const{ return false ; } };

template<class T>
struct uf_is_inf{                 __device__  __host__     bool operator()(const T& t)            const{ return (bool)!isfinite(t); } };
template<>                                                                                        
struct uf_is_inf<int>{            __device__  __host__     bool operator()(const int t)           const{ return false; } };
template<>                                                                                        
struct uf_is_inf<signed char>{    __device__  __host__     bool operator()(const signed char t)   const{ return false; } };
template<>
struct uf_is_inf<unsigned char>{  __device__  __host__     bool operator()(const unsigned char t) const{ return false; } };

template<class T>
struct uf_poslin{  __device__  __host__     T operator()(const T& t)      const{ return (t > 0)*t; } };


template<class T>
struct bf_sigm_temp{ __device__  __host__       T operator()(const T& t, const T& temp)           const{ return ((T)1)/(((T)1)+expf(-t / (T)(temp))); } };

template<class T>
struct tf_tanh{  __device__  __host__       T operator()(const T& x, const T& a, const T& b)      const{ return a * tanhf(b * x); } };
template<class T>
struct tf_dtanh{  __device__  __host__      T operator()(const T& x, const T& a, const T& b)      const{ return b/a * (a+x) * (a-x); } };

// rectifying transfer function. a is param beta
template<class T, class A>
struct bf_rect{  __device__  __host__       T operator()(const T& x, const A& a)      const{
	T ax = a*x;
	if(ax > 87.33f)
		return (T) x;
	return log(1.0f+expf(ax))/a;
}};
/*template<class T, class A>*/
/*struct bf_rect{  __device__  __host__       T operator()(const T& x, const A& a)      const{ return (T) log(1.0 + (double)exp((double)a*x))/a; } };*/
template<class T, class A>
struct bf_drect{  __device__  __host__      T operator()(const T& x, const A& a)      const{ return 1-1/(x*expf(a)); } };


template<class T, class binary_functor>
struct uf_base_op{
  const T x;
  const binary_functor bf;
  uf_base_op(const T& _x):x(_x),bf(){};
  __device__ __host__
  T operator()(const T& t){ return bf(t,x); }
};
template<class T, class ternary_functor>
struct uf_base_op3{
  const T x,y;
  const ternary_functor tf;
  uf_base_op3(const T& _x, const T& _y):x(_x),y(_y),tf(){};
  __device__ __host__
  T operator()(const T& t){ return tf(t,x,y); }
};

/*
 * Binary Functors
 */

// functors without parameter
template<class T, class U>
struct bf_plus : binary_functor {  __device__  __host__       T operator()(const T& t, const U& u)      const{ return  t + (T)u; } };
template<class T, class U>
struct bf_minus: binary_functor {  __device__  __host__      T operator()(const T& t, const U& u)      const{ return  t - (T)u; } };
template<class T, class U>
struct bf_multiplies: binary_functor {  __device__  __host__ T operator()(const T& t, const U& u)      const{ return  t * (T)u; } };
template<class T, class U>
struct bf_divides: binary_functor {  __device__  __host__    T operator()(const T& t, const U& u)      const{ return  t / (T)u; } };
template<class T, class U>
struct bf_squared_diff: binary_functor {__device__ __host__  T operator()(const T& t, const U& u)      const{ T ret =  t - (T)u; return ret*ret; } };
template<class T, class U>
struct bf_add_square: binary_functor {__device__ __host__  T operator()(const T& t, const U& u)      const{ return t + (T)(u*u);} };
template<class T, class U>
struct bf_and: binary_functor {__device__ __host__   T operator()(const T& t, const U& u)      const{ return t && u; } };
template<class T, class U>
struct bf_or: binary_functor { __device__ __host__   T operator()(const T& t, const U& u)      const{ return t || u; } };
template<class T, class U>
struct bf_min: binary_functor { __device__ __host__  T operator()(const T& t, const U& u)      const{ return t<u ? t : u; } };
template<class T, class U>
struct bf_max: binary_functor { __device__ __host__  T operator()(const T& t, const U& u)      const{ return t>u ? t : u; } };

// functors with parameter
template<class T, class U>
struct bf_axpy: binary_functor {  
	const T a;
	bf_axpy(const T& _a):a(_a){}
	__device__  __host__       T operator()(const T& t, const U& u) const{ return  a*t+(T)u; } 
};
template<class T, class U>
struct bf_xpby: binary_functor {  
	const T b;
	bf_xpby(const T& _b):b(_b){}
	__device__  __host__       T operator()(const T& t, const U& u) const{ return  t+b*(T)u; } 
};
template<class T, class U>
struct bf_axpby: binary_functor {  
	const T a;
	const T b;
	bf_axpby(const T& _a, const T& _b):a(_a),b(_b){}
	__device__  __host__       T operator()(const T& t, const U& u) const{ return  a*t + b*((T)u); } 
};

template< class J, class V, class I>
struct reduce_argmax : quadrary_functor {  
	__device__  __host__    void    operator()(V& t, J& i, const V& u, const I& j) const{
	   if (t > u) {
		  t = u;
		  i = (I) j;
	   }
	} 
};

template< class V, class I, class J>
struct reduce_argmin : quadrary_functor {  
	__device__  __host__    void    operator()(V& t, I& i,const  V& u, const J& j) const{
	   if (t < u) {
		  t = u;
		  i = (I) j;
	   }
	} 
};
// for reduce functors: set initial value of shared memory
template<class T, class FUNC>
struct reduce_functor_traits{ 
	static const T init_value = 0;     
	static const bool is_simple=false;
	static const bool returns_index = false;
	typedef T result_type;
};

template<class T>
struct reduce_functor_traits<T,bf_max<T,T> >{
	static const T init_value = -INT_MAX;    
	static const bool returns_index = false;
	typedef T result_type;

};

template<class T>
struct reduce_functor_traits<T,bf_min<T,T> >{  
	static const T init_value = INT_MAX;     
	static const bool returns_index = false;
	typedef T result_type;
};

template<class I, class T, class J>
struct reduce_functor_traits<T,reduce_argmax<I,T,J> >{  
	static const T init_value = -INT_MAX;     
	static const bool returns_index=true;
	typedef unsigned int result_type;	
};

template<class I, class T, class J>
struct reduce_functor_traits<T,reduce_argmin<I,T,J> >{  
	static const T init_value = INT_MAX;     
	static const bool returns_index=true;
	typedef unsigned int result_type;	
};

template<class F>
struct functor_dispatcher{
	__device__  __host__       void operator()() { cuvAssert(false); } 
};

template<>
struct functor_dispatcher<binary_functor_tag>{
	template<class T, class V, class I, class J>
	__device__  __host__   void operator()(const T& bf, V &t, I &i, const V &u, const J &j )const {
		t =  bf(t,u);
	} 
};

template<>
struct functor_dispatcher<quadrary_functor_tag>{
	template<class T, class V, class I, class J>
	__device__  __host__   void operator()(const T& qf, V &t, I &i, const V &u, const J &j ) const{
		qf(t,i,u,j);
	} 
};
};//namespace cuv
