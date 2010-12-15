#include <cmath>
#include <cuv_general.hpp>

#define sgn(a) (copysign(1.f,a))

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
struct bf_plus{  __device__  __host__       T operator()(const T& t, const U& u)      const{ return  t + (T)u; } };
template<class T, class U>
struct bf_minus{  __device__  __host__      T operator()(const T& t, const U& u)      const{ return  t - (T)u; } };
template<class T, class U>
struct bf_multiplies{  __device__  __host__ T operator()(const T& t, const U& u)      const{ return  t * (T)u; } };
template<class T, class U>
struct bf_divides{  __device__  __host__    T operator()(const T& t, const U& u)      const{ return  t / (T)u; } };
template<class T, class U>
struct bf_squared_diff{__device__ __host__  T operator()(const T& t, const U& u)      const{ T ret =  t - (T)u; return ret*ret; } };
template<class T, class U>
struct bf_and{__device__ __host__   T operator()(const T& t, const U& u)      const{ return t && u; } };
template<class T, class U>
struct bf_or{ __device__ __host__   T operator()(const T& t, const U& u)      const{ return t || u; } };
template<class T, class U>
struct bf_min{ __device__ __host__  T operator()(const T& t, const U& u)      const{ return t<u ? t : u; } };
template<class T, class U>
struct bf_max{ __device__ __host__  T operator()(const T& t, const U& u)      const{ return t>u ? t : u; } };

// functors with parameter
template<class T, class U>
struct bf_axpy{  
	const T a;
	bf_axpy(const T& _a):a(_a){}
	__device__  __host__       T operator()(const T& t, const U& u) const{ return  a*t+(T)u; } 
};
template<class T, class U>
struct bf_xpby{  
	const T b;
	bf_xpby(const T& _b):b(_b){}
	__device__  __host__       T operator()(const T& t, const U& u) const{ return  t+b*(T)u; } 
};
template<class T, class U>
struct bf_axpby{  
	const T a;
	const T b;
	bf_axpby(const T& _a, const T& _b):a(_a),b(_b){}
	__device__  __host__       T operator()(const T& t, const U& u) const{ return  a*t + b*((T)u); } 
};

