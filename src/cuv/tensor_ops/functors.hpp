#include <boost/numeric/conversion/bounds.hpp>
#include <boost/static_assert.hpp>
#include <functional>

#include <cmath>
#include <cuv/tools/meta_programming.hpp>
#include <cuv/tools/cuv_general.hpp>
#include <cuv/tensor_ops/type_traits.hpp>

#define sgn(a) (copysign(1.f,(float)a))

namespace cuv {
/**
 * @defgroup UnaryFunctors
 * @{
 */

/** 
 * Unary functor base class
 */
template<class R, class T>
struct unary_functor{ 
	typedef R result_type;             /// the return type of this functor
	typedef T first_argument_type;     /// the only argument of this functor
};
/** 
 * Binary functor base class
 */
template<class R, class T, class S>
struct binary_functor{ 
	typedef R result_type;              /// the return type of this functor
	typedef T first_argument_type;      /// the type of the 1st argument 
	typedef S second_argument_type;     /// the type of the 2nd argument
};
/** 
 * Ternary functor base class
 */
template<class R, class T, class S, class U>
struct ternary_functor{ 
	typedef R result_type;              /// the type of the result of this funtcor
	typedef T first_argument_type;      /// the type of the 1st argument
	typedef S second_argument_type;     /// the type of the 2nd argument
	typedef U third_argument_type;      /// the type of the 3rd argument
};
/** 
 * functor with arity 4, base class
 */
template<class R, class T, class S, class U, class V>
struct fourary_functor{ 
	typedef R result_type;              /// the type of the result of this funtcor
	typedef T first_argument_type;      /// the type of the 1st argument
	typedef S second_argument_type;     /// the type of the 2nd argument
	typedef U third_argument_type;      /// the type of the 3rd argument
	typedef V fourth_argument_type;     /// the type of the 4th argument
};

/// calculates exp(x)
template<class R, class T>
struct uf_exp:unary_functor<R,T>{  inline __host__ __device__         R operator()(const T& t)const{ return expf(t);    } };
/// calculates exp(x) (deprecated)
template<class R, class T>
struct uf_exact_exp:unary_functor<R,T>{  inline __device__ __host__   R operator()(const T& t)const{ return expf(t);    } };
/// calculates sin(x)
template<class R, class T>
struct uf_sin:unary_functor<R,T>{  inline __host__ __device__         R operator()(const T& t)const{ return sinf(t);    } };
/// calculates cos(x)
template<class R, class T>
struct uf_cos:unary_functor<R,T>{  inline __host__ __device__         R operator()(const T& t)const{ return cosf(t);    } };
/// calculates log(x)
template<class R, class T>
struct uf_log:unary_functor<R,T>{  inline __device__ __host__         R operator()(const T& t)      const{ return logf(t);    } };
/// calculates log(1+x) using a stable numeric variant
template<class R, class T>
struct uf_log1p:unary_functor<R,T>{  inline __device__ __host__       R operator()(const T& t)      const{
	volatile float y;
	y = 1.f + t;
	return logf(y) - ((y-1.f)-t)/y;
} };
/// calculates signum(x)
template<class R, class T>
struct uf_sign:unary_functor<R,T>{  inline __device__ __host__        R operator()(const T& t)      const{ return sgn((float)t);    } };
/// calculates returns the absolute value of x
template<class R, class T>
struct uf_abs:unary_functor<R,T>{  inline __device__ __host__        R operator()(const T& t)      const{ return t<0 ? -t : t;    } };
/// calculates returns the absolute value of x (for unsigned char, this is identity!)
template<class R>
struct uf_abs<R, unsigned char>:unary_functor<R,unsigned char>{  inline __device__ __host__        R operator()(const unsigned char& t)      const{ return t;    } };
/// calculates returns the absolute value of x (for unsigned int, this is identity!)
template<class R>
struct uf_abs<R, unsigned int>:unary_functor<R,unsigned int>{  inline __device__ __host__         R operator()(const unsigned int& t)      const{ return t;    } };
/// calculates the logistic function 1/(1+exp(-x)
template<class R, class T>
struct uf_sigm:unary_functor<R,T>{  inline __device__  __host__ R operator()(const T& t)      const{ return ((R)1)/(((R)1)+expf(-t));    } };
/// calculates the derivative of logistic(x) as x(1-x)
template<class R, class T>
struct uf_dsigm:unary_functor<R,T>{  inline __device__ __host__       R operator()(const T& t)      const{ return t * (((T)1)-t); } };
/// calculates the hyperbolic tangent of x
template<class R, class T>
struct uf_tanh:unary_functor<R,T>{  inline __device__  __host__       R operator()(const T& t)      const{ return tanhf(t); } };
/// calculates the derivative of the hyperbolic tangent as 1-x*x
template<class R, class T>
struct uf_dtanh:unary_functor<R,T>{  inline __device__  __host__      R operator()(const T& t)      const{ return ((R)1) - (t*t); } };
/// squares x
template<class R, class T>
struct uf_square:unary_functor<R,T>{  inline __device__  __host__     R operator()(const T& t)      const{ return t*t; } };
/// calculates 1-x
template<class R, class T>
struct uf_sublin:unary_functor<R,T>{  inline __device__  __host__     R operator()(const T& t)      const{ return ((R)1)-t; } };
/// calculates -log(x)
template<class R, class T>
struct uf_energ:unary_functor<R,T>{  inline __device__  __host__      R operator()(const T& t)      const{ return -logf(t); } };
/// calculates 1/(x+epsilon)
template<class R, class T>
struct uf_inv:unary_functor<R,T>{  inline __device__  __host__        R operator()(const T& t)      const{ return ((R)1)/(t+((R)0.00000001)); } };
/// calculates sqrt(x)
template<class R, class T>
struct uf_sqrt:unary_functor<R,T>{  inline __device__  __host__       R operator()(const T& t)      const{ return sqrtf(t); } };
/// calculates (1/x-1)*x, useful for softmax
template<class R, class T>
struct uf_smax:unary_functor<R,T>{  inline __device__  __host__      R operator()(const T& t)      const{ return (((R)1)/t - (R) 1) * t; } };

/// calculates whether x is not-a-number
template<class T>
struct uf_is_nan:unary_functor<bool,T>{                 inline __device__  __host__   bool operator()(const T& t)             const{ return (t!=t) ; } };
/// calculates whether x is not-a-number (for int, this is false)
template<>
struct uf_is_nan<int>:unary_functor<bool,int>{            inline __device__  __host__   bool operator()(const int& t)           const{ return false ; } };
/// calculates whether x is not-a-number (for unsigned int, this is false)
template<>
struct uf_is_nan<unsigned int>:unary_functor<bool,unsigned int>{  inline __device__  __host__   bool operator()(const unsigned int& t) const{ return false ; } };
/// calculates whether x is not-a-number (for unsigned char, this is false)
template<>
struct uf_is_nan<unsigned char>:unary_functor<bool,unsigned char>{  inline __device__  __host__   bool operator()(const unsigned char& t) const{ return false ; } };
/// calculates whether x is not-a-number (for signed char, this is false)
template<>
struct uf_is_nan<signed char>:unary_functor<bool,signed char>{    inline __device__  __host__   bool operator()(const signed char& t)   const{ return false ; } };

/// calculates whether x is infinity
template<class T>
struct uf_is_inf:unary_functor<bool,T>{                 inline __device__  __host__     bool operator()(const T& t)            const{ return (bool)isinf(t); } };
/// calculates whether x is infinity (for int, this is false)
template<>                                                                                        
struct uf_is_inf<int>:unary_functor<bool,int>{            inline __device__  __host__     bool operator()(const int t)           const{ return false; } };
/// calculates whether x is infinity (for signed char, this is false)
template<>                                                                                        
struct uf_is_inf<signed char>:unary_functor<bool,signed char>{    inline __device__  __host__     bool operator()(const signed char t)   const{ return false; } };
/// calculates whether x is infinity (for unsigned int, this is false)
template<>
struct uf_is_inf<unsigned int>:unary_functor<bool,unsigned int>{  inline __device__  __host__     bool operator()(const unsigned int t) const{ return false; } };
/// calculates whether x is infinity (for unsigned char, this is false)
template<>
struct uf_is_inf<unsigned char>:unary_functor<bool,unsigned char>{  inline __device__  __host__     bool operator()(const unsigned char t) const{ return false; } };

/// calculates (x>0)*x
template<class R, class T>
struct uf_poslin:unary_functor<R,T>{  inline __device__  __host__     R operator()(const T& t)      const{ return (t > 0)*t; } };

/// calculates the logistic function with a temperature, 1/(1+exp(-x/temp))
template<class R, class T>
struct bf_sigm_temp:binary_functor<R,T,T>{ inline __device__  __host__       R operator()(const T& t, const T& temp)           const{ return ((T)1)/(((T)1)+expf(-t / (T)(temp))); } };

/// calculates the hyperbolic tangent with parameters, a*tanh(b*x)
template<class R, class T=R>
struct tf_tanh:ternary_functor<R,T,T,T>{  inline __device__  __host__       T operator()(const T& x, const T& a, const T& b)      const{ return a * tanhf(b * x); } };
/// calculates the derivative of the hyperbolic tangent with parameters, as b/a*(a+x)*(a-x)
template<class R, class T=R>
struct tf_dtanh:ternary_functor<R,T,T,T>{  inline __device__  __host__      T operator()(const T& x, const T& a, const T& b)      const{ return b/a * (a+x) * (a-x); } };

template<class R, class T=R>
struct tf_sqsquared_loss:ternary_functor<R,T,T,T>{  inline __device__  __host__      T operator()(const T& x, const T& x_hat, const T& m)      const{
    T v1 = max((T)0,(T)1-x_hat-m); 
    T v2 = max((T)0,(T)  x_hat-m); 
    return x*v1*v1 + ((T)1-x)*v2*v2;
}};
template<class R, class T=R>
struct tf_dsqsquared_loss:ternary_functor<R,T,T,T>{  inline __device__  __host__      T operator()(const T& x, const T& x_hat, const T& m)      const{
               //T v1 = max((T)0,(T)1-x_hat-m); 
               //T v2 = max((T)0,(T)  x_hat-m); 
	       //return x*v1*v1 + ((T)1-x)*v2*v2;
	       //return -x*v1 + ((T)1-x)*v2; // this is the _DERVIATIVE_ of the loss function
	uf_abs<T,T> absfunc;
	T diff = x_hat-x;
	//T absdiff = diff <0 ? -diff : diff;
	T absdiff = absfunc(diff);
	T v = max((T)0, (T) (absdiff-m));
	return sgn(diff) * v;
}};

/// calculates the rectifying transfer function log(1+expf(a*x))/a using a numerically stable variant
template<class R, class T, class A>
struct bf_rect:binary_functor<R,T,A>{  inline __device__  __host__       R operator()(const T& x, const A& a)      const{
	T ax = a*x;
	if(ax > 87.33f)
		return (T) x;
	return log(1.0f+expf(ax))/a;
}};
/// calculates the derivative of the rectifying transfer function 1-1/(x*exp(a))
template<class R, class T, class A>
struct bf_drect:binary_functor<R,T,A>{  inline __device__  __host__      R operator()(const T& x, const A& a)      const{ return 1-1/(x*expf(a)); } };

/// calculates pow(x,y)
template<class R, class T, class A>
struct bf_pow:binary_functor<R,T,A>{  inline __device__  __host__      R operator()(const T& x, const A& y)      const{ return pow((float)x,(float)y); } };

/// calculates 1/y * x^(y-1)
template<class R, class T, class A>
struct bf_dpow:binary_functor<R,T,A>{  inline __device__  __host__      R operator()(const T& x, const A& y)      const{ return ((float)y) * pow((float)x,(float)y-1.f); } };

/// calculates atan2(y,x)
template<class R, class T, class A>
struct bf_atan2:binary_functor<R,T,A>{  inline __device__  __host__      R operator()(const T& y, const A& x)      const{ return atan2((float)y,(float)x); } };
//struct bf_atan2:binary_functor<R,T,A>{  inline __device__  __host__      R operator()(const T& y, const A& x)      const{ return 2.f*atan(y/(sqrt(x*x+y*y)+x)); } };

/// calculates the norm of the arguments as sqrt(y*y+x*x)
template<class R, class T, class A>
struct bf_norm:binary_functor<R,T,A>{  inline __device__  __host__      R operator()(const T& x, const A& y)      const{ return sqrtf(y*y+x*x); } };


/// binds the 1st argument of a binary functor, yielding a unary functor
template<class __binary_functor>
struct uf_bind1st : unary_functor<typename __binary_functor::result_type,typename __binary_functor::second_argument_type>{
  typedef typename __binary_functor::first_argument_type first_argument_type;
  typedef typename __binary_functor::second_argument_type second_argument_type;
  typedef typename __binary_functor::result_type result_type;
  const  first_argument_type x; /// the encapsulated, constant 2nd argument of bf
  const __binary_functor bf; /// the encapsulated binary functor
  uf_bind1st(const __binary_functor& _bf, const second_argument_type& _x):x(_x),bf(_bf){};  
  inline __device__ __host__
  result_type operator()(const second_argument_type& s){ return bf(x,s); } /// calls bf with s and x
};

/// binds the 2nd argument of a binary functor, yielding a unary functor
template<class __binary_functor>
struct uf_bind2nd : unary_functor<typename __binary_functor::result_type,typename __binary_functor::first_argument_type>{
  typedef typename __binary_functor::first_argument_type first_argument_type;
  typedef typename __binary_functor::second_argument_type second_argument_type;
  typedef typename __binary_functor::result_type result_type;
  const  second_argument_type x; /// the encapsulated, constant 2nd argument of bf
  const __binary_functor bf; /// the encapsulated binary functor
  uf_bind2nd(const __binary_functor& _bf, const second_argument_type& _x):x(_x),bf(_bf){};  
  inline __device__ __host__
  result_type operator()(const first_argument_type& s){ return bf(s,x); } /// calls bf with s and x
};

/// creates a unary functor from a binary functor and a fixed first argument
template<class __binary_functor>
uf_bind1st<__binary_functor>
make_bind1st(const __binary_functor& bf, const typename __binary_functor::first_argument_type& x){ return uf_bind1st<__binary_functor>(bf, x); }

/// creates a unary functor from a binary functor and a fixed second argument
template<class __binary_functor>
uf_bind2nd<__binary_functor>
make_bind2nd(const __binary_functor& bf, const typename __binary_functor::second_argument_type& x){ return uf_bind2nd<__binary_functor>(bf, x); }

/// bind 2nd and 3rd arg of a ternary functor, yielding a unary functor
template<class ternary_functor>
struct bf_bind3rd:binary_functor<typename ternary_functor::result_type,typename ternary_functor::first_argument_type,typename ternary_functor::second_argument_type>{
  typedef typename ternary_functor::result_type result_type;
  typedef typename ternary_functor::first_argument_type first_argument_type;
  typedef typename ternary_functor::second_argument_type second_argument_type;
  typedef typename ternary_functor::third_argument_type third_argument_type;
  const third_argument_type y;  /// the encapsulated constant 3rd argument of tf
  const ternary_functor tf;     /// the encapsulated ternary functor
  bf_bind3rd(const ternary_functor& _tf, const third_argument_type& _y):y(_y),tf(_tf){};
  inline __device__ __host__
  result_type operator()(const first_argument_type& s, const second_argument_type& t){ return tf(s,t,y); } /// calls tf with s, t and y
};
/// creates a unary functor from a ternary functor and two fixed arguments
template<class __ternary_functor>
bf_bind3rd<__ternary_functor>
make_bind3rd(const __ternary_functor& tf, const typename __ternary_functor::third_argument_type& y){ return bf_bind3rd<__ternary_functor>(tf,y); }

/// bind 2nd and 3rd arg of a ternary functor, yielding a unary functor
template<class ternary_functor>
struct uf_bind2nd3rd:unary_functor<typename ternary_functor::result_type,typename ternary_functor::first_argument_type>{
  typedef typename ternary_functor::result_type result_type;
  typedef typename ternary_functor::first_argument_type first_argument_type;
  typedef typename ternary_functor::second_argument_type second_argument_type;
  typedef typename ternary_functor::third_argument_type third_argument_type;
  const second_argument_type x; /// the encapsulated constant 2nd argument of tf
  const third_argument_type y;  /// the encapsulated constant 3rd argument of tf
  const ternary_functor tf;     /// the encapsulated ternary functor
  uf_bind2nd3rd(const ternary_functor& _tf, const second_argument_type& _x, const third_argument_type& _y):x(_x),y(_y),tf(_tf){};
  inline __device__ __host__
  result_type operator()(const first_argument_type& s){ return tf(s,x,y); } /// calls tf with s, x and y
};
/// creates a unary functor from a ternary functor and two fixed arguments
template<class __ternary_functor>
uf_bind2nd3rd<__ternary_functor>
make_bind2nd3rd(const __ternary_functor& tf, const typename __ternary_functor::second_argument_type& x, const typename __ternary_functor::third_argument_type& y){ return uf_bind2nd3rd<__ternary_functor>(tf,x,y); }
/**
 * @}
 */

/**
 * @defgroup BinaryFunctors
 * @{
 */

/// calculates x==y
template<class R, class T, class U>
struct bf_equals : binary_functor<R,T,U> {  inline __device__  __host__       R operator()(const T& t, const U& u)      const{ return  t == (T)u; } };
/// calculates x+y
template<class R, class T, class U>
struct bf_plus : binary_functor<R,T,U> {  inline __device__  __host__       R operator()(const T& t, const U& u)      const{ return  t + (T)u; } };
/// calculates x-y
template<class R, class T, class U>
struct bf_minus: binary_functor<R,T,U> {  inline __device__  __host__      R operator()(const T& t, const U& u)      const{ return  t - (T)u; } };
/// calculates x*y
template<class R, class T, class U>
struct bf_multiplies: binary_functor<R,T,U> {  inline __device__  __host__ R operator()(const T& t, const U& u)      const{ return  t * (T)u; } };
/// calculates x/y
template<class R, class T, class U>
struct bf_divides: binary_functor<R,T,U> {  inline __device__  __host__    R operator()(const T& t, const U& u)      const{ return  t / (T)u; } };
/// calculates (x-y)^2
template<class R, class T, class U>
struct bf_squared_diff: binary_functor<R,T,U> {inline __device__ __host__  R operator()(const T& t, const U& u)      const{ T ret =  t - (T)u; return ret*ret; } };
/// calculates x+log(y)
template<class R, class T, class U>
struct bf_add_log: binary_functor<R,T,U> {inline __device__ __host__  R operator()(const T& t, const U& u)      const{ return t + (T)logf(u);} };
/// calculates x+y^2
template<class R, class T, class U>
struct bf_add_square: binary_functor<R,T,U> {inline __device__ __host__  R operator()(const T& t, const U& u)      const{ return t + (T)(u*u);} };
/// calculates x && y
template<class R, class T, class U>
struct bf_and: binary_functor<R,T,U> {inline __device__ __host__   R operator()(const T& t, const U& u)      const{ return t && u; } };
/// calculates x || y
template<class R, class T, class U>
struct bf_or: binary_functor<R,T,U> { inline __device__ __host__   R operator()(const T& t, const U& u)      const{ return t || u; } };
/// calculates the minimum of x and y
template<class R, class T, class U>
struct bf_min: binary_functor<R,T,U> { inline __device__ __host__  R operator()(const T& t, const U& u)      const{ return t<u ? t : u; } };
/// calculates the maximum of x and y
template<class R, class T, class U>
struct bf_max: binary_functor<R,T,U> { inline __device__ __host__  R operator()(const T& t, const U& u)      const{ return t>u ? t : u; } };
/// calculates the robust absolute value of x
template<class R, class T, class U>
struct bf_robust_abs: binary_functor<R,T,U> { inline __device__ __host__  R operator()(const T& t, const U& u)      const{ return sqrt((float)t*(float)t+u); } };
/// calculates the derivative of robust absolute value of x w.r.t. x 
template<class R, class T, class U>
struct bf_drobust_abs: binary_functor<R,T,U> { inline __device__ __host__  R operator()(const T& t, const U& u)      const{ return t / sqrt((float)t*(float)t+u); } };

/// calculates a*x+y for fixed a
template<class R, class T, class U>
struct bf_axpy: binary_functor<R,T,U> {  
	const T a;
	bf_axpy(const T& _a):a(_a){}
	inline __device__  __host__       R operator()(const T& t, const U& u) const{ return  a*t+(T)u; } 
};
/// calculates x+b*y for fixed b
template<class R, class T, class U>
struct bf_xpby: binary_functor<R,T,U> {  
	const T b;
	bf_xpby(const T& _b):b(_b){}
	inline __device__  __host__       R operator()(const T& t, const U& u) const{ return  t+b*(T)u; } 
};
/// calculates a*x+b*y for fixed a, b
template<class R, class T, class U>
struct bf_axpby: binary_functor<R,T,U> {  
	const T a;
	const T b;
	bf_axpby(const T& _a, const T& _b):a(_a),b(_b){}
	inline __device__  __host__       R operator()(const T& t, const U& u) const{ return  a*t + b*((T)u); } 
};
/** @} */

/**
 * @defgroup AxisReduceFunctors
 * @{
 */

/**
 * Generic axis reduction functor.
 *
 * The two binary functor arguments are used for the two stages of reduction.
 * RV is used to combine two raw data values, whereas
 * RR is used to combine two already reduced values.
 *
 * For example, to calculate the sum of squares,
 * RV is x + y*y
 * and 
 * RR is x + y.
 */
template<class RV, class RR=RV>
struct axis_reduce_functor{
	typedef RV result_value_functor_type;
	typedef RR result_result_functor_type;
	typedef typename RV::first_argument_type    result_type;
	typedef typename RV::second_argument_type   value_type;
	RV mRV;  /// result-type, value-type
	RR mRR;  /// result-type, result-type
	axis_reduce_functor(const RV t, const RR s) :mRV(t), mRR(s) { }
	template<class T>
	axis_reduce_functor(const T t) :mRV(t), mRR(t) { }
	/// combine a reduce value and a data value
	template<class R1, class I, class R2>
	inline __device__ __host__ void rv(R1& r, I& idx1, const R2&  v, const I& idx2) { r = mRV(r,v);  }
	/// combine two reduce values
	template<class R1, class I, class R2>
	inline __device__ __host__ void rr(R1& r, I& idx1, const R2& r2, const I& idx2){ r = mRR(r,r2); }
};

/**
 * Generic axis argument reduction functor.
 *
 * The two four-ary functor arguments are used for the two stages of reduction.
 * RV is used to combine two raw data values, whereas
 * RR is used to combine two already reduced values.
 *
 * The four-ary functors get the value and the index, respectively, as references.
 * They are expected to change the first values as needed. E.g., to calculate arg-max
 *
 * RV(r,idx1,v,idx2) calculates 
 *
 *   r    = max(r,v)
 *   idx1 = r>v ? idx1 : idx2
 *
 * and RR does the same.
 *
 */
template<class RV, class RR=RV>
struct axis_arg_reduce_functor{
	typedef RV result_value_functor_type;
	typedef RR result_result_functor_type;
	typedef typename RV::first_argument_type    value_type;
	typedef typename RV::second_argument_type   index_type;
	RV mRV;  // result-type, value-type
	RR mRR;  // result-type, result-type
	axis_arg_reduce_functor(const RV t, const RR s) :mRV(t), mRR(s) { }
	template<class T>
	axis_arg_reduce_functor(const T t) :mRV(t), mRR(t) { }
	inline __device__ __host__ void rv(value_type& r, index_type& idx1, const value_type&  v, const index_type& idx2) { mRV(r,idx1,v,idx2);  }
	inline __device__ __host__ void rr(value_type& r, index_type& idx1, const value_type& r2, const index_type& idx2) { mRR(r,idx1,r2,idx2); }
};
/// construct axis-reduce functors conveniently
template<class RV, class RR>
axis_reduce_functor<RV,RR> make_reduce_functor(const RV&t, const RR& s){ return axis_reduce_functor<RV,RR>(t,s); }
/// construct axis-reduce functors conveniently
template<class RV>
axis_reduce_functor<RV,RV> make_reduce_functor(const RV&t)             { return axis_reduce_functor<RV,RV>(t); }

/// construct arg-axis-reduce functors conveniently
template<class RV, class RR>
axis_arg_reduce_functor<RV,RR> make_arg_reduce_functor(const RV&t, const RR& s){ return axis_arg_reduce_functor<RV,RR>(t,s); }
/// construct arg-axis-reduce functors conveniently
template<class RV>
axis_arg_reduce_functor<RV,RV> make_arg_reduce_functor(const RV&t)             { return axis_arg_reduce_functor<RV,RV>(t); }

/// logarithm of the sum of exponentiations of the inputs in a numerically stable way. log(exp(x)+exp(y))
template<class T>
struct bf_logaddexp : binary_functor<float,T, T> {  
	inline __device__  __host__    float    operator()(const T& t, const T& u) const{
		const float diff = (float)t - (float) u;
		uf_log1p<float,float> log1p;
		if(diff > 0)
			return t + log1p(expf(-diff));
		else if(diff<=0)
			return u + log1p(expf(diff));
		else
			return t+u;
	} 
};

/// computes the negative log of cross-entropy \f$-x\log(z)-(1-x)\log(1-z)\f$ of logistic \f$z=1/(1+\exp(-y))\f$
template<class R, class T, class U>
struct bf_logce_of_logistic:binary_functor<R,T,U>{ inline __device__  __host__       R operator()(const T& x, const T& y)           const{ 
    bf_logaddexp<float> lae;
    return  x*lae(0.f,-y)+(1.f-x)*lae(0.f,y);
} };

// BF_BERNOULLI_KL  computes Kullback-Leibler divergence of two bernoulli variables \f$x\log(x/y)+(1-x)\log\frac{1-x}{1-y}\f$
template<class R, class T, class U>
struct bf_bernoulli_kl:binary_functor<R,T,U>{ inline __device__  __host__       R operator()(const T& x_, const T& y_)           const{ 
    float y = max(0.0001f,(float)y_);
    float x = max(0.0001f,(float)x_);
    return  x*log(x/y)+(1.f-x)*log((1-x)/(1-y));
} };
// BF_DBERNOULLI_KL computes derivative of Kullback-Leibler divergence of two bernoulli variables w.r.t. y: \f$\frac{x-y}{y(y-1)}\f$
template<class R, class T, class U>
struct bf_dbernoulli_kl:binary_functor<R,T,U>{ inline __device__  __host__       R operator()(const T& x_, const T& y_)           const{ 
    float y = max(0.0001f,(float)y_);
    float x = max(0.0001f,(float)x_);
    return  (x-y)/(y*y-y);
} };


/// calculates arg-max of two values and their indices
template<class V, class I>
struct reduce_argmax : fourary_functor<void,V,I,V,I> {  
	inline __device__  __host__    void    operator()(V& t, I& i, const V& u, const I& j) const{
	   if (u > t) {
		  t = u;
		  i = (I) j;
	   }
	} 
};

/// calculates arg-min of two values and their indices
template< class V, class I>
struct reduce_argmin : fourary_functor<void,V,I,V,I> {  
	inline __device__  __host__    void    operator()(V& t, I& i,const  V& u, const I& j) const{
	   if (u < t) {
		  t = u;
		  i = (I) j;
	   }
	} 
};

/** 
 * Generic traits class to get basic info about functors.
 *
 * We assume unless specialized, that initial value is 0 and the function does not return an index.
 */
template<class FUNC>
struct reduce_functor_traits{ 
	static const typename FUNC::result_type init_value(){return 0;}
	static const bool returns_index = false;
};

/// specialization of reduce_functor_traits for max functor: initial value is lowest value of this type
template<class T>
struct reduce_functor_traits<bf_max<T,T,T> >{
	static const T init_value(){return boost::numeric::bounds<T>::lowest();}
	static const bool returns_index = false;
};

/// specialization of reduce_functor_traits for min functor: initial value is highest value of this type
template<class T>
struct reduce_functor_traits<bf_min<T,T,T> >{  
	static const T init_value(){return boost::numeric::bounds<T>::highest();}
	static const bool returns_index = false;
};

/// specialization of reduce_functor_traits for logaddexp functor: initial value is negative infinity
template<class T>
struct reduce_functor_traits<bf_logaddexp<T> >{
	static const T init_value(){return -std::numeric_limits<T>::infinity();}
	static const bool returns_index = false;
};


//template<class T>
//struct reduce_functor_traits<bf_logaddexp<T> >{  // this one is symmetric!!!
// ...
//};

/// arg_max also starts with lowest value for initialization and returns an index
template<class I, class T>
struct reduce_functor_traits<reduce_argmax<T,I> >{  
	static const T init_value(){return boost::numeric::bounds<T>::lowest();}
	static const bool returns_index=true;
};

/// arg_min also starts with lowest value for initialization and returns an index
template<class I, class T>
struct reduce_functor_traits<reduce_argmin<T,I> >{  
	static const T init_value(){return boost::numeric::bounds<T>::highest();}
	static const bool returns_index=true;
};

/**
 * @}
 */

/**
 * @defgroup VectorDataTypeFunctors
 * @{
 */
namespace detail
{

template<class R, class T, class UF, bool TIsBaseType, int D>
struct apply_uf_to_vec {
	inline __host__ __device__ R operator()(const T& t, UF& uf){
		return R();
	}
};
template<class R, class T, class UF>
struct apply_uf_to_vec<R,T,UF,false,1> {
	inline __host__ __device__ R operator()(const T& t, UF& uf){
		R r;
		r.x = uf(t.x);
		return r; 
	} 
};
template<class R, class T, class UF>
struct apply_uf_to_vec<R,T,UF,false,2> {
	inline __host__ __device__ R operator()(const T& t, UF& uf){
		R r;
		r.x = uf(t.x);
		r.y = uf(t.y);
		return r; 
	} 
};
template<class R, class T, class UF>
struct apply_uf_to_vec<R,T,UF,false,3> {
	inline __host__ __device__ R operator()(const T& t, UF& uf){
		R r;
		r.x = uf(t.x);
		r.y = uf(t.y);
		r.z = uf(t.z);
		return r; 
	} 
};
template<class R, class T, class UF>
struct apply_uf_to_vec<R,T,UF,false,4> {
	inline __host__ __device__ R operator()(const T& t, UF& uf){
		R r;
		r.x = uf(t.x);
		r.y = uf(t.y);
		r.z = uf(t.z);
		r.w = uf(t.w);
		return r; 
	} 
};
template<class R, class T, class UF>
struct apply_uf_to_vec<R,T,UF,true,1> {
	inline __host__ __device__ R operator()(const T& t, UF& uf){
		R r;
		r.x = uf(t);
		return r; 
	} 
};
template<class R, class T, class UF>
struct apply_uf_to_vec<R,T,UF,true,2> {
	inline __host__ __device__ R operator()(const T& t, UF& uf){
		R r;
		r.x = r.y = uf(t);
		return r; 
	} 
};
template<class R, class T, class UF>
struct apply_uf_to_vec<R,T,UF,true,3> {
	inline __host__ __device__ R operator()(const T& t, UF& uf){
		R r;
		r.x = r.y = r.z = uf(t);
		return r; 
	} 
};
template<class R, class T, class UF>
struct apply_uf_to_vec<R,T,UF,true,4> {
	inline __host__ __device__ R operator()(const T& t, UF& uf){
		R r;
		r.x = r.y = r.z = r.w = uf(t.w);
		return r; 
	} 
};


/**
 * apply a binary functor to a vector
 * where the second vector is either vector-valued 
 * or scalar
 */
template<class R, class T, class S,  class BF, bool SIsBaseType, int D>
struct apply_bf_to_vec { };
template<class R, class T, class S, class BF>
struct apply_bf_to_vec<R,T,S,BF,false,1> {
	inline __host__ __device__ R operator()(const T& t, const S& s, BF& bf){
		R r;
		r.x = bf(t.x, s.x);
		return r; } };
template<class R, class T, class S, class BF>
struct apply_bf_to_vec<R,T,S,BF,false,2> {
	inline __host__ __device__ R operator()(const T& t, const S& s, BF& bf){
		R r;
		r.x = bf(t.x,s.x);
		r.y = bf(t.y,s.y);
		return r; } };
template<class R, class T,class S, class BF>
struct apply_bf_to_vec<R,T,S,BF,false,3> {
	inline __host__ __device__ R operator()( const T& t, const S& s, BF& bf){
		R r;
		r.x = bf(t.x,s.x);
		r.y = bf(t.y,s.y);
		r.z = bf(t.z,s.z);
		return r; } };
template<class R, class T,class S, class BF>
struct apply_bf_to_vec<R,T,S,BF,false,4> {
	inline __host__ __device__ R operator()(const T& t, const S& s, BF& bf){
		R r;
		r.x = bf(t.x,s.x);
		r.y = bf(t.y,s.y);
		r.z = bf(t.z,s.z);
		r.w = bf(t.w,s.w);
		return r; } };
template<class R, class T, class S,class BF>
struct apply_bf_to_vec<R,T,S,BF,true,1> {
	inline __host__ __device__ R operator()(const T& t, const S& s,  BF& bf){
		R r;
		r = bf(t,s);
		return r; } };
template<class R, class T,class S, class BF>
struct apply_bf_to_vec<R,T,S,BF,true,2> {
	inline __host__ __device__ R operator()(const T& t, const S& s,  BF& bf){
		R r;
		r.x = bf(t.x,s);
		r.y = bf(t.y,s);
		return r; } };
template<class R, class T,class S, class BF>
struct apply_bf_to_vec<R,T,S,BF,true,3> {
	inline __host__ __device__ R operator()(const T& t, const S& s,  BF& bf){
		R r;
		r.x = bf(t.x,s);
		r.y = bf(t.y,s);
		r.z = bf(t.z,s);
		return r; } };
template<class R, class T,class S, class BF>
struct apply_bf_to_vec<R,T,S,BF,true,4> {
	inline __host__ __device__ R operator()(const T& t, const S& s,  BF& bf){
		R r;
		r.x = bf(t.x,s);
		r.y = bf(t.y,s);
		r.z = bf(t.z,s);
		r.w = bf(t.w,s);
		return r; } };

}

/**
 * apply a scalar functor pointwise to each component of a vector datatype
 */
template<class R, class T, class UF>
struct uf_vd_vd
:	public unary_functor<R,T>
{
	typedef unary_functor<R,T> super_type;
	typedef R result_type;
	typedef T first_argument_type;
	BOOST_STATIC_ASSERT((IsSame<typename vector_type_traits<T>::base_type,typename UF::first_argument_type>::Result::value));
	BOOST_STATIC_ASSERT((IsSame<typename vector_type_traits<R>::base_type,typename UF::result_type>::Result::value));

	UF m_uf;
	uf_vd_vd(UF uf):m_uf(uf){}
	detail::apply_uf_to_vec<R,T,UF,
				       vector_type_traits<first_argument_type>::is_base_type,
				       vector_type_traits<first_argument_type>::dim> m_applier;

	inline __host__ __device__ R operator()(first_argument_type& t){
		return m_applier(t, m_uf); }
};
template<int D, class UF>
uf_vd_vd<typename vector_type_traits<typename UF::result_type>::template vector<D>::type,
	typename vector_type_traits<typename UF::first_argument_type>::template vector<D>::type,
	UF>
make_uf_vd_vd(UF uf){ 
	return uf_vd_vd< typename vector_type_traits<typename UF::result_type>::template vector<D>::type,
			 typename vector_type_traits<typename UF::first_argument_type>::template vector<D>::type,
			 UF>(uf);
}

/**
 * apply a binary functor pointwise to each component of a vector datatype
 */
template<class R, class T, class S, class BF>
struct bf_vd_vd
:	public binary_functor<R,T,S>
{
	typedef binary_functor<R,T,S> super_type;
	typedef R result_type;
	typedef T first_argument_type;
	typedef S second_argument_type;
	BOOST_STATIC_ASSERT((IsSame<typename vector_type_traits<S>::base_type,typename BF::second_argument_type>::Result::value));
	BOOST_STATIC_ASSERT((IsSame<typename vector_type_traits<T>::base_type,typename BF::first_argument_type>::Result::value));
	BOOST_STATIC_ASSERT((IsSame<typename vector_type_traits<R>::base_type,typename BF::result_type>::Result::value));

	BF m_bf;
	bf_vd_vd(BF bf):m_bf(bf){}
	detail::apply_bf_to_vec<R,T,S,BF,
				       vector_type_traits<second_argument_type>::is_base_type,
				       vector_type_traits<result_type>::dim> m_applier;

	inline __host__ __device__ R operator()(first_argument_type& t, second_argument_type& s){
		return m_applier(t, s, m_bf); }
};
template<int D1,int D2,class BF>
bf_vd_vd<
	typename vector_type_traits<typename BF::result_type>::template          vector<D1>::type,
	typename vector_type_traits<typename BF::first_argument_type>::template  vector<D1>::type,
	typename vector_type_traits<typename BF::second_argument_type>::template vector<D2>::type,
	BF>
make_bf_vd_vd(BF bf){ 
	return bf_vd_vd< 
		typename vector_type_traits<typename BF::result_type>::template          vector<D1>::type,
		typename vector_type_traits<typename BF::first_argument_type>::template  vector<D1>::type,
		typename vector_type_traits<typename BF::second_argument_type>::template vector<D2>::type,
			 BF>(bf);
}

/**
 * @}
 */

};//namespace cuv
