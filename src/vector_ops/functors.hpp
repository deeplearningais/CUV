#include <boost/numeric/conversion/bounds.hpp>
#include <boost/static_assert.hpp>
#include <functional>

#include <cmath>
#include <cuv_general.hpp>

#define sgn(a) (copysign(1.f,a))

namespace cuv {
	/** 
	 * @brief Adds return_type to all functors
	 */
template<class R, class T>
struct unary_functor{ 
	typedef R result_type; 
	typedef T first_argument_type; 
};
template<class R, class T, class S>
struct binary_functor{ 
	typedef R result_type; 
	typedef T first_argument_type; 
	typedef S second_argument_type; 
};
template<class R, class T, class S, class U>
struct ternary_functor{ 
	typedef R result_type; 
	typedef T first_argument_type; 
	typedef S second_argument_type; 
	typedef U third_argument_type; 
};
template<class R, class T, class S, class U, class V>
struct fourary_functor{ 
	typedef R result_type; 
	typedef T first_argument_type; 
	typedef S second_argument_type; 
	typedef U third_argument_type; 
	typedef V fourth_argument_type; 
};

template<class R, class T>
struct uf_exp:unary_functor<R,T>{  __host__ __device__         R operator()(const T& t)const{ return expf(t);    } };
template<class R, class T>
struct uf_exact_exp:unary_functor<R,T>{  __device__ __host__   R operator()(const T& t)const{ return expf(t);    } };
template<class R, class T>
struct uf_log:unary_functor<R,T>{  __device__ __host__         R operator()(const T& t)      const{ return logf(t);    } };
template<class R, class T>
struct uf_log1p:unary_functor<R,T>{  __device__ __host__       R operator()(const T& t)      const{
	volatile float y;
	y = 1.f + t;
	return logf(y) - ((y-1.f)-t)/y;
} };
template<class R, class T>
struct uf_sign:unary_functor<R,T>{  __device__ __host__        R operator()(const T& t)      const{ return sgn((float)t);    } };
template<class R, class T>
struct uf_abs:unary_functor<R,T>{  __device__ __host__        R operator()(const T& t)      const{ return t<0 ? -t : t;    } };
template<class R>
struct uf_abs<R, unsigned char>:unary_functor<R,unsigned char>{  __device__ __host__        R operator()(const unsigned char& t)      const{ return t;    } };
template<class R>
struct uf_abs<R, unsigned int>:unary_functor<R,unsigned int>{  __device__ __host__         R operator()(const unsigned int& t)      const{ return t;    } };
template<class R, class T>
struct uf_sigm:unary_functor<R,T>{  __device__  __host__ R operator()(const T& t)      const{ return ((R)1)/(((R)1)+expf(-t));    } };
template<class R, class T>
struct uf_dsigm:unary_functor<R,T>{  __device__ __host__       R operator()(const T& t)      const{ return t * (((T)1)-t); } };
template<class R, class T>
struct uf_tanh:unary_functor<R,T>{  __device__  __host__       R operator()(const T& t)      const{ return tanhf(t); } };
template<class R, class T>
struct uf_dtanh:unary_functor<R,T>{  __device__  __host__      R operator()(const T& t)      const{ return ((R)1) - (t*t); } };
template<class R, class T>
struct uf_square:unary_functor<R,T>{  __device__  __host__     R operator()(const T& t)      const{ return t*t; } };
template<class R, class T>
struct uf_sublin:unary_functor<R,T>{  __device__  __host__     R operator()(const T& t)      const{ return ((R)1)-t; } };
template<class R, class T>
struct uf_energ:unary_functor<R,T>{  __device__  __host__      R operator()(const T& t)      const{ return -logf(t); } };
template<class R, class T>
struct uf_inv:unary_functor<R,T>{  __device__  __host__        R operator()(const T& t)      const{ return ((R)1)/(t+((R)0.00000001)); } };
template<class R, class T>
struct uf_sqrt:unary_functor<R,T>{  __device__  __host__       R operator()(const T& t)      const{ return sqrtf(t); } };
template<class R, class T>
struct uf_smax:unary_functor<R,T>{  __device__  __host__      R operator()(const T& t)      const{ return (((R)1)/t - (R) 1) * t; } };

template<class T>
struct uf_is_nan:unary_functor<bool,T>{                 __device__  __host__   bool operator()(const T& t)             const{ return (t!=t) ; } };
template<>
struct uf_is_nan<int>:unary_functor<bool,int>{            __device__  __host__   bool operator()(const int& t)           const{ return false ; } };
template<>
struct uf_is_nan<unsigned int>:unary_functor<bool,unsigned int>{  __device__  __host__   bool operator()(const unsigned int& t) const{ return false ; } };
template<>
struct uf_is_nan<unsigned char>:unary_functor<bool,unsigned char>{  __device__  __host__   bool operator()(const unsigned char& t) const{ return false ; } };
template<>
struct uf_is_nan<signed char>:unary_functor<bool,signed char>{    __device__  __host__   bool operator()(const signed char& t)   const{ return false ; } };

template<class T>
struct uf_is_inf:unary_functor<bool,T>{                 __device__  __host__     bool operator()(const T& t)            const{ return (bool)isinf(t); } };
template<>                                                                                        
struct uf_is_inf<int>:unary_functor<bool,int>{            __device__  __host__     bool operator()(const int t)           const{ return false; } };
template<>                                                                                        
struct uf_is_inf<signed char>:unary_functor<bool,signed char>{    __device__  __host__     bool operator()(const signed char t)   const{ return false; } };
template<>
struct uf_is_inf<unsigned int>:unary_functor<bool,unsigned int>{  __device__  __host__     bool operator()(const unsigned int t) const{ return false; } };
template<>
struct uf_is_inf<unsigned char>:unary_functor<bool,unsigned char>{  __device__  __host__     bool operator()(const unsigned char t) const{ return false; } };

template<class R, class T>
struct uf_poslin:unary_functor<R,T>{  __device__  __host__     R operator()(const T& t)      const{ return (t > 0)*t; } };


template<class R, class T>
struct bf_sigm_temp:binary_functor<R,T,T>{ __device__  __host__       R operator()(const T& t, const T& temp)           const{ return ((T)1)/(((T)1)+expf(-t / (T)(temp))); } };

template<class R, class T=R>
struct tf_tanh:ternary_functor<R,T,T,T>{  __device__  __host__       T operator()(const T& x, const T& a, const T& b)      const{ return a * tanhf(b * x); } };
template<class R, class T=R>
struct tf_dtanh:ternary_functor<R,T,T,T>{  __device__  __host__      T operator()(const T& x, const T& a, const T& b)      const{ return b/a * (a+x) * (a-x); } };

// rectifying transfer function. a is param beta
template<class R, class T, class A>
struct bf_rect:binary_functor<R,T,A>{  __device__  __host__       R operator()(const T& x, const A& a)      const{
	T ax = a*x;
	if(ax > 87.33f)
		return (T) x;
	return log(1.0f+expf(ax))/a;
}};
/*template<class T, class A>*/
/*struct bf_rect{  __device__  __host__       T operator()(const T& x, const A& a)      const{ return (T) log(1.0 + (double)exp((double)a*x))/a; } };*/
template<class R, class T, class A>
struct bf_drect:binary_functor<R,T,A>{  __device__  __host__      R operator()(const T& x, const A& a)      const{ return 1-1/(x*expf(a)); } };


// binds the 2nd argument of a binary functor, yielding a unary functor
template<class __binary_functor>
struct uf_bind2nd : unary_functor<typename __binary_functor::result_type,typename __binary_functor::first_argument_type>{
  typedef typename __binary_functor::first_argument_type first_argument_type;
  typedef typename __binary_functor::second_argument_type second_argument_type;
  typedef typename __binary_functor::result_type result_type;
  const  second_argument_type x;
  const __binary_functor bf;
  uf_bind2nd(const __binary_functor& _bf, const second_argument_type& _x):x(_x),bf(_bf){};
  __device__ __host__
  result_type operator()(const first_argument_type& s){ return bf(s,x); }
};
template<class __binary_functor>
uf_bind2nd<__binary_functor>
make_bind2nd(const __binary_functor& bf, const typename __binary_functor::second_argument_type& x){ return uf_bind2nd<__binary_functor>(bf, x); }

// bind 2nd and 3rd arg of a ternary functor, yielding a unary functor
template<class ternary_functor>
struct uf_bind2nd3rd:unary_functor<typename ternary_functor::result_type,typename ternary_functor::first_argument_type>{
  typedef typename ternary_functor::result_type result_type;
  typedef typename ternary_functor::first_argument_type first_argument_type;
  typedef typename ternary_functor::second_argument_type second_argument_type;
  typedef typename ternary_functor::third_argument_type third_argument_type;
  const second_argument_type x;
  const third_argument_type y;
  const ternary_functor tf;
  uf_bind2nd3rd(const ternary_functor& _tf, const second_argument_type& _x, const third_argument_type& _y):x(_x),y(_y),tf(_tf){};
  __device__ __host__
  result_type operator()(const first_argument_type& s){ return tf(s,x,y); }
};
template<class __ternary_functor>
uf_bind2nd3rd<__ternary_functor>
make_bind2nd3rd(const __ternary_functor& tf, const typename __ternary_functor::second_argument_type& x, const typename __ternary_functor::third_argument_type& y){ return uf_bind2nd3rd<__ternary_functor>(tf,x,y); }

/*
 * Binary Functors
 */

// functors without parameter
template<class R, class T, class U>
struct bf_plus : binary_functor<R,T,U> {  __device__  __host__       R operator()(const T& t, const U& u)      const{ return  t + (T)u; } };
template<class R, class T, class U>
struct bf_minus: binary_functor<R,T,U> {  __device__  __host__      R operator()(const T& t, const U& u)      const{ return  t - (T)u; } };
template<class R, class T, class U>
struct bf_multiplies: binary_functor<R,T,U> {  __device__  __host__ R operator()(const T& t, const U& u)      const{ return  t * (T)u; } };
template<class R, class T, class U>
struct bf_divides: binary_functor<R,T,U> {  __device__  __host__    R operator()(const T& t, const U& u)      const{ return  t / (T)u; } };
template<class R, class T, class U>
struct bf_squared_diff: binary_functor<R,T,U> {__device__ __host__  R operator()(const T& t, const U& u)      const{ T ret =  t - (T)u; return ret*ret; } };
template<class R, class T, class U>
struct bf_add_log: binary_functor<R,T,U> {__device__ __host__  R operator()(const T& t, const U& u)      const{ return t + (T)logf(u);} };
template<class R, class T, class U>
struct bf_add_square: binary_functor<R,T,U> {__device__ __host__  R operator()(const T& t, const U& u)      const{ return t + (T)(u*u);} };
template<class R, class T, class U>
struct bf_and: binary_functor<R,T,U> {__device__ __host__   R operator()(const T& t, const U& u)      const{ return t && u; } };
template<class R, class T, class U>
struct bf_or: binary_functor<R,T,U> { __device__ __host__   R operator()(const T& t, const U& u)      const{ return t || u; } };
template<class R, class T, class U>
struct bf_min: binary_functor<R,T,U> { __device__ __host__  R operator()(const T& t, const U& u)      const{ return t<u ? t : u; } };
template<class R, class T, class U>
struct bf_max: binary_functor<R,T,U> { __device__ __host__  R operator()(const T& t, const U& u)      const{ return t>u ? t : u; } };

// functors with parameter
template<class R, class T, class U>
struct bf_axpy: binary_functor<R,T,U> {  
	const T a;
	bf_axpy(const T& _a):a(_a){}
	__device__  __host__       R operator()(const T& t, const U& u) const{ return  a*t+(T)u; } 
};
template<class R, class T, class U>
struct bf_xpby: binary_functor<R,T,U> {  
	const T b;
	bf_xpby(const T& _b):b(_b){}
	__device__  __host__       R operator()(const T& t, const U& u) const{ return  t+b*(T)u; } 
};
template<class R, class T, class U>
struct bf_axpby: binary_functor<R,T,U> {  
	const T a;
	const T b;
	bf_axpby(const T& _a, const T& _b):a(_a),b(_b){}
	__device__  __host__       R operator()(const T& t, const U& u) const{ return  a*t + b*((T)u); } 
};

template<class RV, class RR=RV>
struct axis_reduce_functor{
	typedef RV result_value_functor_type;
	typedef RR result_result_functor_type;
	typedef typename RV::first_argument_type    result_type;
	typedef typename RV::second_argument_type   value_type;
	RV mRV;  // result-type, value-type
	RR mRR;  // result-type, result-type
	axis_reduce_functor(const RV t, const RR s) :mRV(t), mRR(s) { }
	template<class T>
	axis_reduce_functor(const T t) :mRV(t), mRR(t) { }
	template<class R1, class I, class R2>
	__device__ __host__ void rv(R1& r, I& idx1, const R2&  v, const I& idx2) { r = mRV(r,v);  }
	template<class R1, class I, class R2>
	__device__ __host__ void rr(R1& r, I& idx1, const R2& r2, const I& idx2){ r = mRR(r,r2); }
};

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
	__device__ __host__ void rv(value_type& r, index_type& idx1, const value_type&  v, const index_type& idx2) { mRV(r,idx1,v,idx2);  }
	__device__ __host__ void rr(value_type& r, index_type& idx1, const value_type& r2, const index_type& idx2) { mRR(r,idx1,r2,idx2); }
};
// construct reduce functors conveniently
template<class RV, class RR>
axis_reduce_functor<RV,RR> make_reduce_functor(const RV&t, const RR& s){ return axis_reduce_functor<RV,RR>(t,s); }
template<class RV>
axis_reduce_functor<RV,RV> make_reduce_functor(const RV&t)             { return axis_reduce_functor<RV,RV>(t); }

// construct arg-reduce functors conveniently
template<class RV, class RR>
axis_arg_reduce_functor<RV,RR> make_arg_reduce_functor(const RV&t, const RR& s){ return axis_arg_reduce_functor<RV,RR>(t,s); }
template<class RV>
axis_arg_reduce_functor<RV,RV> make_arg_reduce_functor(const RV&t)             { return axis_arg_reduce_functor<RV,RV>(t); }

template<class T>
struct bf_logaddexp : binary_functor<float,T, T> {  
	__device__  __host__    float    operator()(const T& t, const T& u) const{
		const float diff = (float)t - (float) u;
		uf_log1p<float,float> log1p;
		if(diff > 0)
			return t + log1p(expf(-diff));
		return u + log1p(expf(diff));
	} 
};

template<class V, class I>
struct reduce_argmax : fourary_functor<void,V,I,V,I> {  
	__device__  __host__    void    operator()(V& t, I& i, const V& u, const I& j) const{
	   if (u > t) {
		  t = u;
		  i = (I) j;
	   }
	} 
};

template< class V, class I>
struct reduce_argmin : fourary_functor<void,V,I,V,I> {  
	__device__  __host__    void    operator()(V& t, I& i,const  V& u, const I& j) const{
	   if (u < t) {
		  t = u;
		  i = (I) j;
	   }
	} 
};

// for reduce functors: set initial value of shared memory
template<class FUNC>
struct reduce_functor_traits{ 
	static const typename FUNC::result_type init_value(){return 0;}
	static const bool returns_index = false;
};

template<class T>
struct reduce_functor_traits<bf_max<T,T,T> >{
	static const T init_value(){return boost::numeric::bounds<T>::lowest();}
	static const bool returns_index = false;
};

template<class T>
struct reduce_functor_traits<bf_min<T,T,T> >{  
	static const T init_value(){return boost::numeric::bounds<T>::highest();}
	static const bool returns_index = false;
};


// the following binary functors are used for reductions, but they behave
// differently when operating on their own results. 
// For example ADD_SQUARED: 
//     x = a+b*b
//     y = c+d*d
//     z = x + y // NOT x + y*y
template<class S,class T, class U>
struct reduce_functor_traits<bf_add_square<S,T,U> >{  
	static const T init_value(){return 0;}
	static const bool returns_index = false;
};
template<class S,class T, class U>
struct reduce_functor_traits<bf_add_log<S,T,U> >{  
	static const T init_value(){return 0;}
	static const bool returns_index = false;
};
//template<class T>
//struct reduce_functor_traits<bf_logaddexp<T> >{  // this one is symmetric!!!
// ...
//};

template<class I, class T>
struct reduce_functor_traits<reduce_argmax<T,I> >{  
	static const T init_value(){return boost::numeric::bounds<T>::lowest();}
	static const bool returns_index=true;
};

template<class I, class T>
struct reduce_functor_traits<reduce_argmin<T,I> >{  
	static const T init_value(){return boost::numeric::bounds<T>::highest();}
	static const bool returns_index=true;
};

};//namespace cuv
