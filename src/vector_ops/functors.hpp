#include <boost/numeric/conversion/bounds.hpp>
#include <boost/static_assert.hpp>

#include <cmath>
#include <cuv_general.hpp>

#define sgn(a) (copysign(1.f,a))

namespace cuv {
	/** 
	 * @brief Adds return_type to all functors
	 */
template<class T>
struct functor{
	typedef T return_type;
};

template<class T>
struct uf_exp{  __host__ __device__         T operator()(const T& t)const{ return expf(t);    } };
template<class T>
struct uf_exact_exp{  __device__ __host__   T operator()(const T& t)const{ return expf(t);    } };
template<class T>
struct uf_log{  __device__ __host__         T operator()(const T& t)      const{ return logf(t);    } };
template<class T>
struct uf_log1p{  __device__ __host__       T operator()(const T& t)      const{
	volatile float y;
	y = 1.f + t;
	return logf(y) - ((y-1.f)-t)/y;
} };
template<class T>
struct uf_sign{  __device__ __host__        T operator()(const T& t)      const{ return sgn((float)t);    } };
template<class T>
struct uf_abs{  __device__ __host__        T operator()(const T& t)      const{ return t<0 ? -t : t;    } };
template<>
struct uf_abs<unsigned char>{  __device__ __host__        unsigned char operator()(const unsigned char& t)      const{ return t;    } };
template<>
struct uf_abs<unsigned int>{  __device__ __host__         unsigned int operator()(const unsigned int& t)      const{ return t;    } };
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
struct uf_is_nan<unsigned int>{  __device__  __host__   bool operator()(const unsigned int& t) const{ return false ; } };
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
struct uf_is_inf<unsigned int>{  __device__  __host__     bool operator()(const unsigned int t) const{ return false; } };
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
struct bf_plus : functor<T> {  __device__  __host__       T operator()(const T& t, const U& u)      const{ return  t + (T)u; } };
template<class T, class U>
struct bf_minus: functor<T> {  __device__  __host__      T operator()(const T& t, const U& u)      const{ return  t - (T)u; } };
template<class T, class U>
struct bf_multiplies: functor<T> {  __device__  __host__ T operator()(const T& t, const U& u)      const{ return  t * (T)u; } };
template<class T, class U>
struct bf_divides: functor<T> {  __device__  __host__    T operator()(const T& t, const U& u)      const{ return  t / (T)u; } };
template<class T, class U>
struct bf_squared_diff: functor<T> {__device__ __host__  T operator()(const T& t, const U& u)      const{ T ret =  t - (T)u; return ret*ret; } };
template<class T, class U>
struct bf_add_log: functor<T> {__device__ __host__  T operator()(const T& t, const U& u)      const{ return t + (T)logf(u);} };
template<class T, class U>
struct bf_add_square: functor<T> {__device__ __host__  T operator()(const T& t, const U& u)      const{ return t + (T)(u*u);} };
template<class T, class U>
struct bf_and: functor<T> {__device__ __host__   T operator()(const T& t, const U& u)      const{ return t && u; } };
template<class T, class U>
struct bf_or: functor<T> { __device__ __host__   T operator()(const T& t, const U& u)      const{ return t || u; } };
template<class T, class U>
struct bf_min: functor<T> { __device__ __host__  T operator()(const T& t, const U& u)      const{ return t<u ? t : u; } };
template<class T, class U>
struct bf_max: functor<T> { __device__ __host__  T operator()(const T& t, const U& u)      const{ return t>u ? t : u; } };

// functors with parameter
template<class T, class U>
struct bf_axpy: functor<T> {  
	const T a;
	bf_axpy(const T& _a):a(_a){}
	__device__  __host__       T operator()(const T& t, const U& u) const{ return  a*t+(T)u; } 
};
template<class T, class U>
struct bf_xpby: functor<T> {  
	const T b;
	bf_xpby(const T& _b):b(_b){}
	__device__  __host__       T operator()(const T& t, const U& u) const{ return  t+b*(T)u; } 
};
template<class T, class U>
struct bf_axpby: functor<T> {  
	const T a;
	const T b;
	bf_axpby(const T& _a, const T& _b):a(_a),b(_b){}
	__device__  __host__       T operator()(const T& t, const U& u) const{ return  a*t + b*((T)u); } 
};

template<class T>
struct bf_logaddexp : functor<float> {  
	__device__  __host__    float    operator()(const T& t, const T& u) const{
		const float diff = (float)t - (float) u;
		uf_log1p<float> log1p;
		if(diff > 0)
			return t + log1p(expf(-diff));
		return u + log1p(expf(diff));
	} 
};

struct rf_result_value_tag{};
struct rf_result_result_tag{};

template<class V, class I>
struct reduce_add : functor<void> {  
	__device__  __host__    void    operator()(V& t, I& i, const V& u, const I& j) const{
	   if (u > t) {
		  t = u;
		  i = (I) j;
	   }
	} 
};
template<class V, class I>
struct reduce_argmax : functor<void> {  
	__device__  __host__    void    operator()(V& t, I& i, const V& u, const I& j) const{
	   if (u > t) {
		  t = u;
		  i = (I) j;
	   }
	} 
};

template< class V, class I>
struct reduce_argmin : functor<void> {  
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
	static const typename FUNC::return_type init_value(){return 0;}
	static const bool returns_index = false;
	typedef FUNC result_result_functor_type;
};

template<class T>
struct reduce_functor_traits<bf_max<T,T> >{
	static const T init_value(){return boost::numeric::bounds<T>::lowest();}
	static const bool returns_index = false;
	typedef bf_max<T,T> result_result_functor_type;
};

template<class T>
struct reduce_functor_traits<bf_min<T,T> >{  
	static const T init_value(){return boost::numeric::bounds<T>::highest();}
	static const bool returns_index = false;
	typedef bf_min<T,T> result_result_functor_type;
};


// the following binary functors are used for reductions, but they behave
// differently when operating on their own results. 
// For example ADD_SQUARED: 
//     x = a+b*b
//     y = c+d*d
//     z = x + y // NOT x + y*y
template<class T>
struct reduce_functor_traits<bf_add_square<T,T> >{  
	static const T init_value(){return 0;}
	static const bool returns_index = false;
	typedef bf_plus<T,T> result_result_functor_type; // !!!
};
template<class T>
struct reduce_functor_traits<bf_add_log<T,T> >{  
	static const T init_value(){return 0;}
	static const bool returns_index = false;
	typedef bf_plus<T,T> result_result_functor_type; // !!!
};
//template<class T>
//struct reduce_functor_traits<bf_logaddexp<T> >{  // this one is symmetric!!!
// ...
//};

template<class I, class T>
struct reduce_functor_traits<reduce_argmax<T,I> >{  
	static const T init_value(){return boost::numeric::bounds<T>::lowest();}
	static const bool returns_index=true;
	typedef reduce_argmax<T,I> result_result_functor_type;
};

template<class I, class T>
struct reduce_functor_traits<reduce_argmin<T,I> >{  
	static const T init_value(){return boost::numeric::bounds<T>::highest();}
	static const bool returns_index=true;
	typedef reduce_argmin<T,I> result_result_functor_type;
};

template<bool functor_on_value_and_index, class phase>
struct rf_dispatcher{
	BOOST_STATIC_ASSERT(sizeof(phase)!=0);
};

template<>
struct rf_dispatcher<false,rf_result_value_tag>{
	template<class T, class V, class I, class J>
	static 
	__device__  __host__   void run(const T& bf, V &t, I &i, const V &u, const J &j ){
		t =  bf(t,u);
	} 
};
template<>
struct rf_dispatcher<false,rf_result_result_tag>{
	template<class T, class V, class I, class J>
	static 
	__device__  __host__   void run(const T& bf, V &t, I &i, const V &u, const J &j ){
		typename reduce_functor_traits<T>::result_result_functor_type func;
		t = func(t,u);
	} 
};

template<>
struct rf_dispatcher<true,rf_result_value_tag>{
	template<class T, class V, class I, class J>
	static
	__device__  __host__   void run(const T& qf, V &t, I &i, const V &u, const J &j ) {
		qf(t,i,u,j);
	} 
};
template<>
struct rf_dispatcher<true,rf_result_result_tag>{
	template<class T, class V, class I, class J>
	static
	__device__  __host__   void run(const T& qf, V &t, I &i, const V &u, const J &j ) {
		typename reduce_functor_traits<T>::result_result_functor_type func;
		func(t,i,u,j);
	} 
};
};//namespace cuv
