#ifndef __CUV_TEST_HPP__
#define __CUV_TEST_HPP__
#include <boost/test/included/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#define MAT_CMP(X,Y) \
	if(1){                                                                                        \
		bool (*matrix_eq_cmp)(const __typeof__(X)&, const __typeof__(Y)&) = make_matrix_eq(X,Y);  \
		BOOST_CHECK_PREDICATE(matrix_eq_cmp,(X)(Y));                                              \
	}

	template<class M, class N>
	bool matrix_eq(const M& w, const N& w2)
	{ 
		boost::test_tools::percent_tolerance_t <double> pt(0.1);
		boost::test_tools::close_at_tolerance<typename M::value_type> cmp(pt);
		for(int i=0;i<w.h();i++){
			for(int j=0;j<w.w();j++){
				if( !cmp(w(i,j), w2(i,j)))
					return false;
			}
		}
		return true;
	}

	template<class M, class N>
	bool (*make_matrix_eq(const M&,const N&))(const M& w, const N& w2){
		return &matrix_eq<M,N>;
	}

#endif

