//*LB*
// Copyright (c) 2010, Hannes Schulz, Andreas Mueller, Dominik Scherer
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 
//  * Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//  * Neither the name of the University of Bonn 
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this software without specific prior written
//    permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//*LE*




#ifndef __CUV_TEST_HPP__
#define __CUV_TEST_HPP__
#include <boost/test/included/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#define MAT_CMP(X,Y,PREC) \
	if(1){                                                                                        \
		bool (*matrix_eq_cmp)(const __typeof__(X)&, const __typeof__(Y)&, float) = make_matrix_eq(X,Y);  \
		BOOST_CHECK_PREDICATE(matrix_eq_cmp,(X)(Y)(PREC));                                              \
	}

	template<class M, class N>
	bool matrix_eq(const M& w, const N& w2, float prec)
	{ 
		boost::test_tools::percent_tolerance_t <float> pt(prec);
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
	bool (*make_matrix_eq(const M&,const N&))(const M& w, const N& w2, float){
		return &matrix_eq<M,N>;
	}

#endif

