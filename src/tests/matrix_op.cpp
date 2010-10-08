//*LB*
// Copyright (c) 2010, University of Bonn, Institute for Computer Science VI
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





#define BOOST_TEST_MODULE example
#include <boost/test/included/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <cuv_general.hpp>
#include <dense_matrix.hpp>
#include <convert.hpp>
#include <matrix_ops.hpp>
#include <matrix_ops/rprop.hpp>
#include <cuv_test.hpp>
#include <../random/random.hpp>

using namespace cuv;

struct MyConfig {
	static const int dev = CUDA_TEST_DEVICE;
	MyConfig()   { 
		printf("Testing on device=%d\n",dev);
		initCUDA(dev); 
	}
	~MyConfig()  { exitCUDA();  }
};

BOOST_GLOBAL_FIXTURE( MyConfig );

struct Fix{
	static const int n=256;
	static const int N=n*n;
	static const int big_images = 384*384*32;
	dense_matrix<float,column_major,dev_memory_space> a,b,u,v,w,d_reduce_big;
	dense_matrix<float,column_major,host_memory_space> s,t,r,x,z,h_reduce_big;
	Fix()
	:   a(1,n),b(1,n),u(n,n),v(n,n),w(n,n), d_reduce_big(32,big_images)
	,   s(1,n),t(1,n),r(n,n),x(n,n),z(n,n), h_reduce_big(32,big_images)
	{
	}
	~Fix(){
	}
};


BOOST_FIXTURE_TEST_SUITE( s, Fix )


BOOST_AUTO_TEST_CASE( vec_ops_unary1 )
{
	apply_scalar_functor(v, SF_EXP);
	//apply_scalar_functor(v, SF_EXACT_EXP);
	apply_scalar_functor(x, SF_EXP);
	//apply_scalar_functor(x, SF_EXACT_EXP);
}

BOOST_AUTO_TEST_CASE( vec_ops_binary1 )
{
	sequence(v.vec());
	sequence(w.vec());
	apply_scalar_functor(v,SF_ADD,1);
	for(int i=0;i<N;i++){
		BOOST_CHECK_EQUAL(v.vec()[i], i + 1);
	}
	apply_binary_functor(v,w, BF_ADD);
	for(int i=0;i<N;i++){
		BOOST_CHECK_EQUAL(v.vec()[i], i + i + 1);
	}
}

BOOST_AUTO_TEST_CASE( vec_ops_binary2 )
{
	apply_binary_functor(v,w, BF_AXPY, 2);
}

BOOST_AUTO_TEST_CASE( vec_ops_copy )
{
	// generate data
	sequence(w);

	// copy data from v to w
	copy(v,w);
	for(int i=0;i<N;i++){
		BOOST_CHECK_EQUAL(v.vec()[i],w.vec()[i]);
	}
}

BOOST_AUTO_TEST_CASE( vec_ops_unary_add )
{
	sequence(v);
	apply_scalar_functor(v,SF_ADD,3.8f);
	for(int i=0;i<N;i++){
		BOOST_CHECK_EQUAL(v.vec()[i], i+3.8f);
	}
}
BOOST_AUTO_TEST_CASE( vec_ops_axpby )
{
	// generate data
	sequence(w);
	sequence(v);
	apply_scalar_functor(v,SF_ADD, 1.f);
	BOOST_CHECK_EQUAL(w.vec()[0],0);
	BOOST_CHECK_EQUAL(v.vec()[0],1);

	// copy data from v to w
	apply_binary_functor(v,w,BF_AXPBY, 2.f,3.f);
	for(int i=0;i<N;i++){
		BOOST_CHECK_EQUAL(v.vec()[i], 2*(i+1) + 3*i );
	}
}


BOOST_AUTO_TEST_CASE( vec_ops_0ary1 )
{
	// test sequence
	sequence(v.vec());
	for(int i=0;i<N;i++){
		BOOST_CHECK_EQUAL(v.vec()[i], i);
	}

	// test fill
	fill(w.vec(),1);
	for(int i=0;i<N;i++){
		BOOST_CHECK_EQUAL(w.vec()[i], 1);
	}
}

BOOST_AUTO_TEST_CASE( vec_ops_norms )
{
	sequence(v.vec());
	float f1 = norm1(v), f1_ = 0;
	float f2 = norm2(v), f2_ = 0;
	for(int i=0;i<N;i++){
		f2_ += v.vec()[i]*v.vec()[i];
		f1_ += fabs(v.vec()[i]);
	}
	f2_ = sqrt(f2_);
	BOOST_CHECK_CLOSE(f1,f1_,0.01f);
	BOOST_CHECK_CLOSE(f2,f2_,0.01f);
}


BOOST_AUTO_TEST_CASE( mat_op_mm )
{
	sequence(v);     apply_scalar_functor(v, SF_MULT, 0.01f);
	sequence(w);     apply_scalar_functor(w, SF_MULT, 0.01f);
	sequence(x);     apply_scalar_functor(x, SF_MULT, 0.01f);
	sequence(z);     apply_scalar_functor(z, SF_MULT, 0.01f);
	prod(u,v,w,'n','t');
	prod(r,x,z,'n','t');

	dense_matrix<float,column_major,host_memory_space> u2(u.h(), u.w());
	convert(u2,u);
	for(int i=0;i<256;i++){
		for(int j=0;j<256;j++){
			BOOST_CHECK_CLOSE( u2(i,j), r(i,j), 0.01 );
		}
	}
}

BOOST_AUTO_TEST_CASE( mat_op_rm_prod )
{
	int m = 234;
	int n = 314;
	int k = 413;

	dense_matrix<float,row_major,host_memory_space> hA(m, k);
	dense_matrix<float,row_major,host_memory_space> hB(k, n);
	dense_matrix<float,row_major,host_memory_space> hC(m, n);

	dense_matrix<float,row_major,dev_memory_space> dA(m, k);
	dense_matrix<float,row_major,dev_memory_space> dB(k, n);
	dense_matrix<float,row_major,dev_memory_space> dC(m, n);

	sequence(hA);     apply_scalar_functor(hA, SF_MULT, 0.01f);
	sequence(hB);     apply_scalar_functor(hB, SF_MULT, 0.01f);
	sequence(hC);     apply_scalar_functor(hC, SF_MULT, 0.01f);

	sequence(dA);     apply_scalar_functor(dA, SF_MULT, 0.01f);
	sequence(dB);     apply_scalar_functor(dB, SF_MULT, 0.01f);
	sequence(dC);     apply_scalar_functor(dC, SF_MULT, 0.01f);

	prod(hC,hA,hB,'n','n');
	prod(dC,dA,dB,'n','n');

	dense_matrix<float,row_major,host_memory_space> c2(dC.h(), dC.w());
	convert(c2,dC);

	for(int i=0;i<m;i++){
		for(int j=0;j<n;j++){
			BOOST_CHECK_CLOSE( hC(i,j), c2(i,j), 0.01 );
		}
	}
}

BOOST_AUTO_TEST_CASE( mat_op_mmdim1 )
{
	sequence(a);     apply_scalar_functor(a, SF_MULT, 0.01f);
	sequence(w);     apply_scalar_functor(w, SF_MULT, 0.01f);
	sequence(s);     apply_scalar_functor(s, SF_MULT, 0.01f);
	sequence(z);     apply_scalar_functor(z, SF_MULT, 0.01f);
	prod(b,a,w,'n','t');
	prod(t,s,z,'n','t');

	dense_matrix<float,column_major,host_memory_space> b2(b.h(), b.w());
	convert(b2,b);

	for(int i=0;i<256;i++) {
		float val = 0.0f;
		for(int j=0;j<256;j++) {
			val += s(0,j) * z(i,j);
		}
		BOOST_CHECK_CLOSE( b2(0,i), val, 0.01 );
		BOOST_CHECK_CLOSE( t(0,i), val, 0.01 );
	}
}

BOOST_AUTO_TEST_CASE( mat_op_mat_plus_vec )
{
	sequence(v); sequence(w);
	sequence(x); sequence(z);
	vector<float,dev_memory_space>   v_vec(n); sequence(v_vec);
	vector<float,host_memory_space>  x_vec(n); sequence(x_vec);
	matrix_plus_col(v,v_vec);
	matrix_plus_col(x,x_vec);
	for(int i=0;i<n;i++){
		for(int j=0;j<n;j++){
			BOOST_CHECK_CLOSE(v(i,j), x(i,j), 0.01);
			BOOST_CHECK_CLOSE(v(i,j), w(i,j)+v_vec[i], 0.01);
			BOOST_CHECK_CLOSE(x(i,j), z(i,j)+x_vec[i], 0.01);
		}
	}

}

BOOST_AUTO_TEST_CASE( mat_op_mat_plus_vec_row_major )
{
	dense_matrix<float,row_major,dev_memory_space> V(v.h(),v.w()); sequence(V);
	dense_matrix<float,row_major,host_memory_space> X(x.h(),x.w()); sequence(X);
	dense_matrix<float,row_major,dev_memory_space> W(v.h(),v.w()); sequence(W);
	dense_matrix<float,row_major,host_memory_space> Z(x.h(),x.w()); sequence(Z);
	vector<float,dev_memory_space>   v_vec(n); sequence(v_vec);
	vector<float,host_memory_space>  x_vec(n); sequence(x_vec);
	matrix_plus_col(V,v_vec);
	matrix_plus_col(X,x_vec);
	for(int i=0;i<n;i++){
		for(int j=0;j<n;j++){
			BOOST_CHECK_CLOSE(V(i,j), X(i,j), 0.01);
			BOOST_CHECK_CLOSE(V(i,j), W(i,j)+v_vec[i], 0.01);
			BOOST_CHECK_CLOSE(X(i,j), Z(i,j)+x_vec[i], 0.01);
		}
	}

}

BOOST_AUTO_TEST_CASE( mat_op_reduce_to_col )
{
	sequence(v);
	sequence(x);
	vector<float,dev_memory_space>  v_col(n); sequence(v_col);
	vector<float,host_memory_space> x_col(n); sequence(x_col);
	reduce_to_col(v_col,v,RF_ADD,1.f,0.5f);
	reduce_to_col(x_col,x,RF_ADD,1.f,0.5f);
	for(int i=0;i<n;i++){
		float v_correct = 0;
		for(int j=0;j<n;j++)
			v_correct += v(i,j);
		BOOST_CHECK_CLOSE(v_correct,v_col[i],0.01);
		BOOST_CHECK_CLOSE(v_col[i],x_col[i],0.01);
	}
}

/*
BOOST_AUTO_TEST_CASE( mat_op_big_reduce_to_col )
{
	sequence(d_reduce_big);
	sequence(h_reduce_big);
	vector<float,dev_memory_space>  v_col(32); sequence(v_col);
	vector<float,host_memory_space> x_col(32); sequence(x_col);
	reduce_to_col(v_col,d_reduce_big,RF_ADD,1.f,0.5f);
	reduce_to_col(x_col,h_reduce_big,RF_ADD,1.f,0.5f);
	for(int i=0;i<32;i++){
		float v_correct = 0;
		for(int j=0;j<big_images;j++)
			v_correct += d_reduce_big(i,j);
		BOOST_CHECK_CLOSE(v_correct,v_col[i],0.01);
		BOOST_CHECK_CLOSE(v_col[i],x_col[i],0.01);
	}
}
*/

BOOST_AUTO_TEST_CASE( mat_op_reduce_to_col_min )
{
	sequence(v);
	sequence(x);
	vector<float,dev_memory_space>  v_col(n); sequence(v_col);
	vector<float,host_memory_space> x_col(n); sequence(x_col);
	reduce_to_col(v_col,v,RF_MAX);
	reduce_to_col(x_col,x,RF_MAX);
	for(int i=0;i<n;i++){
		float v_correct = -INT_MAX;
		for(int j=0;j<n;j++)
			v_correct = std::max(v_correct,v(i,j));
		BOOST_CHECK_CLOSE(v_correct,v_col[i],0.01);
		BOOST_CHECK_CLOSE(v_col[i],x_col[i],0.01);
	}
}

BOOST_AUTO_TEST_CASE( mat_op_divide_col )
{
	sequence(v);
	sequence(x);
	sequence(z);
	vector<float,dev_memory_space>  v_col(n); sequence(v_col); apply_scalar_functor(v_col, SF_ADD, 1.0f);
	vector<float,host_memory_space> x_col(n); sequence(x_col); apply_scalar_functor(x_col, SF_ADD, 1.0f);

	matrix_divide_col(v, v_col);
	matrix_divide_col(x, x_col);

	for(int i=0;i<n;i++)
		for(int j=0; j<n; j++) {
			BOOST_CHECK_CLOSE(v(i,j),x(i,j),0.01);
			BOOST_CHECK_CLOSE(x(i,j),z(i,j)/x_col[i],0.01);
		}
}

BOOST_AUTO_TEST_CASE( mat_op_reduce_rm_to_col )
{
	const int m = 400;
	const int n = 300;

	float factOld = 1.33f;
	float factNew = 0.2f;

	dense_matrix<float,row_major,dev_memory_space> dA(m, n);
	vector<float,dev_memory_space> dV(m);
	dense_matrix<float,row_major,host_memory_space> hA(m, n);
	vector<float,host_memory_space> hV(m);

	sequence(dA); apply_scalar_functor(dA, SF_MULT, 0.01f);
	sequence(dV);
	sequence(hA); apply_scalar_functor(hA, SF_MULT, 0.01f);
	sequence(hV);

	reduce_to_col(dV,dA,RF_ADD,factNew,factOld);
	reduce_to_col(hV,hA,RF_ADD,factNew,factOld);

	vector<float,host_memory_space> hV2(dV.size());
	convert(hV2, dV);

	for(int i=0; i<m; i++)
		BOOST_CHECK_CLOSE(hV2[i],hV[i],0.1);
}

BOOST_AUTO_TEST_CASE( mat_op_reduce_to_row )
{
	dense_matrix<float,column_major,dev_memory_space> dA(40, 30);
	vector<float,dev_memory_space> dV(30);
	dense_matrix<float,column_major,host_memory_space> hA(40, 30);
	vector<float,host_memory_space> hV(30);

	sequence(dA);
	sequence(dV);
	sequence(hA);
	sequence(hV);

	reduce_to_row(dV,dA,RF_ADD,1.f,0.5f);
	reduce_to_row(hV,hA,RF_ADD,1.f,0.5f);

	vector<float,host_memory_space> hV2(dV.size());
	convert(hV2, dV);

	for(int i=0;i<30;i++)
		BOOST_CHECK_CLOSE(hV2[i],hV[i],0.1);
}

BOOST_AUTO_TEST_CASE( mat_op_reduce_rm_to_row )
{
	dense_matrix<float,row_major,dev_memory_space> dA(40, 30);
	vector<float,dev_memory_space> dV(30);
	dense_matrix<float,row_major,host_memory_space> hA(40, 30);
	vector<float,host_memory_space> hV(30);

	sequence(dA);
	sequence(dV);
	sequence(hA);
	sequence(hV);

	reduce_to_row(dV,dA,RF_ADD,1.f,0.5f);
	reduce_to_row(hV,hA,RF_ADD,1.f,0.5f);

	vector<float,host_memory_space> hV2(dV.size());
	convert(hV2, dV);

	for(int i=0;i<30;i++)
		BOOST_CHECK_CLOSE(hV2[i],hV[i],0.01);
}

/*
BOOST_AUTO_TEST_CASE( mat_op_reduce_big_rm_to_row )
{
	dense_matrix<float,row_major,dev_memory_space> dA(32, 1179648);
	vector<float,dev_memory_space> dV(1179648);
	dense_matrix<float,row_major,host_memory_space> hA(32, 1179648);
	vector<float,host_memory_space> hV(1179648);

	sequence(dA);
	sequence(dV);
	sequence(hA);
	sequence(hV);

	reduce_to_row(dV,dA,RF_ADD, 1.0f, 1.0f);
	reduce_to_row(hV,hA,RF_ADD, 1.0f, 1.0f);

	vector<float,host_memory_space> hV2(dV.size());
	convert(hV2, dV);

	for(int i=0;i<1179648;i++){
		BOOST_CHECK_CLOSE(hV[i],hV2[i],0.01);
	}

}
*/

BOOST_AUTO_TEST_CASE( mat_op_view )
{
	dense_matrix<float,column_major,host_memory_space>* h2 = blockview(x,(unsigned int)0,(unsigned int)n,(unsigned int)1,(unsigned int)2);
	dense_matrix<float,column_major,dev_memory_space>*  d2 = blockview(v,(unsigned int)0,(unsigned int)n,(unsigned int)1,(unsigned int)2);
	sequence(x);
	sequence(v);
	BOOST_CHECK_EQUAL(h2->h(), x.h());
	BOOST_CHECK_EQUAL(d2->h(), x.h());
	BOOST_CHECK_EQUAL(h2->w(), 2);
	BOOST_CHECK_EQUAL(d2->w(), 2);
	for(int i=0;i<n;i++)
		for(int j=0;j<2;j++){
			BOOST_CHECK_CLOSE((*h2)(i,j),(*d2)(i,j),0.01);
			BOOST_CHECK_CLOSE((*h2)(i,j),x(i, j+1),0.01);
		}
}


BOOST_AUTO_TEST_CASE( mat_op_transpose )
{
	const int n = 8;
	const int m = 3;

	dense_matrix<float,column_major,host_memory_space> hA(n, m), hB(m, n);
	dense_matrix<float,column_major,dev_memory_space>  dA(n, m), dB(m, n);
	dense_matrix<float,row_major,host_memory_space> hC(n, m), hD(m, n);
	dense_matrix<float,row_major,dev_memory_space>  dC(n, m), dD(m, n);

	sequence(hB); sequence(dB);
	sequence(hD); sequence(dD);

	transpose(hA, hB);
	transpose(dA, dB);
	transpose(hC, hD);
	transpose(dC, dD);

	dense_matrix<float,column_major,host_memory_space> h2A(dA.w(), dA.h()); convert(h2A, dA);
	dense_matrix<float,row_major,host_memory_space> h2C(dC.w(), dC.h()); convert(h2C, dC);

	for(int i=0;i<n;i++)
		for(int j=0;j<m;j++){
			BOOST_CHECK_EQUAL(hA(i,j), hB(j,i));
			BOOST_CHECK_EQUAL(hC(i,j), hD(j,i));
		}

	MAT_CMP(hA, h2A, 0.1);
	MAT_CMP(hC, h2C, 0.1);
}

BOOST_AUTO_TEST_CASE( mat_op_argmax )
{
	const int n = 517;
	const int m = 212;

	dense_matrix<float,column_major,host_memory_space> hA(n, m);
	dense_matrix<float,column_major,dev_memory_space>  dA(n, m);
	vector<int,host_memory_space> v(m);
	vector<int,dev_memory_space> x(m);

	dense_matrix<float,row_major,host_memory_space> hB(m, n);
	dense_matrix<float,row_major,dev_memory_space>  dB(m, n);
	vector<int,host_memory_space> w(m);
	vector<int,dev_memory_space> y(m);

	fill_rnd_uniform(hA.vec());
	fill_rnd_uniform(hB.vec());
	convert(dA, hA);
	convert(dB, hB);

	argmax_to_row(v, hA);
	argmax_to_row(x, dA);

	argmax_to_column(w, hB);
	argmax_to_column(y, dB);

	for(int i=0; i<m; i++) {
		BOOST_CHECK_EQUAL(v[i], x[i]);
		BOOST_CHECK_EQUAL(w[i], y[i]);
	}
}

BOOST_AUTO_TEST_SUITE_END()
