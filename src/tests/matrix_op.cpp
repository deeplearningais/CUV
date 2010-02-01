#define BOOST_TEST_MODULE example
#include <boost/test/included/unit_test.hpp>
#include <boost/test/floating_point_comparison.hpp>

#include <cuv_general.hpp>
#include <dev_dense_matrix.hpp>
#include <host_dense_matrix.hpp>
#include <convert.hpp>
#include <matrix_ops.hpp>
#include <matrix_ops/rprop.hpp>
#include <cuv_test.hpp>
#include <../random/random.hpp>

using namespace cuv;

struct Fix{
	static const int n=256;
	static const int N=n*n;
	dev_dense_matrix<float> a,b,u,v,w;
	host_dense_matrix<float> s,t,r,x,z;
	Fix()
	:   a(1,n),b(1,n),u(n,n),v(n,n),w(n,n)
	,   s(1,n),t(1,n),r(n,n),x(n,n),z(n,n)
	{
	}
	~Fix(){
	}
};


BOOST_FIXTURE_TEST_SUITE( s, Fix )


BOOST_AUTO_TEST_CASE( vec_ops_unary1 )
{
	apply_scalar_functor(v, SF_EXP);
	apply_scalar_functor(v, SF_EXACT_EXP);
	apply_scalar_functor(x, SF_EXP);
	apply_scalar_functor(x, SF_EXACT_EXP);
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

	host_dense_matrix<float> u2(u.h(), u.w());
	convert(u2,u);
	for(int i=0;i<256;i++){
		for(int j=0;j<256;j++){
			BOOST_CHECK_CLOSE( u2(i,j), r(i,j), 0.01 );
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

	host_dense_matrix<float> b2(b.h(), b.w());
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
	dev_vector<float>   v_vec(n); sequence(v_vec);
	host_vector<float>  x_vec(n); sequence(x_vec);
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
	dev_dense_matrix<float,  row_major> V(v.h(),v.w()); sequence(V);
	host_dense_matrix<float, row_major> X(x.h(),x.w()); sequence(X);
	dev_dense_matrix<float,  row_major> W(v.h(),v.w()); sequence(W);
	host_dense_matrix<float, row_major> Z(x.h(),x.w()); sequence(Z);
	dev_vector<float>   v_vec(n); sequence(v_vec);
	host_vector<float>  x_vec(n); sequence(x_vec);
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
	dev_vector<float>  v_col(n); sequence(v_col);
	host_vector<float> x_col(n); sequence(x_col);
	reduce_to_col(v_col,v,1.f,0.5f);
	reduce_to_col(x_col,x,1.f,0.5f);
	for(int i=0;i<n;i++){
		float v_correct = 0;
		for(int j=0;j<n;j++)
			v_correct += v(i,j);
		BOOST_CHECK_CLOSE(v_correct,v_col[i],0.01);
		BOOST_CHECK_CLOSE(v_col[i],x_col[i],0.01);
	}
}

BOOST_AUTO_TEST_CASE( mat_op_divide_col )
{
	sequence(v);
	sequence(x);
	sequence(z);
	dev_vector<float>  v_col(n); sequence(v_col); apply_scalar_functor(v_col, SF_ADD, 1.0f);
	host_vector<float> x_col(n); sequence(x_col); apply_scalar_functor(x_col, SF_ADD, 1.0f);

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
	dev_dense_matrix<float,row_major> dA(40, 30);
	dev_vector<float> dV(40);
	host_dense_matrix<float,row_major> hA(40, 30);
	host_vector<float> hV(40);

	sequence(dA);
	sequence(dV);
	sequence(hA);
	sequence(hV);

	reduce_to_col(dV,dA,1.f,0.5f);
	reduce_to_col(hV,hA,1.f,0.5f);

	host_vector<float> hV2(dV.size());
	convert(hV2, dV);

	for(int i=0;i<30;i++)
		BOOST_CHECK_CLOSE(hV2[i],hV[i],0.01);
}

BOOST_AUTO_TEST_CASE( mat_op_reduce_to_row )
{
	dev_dense_matrix<float> dA(40, 30);
	dev_vector<float> dV(30);
	host_dense_matrix<float> hA(40, 30);
	host_vector<float> hV(30);

	sequence(dA);
	sequence(dV);
	sequence(hA);
	sequence(hV);

	reduce_to_row(dV,dA,1.f,0.5f);
	reduce_to_row(hV,hA,1.f,0.5f);

	host_vector<float> hV2(dV.size());
	convert(hV2, dV);

	for(int i=0;i<30;i++)
		BOOST_CHECK_CLOSE(hV2[i],hV[i],0.01);
}

BOOST_AUTO_TEST_CASE( mat_op_reduce_rm_to_row )
{
	dev_dense_matrix<float,row_major> dA(40, 30);
	dev_vector<float> dV(30);
	host_dense_matrix<float,row_major> hA(40, 30);
	host_vector<float> hV(30);

	sequence(dA);
	sequence(dV);
	sequence(hA);
	sequence(hV);

	reduce_to_row(dV,dA,1.f,0.5f);
	reduce_to_row(hV,hA,1.f,0.5f);

	host_vector<float> hV2(dV.size());
	convert(hV2, dV);

	for(int i=0;i<30;i++)
		BOOST_CHECK_CLOSE(hV2[i],hV[i],0.01);
}


BOOST_AUTO_TEST_CASE( mat_op_view )
{
	host_dense_matrix<float>* h2 = blockview(x,0,n,1,2);
	dev_dense_matrix<float>*  d2 = blockview(v,0,n,1,2);
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

	host_dense_matrix<float> hA(n, m), hB(m, n);
	dev_dense_matrix<float>  dA(n, m), dB(m, n);
	host_dense_matrix<float, row_major> hC(n, m), hD(m, n);
	dev_dense_matrix<float, row_major>  dC(n, m), dD(m, n);

	sequence(hB); sequence(dB);
	sequence(hD); sequence(dD);

	transpose(hA, hB);
	transpose(dA, dB);
	transpose(hC, hD);
	transpose(dC, dD);

	host_dense_matrix<float> h2A(dA.w(), dA.h()); convert(h2A, dA);
	host_dense_matrix<float, row_major> h2C(dC.w(), dC.h()); convert(h2C, dC);

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

	host_dense_matrix<float> hA(n, m);
	dev_dense_matrix<float>  dA(n, m);
	host_vector<int> v(m);
	dev_vector<int> x(m);

	host_dense_matrix<float,row_major> hB(m, n);
	dev_dense_matrix<float,row_major>  dB(m, n);
	host_vector<int> w(m);
	dev_vector<int> y(m);

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
