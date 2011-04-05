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
#include <boost/assign.hpp>
#include <list>
using namespace boost::assign;

#include <cuv/tools/cuv_general.hpp>
#include <cuv/basics/dense_matrix.hpp>
#include <cuv/convert/convert.hpp>
#include <cuv/matrix_ops/matrix_ops.hpp>
#include <cuv/matrix_ops/rprop.hpp>
#include <cuv/tools/cuv_test.hpp>
#include <cuv/random/random.hpp>

using namespace cuv;

struct MyConfig {
	static const int dev = CUDA_TEST_DEVICE;
	MyConfig()   { 
		printf("Testing on device=%d\n",dev);
		initCUDA(dev); 
		initialize_mersenne_twister_seeds();
	}
	~MyConfig()  { exitCUDA();  }
};

BOOST_GLOBAL_FIXTURE( MyConfig );

struct Fix{
	static const int n=128;
	static const int N=n*n;
	static const int big_images = 384*384*2;
	dense_matrix<float,dev_memory_space,column_major> a,b,u,v,w,d_reduce_big;
	dense_matrix<float,host_memory_space,column_major> s,t,r,x,z,h_reduce_big;
	Fix()
	:   a(1,n),b(1,n),u(n,n),v(n,n),w(n,n), d_reduce_big(32,big_images)
	,   s(1,n),t(1,n),r(n,n),x(n,n),z(n,n), h_reduce_big(32,big_images)
	{
	}
	~Fix(){
	}
};

template<class VT2, class VT, class ML, class I>
std::pair<tensor<VT2,host_memory_space,I>*,    // host result
	 tensor<VT2,host_memory_space,I>*>   // dev  result
test_reduce(
	int dim,
	dense_matrix<VT,ML,dev_memory_space,I>&   d_mat,
	cuv::reduce_functor rf
){
	dense_matrix<VT,ML,host_memory_space> h_mat(d_mat.h(), d_mat.w());
	convert(h_mat,d_mat);

	unsigned int len = d_mat.h();
	if(dim==0) len = d_mat.w();
	tensor<VT2,host_memory_space>* v_host1= new tensor<VT2,host_memory_space> (len);
	tensor<VT2,host_memory_space>* v_host2= new tensor<VT2,host_memory_space> (len);
	tensor<VT2,dev_memory_space>   v_dev(len);
	if(dim==0){
		reduce_to_row(*v_host1, h_mat,rf);
		reduce_to_row( v_dev,   d_mat,rf);
	}else if(dim==1){
		reduce_to_col(*v_host1, h_mat,rf);
		reduce_to_col( v_dev,   d_mat,rf);
	}
	convert(*v_host2,v_dev);
	return std::make_pair(v_host1, v_host2);
}


BOOST_FIXTURE_TEST_SUITE( s, Fix )


BOOST_AUTO_TEST_CASE( vec_ops_unary1 )
{
	apply_scalar_functor(v, SF_EXP);
	//apply_scalar_functor(v, SF_EXACT_EXP);
	apply_scalar_functor(x, SF_EXP);
	//apply_scalar_functor(x, SF_EXACT_EXP);
}

BOOST_AUTO_TEST_CASE( binary_operators )
{
  dense_matrix<float,dev_memory_space,column_major> j(32,32);
  dense_matrix<float,dev_memory_space,column_major> k(32,32);
  j = k = 1.f;
  const dense_matrix<float,dev_memory_space,column_major>& j_ = j;
  const dense_matrix<float,dev_memory_space,column_major>& k_ = k;
  const dense_matrix<float,dev_memory_space,column_major> l = j_+k_;
  for(int i=0;i<32*32;i++){
	  BOOST_CHECK_EQUAL(l[i], 2.f);
  }
}

BOOST_AUTO_TEST_CASE( vec_ops_binary1 )
{
	sequence(v);
	w=v;
	sequence(a);
	//apply_scalar_functor(v,SF_ADD,1);
	v+= (float)1.0;
	for(int i=0;i<N;i++){
		BOOST_CHECK_EQUAL(v[i], i + 1);
	}
	//apply_binary_functor(v,w, BF_ADD);
	a=v;
	a=v+w;
	v=a;
	for(int i=0;i<N;i++){
		BOOST_CHECK_EQUAL(v[i], i + i + 1);
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
	// copy(v,w);
	v=w;
	for(int i=0;i<N;i++){
		BOOST_CHECK_EQUAL(v[i],w[i]);
	}
}

BOOST_AUTO_TEST_CASE( vec_ops_unary_add )
{
	sequence(v);
	apply_scalar_functor(v,SF_ADD,3.8f);
	for(int i=0;i<N;i++){
		BOOST_CHECK_EQUAL(v[i], i+3.8f);
	}
}
BOOST_AUTO_TEST_CASE( vec_ops_axpby )
{
	// generate data
	sequence(w);
	sequence(v);
	apply_scalar_functor(v,SF_ADD, 1.f);
	BOOST_CHECK_EQUAL(w[0],0);
	BOOST_CHECK_EQUAL(v[0],1);

	// copy data from v to w
	apply_binary_functor(v,w,BF_AXPBY, 2.f,3.f);
	for(int i=0;i<N;i++){
		BOOST_CHECK_EQUAL(v[i], 2*(i+1) + 3*i );
	}
}


BOOST_AUTO_TEST_CASE( vec_ops_0ary1 )
{
	// test sequence
	sequence(v);
	for(int i=0;i<N;i++){
		BOOST_CHECK_EQUAL(v[i], i);
	}

	// test fill
	w = 1.f;
	for(int i=0;i<N;i++){
		BOOST_CHECK_EQUAL(w[i], 1);
	}
}

BOOST_AUTO_TEST_CASE( vec_ops_norms )
{
	sequence(v);
	float f1 = norm1(v), f1_ = 0;
	float f2 = norm2(v), f2_ = 0;
	for(int i=0;i<N;i++){
		f2_ += v[i]*v[i];
		f1_ += fabs(v[i]);
	}
	f2_ = sqrt(f2_);
	BOOST_CHECK_CLOSE((float)f1,(float)f1_,0.01f);
	BOOST_CHECK_CLOSE((float)f2,(float)f2_,0.01f);
}


BOOST_AUTO_TEST_CASE( mat_op_mm )
{
	sequence(v);     apply_scalar_functor(v, SF_MULT, 0.01f);
	sequence(w);     apply_scalar_functor(w, SF_MULT, 0.01f);
	sequence(x);     apply_scalar_functor(x, SF_MULT, 0.01f);
	sequence(z);     apply_scalar_functor(z, SF_MULT, 0.01f);
	prod(u,v,w,'n','t');
	prod(r,x,z,'n','t');

	dense_matrix<float,host_memory_space,column_major> u2(u.h(), u.w());
	convert(u2,u);
	for(int i=0;i<u2.h();i++){
		for(int j=0;j<u2.h();j++){
			BOOST_CHECK_CLOSE( (float)u2(i,j), (float)r(i,j), 0.01 );
		}
	}
}

BOOST_AUTO_TEST_CASE( mat_op_rm_prod )
{
	int m = 234;
	int n = 314;
	int k = 413;

	dense_matrix<float,host_memory_space,row_major> hA(m, k);
	dense_matrix<float,host_memory_space,row_major> hB(k, n);
	dense_matrix<float,host_memory_space,row_major> hC(m, n);

	dense_matrix<float,dev_memory_space,row_major> dA(m, k);
	dense_matrix<float,dev_memory_space,row_major> dB(k, n);
	dense_matrix<float,dev_memory_space,row_major> dC(m, n);

	sequence(hA);     apply_scalar_functor(hA, SF_MULT, 0.01f);
	sequence(hB);     apply_scalar_functor(hB, SF_MULT, 0.01f);
	sequence(hC);     apply_scalar_functor(hC, SF_MULT, 0.01f);

	sequence(dA);     apply_scalar_functor(dA, SF_MULT, 0.01f);
	sequence(dB);     apply_scalar_functor(dB, SF_MULT, 0.01f);
	sequence(dC);     apply_scalar_functor(dC, SF_MULT, 0.01f);

	prod(hC,hA,hB,'n','n');
	prod(dC,dA,dB,'n','n');

	dense_matrix<float,host_memory_space,row_major> c2(dC.h(), dC.w());
	convert(c2,dC);

	for(int i=0;i<m;i++){
		for(int j=0;j<n;j++){
			BOOST_CHECK_CLOSE( (float)hC(i,j), (float)c2(i,j), 0.01 );
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

	dense_matrix<float,host_memory_space,column_major> b2(b.h(), b.w());
	convert(b2,b);

	for(int i=0;i<z.h();i++) {
		float val = 0.0f;
		for(int j=0;j<z.w();j++) {
			val += s(0,j) * z(i,j);
		}
		BOOST_CHECK_CLOSE( (float)b2(0,i), (float)val, 0.01 );
		BOOST_CHECK_CLOSE( (float)t(0,i), (float)val, 0.01 );
	}
}

BOOST_AUTO_TEST_CASE( mat_op_mat_plus_row )
{
	sequence(v); sequence(w);
	sequence(x); sequence(z);
	tensor<float,dev_memory_space>   v_vec(n); sequence(v_vec);
	tensor<float,host_memory_space>  x_vec(n); sequence(x_vec);
	matrix_plus_row(v,v_vec);
	matrix_plus_row(x,x_vec);
	for(int i=0;i<n;i++){
		for(int j=0;j<n;j++){
			BOOST_CHECK_CLOSE((float)v(i,j), (float)x(i,j), 0.01);
			BOOST_CHECK_CLOSE((float)v(i,j), w(i,j)+v_vec[j], 0.01);
			BOOST_CHECK_CLOSE((float)x(i,j), z(i,j)+x_vec[j], 0.01);
		}
	}

}
BOOST_AUTO_TEST_CASE( mat_op_mat_plus_col )
{
	sequence(v); sequence(w);
	sequence(x); sequence(z);
	tensor<float,dev_memory_space>   v_vec(n); sequence(v_vec);
	tensor<float,host_memory_space>  x_vec(n); sequence(x_vec);
	matrix_plus_col(v,v_vec);
	matrix_plus_col(x,x_vec);
	for(int i=0;i<n;i++){
		for(int j=0;j<n;j++){
			BOOST_CHECK_CLOSE((float)v(i,j), (float)x(i,j), 0.01);
			BOOST_CHECK_CLOSE((float)v(i,j), w(i,j)+v_vec[i], 0.01);
			BOOST_CHECK_CLOSE((float)x(i,j), z(i,j)+x_vec[i], 0.01);
		}
	}

}

BOOST_AUTO_TEST_CASE( mat_op_mat_plus_vec_row_major )
{
	dense_matrix<float,dev_memory_space,row_major> V(v.h(),v.w()); sequence(V);
	dense_matrix<float,host_memory_space,row_major> X(x.h(),x.w()); sequence(X);
	dense_matrix<float,dev_memory_space,row_major> W(v.h(),v.w()); sequence(W);
	dense_matrix<float,host_memory_space,row_major> Z(x.h(),x.w()); sequence(Z);
	tensor<float,dev_memory_space>   v_vec(n); sequence(v_vec);
	tensor<float,host_memory_space>  x_vec(n); sequence(x_vec);
	matrix_plus_col(V,v_vec);
	matrix_plus_col(X,x_vec);
	for(int i=0;i<n;i++){
		for(int j=0;j<n;j++){
			BOOST_CHECK_CLOSE((float)V(i,j), (float)X(i,j), 0.01);
			BOOST_CHECK_CLOSE((float)V(i,j), W(i,j)+v_vec[i], 0.01);
			BOOST_CHECK_CLOSE((float)X(i,j), Z(i,j)+x_vec[i], 0.01);
		}
	}

}

/*
BOOST_AUTO_TEST_CASE( mat_op_big_reduce_to_col )
{
	sequence(d_reduce_big);
	sequence(h_reduce_big);
	tensor<float,dev_memory_space>  v_col(32); sequence(v_col);
	tensor<float,host_memory_space> x_col(32); sequence(x_col);
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


BOOST_AUTO_TEST_CASE( mat_op_divide_col )
{
	sequence(v);
	sequence(x);
	sequence(z);
	tensor<float,dev_memory_space>  v_col(n); sequence(v_col); apply_scalar_functor(v_col, SF_ADD, 1.0f);
	tensor<float,host_memory_space> x_col(n); sequence(x_col); apply_scalar_functor(x_col, SF_ADD, 1.0f);

	matrix_divide_col(v, v_col);
	matrix_divide_col(x, x_col);

	for(int i=0;i<n;i++)
		for(int j=0; j<n; j++) {
			BOOST_CHECK_CLOSE((float)v(i,j),(float)x(i,j),0.01);
			BOOST_CHECK_CLOSE((float)x(i,j),z(i,j)/x_col[i],0.01);
		}
}


/*
BOOST_AUTO_TEST_CASE( mat_op_reduce_big_rm_to_row )
{
	dense_matrix<float,dev_memory_space,row_major> dA(32, 1179648);
	tensor<float,dev_memory_space> dV(1179648);
	dense_matrix<float,host_memory_space,row_major> hA(32, 1179648);
	tensor<float,host_memory_space> hV(1179648);

	sequence(dA);
	sequence(dV);
	sequence(hA);
	sequence(hV);

	reduce_to_row(dV,dA,RF_ADD, 1.0f, 1.0f);
	reduce_to_row(hV,hA,RF_ADD, 1.0f, 1.0f);

	tensor<float,host_memory_space> hV2(dV.size());
	convert(hV2, dV);

	for(int i=0;i<1179648;i++){
		BOOST_CHECK_CLOSE(hV[i],hV2[i],0.01);
	}

}
*/

BOOST_AUTO_TEST_CASE( mat_op_view )
{
	dense_matrix<float,host_memory_space,column_major>* h2 = blockview(x,(unsigned int)0,(unsigned int)n,(unsigned int)1,(unsigned int)2);
	dense_matrix<float,dev_memory_space,column_major>*  d2 = blockview(v,(unsigned int)0,(unsigned int)n,(unsigned int)1,(unsigned int)2);
	sequence(x);
	sequence(v);
	BOOST_CHECK_EQUAL(h2->h(), x.h());
	BOOST_CHECK_EQUAL(d2->h(), x.h());
	BOOST_CHECK_EQUAL(h2->w(), 2);
	BOOST_CHECK_EQUAL(d2->w(), 2);
	for(int i=0;i<n;i++)
		for(int j=0;j<2;j++){
			BOOST_CHECK_CLOSE((float)(*h2)(i,j),(float)(*d2)(i,j),0.01);
			BOOST_CHECK_CLOSE((float)(*h2)(i,j),(float)x(i, j+1),0.01);
		}
}


BOOST_AUTO_TEST_CASE( mat_op_transpose )
{
	const int n = 8;
	const int m = 3;

	dense_matrix<float,host_memory_space,column_major> hA(n, m), hB(m, n);
	dense_matrix<float,dev_memory_space,column_major>  dA(n, m), dB(m, n);
	dense_matrix<float,host_memory_space,row_major> hC(n, m), hD(m, n);
	dense_matrix<float,dev_memory_space,row_major>  dC(n, m), dD(m, n);

	sequence(hB); sequence(dB);
	sequence(hD); sequence(dD);

	transpose(hA, hB);
	transpose(dA, dB);
	transpose(hC, hD);
	transpose(dC, dD);

	dense_matrix<float,host_memory_space,column_major> h2A(dA.w(), dA.h()); convert(h2A, dA);
	dense_matrix<float,host_memory_space,row_major> h2C(dC.w(), dC.h()); convert(h2C, dC);

	for(int i=0;i<n;i++)
		for(int j=0;j<m;j++){
			BOOST_CHECK_EQUAL(hA(i,j), hB(j,i));
			BOOST_CHECK_EQUAL(hC(i,j), hD(j,i));
		}

	MAT_CMP(hA, h2A, 0.1);
	MAT_CMP(hC, h2C, 0.1);
}

BOOST_AUTO_TEST_CASE( all_reduce )
{
	const int n = 270;
	const int m = 270;

	std::list<reduce_functor> rf_arg;
	std::list<reduce_functor> rf_val;
	std::list<reduce_functor> rf_rp; // reduced precision
	rf_arg += RF_ARGMAX, RF_ARGMIN;
	rf_val += RF_ADD, RF_MAX, RF_MIN, RF_LOGADDEXP;
	rf_rp  += RF_LOGADDEXP, RF_MULT;

	for(int dim=0;dim<2;dim++){ 
	if(1){ // column-major
		std::cout << "Column Major"<<std::endl;
		dense_matrix<float,dev_memory_space,column_major>  dA(n, m);
		fill_rnd_uniform(dA);
		dA *= 2.f;

		for(std::list<reduce_functor>::iterator it=rf_arg.begin(); it!=rf_arg.end(); it++)
		{   std::cout << "Functor: "<<(*it)<<std::endl;
			std::pair<tensor<unsigned int,host_memory_space>*,
				tensor<unsigned int,host_memory_space>*> p = test_reduce<unsigned int>(dim,dA,*it);
			for(unsigned int i=0; i<m; i++) {
				BOOST_CHECK_EQUAL((*p.first)[i], (*p.second)[i]);
			}
			delete p.first; delete p.second;
		}
		for(std::list<reduce_functor>::iterator it=rf_val.begin(); it!=rf_val.end(); it++)
		{   std::cout << "Functor: "<<(*it)<<std::endl;
		    std::pair<tensor<float,host_memory_space>*,
				tensor<float,host_memory_space>*> p = test_reduce<float>(dim,dA,*it);
			const float prec = find(rf_rp.begin(), rf_rp.end(), *it)==rf_rp.end() ? 0.1f : 4.5f;
			for(unsigned int i=0; i<m; i++) {
				BOOST_CHECK_CLOSE((float)(*p.first)[i], (float)(*p.second)[i],prec);
			}
			delete p.first; delete p.second;
		}
	}
	if(1){ // row-major
		std::cout << "Row Major"<<std::endl;
		dense_matrix<float,dev_memory_space,row_major>  dA(n, m);
		fill_rnd_uniform(dA);
		dA *= 2.f;

		for(std::list<reduce_functor>::iterator it=rf_arg.begin(); it!=rf_arg.end(); it++)
		{   std::cout << "Functor: "<<(*it)<<std::endl;
		    std::pair<tensor<unsigned int,host_memory_space>*,
				tensor<unsigned int,host_memory_space>*> p = test_reduce<unsigned int>(dim,dA,*it);
			for(unsigned int i=0; i<m; i++) {
				BOOST_CHECK_EQUAL((*p.first)[i], (*p.second)[i]);
			}
			delete p.first; delete p.second;
		}
		for(std::list<reduce_functor>::iterator it=rf_val.begin(); it!=rf_val.end(); it++)
		{   std::cout << "Functor: "<<(*it)<<std::endl;
		    std::pair<tensor<float,host_memory_space>*,
				tensor<float,host_memory_space>*> p = test_reduce<float>(dim,dA,*it);
			const float prec = find(rf_rp.begin(), rf_rp.end(), *it)==rf_rp.end() ? 0.1f : 4.5f;
			for(unsigned int i=0; i<m; i++) {
				BOOST_CHECK_CLOSE((float)(*p.first)[i], (float)(*p.second)[i], prec);
			}
			delete p.first; delete p.second;
		}
	}
	}
}
BOOST_AUTO_TEST_SUITE_END()
