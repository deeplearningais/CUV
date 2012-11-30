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
#include <limits>

#include <cuv/tools/cuv_general.hpp>
#include <cuv/random/random.hpp>
#include <cuv/matrix_ops/matrix_ops.hpp>
#include <cuv/tensor_ops/tensor_ops.hpp>
#include <cuv/tensor_ops/rprop.hpp>
#include <cuv/libs/opt/opt.hpp>

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

template<class M>
void lswd_optimization(int N){
	tensor<float,M>       W(N);
	tensor<float,M>       dW(N);

	// we will optimize the function f(W) = 2 * (W-optimum)^2
	tensor<float,M>       optimum(N);
	fill_rnd_uniform(optimum);

	// start at random value
	fill_rnd_uniform(W);

	
	for (int iter = 0; iter < 1000; ++iter) {
		dW = W-optimum;
		learn_step_weight_decay(W,dW,0.01,0.0);
	}

	tensor<float,M>       f = (W-optimum);
	f *= f;
	f *= 2.f;
	double error = mean(f);
	BOOST_CHECK_CLOSE((float)error+1.f,(float)1.f,0.01f);


	for(int i=0;i<N;i++){
	       BOOST_CHECK_CLOSE((float)W[i]+1.f,(float)optimum[i]+1.f,0.01f);
	}
}

template<class M>
void lswd_momentum_optimization(int N){
	tensor<float,M>       W(N);
	tensor<float,M>       dW(N);
	tensor<float,M>       mom(N);

	// we will optimize the function f(W) = 2 * (W-optimum)^2
	tensor<float,M>       optimum(N);
	fill_rnd_uniform(optimum);

	// start at random value
	fill_rnd_uniform(W);
    fill(mom,0.f);

	
    int max_iter = 1000;
	for (int iter = 0; iter < max_iter; ++iter) {
		dW = W-optimum;
        cuvAssert(!has_nan(W));
        cuvAssert(!has_nan(dW));
        cuvAssert(!has_nan(mom));
		learn_step_weight_decay_momentum(W,mom,dW,0.01, 1.f - iter/(float)max_iter,0.0);
	}

	tensor<float,M>       f = (W-optimum);
	f *= f;
	f *= 2.f;
	double error = mean(f);
	BOOST_CHECK_CLOSE((float)error+1.f,(float)1.f,0.01f);


	for(int i=0;i<N;i++){
	       BOOST_CHECK_CLOSE((float)W[i]+1.f,(float)optimum[i]+1.f,0.01f);
	}
}

template<class M>
void adagrad_optimization(int N, bool l1decay, bool rmsprop){
	tensor<float,M>       W(N);
	tensor<float,M>       dW(N);
	tensor<float,M>       sW(N);

	// we will optimize the function f(W) = 2 * (W-optimum)^2
	tensor<float,M>       optimum(N);
	fill_rnd_uniform(optimum);
	optimum -= 0.5f;

	// start at random value
	fill_rnd_uniform(W);
	W -= 0.5f;
	fill(sW, 0.0f); // sum of squares
	
    float l1d = l1decay ? 0.001f : 0.0f;
    float learningrate = 1.0;
    int max_iter = 1000;
    float delta = 0.00001f;
	for (int iter = 0; iter < max_iter; ++iter) {
		dW = W-optimum;
        if(iter == 0 && rmsprop)
            sW = dW * dW; // initialize squared grad avg
        float lr = learningrate * (1.f - iter / (float) max_iter); // decrease linearly
        if(rmsprop)
            cuv::libs::opt::rmsprop(W,dW,sW,lr, delta, 0.f, l1d);
        else
            cuv::libs::opt::adagrad(W,dW,sW,lr, delta, 0.f, l1d);
	}

	std::cout << "Adagrad: L1-Norm of W (l1-reg'n: "<<l1decay<<"): " << cuv::norm1(W)<<std::endl;
	std::cout << " number exact zeros: " << cuv::count(W, 0.f)<<std::endl;
    if(l1decay) {
        BOOST_CHECK_GT(cuv::count(W, 0.f), 0); // number of exact zeros should be >0 in l1-optimization!
    }
	tensor<float,M>   f = W-optimum;
	double tendency = norm1(f*f) - norm1(optimum*optimum);
	BOOST_CHECK_LT(tendency,0); // weights should be a bit too cose to 0
	f *= f;
	f *= 2.f;
	double error = mean(f);
    if(l1decay)
        BOOST_CHECK_CLOSE(error+1.0,1.0,1.00); // less strict if l1decay used
    else
        BOOST_CHECK_CLOSE(error+1.0,1.0,0.01);

	for(int i=0;i<N;i++){
        if(l1decay)
	       BOOST_CHECK_CLOSE((float)W[i] + 10.f,(float)optimum[i]+ 10.f,1.0f); // less strict if l1decay used
        else
	       BOOST_CHECK_CLOSE((float)W[i] + 10.f,(float)optimum[i]+ 10.f,0.1f);
	}
}
template<class M>
void rprop_optimization_decay_l1(int N){
	tensor<signed char,M> dW_old(N);
	tensor<float,M>       W(N);
	tensor<float,M>       dW(N);
	tensor<float,M>       rate(N);

	// we will optimize the function f(W) = 2 * (W-optimum)^2
	tensor<float,M>       optimum(N);
	fill_rnd_uniform(optimum);
	optimum -= 0.5f;

	// start at random value
	fill_rnd_uniform(W);
	W -= 0.5f;
	fill(dW_old, 0);    // initialize gradient
	fill(rate, 0.001f); // initialize learning rates
	
	for (int iter = 0; iter < 300; ++iter) {
		dW = W-optimum;
		rprop(W,dW,dW_old,rate,0.00,0.001);
	}

	std::cout << "L1-Norm of W (w/ l1-reg'n): " << cuv::norm1(W)<<std::endl;
	std::cout << "number exact zeros: " << cuv::count(W, 0.f)<<std::endl;
	BOOST_CHECK_GT(cuv::count(W, 0.f), 0); // number of exact zeros should be >0 in l1-optimization!
	tensor<float,M>   f = W-optimum;
	double tendency = norm1(f*f) - norm1(optimum*optimum);
	BOOST_CHECK_LT(tendency,0); // weights should be a bit too cose to 0
	f *= f;
	f *= 2.f;
	double error = mean(f);
	BOOST_CHECK_CLOSE(error+1.0,1.0,0.01);

	for(int i=0;i<N;i++){
	       BOOST_CHECK_CLOSE((float)W[i] + 10.f,(float)optimum[i]+ 10.f,0.1f);
	}
}

template<class M>
void rprop_optimization(int N){
	tensor<float,M> dW_old(N);
	tensor<float,M>       W(N);
	tensor<float,M>       dW(N);
	tensor<float,M>       rate(N);

	// we will optimize the function f(W) = 2 * (W-optimum)^2
	tensor<float,M>       optimum(N);
	fill_rnd_uniform(optimum);
	optimum -= 0.5f;

	// start at random value
	fill_rnd_uniform(W);
	fill(dW_old, 0);     // initialize gradient
	fill(rate, 0.001f); // initialize learning rates
	
	for (int iter = 0; iter < 300; ++iter) {
		dW = W-optimum;
		rprop(W,dW,dW_old,rate);
	}

	std::cout << "L1-Norm of W (no l1-reg'n): " << cuv::norm1(W)<<std::endl;

	tensor<float,M>       f = (W-optimum);
	f *= f;
	f *= 2.f;
	double error = mean(f);
	BOOST_CHECK_CLOSE(error+1.0,1.0,0.001);

	for(int i=0;i<N;i++){
	       BOOST_CHECK_CLOSE((float)W[i] + 1.f,(float)optimum[i]+ 1.f,0.01f);
	}
}

template<class M,class L>
void softmax_derivative(int n_var, int n_val){
    const float eps = 0.001f;

    tensor<float,M,L> X(extents[n_val][n_var]); fill_rnd_uniform(X); // inputs
    tensor<float,M,L> Y(extents[n_val][n_var]); Y = 0.f; // softmax result
    tensor<float,M,L> D(extents[n_val][n_var]); D = 0.f; // delta
    tensor<float,M,L> R(extents[n_val][n_var]); fill_rnd_uniform(R); // residual
    X+=1.1f;
    R+=1.3f;

    cuv::libs::opt::softmax(Y,X,1);
    cuv::libs::opt::softmax_derivative(D,Y,R,1);
    tensor<float,M,L> Jtilde(extents[n_var*n_val][n_var*n_val]);
    for(int i=0;i<n_val*n_var;i++){
            tensor<float,M,L> X_ = X.copy();
            tensor<float,M,L> Y_minus(X.shape());
            tensor<float,M,L> Y_plus (X.shape());
            X_[i] += eps;
            cuv::libs::opt::softmax(Y_plus,X_,1);
            X_[i] -= 2*eps;
            cuv::libs::opt::softmax(Y_minus,X_,1);
            Y_plus.reshape(extents[X.size()]);
            Y_minus.reshape(extents[X.size()]);
            tensor_view<float,M,L> finite_diff(indices[index_range(i,i+1)][index_range()], Jtilde);
            finite_diff = (Y_plus-Y_minus)/(2*eps);
    }
    tensor<float,M,L> D2(D.shape());
    R .reshape(extents[R.size()][1]);
    D .reshape(extents[D.size()][1]);
    D2.reshape(extents[D.size()][1]);

    cuv::prod(D2,Jtilde,R,'t','n');
    for(int i=0;i<D2.size();i++){
            BOOST_CHECK_CLOSE((float)D[i] + 1.f, (float)D2[i] + 1.f, 1.0f); // usually below 1.5%, but 5% stop this from failing occasionally
    }

}

double logaddexp(double x,double y){
    double m = std::max(x,y) ;

}
template<class M,class L>
void softmax(int n_var, int n_val){
    const float eps = 0.001;

    tensor<float,M,L> X(extents[n_val][n_var]); fill_rnd_uniform(X); X*=10.f;// inputs
    tensor<float,M,L> Y(extents[n_val][n_var]);          // softmax result

    cuv::libs::opt::softmax(Y,X, true);

    for(int i=0;i<n_var;i++){
        double normalizer = 0.f;
        double m=-1E9;
        for(int j=0;j<n_val;j++)
            m = std::max(m, (double)X(j,i));
        for(int j=0;j<n_val;j++)
            normalizer = normalizer + exp(X(j,i)-m);
        normalizer = m + log(normalizer);

        double sum=0.0;
        for(int j=0;j<n_val;j++){
            sum += Y(j,i);
            BOOST_CHECK_CLOSE(exp(X(j,i)-normalizer),  (double)(float)Y(j,i), 0.01);
        }
        BOOST_CHECK_CLOSE(sum, 1.0 , 0.01);
    }
}


struct Fix{
	static const int N = 8092;
	Fix()
	{
		initialize_mersenne_twister_seeds();
	}
	~Fix(){
	}
};


BOOST_FIXTURE_TEST_SUITE( s, Fix )

BOOST_AUTO_TEST_CASE( test_rprop_optimization_host )
{
   rprop_optimization<host_memory_space>(N);
}

BOOST_AUTO_TEST_CASE( test_rprop_optimization_dev )
{
   rprop_optimization<dev_memory_space>(N);
}




BOOST_AUTO_TEST_CASE( test_rmsprop_optimization_host )
{
   adagrad_optimization<host_memory_space>(N,false, true);
}
BOOST_AUTO_TEST_CASE( test_rmsprop_optimization_dev )
{
   adagrad_optimization<dev_memory_space>(N,false, true);
}



BOOST_AUTO_TEST_CASE( test_adagrad_optimization_l1_host )
{
   adagrad_optimization<host_memory_space>(N,true, false);
}
BOOST_AUTO_TEST_CASE( test_adagrad_optimization_host )
{
   adagrad_optimization<host_memory_space>(N,false, false);
}

BOOST_AUTO_TEST_CASE( test_adagrad_optimization_l1_dev )
{
   adagrad_optimization<dev_memory_space>(N,true, false);
}
BOOST_AUTO_TEST_CASE( test_adagrad_optimization_dev )
{
   adagrad_optimization<dev_memory_space>(N,false, false);
}
BOOST_AUTO_TEST_CASE( test_lswd_mom_optimization_host )
{
   lswd_momentum_optimization<host_memory_space>(N);
}

BOOST_AUTO_TEST_CASE( test_lswd_mom_optimization_dev )
{
   lswd_momentum_optimization<dev_memory_space>(N);
}
BOOST_AUTO_TEST_CASE( test_lswd_optimization_host )
{
   lswd_optimization<host_memory_space>(N);
}

BOOST_AUTO_TEST_CASE( test_lswd_optimization_dev )
{
   lswd_optimization<dev_memory_space>(N);
}

BOOST_AUTO_TEST_CASE( test_rprop_optimization_l1_host )
{
   rprop_optimization_decay_l1<host_memory_space>(N);
}
BOOST_AUTO_TEST_CASE( test_rprop_optimization_l1_dev )
{
   rprop_optimization_decay_l1<dev_memory_space>(N);
}

BOOST_AUTO_TEST_CASE( test_softmax )
{
    softmax<host_memory_space,row_major>(16,10);
}
BOOST_AUTO_TEST_CASE( test_softmax_derivative )
{
    softmax_derivative<host_memory_space,row_major>(16,4);
}

BOOST_AUTO_TEST_SUITE_END()
