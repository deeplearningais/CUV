#include <cuv/basics/tensor.hpp>
#include <cuv/matrix_ops/matrix_ops.hpp>
using namespace cuv;

int main(void){
    tensor<float,dev_memory_space,column_major> C(2048,2048),A(2048,2048),B(2048,2048);

    fill(C,0);         // initialize to some defined value, not strictly necessary here
    sequence(A);
    sequence(B);

    apply_binary_functor(A,B,BF_MULT);  // elementwise multiplication
    A *= B;                             // operators also work (elementwise)
    prod(C,A,B, 'n','t');               // matrix multiplication
}
