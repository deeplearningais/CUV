#include <cuv.hpp>
using namespace cuv;

int main(void){
    tensor<float,host_memory_space> h(256);  // reserves space in host memory
    tensor<float,dev_memory_space>  d(256);  // reserves space in device memory

    fill(h,0);                          // terse form
    apply_0ary_functor(h,NF_FILL,0.f);    // more verbose

    d=h;                                // push to device
    sequence(d);                        // fill device vector with a sequence

    h=d;                                // pull to host
    for(int i=0;i<h.size();i++)
    {
        assert(d[i] == h[i]);
    }
}
