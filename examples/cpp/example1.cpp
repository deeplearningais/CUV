#include <cuv.hpp>
using namespace cuv;

int main(void){
    tensor<float,host_memory_space> h(extents[8][5]);  // reserves space in host memory
    tensor<float,dev_memory_space>  d(extents[8][5]);  // reserves space in device memory

    h = 0;                              // set all values to 0

    d=h;                                // push to device
    sequence(d);                        // fill device vector with a sequence

    h=d;                                // pull to host
    for(int i=0;i<h.size();i++) {
        assert(d[i] == h[i]);
    }

    for(int i=0;i<h.shape(0);i++)
        for(int j=0;j<h.shape(1);j++) {
            assert(d(i,j) == h(i,j));
        }
}
