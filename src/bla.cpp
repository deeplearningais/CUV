#include <iostream>
#include <cstring>
#include "cuv/basics/tensor2.hpp"
using namespace cuv;

    int
main(int argc, char **argv)
{
    linear_memory<float,host_memory_space> ms(3), mt;
    ms[0] = 4;
    mt=ms;
    std::cout << ms<<std::endl;
    std::cout << mt<<std::endl;

    pitched_memory<float, host_memory_space> mp(3,4), mq;
    mp[0] = 4;
    mq = mp;
    std::cout << mp<<std::endl;
    std::cout << mq<<std::endl;

    tensor<float,host_memory_space> s(extents[6][4]);
    tensor<float,host_memory_space> t(extents[1][6][4], pitched_memory_tag());

    //std::cout << s.is_2dcopyable()<<std::endl;
    std::cout << t.is_2dcopyable()<<std::endl;

    for(unsigned int i=0;i<t.shape(2);i++){
        for(unsigned int j=0;j<t.shape(1);j++)
            for(unsigned int k=0;k<t.shape(0);k++){
                t(k,j,i) = k*t.shape(1)*t.shape(2) + j*t.shape(2)+i;
                s(  j,i) =                         + j*s.shape(1)+i;
            }
    }

    tensor<float,host_memory_space> u(t, linear_memory_tag());
    tensor<float,host_memory_space> v(t, pitched_memory_tag());
    tensor<float,dev_memory_space> w(t, pitched_memory_tag());

    tensor<float,host_memory_space,column_major> x = s;

    std::cout << "t Shape  : "<<t.info().host_shape<<std::endl;
    std::cout << "t Strides: "<<t.info().host_stride<<std::endl;
    std::cout << "u Shape  : "<<u.info().host_shape<<std::endl;
    std::cout << "u Strides: "<<u.info().host_stride<<std::endl;
    std::cout << "v Shape  : "<<v.info().host_shape<<std::endl;
    std::cout << "v Strides: "<<v.info().host_stride<<std::endl;
    std::cout << "w Shape  : "<<w.info().host_shape<<std::endl;
    std::cout << "w Strides: "<<w.info().host_stride<<std::endl;
    std::cout << t <<std::endl;
    std::cout << u <<std::endl;
    std::cout << v <<std::endl;
    std::cout << w <<std::endl;
    std::cout << "---------"<<std::endl;
    std::cout << s <<std::endl;
    std::cout << x <<std::endl;

    // a does not have a m_memory
    tensor<float,host_memory_space> a(extents[1][6][4],t.ptr());
    // copies 0-ptr
    t = a;

    std::cout << "---------"<<std::endl;
    tensor<float,host_memory_space> y(t,indices[0][index_range().stride(2)][index_range()]);
    tensor<float,host_memory_space> z = t[indices[0][index_range().stride(2)][index_range()]];
    std::cout << t <<std::endl;
    std::cout << y <<std::endl;
    std::cout << z <<std::endl;
    z(0,0) = -1;
    std::cout << t <<std::endl;
    std::cout << y <<std::endl;
    std::cout << z <<std::endl;
    z = z.copy();
    z(0,0) = 1;
    std::cout << y <<std::endl;

    return 0;
}
