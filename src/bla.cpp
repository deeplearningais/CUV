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

    //tensor<float,host_memory_space> s(extents[5][4]);
    tensor<float,host_memory_space> t(extents[2][5][4], pitched_memory_tag());
    //tensor<float,host_memory_space> u(s);
    //tensor<float,host_memory_space> v(s, pitched_memory_tag());

    //std::cout << s.is_2dcopyable()<<std::endl;
    std::cout << t.is_2dcopyable()<<std::endl;

    for(unsigned int i=0;i<t.shape(2);i++)
        for(unsigned int j=0;j<t.shape(1);j++)
            for(unsigned int k=0;k<t.shape(0);k++)
                t(k,j,i) = k*t.shape(1)*t.shape(2) + j*t.shape(2)+i;
    std::cout << "Shape  : "<<t.info().host_shape<<std::endl;
    std::cout << "Strides: "<<t.info().host_stride<<std::endl;
    std::cout << "-------------------------------------------"<<std::endl;
    std::cout << t(1,2,3)<<std::endl;
    //std::cout << t[0] <<std::endl;
    std::cout << t <<std::endl;

    return 0;
}
