template void apply_0ary_functor<vector<float,host_memory_space> >(vector<float,host_memory_space>&, const NullaryFunctor&);
template void apply_0ary_functor<vector<float,host_memory_space> >(vector<float,host_memory_space>&, const NullaryFunctor&, const vector<float,host_memory_space>::value_type&);
template void apply_0ary_functor<vector<unsigned int,host_memory_space> >(vector<unsigned int,host_memory_space>&, const NullaryFunctor&);
template void apply_0ary_functor<vector<unsigned int,host_memory_space> >(vector<unsigned int,host_memory_space>&, const NullaryFunctor&, const vector<unsigned int,host_memory_space>::value_type&);
template void apply_0ary_functor<vector<int,host_memory_space> >(vector<int,host_memory_space>&, const NullaryFunctor&);
template void apply_0ary_functor<vector<int,host_memory_space> >(vector<int,host_memory_space>&, const NullaryFunctor&, const vector<int,host_memory_space>::value_type&);
template void apply_0ary_functor<vector<unsigned char,host_memory_space> >(vector<unsigned char,host_memory_space>&, const NullaryFunctor&);
template void apply_0ary_functor<vector<unsigned char,host_memory_space> >(vector<unsigned char,host_memory_space>&, const NullaryFunctor&, const vector<unsigned char,host_memory_space>::value_type&);
template void apply_0ary_functor<vector<signed char,host_memory_space> >(vector<signed char,host_memory_space>&, const NullaryFunctor&);
template void apply_0ary_functor<vector<signed char,host_memory_space> >(vector<signed char,host_memory_space>&, const NullaryFunctor&, const vector<signed char,host_memory_space>::value_type&);
namespace detail{ template void apply_scalar_functor<vector<float,host_memory_space>,vector<float,host_memory_space>,float >(vector<float,host_memory_space>&,const vector<float,host_memory_space>&, const ScalarFunctor&,const int&, const float&, const float&);}
namespace detail{ template void apply_scalar_functor<vector<unsigned int,host_memory_space>,vector<unsigned int,host_memory_space>,unsigned int >(vector<unsigned int,host_memory_space>&,const vector<unsigned int,host_memory_space>&, const ScalarFunctor&,const int&, const unsigned int&, const unsigned int&);}
namespace detail{ template void apply_scalar_functor<vector<int,host_memory_space>,vector<int,host_memory_space>,int >(vector<int,host_memory_space>&,const vector<int,host_memory_space>&, const ScalarFunctor&,const int&, const int&, const int&);}
namespace detail{ template void apply_scalar_functor<vector<unsigned char,host_memory_space>,vector<unsigned char,host_memory_space>,unsigned char >(vector<unsigned char,host_memory_space>&,const vector<unsigned char,host_memory_space>&, const ScalarFunctor&,const int&, const unsigned char&, const unsigned char&);}
namespace detail{ template void apply_scalar_functor<vector<signed char,host_memory_space>,vector<signed char,host_memory_space>,signed char >(vector<signed char,host_memory_space>&,const vector<signed char,host_memory_space>&, const ScalarFunctor&,const int&, const signed char&, const signed char&);}
namespace detail{ template void apply_scalar_functor<vector<unsigned char,host_memory_space>,vector<float,host_memory_space>,float >(vector<unsigned char,host_memory_space>&,const vector<float,host_memory_space>&, const ScalarFunctor&,const int&, const float&, const float&);}
namespace detail{ template void apply_scalar_functor<vector<unsigned char,host_memory_space>,vector<unsigned int,host_memory_space>,unsigned int >(vector<unsigned char,host_memory_space>&,const vector<unsigned int,host_memory_space>&, const ScalarFunctor&,const int&, const unsigned int&, const unsigned int&);}
namespace detail{ template void apply_scalar_functor<vector<unsigned char,host_memory_space>,vector<int,host_memory_space>,int >(vector<unsigned char,host_memory_space>&,const vector<int,host_memory_space>&, const ScalarFunctor&,const int&, const int&, const int&);}
namespace detail{ template void apply_scalar_functor<vector<unsigned char,host_memory_space>,vector<signed char,host_memory_space>,signed char >(vector<unsigned char,host_memory_space>&,const vector<signed char,host_memory_space>&, const ScalarFunctor&,const int&, const signed char&, const signed char&);}
namespace detail{ template void apply_binary_functor<vector<float,host_memory_space>,vector<float,host_memory_space>,vector<float,host_memory_space>,float >(vector<float,host_memory_space>&,const vector<float,host_memory_space>&,const vector<float,host_memory_space>&, const BinaryFunctor&,const int&, const float&, const float&);}
namespace detail{ template void apply_binary_functor<vector<unsigned int,host_memory_space>,vector<unsigned int,host_memory_space>,vector<unsigned int,host_memory_space>,unsigned int >(vector<unsigned int,host_memory_space>&,const vector<unsigned int,host_memory_space>&,const vector<unsigned int,host_memory_space>&, const BinaryFunctor&,const int&, const unsigned int&, const unsigned int&);}
namespace detail{ template void apply_binary_functor<vector<int,host_memory_space>,vector<int,host_memory_space>,vector<int,host_memory_space>,int >(vector<int,host_memory_space>&,const vector<int,host_memory_space>&,const vector<int,host_memory_space>&, const BinaryFunctor&,const int&, const int&, const int&);}
namespace detail{ template void apply_binary_functor<vector<unsigned char,host_memory_space>,vector<unsigned char,host_memory_space>,vector<unsigned char,host_memory_space>,unsigned char >(vector<unsigned char,host_memory_space>&,const vector<unsigned char,host_memory_space>&,const vector<unsigned char,host_memory_space>&, const BinaryFunctor&,const int&, const unsigned char&, const unsigned char&);}
namespace detail{ template void apply_binary_functor<vector<signed char,host_memory_space>,vector<signed char,host_memory_space>,vector<signed char,host_memory_space>,signed char >(vector<signed char,host_memory_space>&,const vector<signed char,host_memory_space>&,const vector<signed char,host_memory_space>&, const BinaryFunctor&,const int&, const signed char&, const signed char&);}
template bool has_inf<vector<float,host_memory_space> >(const vector<float,host_memory_space>&);
template bool has_nan<vector<float,host_memory_space> >(const vector<float,host_memory_space>&);
template float minimum<vector<float,host_memory_space> >(const vector<float,host_memory_space>&);
template float maximum<vector<float,host_memory_space> >(const vector<float,host_memory_space>&);
template float sum<vector<float,host_memory_space> >(const vector<float,host_memory_space>&);
template float norm1<vector<float,host_memory_space> >(const vector<float,host_memory_space>&);
template float norm2<vector<float,host_memory_space> >(const vector<float,host_memory_space>&);
template float mean<vector<float,host_memory_space> >(const vector<float,host_memory_space>&);
template float var<vector<float,host_memory_space> >(const vector<float,host_memory_space>&);
template typename vector<float,host_memory_space>::index_type     arg_max<vector<float,host_memory_space> >(const vector<float,host_memory_space>&);
template typename vector<float,host_memory_space>::index_type     arg_min<vector<float,host_memory_space> >(const vector<float,host_memory_space>&);
template bool has_inf<vector<unsigned int,host_memory_space> >(const vector<unsigned int,host_memory_space>&);
template bool has_nan<vector<unsigned int,host_memory_space> >(const vector<unsigned int,host_memory_space>&);
template float minimum<vector<unsigned int,host_memory_space> >(const vector<unsigned int,host_memory_space>&);
template float maximum<vector<unsigned int,host_memory_space> >(const vector<unsigned int,host_memory_space>&);
template float sum<vector<unsigned int,host_memory_space> >(const vector<unsigned int,host_memory_space>&);
template float norm1<vector<unsigned int,host_memory_space> >(const vector<unsigned int,host_memory_space>&);
template float norm2<vector<unsigned int,host_memory_space> >(const vector<unsigned int,host_memory_space>&);
template float mean<vector<unsigned int,host_memory_space> >(const vector<unsigned int,host_memory_space>&);
template float var<vector<unsigned int,host_memory_space> >(const vector<unsigned int,host_memory_space>&);
template typename vector<unsigned int,host_memory_space>::index_type     arg_max<vector<unsigned int,host_memory_space> >(const vector<unsigned int,host_memory_space>&);
template typename vector<unsigned int,host_memory_space>::index_type     arg_min<vector<unsigned int,host_memory_space> >(const vector<unsigned int,host_memory_space>&);
template bool has_inf<vector<int,host_memory_space> >(const vector<int,host_memory_space>&);
template bool has_nan<vector<int,host_memory_space> >(const vector<int,host_memory_space>&);
template float minimum<vector<int,host_memory_space> >(const vector<int,host_memory_space>&);
template float maximum<vector<int,host_memory_space> >(const vector<int,host_memory_space>&);
template float sum<vector<int,host_memory_space> >(const vector<int,host_memory_space>&);
template float norm1<vector<int,host_memory_space> >(const vector<int,host_memory_space>&);
template float norm2<vector<int,host_memory_space> >(const vector<int,host_memory_space>&);
template float mean<vector<int,host_memory_space> >(const vector<int,host_memory_space>&);
template float var<vector<int,host_memory_space> >(const vector<int,host_memory_space>&);
template typename vector<int,host_memory_space>::index_type     arg_max<vector<int,host_memory_space> >(const vector<int,host_memory_space>&);
template typename vector<int,host_memory_space>::index_type     arg_min<vector<int,host_memory_space> >(const vector<int,host_memory_space>&);
template bool has_inf<vector<unsigned char,host_memory_space> >(const vector<unsigned char,host_memory_space>&);
template bool has_nan<vector<unsigned char,host_memory_space> >(const vector<unsigned char,host_memory_space>&);
template float minimum<vector<unsigned char,host_memory_space> >(const vector<unsigned char,host_memory_space>&);
template float maximum<vector<unsigned char,host_memory_space> >(const vector<unsigned char,host_memory_space>&);
template float sum<vector<unsigned char,host_memory_space> >(const vector<unsigned char,host_memory_space>&);
template float norm1<vector<unsigned char,host_memory_space> >(const vector<unsigned char,host_memory_space>&);
template float norm2<vector<unsigned char,host_memory_space> >(const vector<unsigned char,host_memory_space>&);
template float mean<vector<unsigned char,host_memory_space> >(const vector<unsigned char,host_memory_space>&);
template float var<vector<unsigned char,host_memory_space> >(const vector<unsigned char,host_memory_space>&);
template typename vector<unsigned char,host_memory_space>::index_type     arg_max<vector<unsigned char,host_memory_space> >(const vector<unsigned char,host_memory_space>&);
template typename vector<unsigned char,host_memory_space>::index_type     arg_min<vector<unsigned char,host_memory_space> >(const vector<unsigned char,host_memory_space>&);
template bool has_inf<vector<signed char,host_memory_space> >(const vector<signed char,host_memory_space>&);
template bool has_nan<vector<signed char,host_memory_space> >(const vector<signed char,host_memory_space>&);
template float minimum<vector<signed char,host_memory_space> >(const vector<signed char,host_memory_space>&);
template float maximum<vector<signed char,host_memory_space> >(const vector<signed char,host_memory_space>&);
template float sum<vector<signed char,host_memory_space> >(const vector<signed char,host_memory_space>&);
template float norm1<vector<signed char,host_memory_space> >(const vector<signed char,host_memory_space>&);
template float norm2<vector<signed char,host_memory_space> >(const vector<signed char,host_memory_space>&);
template float mean<vector<signed char,host_memory_space> >(const vector<signed char,host_memory_space>&);
template float var<vector<signed char,host_memory_space> >(const vector<signed char,host_memory_space>&);
template typename vector<signed char,host_memory_space>::index_type     arg_max<vector<signed char,host_memory_space> >(const vector<signed char,host_memory_space>&);
template typename vector<signed char,host_memory_space>::index_type     arg_min<vector<signed char,host_memory_space> >(const vector<signed char,host_memory_space>&);
template void apply_0ary_functor<vector<float, dev_memory_space> >(vector<float, dev_memory_space>&, const NullaryFunctor&);
template void apply_0ary_functor<vector<float, dev_memory_space> >(vector<float, dev_memory_space>&, const NullaryFunctor&, const vector<float, dev_memory_space>::value_type&);
template void apply_0ary_functor<vector<unsigned int, dev_memory_space> >(vector<unsigned int, dev_memory_space>&, const NullaryFunctor&);
template void apply_0ary_functor<vector<unsigned int, dev_memory_space> >(vector<unsigned int, dev_memory_space>&, const NullaryFunctor&, const vector<unsigned int, dev_memory_space>::value_type&);
template void apply_0ary_functor<vector<int, dev_memory_space> >(vector<int, dev_memory_space>&, const NullaryFunctor&);
template void apply_0ary_functor<vector<int, dev_memory_space> >(vector<int, dev_memory_space>&, const NullaryFunctor&, const vector<int, dev_memory_space>::value_type&);
template void apply_0ary_functor<vector<unsigned char, dev_memory_space> >(vector<unsigned char, dev_memory_space>&, const NullaryFunctor&);
template void apply_0ary_functor<vector<unsigned char, dev_memory_space> >(vector<unsigned char, dev_memory_space>&, const NullaryFunctor&, const vector<unsigned char, dev_memory_space>::value_type&);
template void apply_0ary_functor<vector<signed char, dev_memory_space> >(vector<signed char, dev_memory_space>&, const NullaryFunctor&);
template void apply_0ary_functor<vector<signed char, dev_memory_space> >(vector<signed char, dev_memory_space>&, const NullaryFunctor&, const vector<signed char, dev_memory_space>::value_type&);
namespace detail{ template void apply_scalar_functor<vector<float, dev_memory_space>,vector<float, dev_memory_space>,float >(vector<float, dev_memory_space>&,const vector<float, dev_memory_space>&, const ScalarFunctor&,const int&, const float&, const float&);}
namespace detail{ template void apply_scalar_functor<vector<unsigned int, dev_memory_space>,vector<unsigned int, dev_memory_space>,unsigned int >(vector<unsigned int, dev_memory_space>&,const vector<unsigned int, dev_memory_space>&, const ScalarFunctor&,const int&, const unsigned int&, const unsigned int&);}
namespace detail{ template void apply_scalar_functor<vector<int, dev_memory_space>,vector<int, dev_memory_space>,int >(vector<int, dev_memory_space>&,const vector<int, dev_memory_space>&, const ScalarFunctor&,const int&, const int&, const int&);}
namespace detail{ template void apply_scalar_functor<vector<unsigned char, dev_memory_space>,vector<unsigned char, dev_memory_space>,unsigned char >(vector<unsigned char, dev_memory_space>&,const vector<unsigned char, dev_memory_space>&, const ScalarFunctor&,const int&, const unsigned char&, const unsigned char&);}
namespace detail{ template void apply_scalar_functor<vector<signed char, dev_memory_space>,vector<signed char, dev_memory_space>,signed char >(vector<signed char, dev_memory_space>&,const vector<signed char, dev_memory_space>&, const ScalarFunctor&,const int&, const signed char&, const signed char&);}
namespace detail{ template void apply_scalar_functor<vector<unsigned char, dev_memory_space>,vector<float, dev_memory_space>,float >(vector<unsigned char, dev_memory_space>&,const vector<float, dev_memory_space>&, const ScalarFunctor&,const int&, const float&, const float&);}
namespace detail{ template void apply_scalar_functor<vector<unsigned char, dev_memory_space>,vector<unsigned int, dev_memory_space>,unsigned int >(vector<unsigned char, dev_memory_space>&,const vector<unsigned int, dev_memory_space>&, const ScalarFunctor&,const int&, const unsigned int&, const unsigned int&);}
namespace detail{ template void apply_scalar_functor<vector<unsigned char, dev_memory_space>,vector<int, dev_memory_space>,int >(vector<unsigned char, dev_memory_space>&,const vector<int, dev_memory_space>&, const ScalarFunctor&,const int&, const int&, const int&);}
namespace detail{ template void apply_scalar_functor<vector<unsigned char, dev_memory_space>,vector<signed char, dev_memory_space>,signed char >(vector<unsigned char, dev_memory_space>&,const vector<signed char, dev_memory_space>&, const ScalarFunctor&,const int&, const signed char&, const signed char&);}
namespace detail{ template void apply_binary_functor<vector<float, dev_memory_space>,vector<float, dev_memory_space>,vector<float, dev_memory_space>,float >(vector<float, dev_memory_space>&,const vector<float, dev_memory_space>&,const vector<float, dev_memory_space>&, const BinaryFunctor&,const int&, const float&, const float&);}
namespace detail{ template void apply_binary_functor<vector<unsigned int, dev_memory_space>,vector<unsigned int, dev_memory_space>,vector<unsigned int, dev_memory_space>,unsigned int >(vector<unsigned int, dev_memory_space>&,const vector<unsigned int, dev_memory_space>&,const vector<unsigned int, dev_memory_space>&, const BinaryFunctor&,const int&, const unsigned int&, const unsigned int&);}
namespace detail{ template void apply_binary_functor<vector<int, dev_memory_space>,vector<int, dev_memory_space>,vector<int, dev_memory_space>,int >(vector<int, dev_memory_space>&,const vector<int, dev_memory_space>&,const vector<int, dev_memory_space>&, const BinaryFunctor&,const int&, const int&, const int&);}
namespace detail{ template void apply_binary_functor<vector<unsigned char, dev_memory_space>,vector<unsigned char, dev_memory_space>,vector<unsigned char, dev_memory_space>,unsigned char >(vector<unsigned char, dev_memory_space>&,const vector<unsigned char, dev_memory_space>&,const vector<unsigned char, dev_memory_space>&, const BinaryFunctor&,const int&, const unsigned char&, const unsigned char&);}
namespace detail{ template void apply_binary_functor<vector<signed char, dev_memory_space>,vector<signed char, dev_memory_space>,vector<signed char, dev_memory_space>,signed char >(vector<signed char, dev_memory_space>&,const vector<signed char, dev_memory_space>&,const vector<signed char, dev_memory_space>&, const BinaryFunctor&,const int&, const signed char&, const signed char&);}
template bool has_inf<vector<float, dev_memory_space> >(const vector<float, dev_memory_space>&);
template bool has_nan<vector<float, dev_memory_space> >(const vector<float, dev_memory_space>&);
template float minimum<vector<float, dev_memory_space> >(const vector<float, dev_memory_space>&);
template float maximum<vector<float, dev_memory_space> >(const vector<float, dev_memory_space>&);
template float sum<vector<float, dev_memory_space> >(const vector<float, dev_memory_space>&);
template float norm1<vector<float, dev_memory_space> >(const vector<float, dev_memory_space>&);
template float norm2<vector<float, dev_memory_space> >(const vector<float, dev_memory_space>&);
template float mean<vector<float, dev_memory_space> >(const vector<float, dev_memory_space>&);
template float var<vector<float, dev_memory_space> >(const vector<float, dev_memory_space>&);
template typename vector<float, dev_memory_space>::index_type     arg_max<vector<float, dev_memory_space> >(const vector<float, dev_memory_space>&);
template typename vector<float, dev_memory_space>::index_type     arg_min<vector<float, dev_memory_space> >(const vector<float, dev_memory_space>&);
template bool has_inf<vector<unsigned int, dev_memory_space> >(const vector<unsigned int, dev_memory_space>&);
template bool has_nan<vector<unsigned int, dev_memory_space> >(const vector<unsigned int, dev_memory_space>&);
template float minimum<vector<unsigned int, dev_memory_space> >(const vector<unsigned int, dev_memory_space>&);
template float maximum<vector<unsigned int, dev_memory_space> >(const vector<unsigned int, dev_memory_space>&);
template float sum<vector<unsigned int, dev_memory_space> >(const vector<unsigned int, dev_memory_space>&);
template float norm1<vector<unsigned int, dev_memory_space> >(const vector<unsigned int, dev_memory_space>&);
template float norm2<vector<unsigned int, dev_memory_space> >(const vector<unsigned int, dev_memory_space>&);
template float mean<vector<unsigned int, dev_memory_space> >(const vector<unsigned int, dev_memory_space>&);
template float var<vector<unsigned int, dev_memory_space> >(const vector<unsigned int, dev_memory_space>&);
template typename vector<unsigned int, dev_memory_space>::index_type     arg_max<vector<unsigned int, dev_memory_space> >(const vector<unsigned int, dev_memory_space>&);
template typename vector<unsigned int, dev_memory_space>::index_type     arg_min<vector<unsigned int, dev_memory_space> >(const vector<unsigned int, dev_memory_space>&);
template bool has_inf<vector<int, dev_memory_space> >(const vector<int, dev_memory_space>&);
template bool has_nan<vector<int, dev_memory_space> >(const vector<int, dev_memory_space>&);
template float minimum<vector<int, dev_memory_space> >(const vector<int, dev_memory_space>&);
template float maximum<vector<int, dev_memory_space> >(const vector<int, dev_memory_space>&);
template float sum<vector<int, dev_memory_space> >(const vector<int, dev_memory_space>&);
template float norm1<vector<int, dev_memory_space> >(const vector<int, dev_memory_space>&);
template float norm2<vector<int, dev_memory_space> >(const vector<int, dev_memory_space>&);
template float mean<vector<int, dev_memory_space> >(const vector<int, dev_memory_space>&);
template float var<vector<int, dev_memory_space> >(const vector<int, dev_memory_space>&);
template typename vector<int, dev_memory_space>::index_type     arg_max<vector<int, dev_memory_space> >(const vector<int, dev_memory_space>&);
template typename vector<int, dev_memory_space>::index_type     arg_min<vector<int, dev_memory_space> >(const vector<int, dev_memory_space>&);
template bool has_inf<vector<unsigned char, dev_memory_space> >(const vector<unsigned char, dev_memory_space>&);
template bool has_nan<vector<unsigned char, dev_memory_space> >(const vector<unsigned char, dev_memory_space>&);
template float minimum<vector<unsigned char, dev_memory_space> >(const vector<unsigned char, dev_memory_space>&);
template float maximum<vector<unsigned char, dev_memory_space> >(const vector<unsigned char, dev_memory_space>&);
template float sum<vector<unsigned char, dev_memory_space> >(const vector<unsigned char, dev_memory_space>&);
template float norm1<vector<unsigned char, dev_memory_space> >(const vector<unsigned char, dev_memory_space>&);
template float norm2<vector<unsigned char, dev_memory_space> >(const vector<unsigned char, dev_memory_space>&);
template float mean<vector<unsigned char, dev_memory_space> >(const vector<unsigned char, dev_memory_space>&);
template float var<vector<unsigned char, dev_memory_space> >(const vector<unsigned char, dev_memory_space>&);
template typename vector<unsigned char, dev_memory_space>::index_type     arg_max<vector<unsigned char, dev_memory_space> >(const vector<unsigned char, dev_memory_space>&);
template typename vector<unsigned char, dev_memory_space>::index_type     arg_min<vector<unsigned char, dev_memory_space> >(const vector<unsigned char, dev_memory_space>&);
template bool has_inf<vector<signed char, dev_memory_space> >(const vector<signed char, dev_memory_space>&);
template bool has_nan<vector<signed char, dev_memory_space> >(const vector<signed char, dev_memory_space>&);
template float minimum<vector<signed char, dev_memory_space> >(const vector<signed char, dev_memory_space>&);
template float maximum<vector<signed char, dev_memory_space> >(const vector<signed char, dev_memory_space>&);
template float sum<vector<signed char, dev_memory_space> >(const vector<signed char, dev_memory_space>&);
template float norm1<vector<signed char, dev_memory_space> >(const vector<signed char, dev_memory_space>&);
template float norm2<vector<signed char, dev_memory_space> >(const vector<signed char, dev_memory_space>&);
template float mean<vector<signed char, dev_memory_space> >(const vector<signed char, dev_memory_space>&);
template float var<vector<signed char, dev_memory_space> >(const vector<signed char, dev_memory_space>&);
template typename vector<signed char, dev_memory_space>::index_type     arg_max<vector<signed char, dev_memory_space> >(const vector<signed char, dev_memory_space>&);
template typename vector<signed char, dev_memory_space>::index_type     arg_min<vector<signed char, dev_memory_space> >(const vector<signed char, dev_memory_space>&);
