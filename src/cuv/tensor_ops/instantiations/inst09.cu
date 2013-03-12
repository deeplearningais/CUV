
/**************************************************
 This is an auto-generated file.
 See instantiate.py to modify the content in here!
 **************************************************/
 #include "../tensor_ops.cuh"
 namespace cuv{
 template bool has_inf<tensor<float, dev_memory_space>::value_type,tensor<float, dev_memory_space>::memory_space_type >(const tensor<float, dev_memory_space>&);
template bool has_nan<tensor<float, dev_memory_space>::value_type,tensor<float, dev_memory_space>::memory_space_type >(const tensor<float, dev_memory_space>&);
template float minimum<tensor<float, dev_memory_space>::value_type,tensor<float, dev_memory_space>::memory_space_type >(const tensor<float, dev_memory_space>&);
template float maximum<tensor<float, dev_memory_space>::value_type,tensor<float, dev_memory_space>::memory_space_type >(const tensor<float, dev_memory_space>&);
template float sum<tensor<float, dev_memory_space>::value_type,tensor<float, dev_memory_space>::memory_space_type >(const tensor<float, dev_memory_space>&);
template unsigned int count<tensor<float, dev_memory_space>::value_type,tensor<float, dev_memory_space>::memory_space_type >(const tensor<float, dev_memory_space>&, const tensor<float, dev_memory_space>::value_type&);
template float norm1<tensor<float, dev_memory_space>::value_type,tensor<float, dev_memory_space>::memory_space_type >(const tensor<float, dev_memory_space>&);
template float norm2<tensor<float, dev_memory_space>::value_type,tensor<float, dev_memory_space>::memory_space_type >(const tensor<float, dev_memory_space>&);
template float diff_norm2<tensor<float, dev_memory_space>::value_type,tensor<float, dev_memory_space>::memory_space_type >(const tensor<float, dev_memory_space>&, const tensor<float, dev_memory_space>&);
template float mean<tensor<float, dev_memory_space>::value_type,tensor<float, dev_memory_space>::memory_space_type >(const tensor<float, dev_memory_space>&);
template float var<tensor<float, dev_memory_space>::value_type,tensor<float, dev_memory_space>::memory_space_type >(const tensor<float, dev_memory_space>&);
template typename tensor<float, dev_memory_space>::index_type     arg_max<tensor<float, dev_memory_space>::value_type,tensor<float, dev_memory_space>::memory_space_type >(const tensor<float, dev_memory_space>&);
template typename tensor<float, dev_memory_space>::index_type     arg_min<tensor<float, dev_memory_space>::value_type,tensor<float, dev_memory_space>::memory_space_type >(const tensor<float, dev_memory_space>&);
template bool has_inf<tensor<unsigned int, dev_memory_space>::value_type,tensor<unsigned int, dev_memory_space>::memory_space_type >(const tensor<unsigned int, dev_memory_space>&);
template bool has_nan<tensor<unsigned int, dev_memory_space>::value_type,tensor<unsigned int, dev_memory_space>::memory_space_type >(const tensor<unsigned int, dev_memory_space>&);
template float minimum<tensor<unsigned int, dev_memory_space>::value_type,tensor<unsigned int, dev_memory_space>::memory_space_type >(const tensor<unsigned int, dev_memory_space>&);
template float maximum<tensor<unsigned int, dev_memory_space>::value_type,tensor<unsigned int, dev_memory_space>::memory_space_type >(const tensor<unsigned int, dev_memory_space>&);
template float sum<tensor<unsigned int, dev_memory_space>::value_type,tensor<unsigned int, dev_memory_space>::memory_space_type >(const tensor<unsigned int, dev_memory_space>&);
template unsigned int count<tensor<unsigned int, dev_memory_space>::value_type,tensor<unsigned int, dev_memory_space>::memory_space_type >(const tensor<unsigned int, dev_memory_space>&, const tensor<unsigned int, dev_memory_space>::value_type&);
template float norm1<tensor<unsigned int, dev_memory_space>::value_type,tensor<unsigned int, dev_memory_space>::memory_space_type >(const tensor<unsigned int, dev_memory_space>&);
template float norm2<tensor<unsigned int, dev_memory_space>::value_type,tensor<unsigned int, dev_memory_space>::memory_space_type >(const tensor<unsigned int, dev_memory_space>&);
template float diff_norm2<tensor<unsigned int, dev_memory_space>::value_type,tensor<unsigned int, dev_memory_space>::memory_space_type >(const tensor<unsigned int, dev_memory_space>&, const tensor<unsigned int, dev_memory_space>&);
template float mean<tensor<unsigned int, dev_memory_space>::value_type,tensor<unsigned int, dev_memory_space>::memory_space_type >(const tensor<unsigned int, dev_memory_space>&);
template float var<tensor<unsigned int, dev_memory_space>::value_type,tensor<unsigned int, dev_memory_space>::memory_space_type >(const tensor<unsigned int, dev_memory_space>&);
template typename tensor<unsigned int, dev_memory_space>::index_type     arg_max<tensor<unsigned int, dev_memory_space>::value_type,tensor<unsigned int, dev_memory_space>::memory_space_type >(const tensor<unsigned int, dev_memory_space>&);
template typename tensor<unsigned int, dev_memory_space>::index_type     arg_min<tensor<unsigned int, dev_memory_space>::value_type,tensor<unsigned int, dev_memory_space>::memory_space_type >(const tensor<unsigned int, dev_memory_space>&);
template bool has_inf<tensor<int, dev_memory_space>::value_type,tensor<int, dev_memory_space>::memory_space_type >(const tensor<int, dev_memory_space>&);
template bool has_nan<tensor<int, dev_memory_space>::value_type,tensor<int, dev_memory_space>::memory_space_type >(const tensor<int, dev_memory_space>&);
template float minimum<tensor<int, dev_memory_space>::value_type,tensor<int, dev_memory_space>::memory_space_type >(const tensor<int, dev_memory_space>&);
template float maximum<tensor<int, dev_memory_space>::value_type,tensor<int, dev_memory_space>::memory_space_type >(const tensor<int, dev_memory_space>&);
template float sum<tensor<int, dev_memory_space>::value_type,tensor<int, dev_memory_space>::memory_space_type >(const tensor<int, dev_memory_space>&);
template unsigned int count<tensor<int, dev_memory_space>::value_type,tensor<int, dev_memory_space>::memory_space_type >(const tensor<int, dev_memory_space>&, const tensor<int, dev_memory_space>::value_type&);
template float norm1<tensor<int, dev_memory_space>::value_type,tensor<int, dev_memory_space>::memory_space_type >(const tensor<int, dev_memory_space>&);
template float norm2<tensor<int, dev_memory_space>::value_type,tensor<int, dev_memory_space>::memory_space_type >(const tensor<int, dev_memory_space>&);
template float diff_norm2<tensor<int, dev_memory_space>::value_type,tensor<int, dev_memory_space>::memory_space_type >(const tensor<int, dev_memory_space>&, const tensor<int, dev_memory_space>&);
template float mean<tensor<int, dev_memory_space>::value_type,tensor<int, dev_memory_space>::memory_space_type >(const tensor<int, dev_memory_space>&);
template float var<tensor<int, dev_memory_space>::value_type,tensor<int, dev_memory_space>::memory_space_type >(const tensor<int, dev_memory_space>&);
template typename tensor<int, dev_memory_space>::index_type     arg_max<tensor<int, dev_memory_space>::value_type,tensor<int, dev_memory_space>::memory_space_type >(const tensor<int, dev_memory_space>&);
template typename tensor<int, dev_memory_space>::index_type     arg_min<tensor<int, dev_memory_space>::value_type,tensor<int, dev_memory_space>::memory_space_type >(const tensor<int, dev_memory_space>&);
template bool has_inf<tensor<unsigned char, dev_memory_space>::value_type,tensor<unsigned char, dev_memory_space>::memory_space_type >(const tensor<unsigned char, dev_memory_space>&);
template bool has_nan<tensor<unsigned char, dev_memory_space>::value_type,tensor<unsigned char, dev_memory_space>::memory_space_type >(const tensor<unsigned char, dev_memory_space>&);
template float minimum<tensor<unsigned char, dev_memory_space>::value_type,tensor<unsigned char, dev_memory_space>::memory_space_type >(const tensor<unsigned char, dev_memory_space>&);
template float maximum<tensor<unsigned char, dev_memory_space>::value_type,tensor<unsigned char, dev_memory_space>::memory_space_type >(const tensor<unsigned char, dev_memory_space>&);
template float sum<tensor<unsigned char, dev_memory_space>::value_type,tensor<unsigned char, dev_memory_space>::memory_space_type >(const tensor<unsigned char, dev_memory_space>&);
template unsigned int count<tensor<unsigned char, dev_memory_space>::value_type,tensor<unsigned char, dev_memory_space>::memory_space_type >(const tensor<unsigned char, dev_memory_space>&, const tensor<unsigned char, dev_memory_space>::value_type&);
template float norm1<tensor<unsigned char, dev_memory_space>::value_type,tensor<unsigned char, dev_memory_space>::memory_space_type >(const tensor<unsigned char, dev_memory_space>&);
template float norm2<tensor<unsigned char, dev_memory_space>::value_type,tensor<unsigned char, dev_memory_space>::memory_space_type >(const tensor<unsigned char, dev_memory_space>&);
template float diff_norm2<tensor<unsigned char, dev_memory_space>::value_type,tensor<unsigned char, dev_memory_space>::memory_space_type >(const tensor<unsigned char, dev_memory_space>&, const tensor<unsigned char, dev_memory_space>&);
template float mean<tensor<unsigned char, dev_memory_space>::value_type,tensor<unsigned char, dev_memory_space>::memory_space_type >(const tensor<unsigned char, dev_memory_space>&);
template float var<tensor<unsigned char, dev_memory_space>::value_type,tensor<unsigned char, dev_memory_space>::memory_space_type >(const tensor<unsigned char, dev_memory_space>&);
template typename tensor<unsigned char, dev_memory_space>::index_type     arg_max<tensor<unsigned char, dev_memory_space>::value_type,tensor<unsigned char, dev_memory_space>::memory_space_type >(const tensor<unsigned char, dev_memory_space>&);
template typename tensor<unsigned char, dev_memory_space>::index_type     arg_min<tensor<unsigned char, dev_memory_space>::value_type,tensor<unsigned char, dev_memory_space>::memory_space_type >(const tensor<unsigned char, dev_memory_space>&);
template bool has_inf<tensor<signed char, dev_memory_space>::value_type,tensor<signed char, dev_memory_space>::memory_space_type >(const tensor<signed char, dev_memory_space>&);
template bool has_nan<tensor<signed char, dev_memory_space>::value_type,tensor<signed char, dev_memory_space>::memory_space_type >(const tensor<signed char, dev_memory_space>&);
template float minimum<tensor<signed char, dev_memory_space>::value_type,tensor<signed char, dev_memory_space>::memory_space_type >(const tensor<signed char, dev_memory_space>&);
template float maximum<tensor<signed char, dev_memory_space>::value_type,tensor<signed char, dev_memory_space>::memory_space_type >(const tensor<signed char, dev_memory_space>&);
template float sum<tensor<signed char, dev_memory_space>::value_type,tensor<signed char, dev_memory_space>::memory_space_type >(const tensor<signed char, dev_memory_space>&);
template unsigned int count<tensor<signed char, dev_memory_space>::value_type,tensor<signed char, dev_memory_space>::memory_space_type >(const tensor<signed char, dev_memory_space>&, const tensor<signed char, dev_memory_space>::value_type&);
template float norm1<tensor<signed char, dev_memory_space>::value_type,tensor<signed char, dev_memory_space>::memory_space_type >(const tensor<signed char, dev_memory_space>&);
template float norm2<tensor<signed char, dev_memory_space>::value_type,tensor<signed char, dev_memory_space>::memory_space_type >(const tensor<signed char, dev_memory_space>&);
template float diff_norm2<tensor<signed char, dev_memory_space>::value_type,tensor<signed char, dev_memory_space>::memory_space_type >(const tensor<signed char, dev_memory_space>&, const tensor<signed char, dev_memory_space>&);
template float mean<tensor<signed char, dev_memory_space>::value_type,tensor<signed char, dev_memory_space>::memory_space_type >(const tensor<signed char, dev_memory_space>&);
template float var<tensor<signed char, dev_memory_space>::value_type,tensor<signed char, dev_memory_space>::memory_space_type >(const tensor<signed char, dev_memory_space>&);
template typename tensor<signed char, dev_memory_space>::index_type     arg_max<tensor<signed char, dev_memory_space>::value_type,tensor<signed char, dev_memory_space>::memory_space_type >(const tensor<signed char, dev_memory_space>&);
template typename tensor<signed char, dev_memory_space>::index_type     arg_min<tensor<signed char, dev_memory_space>::value_type,tensor<signed char, dev_memory_space>::memory_space_type >(const tensor<signed char, dev_memory_space>&);
}
