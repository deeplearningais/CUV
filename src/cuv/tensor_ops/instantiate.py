#!/usr/bin/python

class vec_t:
	def __init__(self, v, m):
		self.types = (v,m)
		self.v     = v
		self.m     = m
	def __str__(self):
		#return "vector<%s,%s,%s> "%self.types
		return "tensor<%s,%s>"%self.types
	def value_type(self): return self.types[0]
	def memory_space_type(self): return self.types[1]

def apply_0ary_functor(types):
	for t in types:
		yield "template void apply_0ary_functor<{0},{1} >({2}&, const NullaryFunctor&);".format(t[0].v,t[0].m,t[0])
		yield "template void apply_0ary_functor<{0},{1} >({2}&, const NullaryFunctor&, const {0}&);".format(t[0].v,t[0].m,t[0])

def apply_scalar_functor(types):
	for t in types:
		yield "namespace detail{{ template void apply_scalar_functor<{0}::value_type,{1}::value_type,{0}::memory_space_type,{2},{2}>({0}&,const {1}&, const ScalarFunctor&,const int&, const tensor<unsigned char, {0}::memory_space_type>*, const {2}&, const {2}&);}}".format(t[0], t[1], t[2])

def apply_binary_functor(types):
	for t in types:
		yield "namespace detail{{ template void apply_binary_functor<{0}::value_type,{1}::value_type,{2}::value_type,{0}::memory_space_type,{3},{3} >({0}&,const {1}&,const {2}&, const BinaryFunctor&,const int&, const {3}&, const {3}&);}}".format(t[0],t[1],t[2],t[3])

def reductions(vecs):
	L= """template bool has_inf<{0}::value_type,{0}::memory_space_type >(const {0}&);
template bool has_nan<{0}::value_type,{0}::memory_space_type >(const {0}&);
template float minimum<{0}::value_type,{0}::memory_space_type >(const {0}&);
template float maximum<{0}::value_type,{0}::memory_space_type >(const {0}&);
template float sum<{0}::value_type,{0}::memory_space_type >(const {0}&);
template float norm1<{0}::value_type,{0}::memory_space_type >(const {0}&);
template float norm2<{0}::value_type,{0}::memory_space_type >(const {0}&);
template float mean<{0}::value_type,{0}::memory_space_type >(const {0}&);
template float var<{0}::value_type,{0}::memory_space_type >(const {0}&);
template typename {0}::index_type     arg_max<{0}::value_type,{0}::memory_space_type >(const {0}&);
template typename {0}::index_type     arg_min<{0}::value_type,{0}::memory_space_type >(const {0}&);""".split("\n")
	for v in vecs:
		for x in L:
			yield x.format(v);


def tensors(value_types, memory_types):
	for v in value_types:
		for m in memory_types:
			yield vec_t(v,m)

def instantiate_memtype(memtype):
	value_types = "float,unsigned int,int,unsigned char,signed char".split(",")
	tensor_types = [x for x in tensors(value_types, [memtype])]
	scalar_types = "float,int".split(",")

	for s in apply_0ary_functor(zip(tensor_types,[x.value_type() for x in tensor_types])):
		yield s

	# operators which have the same type before and after the operation
	for s in apply_scalar_functor(zip(tensor_types, tensor_types, [x.value_type() for x in tensor_types])):
		yield s
	# boolean predicates
	for s in apply_scalar_functor(zip([vec_t("unsigned char",memtype) for v in tensor_types], tensor_types, [x.value_type() for x in tensor_types])):
		yield s

	# operators where all operands have the same type
	for s in apply_binary_functor(zip(tensor_types,tensor_types,tensor_types,[x.value_type() for x in tensor_types])):
		yield s
	for s in reductions(tensor_types):
		yield s


def f7(seq):
    """ uniquify a list """
    seen = set()
    seen_add = seen.add
    return [ x for x in seq if x not in seen and not seen_add(x)]

if __name__ == "__main__":
	hd_types    = "host_memory_space, dev_memory_space".split(",")
	L = []
	for m in hd_types:
		for s in instantiate_memtype(m):
			L.append(s)
	print "\n".join(f7(L))



