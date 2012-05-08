.PHONY:tags filelist cs ctags_hilite
all: filelist tags cs 
filelist:
	find . -name '*.cpp' -or -name '*.cu' -or -name '*.h' -or -name '*.hpp'  | grep -v ConvNetwork | grep -v Mlp |grep -v build > cscope.files
tags: Makefile
	ctags -L cscope.files -R --c++-kinds=+p --fields=+iaS --extra=+q -I __global__ -I __device__ -I __shared__ --langmap='c++:+.cu.cuh' .
cs:
	cscope -b
bob: cs
	#bob -L cscope.files --make-ctags
	bob -L cscope.files --call-tags
ctags_hilite:
	vim '+UpdateTypesFile!' '+quit'
