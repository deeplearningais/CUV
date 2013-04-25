#include "theano_ops.hpp" 
#include "../../../3rd_party/cuda_ndarray/conv.cu"
#include "../../../3rd_party/cuda_ndarray/cuda_ndarray.cuh"
#include <vector>
CudaNdarray* cnda_flip_dims(CudaNdarray* self, bool * flip_dims);

int  CudaNdarray_reshape_2(CudaNdarray * self, CudaNdarray * rval, int * rval_dims, unsigned int rval_nd);
namespace cuv{

namespace theano_ops{

PyMODINIT_FUNC initcuda_ndarray(void);
PyObject * CudaNdarray_dimshuffle(PyObject* _unused, PyObject* args);


void initcuda(){
    std::cout << "init cuda and py" << std::endl;
    Py_Initialize();
    initcuda_ndarray();
}

void finalize_cuda(){
   Py_Finalize();
}

void view(CudaNdarray*& nda, cuv::tensor<float,cuv::dev_memory_space>& ct){
    int nd = ct.ndim();
    nda = (CudaNdarray*)CudaNdarray_New(nd); // same number of dimensions
    int size = 1; // strides in contiguous tensor
    for(int i=nd-1;i>=0;--i){
        /*CudaNdarray_set_stride(nda, i, ct.shape(i)==1 ? 0: size);*/
        CudaNdarray_set_stride(nda, i, ct.stride(i));
        CudaNdarray_set_dim(nda, i, ct.shape(i));
        size = size * ct.shape(i);
    }
    cnda_copy_structure_to_device(nda);
    nda->devdata = ct.ptr();
}


void dim_shuffle_vec(cuv::tensor<float,cuv::dev_memory_space>& dst, const cuv::tensor<float,cuv::dev_memory_space>& src, std::vector<int> pattern){
    unsigned int size = pattern.size();
    int new_dims[size];
    for (unsigned int i = 0; i < size; ++i)
    {
        new_dims[i] = pattern[i];
    }
    dim_shuffle2(dst,src, new_dims, size);
}



void dim_shuffle2(cuv::tensor<float,cuv::dev_memory_space>& dst, const cuv::tensor<float,cuv::dev_memory_space>& src_, int new_dims[], unsigned int size){
    cuv::tensor<float,cuv::dev_memory_space> src = src_;
    assert(src.ndim() == size);
    CudaNdarray *csrc;
    CudaNdarray *cdst;
    view(csrc, src);

    // shuffles the dims
    if(0 != CudaNdarray_dimshuffle(csrc, size,new_dims))
        throw std::runtime_error("could not dimshuffle tensor");

    // determines a new shape of a tensor
    std::vector<unsigned int> new_shape(size);
    int shape[size];
    for(unsigned int i = 0; i < size; i++){
       new_shape[i] = src.shape(new_dims[i]);
       shape[i] = new_shape[i];
    }

    dst.reshape(new_shape);
    view(cdst, dst);
    // reshapes to row_major
    if(1 !=CudaNdarray_reshape_2(csrc,cdst, shape, size))
       throw std::runtime_error("could not reshape tensor");


    Py_DECREF(csrc);
    Py_DECREF(cdst);
}

void flip_dims(cuv::tensor<float,cuv::dev_memory_space>& dst, const cuv::tensor<float,cuv::dev_memory_space>& src_, bool * flip_dims){
    cuv::tensor<float,cuv::dev_memory_space> src = src_;
    CudaNdarray *cout;
    CudaNdarray *cflipped;
    CudaNdarray *cdst;
    view(cout, src);
    view(cdst, dst);

    /*cflipped = cnda_flip_dims2and3(cout);*/
    cflipped = cnda_flip_dims(cout, flip_dims);


    unsigned int size = dst.ndim();
    int shape[size];
    for(unsigned int i = 0; i < size; i++){
        shape[i] = dst.shape(i);
    }

    if(1 !=CudaNdarray_reshape_2(cflipped,cdst, shape, size)){
      Py_DECREF(cout);
      Py_DECREF(cflipped);
      Py_DECREF(cdst);
      throw std::runtime_error("could not reshape tensor");
    }


    Py_DECREF(cout);
    Py_DECREF(cflipped);
    Py_DECREF(cdst);
}


}

}
