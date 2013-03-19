/*#include <ctime>*/
#include <unistd.h>
/*#include <sys/time.h>*/

/*#include <cuv.hpp>*/
#include <cuv/convolution_ops/convolution_ops_theano.hpp>
#include"conv.cu"
#include"cuda_ndarray.cuh"

CudaNdarray* cnda_flip_dims2and3(CudaNdarray* self);
namespace cuv{

namespace theano_conv{


PyMODINIT_FUNC initcuda_ndarray(void);
PyObject * CudaNdarray_Dimshuffle(PyObject* _unused, PyObject* args);

void initcuda(){
    std::cout << "init cuda and py" << std::endl;
    Py_Initialize();
    initcuda_ndarray();
}

void finalize_cuda(){
   Py_Finalize();
}

void empty_like(CudaNdarray*& nda, cuv::tensor<float,cuv::dev_memory_space>& ct){
    int nd = ct.ndim();
    nda = (CudaNdarray*)CudaNdarray_New(nd); // same number of dimensions
    CudaNdarray_alloc_contiguous(nda, nd, &ct.shape()[0]);
    cnda_copy_structure_to_device(nda);
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

void convolve_2d(cuv::tensor<float,cuv::dev_memory_space>& out, const cuv::tensor<float,cuv::dev_memory_space>& images, const cuv::tensor<float,cuv::dev_memory_space>& kern, const std::string& mode, int version){
    cuvAssert(images.shape(0)==out.shape(0));
    cuvAssert(kern.shape(0)==out.shape(1));
    if(mode=="valid"){
        cuvAssert(out.shape(2) == images.shape(2)-kern.shape(2)+1);
        cuvAssert(out.shape(3) == images.shape(3)-kern.shape(3)+1);
    }else if(mode=="full"){
        cuvAssert(out.shape(2) == images.shape(2)+kern.shape(2)-1);
        cuvAssert(out.shape(3) == images.shape(3)+kern.shape(3)-1);
    }else{
        throw std::runtime_error("undefined convolution mode `"+mode+"'");
    }

    cuv::tensor<float,cuv::dev_memory_space> img = images;
    cuv::tensor<float,cuv::dev_memory_space> krn = kern;
    CudaNdarray *cimages, *ckern, *cout;

    view(cimages, img);
    view(ckern, krn);
    view(cout, out);

    if(mode=="valid")
        CudaNdarray_conv_valid(cimages, ckern, cout, 1, 1, version, 0); // ssrows sscols version verbose
    else
        CudaNdarray_conv_full(cimages, ckern, cout, 1, 1, version, 0); // ssrows sscols version verbose
    cuvAssert(CudaNdarray_is_c_contiguous(cout));
    Py_DECREF(cimages);
    Py_DECREF(ckern);
    Py_DECREF(cout);
}




void d_convolve_d_images(cuv::tensor<float,cuv::dev_memory_space>& images, const cuv::tensor<float,cuv::dev_memory_space>& out, const cuv::tensor<float,cuv::dev_memory_space>& kern, const std::string& mode){
    cuvAssert(images.shape(0)==out.shape(0));
    cuvAssert(kern.shape(0)==out.shape(1));
    if(mode=="valid"){
        cuvAssert(out.shape(2) == images.shape(2)-kern.shape(2)+1);
        cuvAssert(out.shape(3) == images.shape(3)-kern.shape(3)+1);
    }else if(mode=="full"){
        cuvAssert(out.shape(2) == images.shape(2)+kern.shape(2)-1);
        cuvAssert(out.shape(3) == images.shape(3)+kern.shape(3)-1);
    }else{
        throw std::runtime_error("undefined convolution mode `"+mode+"'");
    }

    cuv::tensor<float,cuv::dev_memory_space> output = out;
    cuv::tensor<float,cuv::dev_memory_space> kernel = kern;
    CudaNdarray *cimages, *ckern, *cout;

    view(cimages, images);
    view(ckern, kernel);
    view(cout, output);

    int kern_dims[] = {1,0,2,3};
    if(0 != CudaNdarray_dimshuffle(ckern, 4,kern_dims))
        throw std::runtime_error("could not dimshuffle tensor");
    CudaNdarray *cflipped_kern = cnda_flip_dims2and3(ckern);

    std::string _mode = mode == "valid" ? "full" : "valid";
    if(_mode=="valid")
        CudaNdarray_conv_valid(cout, cflipped_kern, cimages, 1, 1, -1, 0); // ssrows sscols version verbose
    else
        CudaNdarray_conv_full(cout, cflipped_kern, cimages, 1, 1, -1, 0); // ssrows sscols version verbose

    cuvAssert(CudaNdarray_is_c_contiguous(cimages));

    Py_DECREF(cflipped_kern);
    Py_DECREF(cimages);
    Py_DECREF(ckern);
    Py_DECREF(cout);
}

void d_convolve_d_kern(cuv::tensor<float,cuv::dev_memory_space>& kern_, const cuv::tensor<float,cuv::dev_memory_space>& images, const cuv::tensor<float,cuv::dev_memory_space>& out, const std::string& mode){
    cuvAssert(images.shape(0)==out.shape(0));
    cuvAssert(kern_.shape(0)==out.shape(1));
    if(mode=="valid"){
        cuvAssert(out.shape(2) == images.shape(2)-kern_.shape(2)+1);
        cuvAssert(out.shape(3) == images.shape(3)-kern_.shape(3)+1);
    }else if(mode=="full"){
        cuvAssert(out.shape(2) == images.shape(2)+kern_.shape(2)-1);
        cuvAssert(out.shape(3) == images.shape(3)+kern_.shape(3)-1);
    }else{
        throw std::runtime_error("undefined convolution mode `"+mode+"'");
    }

    cuv::tensor<float,cuv::dev_memory_space> img = images;
    cuv::tensor<float,cuv::dev_memory_space> output = out;
    CudaNdarray *cimages, *ckern, *ckern_, *cout;

    view(cimages, img);
    view(ckern_,      kern_);
    view(cout, output);

    

    int new_dims[] = {1,0,2,3};
    if(0 != CudaNdarray_dimshuffle(cout, 4,new_dims))              // shuffle out
        throw std::runtime_error("could not dimshuffle tensor");
    if(0 != CudaNdarray_dimshuffle(cimages, 4,new_dims))           // shuffle images
        throw std::runtime_error("could not dimshuffle tensor");

    if(mode=="valid") {
        CudaNdarray *cflipped_cout = cnda_flip_dims2and3(cout);       // flip kern
        ckern = (CudaNdarray*) CudaNdarray_New();
        {
            // create ckern like kern_, but switch the 1st and 2nd dimension
            const int ckern_dim[] = {kern_.shape(1), kern_.shape(0), kern_.shape(2), kern_.shape(3)};
            CudaNdarray_alloc_contiguous(ckern, 4, ckern_dim);
        }

        CudaNdarray_conv_valid(cimages, cflipped_cout, ckern,  1, 1, -1, 0); // ssrows sscols version verbose

        Py_DECREF(cflipped_cout);
    }
    else {
        view(ckern, kern_);
        CudaNdarray *cflipped_cimg = cnda_flip_dims2and3(cimages);       // flip kern
        CudaNdarray_conv_valid(cout, cflipped_cimg, ckern, 1, 1, -1, 0); // ssrows sscols version verbose
        Py_DECREF(cflipped_cimg);
    }

    if(mode=="valid"){
        if(0 != CudaNdarray_dimshuffle(ckern, 4,new_dims))              // shuffle result
            throw std::runtime_error("could not dimshuffle tensor");

        CudaNdarray *cflipped_ckern = cnda_flip_dims2and3(ckern);       // flip kern

        if(CudaNdarray_CopyFromCudaNdarray(ckern_, cflipped_ckern))// ckern is not c-contiguous, so we need to copy :/
            throw std::runtime_error("could not copy ckern");

        Py_DECREF(ckern);
        ckern = cflipped_ckern;
    }
    
        
    cuvAssert(CudaNdarray_is_c_contiguous(ckern_));

    Py_DECREF(cimages);
    Py_DECREF(ckern);
    Py_DECREF(ckern_);
    Py_DECREF(cout);
}

}
}
