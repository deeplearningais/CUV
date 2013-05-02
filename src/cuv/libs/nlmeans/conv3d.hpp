#ifndef __CONV3D_HPP__
#define __CONV3D_HPP__
#include<cuv/basics/tensor.hpp>

namespace cuv{
	namespace libs
	{
        /// non-local means
		namespace nlmeans
		{
            /**
             * @addtogroup libs
             * @{
             * @addtogroup nonlocal_means
             * @{
             */
			
            /// fix a kernel for horizontal convolution in constant memory
            /// @param src the kernel to be loaded in constant memory
			void setConvolutionKernel_horizontal(const cuv::tensor<float,dev_memory_space>&src);


            /** convolve along 1st axis of a 3D array
             * @param d_Dst where to write results
             * @param d_Src source matrix
             * @param kernel_radius radius of the kernel (kernel size = 2*r+1)
             */
			void convolutionRows(
					cuv::tensor<float,dev_memory_space> &d_Dst,
					const cuv::tensor<float,dev_memory_space> &d_Src,
					const cuv::tensor<float,dev_memory_space> &kernel
					);
            /**
             * convolve along 2nd axis of a 3D array
             * @param d_Dst where to write results
             * @param d_Src source matrix
             * @param kernel_radius radius of the kernel (kernel size = 2*r+1)
             */
			void convolutionColumns(
					cuv::tensor<float,dev_memory_space> & d_Dst,
					const cuv::tensor<float,dev_memory_space> & d_Src,
					const cuv::tensor<float,dev_memory_space> &kernel
					);
            /**
             * convolve along 3rd axis of a 3D array
             * @param d_Dst where to write results
             * @param d_Src source matrix
             * @param kernel_radius radius of the kernel (kernel size = 2*r+1)
             */
			void convolutionDepth(
					cuv::tensor<float,dev_memory_space>& d_Dst,
					const cuv::tensor<float,dev_memory_space>& d_Src,
					const cuv::tensor<float,dev_memory_space> &kernel
					);
            /**
             * determine hessian magnitude of 3D array
             *
             * @warning this is not well tested!
             */
			void hessian
				(
				 cuv::tensor<float,dev_memory_space>& d_output,
				 const cuv::tensor<float,dev_memory_space>& d_gxx,
				 const cuv::tensor<float,dev_memory_space>& d_gxy,
				 const cuv::tensor<float,dev_memory_space>& d_gxz,
				 const cuv::tensor<float,dev_memory_space>& d_gyy,
				 const cuv::tensor<float,dev_memory_space>& d_gyz,
				 const cuv::tensor<float,dev_memory_space>& d_gzz,
				 float sigma
				);
            /**
             * determine hessian orientation of 3D array
             *
             * @warning this is not well tested!
             */
			void hessian_orientation
				(
				 cuv::tensor<float,dev_memory_space> &d_Output,
				 cuv::tensor<float,dev_memory_space> &d_Output_theta,
				 cuv::tensor<float,dev_memory_space> &d_Output_phi,
				 const cuv::tensor<float,dev_memory_space> &d_gxx,
				 const cuv::tensor<float,dev_memory_space> &d_gxy,
				 const cuv::tensor<float,dev_memory_space> &d_gxz,
				 const cuv::tensor<float,dev_memory_space> &d_gyy,
				 const cuv::tensor<float,dev_memory_space> &d_gyz,
				 const cuv::tensor<float,dev_memory_space> &d_gzz,
				 float sigma
				);
            /**
             * @}
             * @}
             */
		}
	}
}

#endif /* __CONV3D_HPP__ */
