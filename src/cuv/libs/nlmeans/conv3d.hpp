#ifndef __CONV3D_HPP__
#define __CONV3D_HPP__
#include<cuv/basics/tensor.hpp>

namespace cuv{
	namespace libs
	{
		namespace nlmeans
		{
			
			void setConvolutionKernel_horizontal(const cuv::tensor<float,host_memory_space>&src);
			void setConvolutionKernel_vertical(const cuv::tensor<float,host_memory_space>&src);
			void setConvolutionKernel_depth(const cuv::tensor<float,host_memory_space>&src);

			void convolutionRows(
					cuv::tensor<float,dev_memory_space> &d_Dst,
					const cuv::tensor<float,dev_memory_space> &d_Src,
					int kernel_radius
					);
			void convolutionColumns(
					cuv::tensor<float,dev_memory_space> & d_Dst,
					const cuv::tensor<float,dev_memory_space> & d_Src,
					int kernel_radius
					);
			void convolutionDepth(
					cuv::tensor<float,dev_memory_space>& d_Dst,
					const cuv::tensor<float,dev_memory_space>& d_Src,
					int kernel_radius
					);
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
		}
	}
}

#endif /* __CONV3D_HPP__ */
