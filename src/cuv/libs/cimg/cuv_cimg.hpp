#ifndef __CUV_CIMG_HPP__
#     define __CUV_CIMG_HPP__

#include <cuv/basics/tensor.hpp>
namespace cuv{
	namespace libs{
		/// interaction with CImg library
	namespace cimg{
		/**
		 * @addtogroup libs
		 * @{
		 * @defgroup cimg Loading, saving and showing images using the CImg library
		 * @{
		 */

		/**
		 * show an image stored in a tensor
		 *
		 * the image has to be either a 2d tensor or a 3d-tensor where the 1st dimension has size 3 (color).
		 * @param m the image
		 * @param name name of the window
		 */
		template<class V, class M>
		void show(const tensor<V,host_memory_space,M>& m, const std::string& name);

		/**
		 * load an image in a tensor
		 *
		 * the image will be either a 2d tensor or a 3d-tensor where the 1st dimension has size 3 (color).
		 * @param m the image
		 * @param name name of the file
		 */
		template<class V, class M>
		void load(      tensor<V,host_memory_space,M>& m, const std::string& name);

		/**
		 * save an image stored in a tensor
		 *
		 * the image has to be either a 2d tensor or a 3d-tensor where the 1st dimension has size 3 (color).
		 * @param m the image
		 * @param name name of the file
		 */
		template<class V, class M>
		void save(      tensor<V,host_memory_space,M>& m, const std::string& name);
		/**
		 * @}
		 * @}
		 */
	}
	}
	
}

#endif /* __CUV_CIMG_HPP__ */
