#ifndef __IMAGE_PYRAMID_HPP__
#define __IMAGE_PYRAMID_HPP__

#include <boost/ptr_container/ptr_vector.hpp>
#include <vector_ops/vector_ops.hpp>
#include <basics/dense_matrix.hpp>
#include <basics/cuda_array.hpp>

namespace cuv{

	template <class __matrix_type>
	class image_pyramid
	{
	public:
		typedef __matrix_type              matrix_type;
		typedef typename matrix_type::value_type    value_type;
		typedef typename matrix_type::memory_layout memory_layout;
		typedef typename matrix_type::index_type    index_type;
		image_pyramid( int img_h, int img_w, int depth, int dim );
		matrix_type* get(int depth, int channel);
		inline int dim(){return m_dim;}
		inline unsigned int base_h(){return m_base_height;}
		inline unsigned int base_w(){return m_base_width;}
		inline unsigned int depth(){return m_matrices.size();}

		/**
		 * @brief build the pyramid given an image.
		 */
		template<class __arg_matrix_type>
		void build(const __arg_matrix_type& src, const unsigned int interleaved_channels){
			typedef typename __arg_matrix_type::value_type   argval_type;
			typedef typename __arg_matrix_type::memory_space_type argmemspace_type;
			typedef typename __arg_matrix_type::index_type   argindex_type;
			typedef cuda_array<argval_type,argmemspace_type,argindex_type> argca_type;

			if(    src.h() == m_base_height*m_dim // the image dimensions match the input --> just copy.
				&& src.w() == m_base_width
			){
				//std::cout << "Copycase"<<std::endl;
					copy(m_matrices[0].vec(),const_cast<__arg_matrix_type&>(src).vec());
			}
			else if(   interleaved_channels == 4
					&& m_dim                == 3
			){
				//std::cout << "Colorcase"<<std::endl;
					argca_type cpy(src,4);
					matrix_type& dst = m_matrices[0];
					gaussian_pyramid_downsample(dst, cpy, interleaved_channels);
			}
			else if(src.h() > m_base_height*m_dim  // the image dimensions are too large: downsample to 1st level of pyramid
				&&  src.w() > m_base_width
			){
				//std::cout << "Multichannel case"<<std::endl;
				for(int i=0;i<m_dim;i++){
					const __arg_matrix_type view(src.h()/m_dim, src.w(),(argval_type*)src.ptr(),true);
					argca_type cpy(view);
					matrix_type* dstview = get(0,i);
					gaussian_pyramid_downsample(*dstview, cpy,1);
					delete dstview;
				}
			}

			for(int i=1;i<m_matrices.size();i++){
				for(int d=0;d<m_dim;d++){
					matrix_type* srcview = get(i-1,d);
					argca_type cpy(*srcview);
					matrix_type* dstview = get(i,  d);
					gaussian_pyramid_downsample(*dstview, cpy,1);
					delete dstview;
					delete srcview;
				}
			}

		}
	private:
		boost::ptr_vector<matrix_type> m_matrices;
		unsigned int m_dim;
		unsigned int m_base_width;
		unsigned int m_base_height;
	};
	
	template <class __matrix_type>
	typename image_pyramid<__matrix_type>::matrix_type*
	image_pyramid<__matrix_type>::get(int depth, int channel){
		cuvAssert(depth   < m_matrices.size());
		cuvAssert(channel < m_dim);
		matrix_type& mat = m_matrices[depth];
		return new matrix_type(mat.h()/m_dim,mat.w(),mat.ptr()+channel*m_base_height*m_base_width,true);
	}

	template <class __matrix_type>
	image_pyramid<__matrix_type>::image_pyramid( int img_h, int img_w, int depth, int dim )
	:m_base_height(img_h)
	,m_base_width(img_w)
	,m_dim(dim)
	{
		m_matrices.clear();
		for(unsigned int i=0; i<depth;i++){
			//std::cout << "Creating Pyramid Level: "<< img_h<<"*"<<m_dim<<"x"<<img_w<<std::endl;
			m_matrices.push_back(new matrix_type(img_h*m_dim, img_w));
			img_h=ceil(img_h/2.f);
			img_w=ceil(img_w/2.f);
		}
	}

template<class T,class S, class I>
void gaussian_pyramid_downsample(
	dense_matrix<T,row_major,S,I>& dst,
	const cuda_array<T,S,I>& src,
	const unsigned int interleaved_channels
);
template<class T,class S, class I>
void gaussian_pyramid_upsample(
	dense_matrix<T,row_major,S,I>& dst,
	const cuda_array<T,S,I>& src
);

}
#endif /* __IMAGE_PYRAMID_HPP__ */
