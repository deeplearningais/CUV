#define cimg_use_png 1
#include <CImg.h>
#include <cuv/tools/meta_programming.hpp>
#include <cuv/matrix_ops/matrix_ops.hpp>
#include <cuv/tensor_ops/tensor_ops.hpp>
#include <cuv/convert/convert.hpp>
#include <cuv/libs/cimg/cuv_cimg.hpp>

namespace cuv{
	namespace libs{
	namespace cimg{
		using namespace cimg_library;
		template<class V,class M>
		void load(tensor<V,host_memory_space,M>& m, const std::string& name, column_major ){
			CImg<V> img(name.c_str());
			if(img.spectrum()>1){
				tensor<V,host_memory_space,row_major> m2(indices[index_range(0,img.width())][index_range(0,img.height())][index_range(0,img.spectrum())],(V*) img.data());
				m = m2;
			}else{
				tensor<V,host_memory_space,row_major> m2(indices[index_range(0,img.height())][index_range(0,img.width())],(V*) img.data());
				m = m2;
			}
		}
		template<class V,class M>
		void load(tensor<V,host_memory_space,M>& m, const std::string& name, row_major ){
			CImg<V> img(name.c_str());
			if(img.spectrum()>1){
				tensor<V,host_memory_space,row_major> m2(indices[index_range(0,img.spectrum())][index_range(0,img.height())][index_range(0,img.width())],(V*) img.data());
				m = m2;
			}
			else{
				tensor<V,host_memory_space,row_major> m2(indices[index_range(0,img.height())][index_range(0,img.width())],(V*) img.data());
				m = m2;
			}
		}
		template<class V,class M>
		void load(tensor<V,host_memory_space,M>& m, const std::string& name){
			load(m,name,M());
		}

		template<class V,class M>
		void save(tensor<V,host_memory_space,M>& m, const std::string& name ){
			typedef typename unconst<V>::type Vuc;
			cuvAssert(m.ndim()==2 || m.ndim()==3);

			if(IsSame<M,column_major>::Result::value){
				if(m.ndim()==2){
					CImg<Vuc> img(const_cast<Vuc*>(m.ptr()),m.shape()[0],m.shape()[1]);
					img.save(name.c_str());
				}else if(m.ndim()==3){
					CImg<Vuc> img(const_cast<Vuc*>(m.ptr()),m.shape()[0],m.shape()[1],1,m.shape()[2]);
					img.save(name.c_str());
				}
			}else{
				if(m.ndim()==2){
					CImg<Vuc> img(const_cast<Vuc*>(m.ptr()),m.shape()[1],m.shape()[0]);
					img.save(name.c_str());
				}else if(m.ndim()==3){
					CImg<Vuc> img(const_cast<Vuc*>(m.ptr()),m.shape()[2],m.shape()[1],1,m.shape()[0]);
					img.save(name.c_str());
				}
			}
		}
		template<class V,class M>
		void show(const tensor<V,host_memory_space,M>& m, const std::string& name ){
			if(IsSame<M,column_major>::Result::value){
                                        show(*transposed_view(m), name);
			}else{
				typedef typename unconst<V>::type Vuc;
                                cuvAssert(m.ndim()==2 || m.ndim()==3);
				CImgDisplay disp;

				if(m.ndim()==2){
					CImg<Vuc> img(const_cast<Vuc*>(m.ptr()),m.shape()[1],m.shape()[0]);
					disp.assign(img, name.c_str());
				}else if(m.ndim()==3){
					CImg<Vuc> img(const_cast<Vuc*>(m.ptr()),m.shape()[2],m.shape()[1],1,m.shape()[0]);
					disp.assign(img, name.c_str());
				}
				
				while (!disp.is_closed() && !disp.is_keyQ() && !disp.is_keyESC()) {
					/*img.resize(disp.display(img).resize(false).wait());*/
					/*if (disp.is_keyCTRLLEFT() && disp.is_keyF())*/
					/*        disp.resize(m.shape()[1],m.shape()[0],false).toggle_fullscreen(false);*/
				}
			}
		}

#define LOAD_INST_FULL(V,M) \
		template void load<V,M>(tensor<V,host_memory_space,M>&, const std::string&); \
		template void save<V,M>(tensor<V,host_memory_space,M>&, const std::string&);
#define LOAD_INST_V(V) \
		LOAD_INST_FULL(V,row_major)    \
		LOAD_INST_FULL(V,column_major)

#define INST_FULL(V,M) \
		template void show<V,M>(const tensor<V,host_memory_space,M>&, const std::string&);
#define INST_V(V) \
		INST_FULL(V,column_major) \
		INST_FULL(V,row_major)

		INST_V(float);
		INST_V(unsigned char);

		LOAD_INST_V(float);
		LOAD_INST_V(unsigned char);
		
	} // namespace cimg
} // namespace libs
} // namespace cuv
