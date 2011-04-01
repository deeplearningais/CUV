#define cimg_use_png 1
#include <CImg.h>
#include <cuv/tools/meta_programming.hpp>
#include <cuv/matrix_ops/matrix_ops.hpp>
#include <cuv/tensor_ops/tensor_ops.hpp>
#include <cuv/convert/convert.hpp>
#include <cuv/libs/cimg/cuv_cimg.hpp>

namespace cuv{
	namespace cimg{
		using namespace cimg_library;
		template<class V,class M,class I>
		void load(dense_matrix<V,host_memory_space,M,I>& m, const std::string& name, column_major ){
			CImg<V> img(name.c_str());
			dense_matrix<V,host_memory_space,row_major,I> m2(img.height(),img.width(),(V*) img.data(),true);
			/*cuv::copy(m,m2);*/
			cuvAssert(false); //copy not implemented yet
		}
		template<class V,class M,class I>
		void load(dense_matrix<V,host_memory_space,M,I>& m, const std::string& name, row_major ){
			CImg<V> img(name.c_str());
			dense_matrix<V,host_memory_space,row_major,I> m2(img.width(),img.height(),(V*) img.data(),true);
			m = m2;
		}
		template<class V,class M,class I>
		void load(dense_matrix<V,host_memory_space,M,I>& m, const std::string& name){
			load(m,name,M());
		}

		template<class V,class M,class I>
		void save(dense_matrix<V,host_memory_space,M,I>& m, const std::string& name ){
			typedef typename unconst<V>::type Vuc;
			if(IsSame<M,column_major>::Result::value){
				CImg<Vuc> img(const_cast<Vuc*>(m.ptr()),m.h(),m.w());
				img.save(name.c_str());
			}else{
				CImg<Vuc> img(const_cast<Vuc*>(m.ptr()),m.w(),m.h());
				img.save(name.c_str());
			}
		}
		template<class V,class M, class I>
		void show(const dense_matrix<V,host_memory_space,M,I>& m, const std::string& name ){
			if(IsSame<M,column_major>::Result::value){
					dense_matrix<const V,host_memory_space,row_major,I> m2(m.w(), m.h(),m.ptr(),true);
					show(m2, name);
			}else{
				typedef typename unconst<V>::type Vuc;
				CImg<Vuc> img(const_cast<Vuc*>(m.ptr()),m.w(),m.h());
				CImgDisplay disp(img, name.c_str());
				
				while (!disp.is_closed() && !disp.is_keyQ() && !disp.is_keyESC()) {
					img.resize(disp.display(img).resize(false).wait());
					if (disp.is_keyCTRLLEFT() && disp.is_keyF())
						disp.resize(m.w(),m.h(),false).toggle_fullscreen(false);
				}
			}
		}

#define LOAD_INST_FULL(V,M,I) \
		template void load<V,M,I>(dense_matrix<V,host_memory_space,M,I>&, const std::string&); \
		template void save<V,M,I>(dense_matrix<V,host_memory_space,M,I>&, const std::string&);
#define LOAD_INST_V(V) \
		LOAD_INST_FULL(V,row_major,unsigned int)    \
		LOAD_INST_FULL(V,column_major,unsigned int)

#define INST_FULL(V,M,I) \
		template void show<V,M,I>(const dense_matrix<V,host_memory_space,M,I>&, const std::string&);
#define INST_V(V) \
		INST_FULL(V,column_major,  unsigned int) \
		INST_FULL(V,row_major,     unsigned int)

		INST_V(float);
		INST_V(unsigned char);

		LOAD_INST_V(float);
		LOAD_INST_V(unsigned char);
		
	} // namespace cimg
} // namespace cuv
