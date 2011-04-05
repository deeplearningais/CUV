#include <cuv/basics/dense_matrix.hpp>
namespace cuv{
	namespace cimg{
		template<class V, class M, class I>
		void show(const dense_matrix<V,host_memory_space,M,I>& m, const std::string& name);

		template<class V, class M, class I>
		void load(      dense_matrix<V,host_memory_space,M,I>& m, const std::string& name);

		template<class V, class M, class I>
		void save(      dense_matrix<V,host_memory_space,M,I>& m, const std::string& name);
	}
	
}

