#include <cuv/basics/tensor.hpp>
namespace cuv{
	namespace cimg{
		template<class V, class M>
		void show(const tensor<V,host_memory_space,M>& m, const std::string& name);

		template<class V, class M>
		void load(      tensor<V,host_memory_space,M>& m, const std::string& name);

		template<class V, class M>
		void save(      tensor<V,host_memory_space,M>& m, const std::string& name);
	}
	
}

