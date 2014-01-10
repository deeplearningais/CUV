#include "allocators.hpp"

#include <boost/format.hpp>
#include <boost/thread/recursive_mutex.hpp>
#include <cuda_runtime_api.h>
#include <sstream>
#include <stdexcept>
#include <thrust/device_ptr.h>
#include <vector>

#include <cuv/tools/cuv_general.hpp>

namespace cuv {

void default_allocator::alloc(void** ptr, size_t memsize, size_t valueSize, host_memory_space) {
    assert(*ptr == 0);
    *ptr = malloc(memsize * valueSize);
    assert(*ptr);
}

void default_allocator::alloc(void** ptr, size_t memsize, size_t valueSize, dev_memory_space) {
    assert(*ptr == 0);
    cuvSafeCall(cudaMalloc(ptr, memsize * valueSize));
    assert(*ptr);
}

void default_allocator::alloc2d(void** ptr, size_t& pitch, size_t height, size_t width, size_t valueSize,
        host_memory_space m) {
    pitch = width * valueSize;
    alloc(ptr, height * width, valueSize, m);
    assert(*ptr);
}

void default_allocator::alloc2d(void** ptr, size_t& pitch, size_t height, size_t width, size_t valueSize,
        dev_memory_space) {
    cuvSafeCall(cudaMallocPitch(ptr, &pitch, valueSize * width, height));
    assert(*ptr);
}

void default_allocator::dealloc(void** ptr, host_memory_space) {
    assert(*ptr != 0);
    free(*ptr);
    *ptr = 0;
}

void default_allocator::dealloc(void** ptr, dev_memory_space) {
    assert(*ptr != 0);
    cuvSafeCall(cudaFree(*ptr));
    *ptr = 0;
}

void cuda_allocator::alloc(void** ptr, size_t memsize, size_t valueSize, host_memory_space) {
    assert(*ptr == 0);
    cuvSafeCall(cudaMallocHost(ptr, memsize * valueSize));
    assert(*ptr != 0);
}

void cuda_allocator::alloc(void** ptr, size_t memsize, size_t valueSize, dev_memory_space m) {
    default_allocator::alloc(ptr, memsize, valueSize, m);
}

void cuda_allocator::alloc2d(void** ptr, size_t& pitch, size_t height, size_t width, size_t valueSize,
        host_memory_space m) {
    pitch = width * valueSize;
    alloc(ptr, height * width, valueSize, m);
}

void cuda_allocator::alloc2d(void** ptr, size_t& pitch, size_t height, size_t width, size_t valueSize,
        dev_memory_space) {
    cuvSafeCall(cudaMallocPitch(ptr, &pitch, valueSize * width, height));
}

void cuda_allocator::dealloc(void** ptr, host_memory_space) {
    assert(*ptr != 0);
    cuvSafeCall(cudaFreeHost(*ptr));
    *ptr = 0;
}

void cuda_allocator::dealloc(void** ptr, dev_memory_space m) {
    default_allocator::dealloc(ptr, m);
}

template<class memory_space>
void pooled_cuda_allocator::collect_garbage(memory_space m) {

    boost::recursive_mutex::scoped_lock pool_lock(get_pool_mutex(m));
    std::map<void*, bool>& pool = get_pool(m);
    std::map<void*, size_t>& pool_sizes = get_pool_sizes(m);

    std::vector<void*> to_delete;
    std::map<void*, bool>::iterator it;
    for (it = pool.begin(); it != pool.end(); it++) {
        if (it->second) {
            to_delete.push_back(it->first);
        }
    }

    for (size_t i = 0; i < to_delete.size(); i++) {
        void* ptr = to_delete[i];
        pool.erase(ptr);
        pool_sizes.erase(ptr);
        cuda_alloc.dealloc(&ptr, m);
    }

    assert(pool_free_count(m) == 0);

    CUV_LOG_DEBUG("garbage collection in memory pool " << m_name << " (" << memtype(m) <<
            "): removed " << to_delete.size() << " elements");
}

template<>
boost::recursive_mutex& pooled_cuda_allocator::get_pool_mutex(dev_memory_space) const {
    // locking/unlocking a mutex does not violate constness of this object
    // unfortunately, the design of the scoped_lock and mutex class requires this hack of a const_cast
    return *(const_cast<boost::recursive_mutex*>(&m_dev_pool_mutex));
}

template<>
boost::recursive_mutex& pooled_cuda_allocator::get_pool_mutex(host_memory_space) const {
    // locking/unlocking a mutex does not violate constness of this object
    // unfortunately, the design of the scoped_lock and mutex class requires this hack of a const_cast
    return *(const_cast<boost::recursive_mutex*>(&m_host_pool_mutex));
}

template<>
std::map<void*, bool>& pooled_cuda_allocator::get_pool(dev_memory_space) {
    return m_dev_pool;
}

template<>
std::map<void*, bool>& pooled_cuda_allocator::get_pool(host_memory_space) {
    return m_host_pool;
}

template<>
const std::map<void*, bool>& pooled_cuda_allocator::get_pool(dev_memory_space) const {
    return m_dev_pool;
}

template<>
const std::map<void*, bool>& pooled_cuda_allocator::get_pool(host_memory_space) const {
    return m_host_pool;
}

template<>
std::map<void*, size_t>& pooled_cuda_allocator::get_pool_sizes(dev_memory_space) {
    return m_dev_pool_sizes;
}

template<>
std::map<void*, size_t>& pooled_cuda_allocator::get_pool_sizes(host_memory_space) {
    return m_host_pool_sizes;
}

template<>
const std::map<void*, size_t>& pooled_cuda_allocator::get_pool_sizes(dev_memory_space) const {
    return m_dev_pool_sizes;
}

template<>
const std::map<void*, size_t>& pooled_cuda_allocator::get_pool_sizes(host_memory_space) const {
    return m_host_pool_sizes;
}

template<class memory_space>
void pooled_cuda_allocator::delete_pool(memory_space m) {

    boost::recursive_mutex::scoped_lock pool_lock(get_pool_mutex(m));
    std::map<void*, bool>& pool = get_pool(m);
    std::map<void*, size_t>& pool_sizes = get_pool_sizes(m);

#ifndef NDEBUG
    size_t free_count = pool_free_count(m);
    size_t count = pool_count(m);
    if (free_count != count) {
        throw std::runtime_error(
                (boost::format("detected potential memory leak in memory pool '%s' (%s): free: %d, count: %d")
                        % m_name % memtype(m) % free_count % count).str());
    }
#endif

    std::map<void*, bool>::iterator it;
    for (it = pool.begin(); it != pool.end(); it++) {
        if (!it->second) {
            throw std::runtime_error(
                    "misuse of allocator. memory was not deallocated before allocator is destroyed. this is a programming failure.");
        }
        void* ptr = it->first;
        cuda_alloc.dealloc(&ptr, m);
    }
    pool.clear();
    pool_sizes.clear();

    CUV_LOG_DEBUG("deleted memory pool " << m_name << " (" << memtype(m) << ")");
}

pooled_cuda_allocator::pooled_cuda_allocator(const std::string& _name) :
        m_name(_name),
                m_dev_pool_mutex(), m_host_pool_mutex(),
                m_dev_pool(), m_dev_pool_sizes(),
                m_host_pool(), m_host_pool_sizes() {
    if (m_name.empty()) {
        std::ostringstream o;
        o << this;
        m_name = o.str();
    }
}

pooled_cuda_allocator::~pooled_cuda_allocator() {
    delete_pool(dev_memory_space());
    delete_pool(host_memory_space());
}

template<class memory_space>
size_t pooled_cuda_allocator::pool_size(memory_space m) const {
    size_t sum = 0;

    boost::recursive_mutex::scoped_lock pool_lock(get_pool_mutex(m));
    const std::map<void*, size_t>& pool_sizes = get_pool_sizes(m);

    std::map<void*, size_t>::const_iterator it;
    for (it = pool_sizes.begin(); it != pool_sizes.end(); it++) {
        sum += it->second;
    }
    return sum;
}

template<class memory_space>
size_t pooled_cuda_allocator::pool_count(memory_space m) const {
    boost::recursive_mutex::scoped_lock pool_lock(get_pool_mutex(m));
    return get_pool_sizes(m).size();
}

template<class memory_space>
size_t pooled_cuda_allocator::pool_free_count(memory_space m) const {
    size_t free = 0;

    boost::recursive_mutex::scoped_lock pool_lock(get_pool_mutex(m));
    const std::map<void*, bool>& pool = get_pool(m);

    std::map<void*, bool>::const_iterator it;
    for (it = pool.begin(); it != pool.end(); it++) {
        if (it->second) {
            free++;
        }
    }
    return free;
}

size_t pooled_cuda_allocator::pool_free_count() const {
    return pool_free_count(dev_memory_space()) + pool_free_count(host_memory_space());
}

size_t pooled_cuda_allocator::pool_size() const {
    return pool_size(dev_memory_space()) + pool_size(host_memory_space());
}

size_t pooled_cuda_allocator::pool_count() const {
    return pool_count(dev_memory_space()) + pool_count(host_memory_space());
}

void pooled_cuda_allocator::alloc(void** ptr, size_t memsize, size_t valueSize, dev_memory_space m) {
    if (memsize * valueSize < MIN_SIZE_DEV) {
        default_alloc.alloc(ptr, memsize, valueSize, m);
    } else {
        alloc_pooled(ptr, memsize, valueSize, m);
    }
}

template<class memory_space>
void pooled_cuda_allocator::alloc_pooled(void** ptr, size_t memsize, size_t valueSize, memory_space m) {

    assert(memsize > 0);

    // try to find memory in the pool that is available and large enough but not too large
    size_t bestSize = 0;
    void* bestPtr = 0;

    boost::recursive_mutex::scoped_lock pool_lock(get_pool_mutex(m));
    std::map<void*, bool>& pool = get_pool(m);
    std::map<void*, size_t>& pool_sizes = get_pool_sizes(m);

    std::map<void*, bool>::iterator it;
    {
        for (it = pool.begin(); it != pool.end(); it++) {
            // available?
            if (!it->second) {
                continue;
            }

            size_t size = pool_sizes[it->first];
            // large enough?
            if (size > memsize * valueSize) {
                if (bestPtr == 0 || size < bestSize) {
                    bestPtr = it->first;
                    bestSize = size;
                }
            }
            // can’t get better
            else if (size == memsize * valueSize) {
                bestPtr = it->first;
                bestSize = size;
                break;
            }
        }

        if (bestPtr) {
            // we take it
            assert(pool[bestPtr]);
            pool[bestPtr] = false;
            *ptr = bestPtr;

            CUV_LOG_DEBUG("reusing " << memsize * valueSize << "/" << pool_sizes[bestPtr] << " bytes in pool "
                    << m_name << " (" << memtype(m) << ")");

            return;
        }
    }

    CUV_LOG_DEBUG("allocating " << memsize << "x" << valueSize << " bytes in pool " << m_name <<
            " (" << memtype(m) << ")");

    // nothing found?
    // allocate new memory
    cuda_alloc.alloc(ptr, memsize, valueSize, m);

    pool[*ptr] = false;
    pool_sizes[*ptr] = memsize * valueSize;

    CUV_LOG_DEBUG("allocated in pool " << m_name << " (" << memtype(m) <<
            "). total bytes: " << pool_size(m) << ". count: " << pool_count(m) << ". free: "
            << pool_free_count(m));

    assert(!pool.empty());
}

void pooled_cuda_allocator::dealloc(void** ptr, dev_memory_space m) {
    do_dealloc(ptr, m);
}

void pooled_cuda_allocator::dealloc(void** ptr, host_memory_space m) {
    do_dealloc(ptr, m);
}

template<class memory_space>
void pooled_cuda_allocator::do_dealloc(void** ptr, memory_space m) {

    assert(*ptr);

    boost::recursive_mutex::scoped_lock pool_lock(get_pool_mutex(m));
    std::map<void*, bool>& pool = get_pool(m);

    std::map<void*, bool>::iterator it = pool.find(*ptr);
    if (it == pool.end()) {
        default_alloc.dealloc(ptr, m);
        return;
    }

    // mark the memory as available
    assert(it->second == false);
    it->second = true;

#ifndef NDEBUG
    std::map<void*, size_t>& pool_sizes = get_pool_sizes(m);

    assert(pool_sizes[*ptr] > 0);

    CUV_LOG_DEBUG(
            "released " << pool_sizes[*ptr] << " bytes in pool " << m_name << " ("
            << memtype(m) << "). total bytes: " << pool_size(m) << ". count: " << pool_count(m) <<", free: " << pool_free_count(m));
#endif

    *ptr = 0;
}

void pooled_cuda_allocator::alloc(void** ptr, size_t memsize, size_t valueSize, host_memory_space m) {
    if (memsize * valueSize < MIN_SIZE_HOST) {
        default_alloc.alloc(ptr, memsize, valueSize, m);
    } else {
        alloc_pooled(ptr, memsize, valueSize, m);
    }
}

void pooled_cuda_allocator::alloc2d(void** ptr, size_t& pitch, size_t height, size_t width, size_t valueSize,
        host_memory_space m) {
    // not yet pooled
    default_alloc.alloc2d(ptr, pitch, height, width, valueSize, m);
}

void pooled_cuda_allocator::alloc2d(void** ptr, size_t& pitch, size_t height, size_t width, size_t valueSize,
        dev_memory_space m) {
    // not yet pooled
    default_alloc.alloc2d(ptr, pitch, height, width, valueSize, m);
}

}

#define CUV_POOLED_CUDA_ALLOCATOR_INST(X) \
    template size_t cuv::pooled_cuda_allocator::pool_count(X) const; \
    template size_t cuv::pooled_cuda_allocator::pool_free_count(X) const; \
    template size_t cuv::pooled_cuda_allocator::pool_size(X) const;

CUV_POOLED_CUDA_ALLOCATOR_INST(cuv::dev_memory_space);
CUV_POOLED_CUDA_ALLOCATOR_INST(cuv::host_memory_space);
