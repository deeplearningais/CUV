#define BOOST_TEST_MODULE example

#include <boost/format.hpp>
#include <boost/test/included/unit_test.hpp>
#include <boost/thread/mutex.hpp>
#include <tbb/parallel_for_each.h>

#include <cuv/basics/allocators.hpp>
#include <cuv/basics/reference.hpp>

using namespace cuv;

BOOST_AUTO_TEST_SUITE(allocators_test)

template<class memory_space>
static void test_pooled_allocator() {
    memory_space m;
    pooled_cuda_allocator allocator;
    int* ptr1 = 0;
    int* ptr2 = 0;

    const int NUM_ELEMENTS = 10000;

    allocator.alloc(reinterpret_cast<void**>(&ptr1), NUM_ELEMENTS, sizeof(int), m);
    allocator.alloc(reinterpret_cast<void**>(&ptr2), NUM_ELEMENTS, sizeof(int), m);
    BOOST_CHECK(ptr1);
    BOOST_CHECK(ptr2);
    BOOST_CHECK_NE(ptr1, ptr2);
    BOOST_CHECK_EQUAL(allocator.pool_count(m), 2);
    BOOST_CHECK_EQUAL(allocator.pool_free_count(m), 0);
    BOOST_CHECK_EQUAL(allocator.pool_size(m), 2 * NUM_ELEMENTS * sizeof(int));

    for (size_t i = 0; i < 10000; i++) {
        reference<int, memory_space> ref(ptr1 + i);
        ref = i;
        BOOST_CHECK_EQUAL(static_cast<int>(ref), i);
    }

    allocator.dealloc(reinterpret_cast<void**>(&ptr1), m);
    BOOST_CHECK(ptr1 == 0);

    BOOST_CHECK_EQUAL(allocator.pool_count(m), 2);
    BOOST_CHECK_EQUAL(allocator.pool_free_count(m), 1);
    BOOST_CHECK_EQUAL(allocator.pool_size(m), 2 * NUM_ELEMENTS * sizeof(int));

    for (size_t i = 0; i < 10000; i++) {
        reference<int, memory_space> ref(ptr2 + i);
        ref = i + 100;
        BOOST_CHECK_EQUAL(static_cast<int>(ref), i + 100);
    }

    allocator.dealloc(reinterpret_cast<void**>(&ptr2), m);

    BOOST_CHECK_EQUAL(allocator.pool_free_count(), allocator.pool_count());
}

template<class memory_space>
static void test_pooled_allocator_multi_threaded() {
    memory_space m;
    pooled_cuda_allocator allocator("allocator_multi_threaded");

    const int ALLOC_SIZE = pooled_cuda_allocator::MIN_SIZE_HOST;

    // boost-test is not thread-safe
    boost::mutex boost_mutex;

    std::vector<void*> pointers(1000, NULL);
    tbb::parallel_for_each(pointers.begin(), pointers.end(),
            [&](void*& ptr) {
                size_t pool_size = allocator.pool_size(m);
                void* ptr1 = NULL;
                void* ptr2 = NULL;
                allocator.alloc(&ptr1, ALLOC_SIZE, 1, m);
                allocator.alloc(&ptr2, 1, 1, m);
                allocator.alloc(&ptr, ALLOC_SIZE, 1, m);

                {
                    boost::mutex::scoped_lock lock(boost_mutex);
                    BOOST_REQUIRE(ptr1);
                    BOOST_REQUIRE(ptr2);
                    BOOST_REQUIRE(ptr);

                    BOOST_REQUIRE_NE(ptr1, ptr2);
                    BOOST_REQUIRE_NE(ptr2, ptr);
                    BOOST_REQUIRE_NE(ptr1, ptr);

                    BOOST_REQUIRE_GE(allocator.pool_count(m), 2lu);
                }

                allocator.dealloc(&ptr1, m);
                allocator.dealloc(&ptr2, m);

                {
                    boost::mutex::scoped_lock lock(boost_mutex);
                    BOOST_REQUIRE_GE(allocator.pool_size(m), pool_size);
                    BOOST_REQUIRE_GE(allocator.pool_free_count(m), 0lu);
                }
            });

    for (size_t i = 0; i < pointers.size(); i++) {
        BOOST_REQUIRE(pointers[i]);
    }

    BOOST_CHECK_GE(allocator.pool_size(m), pointers.size() * ALLOC_SIZE);
    BOOST_CHECK_LE(allocator.pool_count(m), 10 * pointers.size());

    size_t count = allocator.pool_count(m);
    BOOST_CHECK_GE(count, pointers.size());

    tbb::parallel_for_each(pointers.begin(), pointers.end(), [&](void*& ptr) {
        allocator.dealloc(&ptr, m);

        {
            boost::mutex::scoped_lock lock(boost_mutex);
            BOOST_CHECK(!ptr);
        }
    });

    BOOST_CHECK_EQUAL(allocator.pool_free_count(), allocator.pool_count());
}

template<class memory_space>
static void test_pooled_allocator_garbage_collection() {
    memory_space m;
    pooled_cuda_allocator allocator;
    int* ptr1 = 0;
    int* ptr2 = 0;
    allocator.alloc(reinterpret_cast<void**>(&ptr1), 10000, sizeof(int), m);
    allocator.alloc(reinterpret_cast<void**>(&ptr2), 10000, sizeof(int), m);

    BOOST_CHECK_EQUAL(allocator.pool_count(m), 2);

    allocator.dealloc(reinterpret_cast<void**>(&ptr1), m);

    BOOST_CHECK_EQUAL(allocator.pool_count(m), 2);
    BOOST_CHECK_EQUAL(allocator.pool_free_count(m), 1);

    allocator.garbage_collection();

    BOOST_CHECK_EQUAL(allocator.pool_count(m), 1);
    BOOST_CHECK_EQUAL(allocator.pool_free_count(m), 0);

    allocator.dealloc(reinterpret_cast<void**>(&ptr2), m);

    BOOST_CHECK_EQUAL(allocator.pool_count(m), 1);
    BOOST_CHECK_EQUAL(allocator.pool_free_count(m), 1);

    allocator.garbage_collection();

    BOOST_CHECK_EQUAL(allocator.pool_count(m), 0);
    BOOST_CHECK_EQUAL(allocator.pool_free_count(m), 0);
}

BOOST_AUTO_TEST_CASE( pooled_cuda_allocator_test_simple ) {
    test_pooled_allocator<dev_memory_space>();
    test_pooled_allocator<host_memory_space>();
}

BOOST_AUTO_TEST_CASE( pooled_cuda_allocator_test_multithreaded ) {
    test_pooled_allocator_multi_threaded<dev_memory_space>();
    test_pooled_allocator_multi_threaded<host_memory_space>();
}

BOOST_AUTO_TEST_CASE( pooled_cuda_allocator_test_garbage_collection ) {
    test_pooled_allocator_garbage_collection<dev_memory_space>();
    test_pooled_allocator_garbage_collection<host_memory_space>();
}
BOOST_AUTO_TEST_SUITE_END()
