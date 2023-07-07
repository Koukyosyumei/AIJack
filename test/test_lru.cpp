#include "mydb/utils/lru.h"
#include <gtest/gtest.h>
#include <mutex>
#include <thread>

TEST(LruTest, InsertAndGet) {
  Lru<int, int> lru(1);
  lru.Insert(10, 100);
  int v = lru.Get(10);
  ASSERT_EQ(v, 100);
}

TEST(LruTest, Evicted) {
  Lru<int, int> lru(1, -99);
  lru.Insert(10, 100);
  lru.Insert(11, 110);

  int recent = lru.Get(11);
  ASSERT_EQ(recent, 110);

  int old = lru.Get(10);
  ASSERT_EQ(old, -99);
}

TEST(LruTest, Concurrency) {
  Lru<int, int> lru(1000);
  std::mutex mutex;

  std::vector<std::thread> threads;
  const int numThreads = 1000;

  for (int i = 0; i < numThreads; i++) {
    threads.emplace_back([&]() {
      std::lock_guard<std::mutex> lock(mutex);
      lru.Insert(i, i);
    });
  }

  for (auto &thread : threads) {
    thread.join();
  }

  ASSERT_EQ(numThreads, lru.Len());
}
