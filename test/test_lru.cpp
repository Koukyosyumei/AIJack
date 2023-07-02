#include "mydb/meta/lru.h"
#include <gtest/gtest.h>

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
