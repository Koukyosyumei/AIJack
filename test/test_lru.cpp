#include "mydb/meta/lru.h"
#include <gtest/gtest.h>

TEST(LruTest, InsertAndGet) {
  Lru<int, int> lru(1);
  lru.Insert(10, 100);
  int v = lru.Get(10);
  ASSERT_EQ(v, 100);
}
