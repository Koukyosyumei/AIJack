#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "src/utils/bptree.h"

TEST(BPlusTreeMapTest, NoSplit) {
  BPlusTreeMap<int, int> btree;

  btree.Insert(1, 1);
  btree.Insert(2, 2);

  auto found = btree.Find(1);
  ASSERT_TRUE(found.first);

  found = btree.Find(2);
  ASSERT_TRUE(found.first);

  found = btree.Find(3);
  ASSERT_FALSE(found.first);
}

TEST(BPlusTreeMapTest, SplitParent) {
  BPlusTreeMap<int, int> btree;

  btree.Insert(1, 1);
  btree.Insert(2, 2);
  btree.Insert(3, 3);

  auto found = btree.Find(1);
  ASSERT_TRUE(found.first);

  found = btree.Find(2);
  ASSERT_TRUE(found.first);

  found = btree.Find(3);
  ASSERT_TRUE(found.first);

  ASSERT_EQ(btree.root->ks[0], 2);
  ASSERT_EQ(btree.root->children[0]->vs[0], 1);
  ASSERT_EQ(btree.root->children[1]->vs[0], 2);
}

TEST(BPlusTreeMapTest, GreaterEq) {
  BPlusTreeMap<int, int> btree;

  btree.Insert(11, 1);
  btree.Insert(2, 2);
  btree.Insert(32, 3);
  btree.Insert(1, 4);
  btree.Insert(5, 5);
  btree.Insert(3, 6);
  btree.Insert(4, 7);
  btree.Insert(8, 8);
  btree.Insert(10, 9);

  std::vector<int> gt = {5, 8, 9, 1, 3};
  std::vector<int> result = btree.FindGreaterEq(5);
  ASSERT_EQ(result.size(), gt.size());
  for (int i = 0; i < gt.size(); i++) {
    ASSERT_EQ(result[i], gt[i]);
  }
}

TEST(BPlusTreeMapTest, LessEq) {
  BPlusTreeMap<int, int> btree;

  btree.Insert(11, 1);
  btree.Insert(2, 2);
  btree.Insert(32, 3);
  btree.Insert(1, 4);
  btree.Insert(5, 5);
  btree.Insert(3, 6);
  btree.Insert(4, 7);
  btree.Insert(8, 8);
  btree.Insert(10, 9);

  std::vector<int> gt = {5, 7, 6, 2, 4};
  std::vector<int> result = btree.FindLessEq(5);
  ASSERT_EQ(result.size(), gt.size());
  for (int i = 0; i < gt.size(); i++) {
    ASSERT_EQ(result[i], gt[i]);
  }
}
