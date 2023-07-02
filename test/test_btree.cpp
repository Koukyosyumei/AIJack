#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "mydb/meta/btree.h"

TEST(BTreeTest, NoSplit) {
  BTree<IntItem> btree;

  btree.Insert(IntItem(1));
  btree.Insert(IntItem(2));

  auto found = btree.Find(IntItem(1));
  ASSERT_TRUE(found.first);

  found = btree.Find(IntItem(2));
  ASSERT_TRUE(found.first);

  found = btree.Find(IntItem(3));
  ASSERT_FALSE(found.first);
}

TEST(BTreeTest, SplitParent) {
  BTree<IntItem> btree;

  btree.Insert(IntItem(1));
  btree.Insert(IntItem(2));
  btree.Insert(IntItem(3));

  auto found = btree.Find(IntItem(1));
  ASSERT_TRUE(found.first);

  found = btree.Find(IntItem(2));
  ASSERT_TRUE(found.first);

  found = btree.Find(IntItem(3));
  ASSERT_TRUE(found.first);

  // test balance
  ASSERT_EQ(btree.GetTop()->Items[0], IntItem(2));
  ASSERT_EQ(btree.GetTop()->Children[0]->Items[0], IntItem(1));
  ASSERT_EQ(btree.GetTop()->Children[1]->Items[0], IntItem(3));
}

TEST(BTreeTest, SplitChild) {
  BTree<IntItem> btree;
  btree.Insert(IntItem(1));
  btree.Insert(IntItem(2));
  btree.Insert(IntItem(3));
  btree.Insert(IntItem(4));
  btree.Insert(IntItem(5));

  auto found = btree.Find(IntItem(1));
  ASSERT_TRUE(found.first);

  found = btree.Find(IntItem(2));
  ASSERT_TRUE(found.first);

  found = btree.Find(IntItem(3));
  ASSERT_TRUE(found.first);

  found = btree.Find(IntItem(4));
  ASSERT_TRUE(found.first);

  found = btree.Find(IntItem(5));
  ASSERT_TRUE(found.first);

  // test balance
  ASSERT_EQ(btree.GetTop()->Items[0], IntItem(2));
  ASSERT_EQ(btree.GetTop()->Items[1], IntItem(4));
  ASSERT_EQ(btree.GetTop()->Children[0]->Items[0], IntItem(1));
  ASSERT_EQ(btree.GetTop()->Children[1]->Items[0], IntItem(3));
  ASSERT_EQ(btree.GetTop()->Children[2]->Items[0], IntItem(5));
}

TEST(BTreeTest, Balanced) {
  BTree<IntItem> btree;
  btree.Insert(IntItem(1));
  btree.Insert(IntItem(2));
  btree.Insert(IntItem(3));
  btree.Insert(IntItem(4));
  btree.Insert(IntItem(5));
  btree.Insert(IntItem(6));
  btree.Insert(IntItem(7));

  auto found = btree.Find(IntItem(1));
  ASSERT_TRUE(found.first);

  found = btree.Find(IntItem(2));
  ASSERT_TRUE(found.first);

  found = btree.Find(IntItem(3));
  ASSERT_TRUE(found.first);

  found = btree.Find(IntItem(4));
  ASSERT_TRUE(found.first);

  found = btree.Find(IntItem(5));
  ASSERT_TRUE(found.first);

  found = btree.Find(IntItem(6));
  ASSERT_TRUE(found.first);

  found = btree.Find(IntItem(7));
  ASSERT_TRUE(found.first);

  // test balance
  ASSERT_EQ(btree.GetTop()->Items[0], IntItem(4));
  ASSERT_EQ(btree.GetTop()->Children[0]->Items[0], IntItem(2));
  ASSERT_EQ(btree.GetTop()->Children[1]->Items[0], IntItem(6));
  ASSERT_EQ(btree.GetTop()->Children[0]->Children[0]->Items[0], IntItem(1));
  ASSERT_EQ(btree.GetTop()->Children[0]->Children[1]->Items[0], IntItem(3));
  ASSERT_EQ(btree.GetTop()->Children[1]->Children[0]->Items[0], IntItem(5));
  ASSERT_EQ(btree.GetTop()->Children[1]->Children[1]->Items[0], IntItem(7));
}

TEST(BTreeTest, BalancedReversed) {
  BTree<IntItem> btree;
  btree.Insert(IntItem(7));
  btree.Insert(IntItem(6));
  btree.Insert(IntItem(5));
  btree.Insert(IntItem(4));
  btree.Insert(IntItem(3));
  btree.Insert(IntItem(2));
  btree.Insert(IntItem(1));

  auto found = btree.Find(IntItem(1));
  ASSERT_TRUE(found.first);

  found = btree.Find(IntItem(2));
  ASSERT_TRUE(found.first);

  found = btree.Find(IntItem(3));
  ASSERT_TRUE(found.first);

  found = btree.Find(IntItem(4));
  ASSERT_TRUE(found.first);

  found = btree.Find(IntItem(5));
  ASSERT_TRUE(found.first);

  found = btree.Find(IntItem(6));
  ASSERT_TRUE(found.first);

  found = btree.Find(IntItem(7));
  ASSERT_TRUE(found.first);

  // test balance
  ASSERT_EQ(btree.GetTop()->Items[0], IntItem(4));
  ASSERT_EQ(btree.GetTop()->Children[0]->Items[0], IntItem(2));
  ASSERT_EQ(btree.GetTop()->Children[1]->Items[0], IntItem(6));
  ASSERT_EQ(btree.GetTop()->Children[0]->Children[0]->Items[0], IntItem(1));
  ASSERT_EQ(btree.GetTop()->Children[0]->Children[1]->Items[0], IntItem(3));
  ASSERT_EQ(btree.GetTop()->Children[1]->Children[0]->Items[0], IntItem(5));
  ASSERT_EQ(btree.GetTop()->Children[1]->Children[1]->Items[0], IntItem(7));
}

TEST(BTreeTest, Get) {
  BTree<IntItem> btree;
  btree.Insert(IntItem(1));
  btree.Insert(IntItem(2));
  btree.Insert(IntItem(3));
  btree.Insert(IntItem(4));
  btree.Insert(IntItem(5));
  btree.Insert(IntItem(6));
  btree.Insert(IntItem(7));

  auto item = btree.Get(IntItem(1));
  ASSERT_EQ(item->value, 1);

  item = btree.Get(IntItem(7));
  ASSERT_EQ(item->value, 7);
}

TEST(BTreeTest, Random) {
  BTree<IntItem> btree;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 999);

  for (int i = 0; i < 10000; ++i) {
    int64_t v = dis(gen);
    btree.Insert(IntItem(v));
  }

  ASSERT_EQ(btree.Len(), 10000);
}

TEST(BTreeTest, Empty) {
  BTree<IntItem> btree;
  auto found = btree.Find(IntItem(1));
  ASSERT_FALSE(found.first);
}

TEST(BTreeTest, Serialize) {
  BTree<IntItem> btree;
  btree.Insert(IntItem(1));

  // std::vector<uint8_t> serialized;
  try {
    btree.SerializeBTree();
  } catch (const std::exception &e) {
    std::cerr << "Serialization error: " << e.what() << std::endl;
    FAIL();
  }

  /*
  BTree<IntItem> newTree;
  try {
    newTree.DeserializeBTree(serialized);
  } catch (const std::exception &e) {
    std::cerr << "Deserialization error: " << e.what() << std::endl;
    FAIL();
  }

  ASSERT_EQ(newTree.GetTop()->Items[0], IntItem(1));
  */
}
