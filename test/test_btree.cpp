#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "mydb/meta/btree.h"

TEST(BTreeTest, NoSplit) {
  BTree<int> btree;

  btree.Insert(ItemType<int>(1));
  btree.Insert(ItemType<int>(2));

  auto found = btree.Find(ItemType<int>(1));
  ASSERT_TRUE(found.first);

  found = btree.Find(ItemType<int>(2));
  ASSERT_TRUE(found.first);

  found = btree.Find(ItemType<int>(3));
  ASSERT_FALSE(found.first);
}

TEST(BTreeTest, SplitParent) {
  BTree<int> btree;

  btree.Insert(ItemType<int>(1));
  btree.Insert(ItemType<int>(2));
  btree.Insert(ItemType<int>(3));

  auto found = btree.Find(ItemType<int>(1));
  ASSERT_TRUE(found.first);

  found = btree.Find(ItemType<int>(2));
  ASSERT_TRUE(found.first);

  found = btree.Find(ItemType<int>(3));
  ASSERT_TRUE(found.first);

  // test balance
  ASSERT_EQ(btree.GetTop()->Items[0].value, 2);
  ASSERT_EQ(btree.GetTop()->Children[0]->Items[0].value, 1);
  ASSERT_EQ(btree.GetTop()->Children[1]->Items[0].value, 3);
}

TEST(BTreeTest, SplitChild) {
  BTree<int> btree;
  btree.Insert(ItemType<int>(1));
  btree.Insert(ItemType<int>(2));
  btree.Insert(ItemType<int>(3));
  btree.Insert(ItemType<int>(4));
  btree.Insert(ItemType<int>(5));

  auto found = btree.Find(ItemType<int>(1));
  ASSERT_TRUE(found.first);

  found = btree.Find(ItemType<int>(2));
  ASSERT_TRUE(found.first);

  found = btree.Find(ItemType<int>(3));
  ASSERT_TRUE(found.first);

  found = btree.Find(ItemType<int>(4));
  ASSERT_TRUE(found.first);

  found = btree.Find(ItemType<int>(5));
  ASSERT_TRUE(found.first);

  // test balance
  ASSERT_EQ(btree.GetTop()->Items[0], ItemType<int>(2));
  ASSERT_EQ(btree.GetTop()->Items[1], ItemType<int>(4));
  ASSERT_EQ(btree.GetTop()->Children[0]->Items[0], ItemType<int>(1));
  ASSERT_EQ(btree.GetTop()->Children[1]->Items[0], ItemType<int>(3));
  ASSERT_EQ(btree.GetTop()->Children[2]->Items[0], ItemType<int>(5));
}

TEST(BTreeTest, Balanced) {
  BTree<int> btree;
  btree.Insert(ItemType<int>(1));
  btree.Insert(ItemType<int>(2));
  btree.Insert(ItemType<int>(3));
  btree.Insert(ItemType<int>(4));
  btree.Insert(ItemType<int>(5));
  btree.Insert(ItemType<int>(6));
  btree.Insert(ItemType<int>(7));

  auto found = btree.Find(ItemType<int>(1));
  ASSERT_TRUE(found.first);

  found = btree.Find(ItemType<int>(2));
  ASSERT_TRUE(found.first);

  found = btree.Find(ItemType<int>(3));
  ASSERT_TRUE(found.first);

  found = btree.Find(ItemType<int>(4));
  ASSERT_TRUE(found.first);

  found = btree.Find(ItemType<int>(5));
  ASSERT_TRUE(found.first);

  found = btree.Find(ItemType<int>(6));
  ASSERT_TRUE(found.first);

  found = btree.Find(ItemType<int>(7));
  ASSERT_TRUE(found.first);

  // test balance
  ASSERT_EQ(btree.GetTop()->Items[0], ItemType<int>(4));
  ASSERT_EQ(btree.GetTop()->Children[0]->Items[0], ItemType<int>(2));
  ASSERT_EQ(btree.GetTop()->Children[1]->Items[0], ItemType<int>(6));
  ASSERT_EQ(btree.GetTop()->Children[0]->Children[0]->Items[0],
            ItemType<int>(1));
  ASSERT_EQ(btree.GetTop()->Children[0]->Children[1]->Items[0],
            ItemType<int>(3));
  ASSERT_EQ(btree.GetTop()->Children[1]->Children[0]->Items[0],
            ItemType<int>(5));
  ASSERT_EQ(btree.GetTop()->Children[1]->Children[1]->Items[0],
            ItemType<int>(7));
}

TEST(BTreeTest, BalancedReversed) {
  BTree<int> btree;
  btree.Insert(ItemType<int>(7));
  btree.Insert(ItemType<int>(6));
  btree.Insert(ItemType<int>(5));
  btree.Insert(ItemType<int>(4));
  btree.Insert(ItemType<int>(3));
  btree.Insert(ItemType<int>(2));
  btree.Insert(ItemType<int>(1));

  auto found = btree.Find(ItemType<int>(1));
  ASSERT_TRUE(found.first);

  found = btree.Find(ItemType<int>(2));
  ASSERT_TRUE(found.first);

  found = btree.Find(ItemType<int>(3));
  ASSERT_TRUE(found.first);

  found = btree.Find(ItemType<int>(4));
  ASSERT_TRUE(found.first);

  found = btree.Find(ItemType<int>(5));
  ASSERT_TRUE(found.first);

  found = btree.Find(ItemType<int>(6));
  ASSERT_TRUE(found.first);

  found = btree.Find(ItemType<int>(7));
  ASSERT_TRUE(found.first);

  // test balance
  ASSERT_EQ(btree.GetTop()->Items[0], ItemType<int>(4));
  ASSERT_EQ(btree.GetTop()->Children[0]->Items[0], ItemType<int>(2));
  ASSERT_EQ(btree.GetTop()->Children[1]->Items[0], ItemType<int>(6));
  ASSERT_EQ(btree.GetTop()->Children[0]->Children[0]->Items[0],
            ItemType<int>(1));
  ASSERT_EQ(btree.GetTop()->Children[0]->Children[1]->Items[0],
            ItemType<int>(3));
  ASSERT_EQ(btree.GetTop()->Children[1]->Children[0]->Items[0],
            ItemType<int>(5));
  ASSERT_EQ(btree.GetTop()->Children[1]->Children[1]->Items[0],
            ItemType<int>(7));
}

TEST(BTreeTest, Get) {
  BTree<int> btree;
  btree.Insert(ItemType<int>(1));
  btree.Insert(ItemType<int>(2));
  btree.Insert(ItemType<int>(3));
  btree.Insert(ItemType<int>(4));
  btree.Insert(ItemType<int>(5));
  btree.Insert(ItemType<int>(6));
  btree.Insert(ItemType<int>(7));

  auto item = btree.Get(1);
  ASSERT_EQ(item->value, 1);

  item = btree.Get(7);
  ASSERT_EQ(item->value, 7);
}

TEST(BTreeTest, Random) {
  BTree<int> btree;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 999);

  for (int i = 0; i < 10000; ++i) {
    int64_t v = dis(gen);
    btree.Insert(ItemType<int>(v));
  }

  ASSERT_EQ(btree.Len(), 10000);
}

TEST(BTreeTest, Empty) {
  BTree<int> btree;
  auto found = btree.Find(ItemType<int>(1));
  ASSERT_FALSE(found.first);
}

TEST(BTreeTest, Serialize) {
  BTree<int> btree;
  btree.Insert(ItemType<int>(1));

  // std::vector<uint8_t> serialized;
  try {
    btree.SerializeBTree();
  } catch (const std::exception &e) {
    std::cerr << "Serialization error: " << e.what() << std::endl;
    FAIL();
  }

  /*
  BTree<ItemType<int>> newTree;
  try {
    newTree.DeserializeBTree(serialized);
  } catch (const std::exception &e) {
    std::cerr << "Deserialization error: " << e.what() << std::endl;
    FAIL();
  }

  ASSERT_EQ(newTree.GetTop()->Items[0], ItemType<int>(1));
  */
}
