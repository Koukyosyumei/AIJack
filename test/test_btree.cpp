#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "src/utils/bptree.h"

TEST(BTreeTest, NoSplit) {
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

TEST(BTreeTest, SplitParent) {
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

TEST(BTreeTest, GreaterEq) {
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

/*
TEST(BTreeTest, Random) {
  BPlusTreeMap<int, int> bpmap;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, 5);

  for (int i = 0; i < 10; ++i) {
    int v = dis(gen);
    std::cout << v << std::endl;
    bpmap.Insert(v, i);
  }

  std::cout << btree.bpmap.Serialize() << std::endl;

  ASSERT_EQ(btree.Len(), 10);
}*/

/*
TEST(BTreeTest, SplitChild) {
  BPlusTreeMap<int, int> btree;
  btree.Insert(1);
  btree.Insert(2);
  btree.Insert(3);
  std::cout << 88 << std::endl;
  std::cout << (btree.top == nullptr) << std::endl;
  btree.Insert(4);
  std::cout << 888 << std::endl;
  btree.Insert(5);

  std::cout << "b" << std::endl;
  auto found = btree.Find(1);
  ASSERT_TRUE(found.first);

  found = btree.Find(2);
  ASSERT_TRUE(found.first);

  found = btree.Find(3);
  ASSERT_TRUE(found.first);

  found = btree.Find(4);
  ASSERT_TRUE(found.first);

  found = btree.Find(5);
  ASSERT_TRUE(found.first);

  std::cout << "b" << std::endl;
  // test balance
  ASSERT_EQ(btree.GetTop()->Items[0], 2);
  ASSERT_EQ(btree.GetTop()->Items[1], 4);
  ASSERT_EQ(btree.GetTop()->Children[0]->Items[0], 1);
  ASSERT_EQ(btree.GetTop()->Children[1]->Items[0], 3);
  ASSERT_EQ(btree.GetTop()->Children[2]->Items[0], 5);
}

TEST(BTreeTest, Balanced) {
  BPlusTreeMap<int, int> btree;
  btree.Insert(1);
  btree.Insert(2);
  btree.Insert(3);
  btree.Insert(4);
  btree.Insert(5);
  btree.Insert(6);
  btree.Insert(7);

  auto found = btree.Find(1);
  ASSERT_TRUE(found.first);

  found = btree.Find(2);
  ASSERT_TRUE(found.first);

  found = btree.Find(3);
  ASSERT_TRUE(found.first);

  found = btree.Find(4);
  ASSERT_TRUE(found.first);

  found = btree.Find(5);
  ASSERT_TRUE(found.first);

  found = btree.Find(6);
  ASSERT_TRUE(found.first);

  found = btree.Find(7);
  ASSERT_TRUE(found.first);

  // test balance
  ASSERT_EQ(btree.GetTop()->Items[0], 4);
  ASSERT_EQ(btree.GetTop()->Children[0]->Items[0], 2);
  ASSERT_EQ(btree.GetTop()->Children[1]->Items[0], 6);
  ASSERT_EQ(btree.GetTop()->Children[0]->Children[0]->Items[0], 1);
  ASSERT_EQ(btree.GetTop()->Children[0]->Children[1]->Items[0], 3);
  ASSERT_EQ(btree.GetTop()->Children[1]->Children[0]->Items[0], 5);
  ASSERT_EQ(btree.GetTop()->Children[1]->Children[1]->Items[0], 7);
}

TEST(BTreeTest, BalancedReversed) {
  BPlusTreeMap<int, int> btree;
  btree.Insert(7);
  btree.Insert(6);
  btree.Insert(5);
  btree.Insert(4);
  btree.Insert(3);
  btree.Insert(2);
  btree.Insert(1);

  auto found = btree.Find(1);
  ASSERT_TRUE(found.first);

  found = btree.Find(2);
  ASSERT_TRUE(found.first);

  found = btree.Find(3);
  ASSERT_TRUE(found.first);

  found = btree.Find(4);
  ASSERT_TRUE(found.first);

  found = btree.Find(5);
  ASSERT_TRUE(found.first);

  found = btree.Find(6);
  ASSERT_TRUE(found.first);

  found = btree.Find(7);
  ASSERT_TRUE(found.first);

  // test balance
  ASSERT_EQ(btree.GetTop()->Items[0], 4);
  ASSERT_EQ(btree.GetTop()->Children[0]->Items[0], 2);
  ASSERT_EQ(btree.GetTop()->Children[1]->Items[0], 6);
  ASSERT_EQ(btree.GetTop()->Children[0]->Children[0]->Items[0], 1);
  ASSERT_EQ(btree.GetTop()->Children[0]->Children[1]->Items[0], 3);
  ASSERT_EQ(btree.GetTop()->Children[1]->Children[0]->Items[0], 5);
  ASSERT_EQ(btree.GetTop()->Children[1]->Children[1]->Items[0], 7);
}

TEST(BTreeTest, Get) {
  BPlusTreeMap<int, int> btree;
  btree.Insert(1);
  btree.Insert(2);
  btree.Insert(3);
  btree.Insert(4);
  btree.Insert(5);
  btree.Insert(6);
  btree.Insert(7);

  auto item = btree.Get(1);
  ASSERT_EQ(item.second, 1);

  item = btree.Get(7);
  ASSERT_EQ(item.second, 7);
}

TEST(BTreeTest, Empty) {
  BPlusTreeMap<int, int> btree;
  auto found = btree.Find(1);
  ASSERT_FALSE(found.first);
}
*/
/*
TEST(BTreeTest, Serialize) {
  BPlusTreeMap<int, int> btree;
  btree.Insert(1);

  // std::vector<uint8_t> serialized;
  try {
    btree.SerializeBTree();
  } catch (const std::exception &e) {
    std::cerr << "Serialization error: " << e.what() << std::endl;
    FAIL();
  }

  BTree<ItemType<int>> newTree;
  try {
    newTree.DeserializeBTree(serialized);
  } catch (const std::exception &e) {
    std::cerr << "Deserialization error: " << e.what() << std::endl;
    FAIL();
  }

  ASSERT_EQ(newTree.GetTop()->Items[0], 1);
 */
