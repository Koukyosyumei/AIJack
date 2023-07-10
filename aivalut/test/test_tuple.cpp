#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "src/storage/tuple.h"

TEST(TupleTest, Serialize) {
  int val = 42;
  Item item(val);
  std::vector<Item> items = {item};
  storage::Tuple *t1 = NewTuple(1, items);

  std::array<uint8_t, 128> buf = SerializeTuple(t1);
  storage::Tuple *t2 = DeserializeTuple(buf);
  ASSERT_EQ(t2->mintxid(), 1);
  ASSERT_EQ(t2->data(0).toi(), val);
}
