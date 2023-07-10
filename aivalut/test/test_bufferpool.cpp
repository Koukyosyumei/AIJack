#include "../src/storage/bufpool.h"
#include <gtest/gtest.h>

TEST(BufferPoolTest, FrontPgid_ReturnsZeroForNonExistingTable) {
  BufferPool *bufferPool = new BufferPool();
  // Test the FrontPgid method for a non-existing table
  std::string tableName = "non_existing_table";
  int frontPgid = bufferPool->FrontPgid(tableName);

  // Assert that the frontPgid is zero
  ASSERT_EQ(frontPgid, 0);
  delete bufferPool;
}

TEST(BufferPoolTest, NewPgid_ReturnsIncrementedValueForEachCall) {
  BufferPool *bufferPool = new BufferPool();
  // Test the NewPgid method for multiple calls
  std::string tableName = "table";
  int initialPgid = bufferPool->NewPgid(tableName);

  // Call NewPgid again and assert that the returned value is incremented
  bufferPool->NewPgid(tableName);
  int newPgid = bufferPool->NewPgid(tableName);
  ASSERT_EQ(newPgid, initialPgid + 1);
  delete bufferPool;
}
