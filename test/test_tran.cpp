#include <atomic>
#include <gtest/gtest.h>
#include <mutex>
#include <thread>
#include <vector>

#include "src/storage/tran.h"

TEST(TransactionManagerTest, TxidAtomicity) {
  std::vector<bool> exists(10001, false);
  TransactionManager manager;

  std::mutex mtx;
  std::vector<std::thread> threads;

  for (int i = 0; i < 10000; i++) {
    threads.emplace_back([&]() {
      uint64_t id = manager.newTxid();
      {
        std::lock_guard<std::mutex> lock(mtx);
        if (exists[id]) {
          FAIL() << "txid duplicated";
        }
        exists[id] = true;
      }
    });
  }

  for (auto &thread : threads) {
    thread.join();
  }

  ASSERT_EQ(10000, manager.GetCurrentTxID());
}
