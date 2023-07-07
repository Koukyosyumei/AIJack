#include "mydb/storage/catalog.h"
#include "mydb/utils/meta.h"
#include <gtest/gtest.h>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

TEST(CatalogTest, SaveCatalog) {
  Catalog ctg;

  ctg.Add(new Scheme{
      .TblName = "users", .ColNames = {"id"}, .ColTypes = {ColType::Int}});
  SaveCatalog("testdata", &ctg);
}

TEST(CatalogTest, LoadCatalog) {
  Catalog ctg;

  ctg.Add(new Scheme{
      .TblName = "users", .ColNames = {"id"}, .ColTypes = {ColType::Int}});

  SaveCatalog(".", &ctg);

  Catalog *out = LoadCatalog(".");
  ASSERT_EQ("users", out->Schemes[0]->TblName);
}

TEST(CatalogTest, AddConcurrency) {
  Catalog ctg;
  std::mutex mutex;

  std::vector<std::thread> threads;
  const int numThreads = 1000;

  for (int i = 0; i < numThreads; i++) {
    threads.emplace_back([&]() {
      std::lock_guard<std::mutex> lock(mutex);
      auto scheme = new ::Scheme{.TblName = std::to_string(i)};
      ctg.Add(scheme);
    });
  }

  for (auto &thread : threads) {
    thread.join();
  }

  ASSERT_EQ(numThreads, ctg.Schemes.size());
}
