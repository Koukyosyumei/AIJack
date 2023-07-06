#pragma once
#include "../meta/bptree.h"
#include "../meta/lru.h"
#include "../meta/meta.h"
#include "page.h"
#include <cstdint>
#include <cstring>
#include <iostream>
#include <map>
#include <vector>

struct BufferTag {
  std::string tableName;
  uint64_t pgid;

  BufferTag(const std::string &tableName, uint64_t pgid)
      : tableName(tableName), pgid(pgid) {}

  uint64_t hash() {
    std::string data = tableName + std::to_string(pgid);
    return std::hash<std::string>{}(data);
  }
};

struct PageDescriptor {
  std::string tableName;
  uint64_t pgid;
  bool dirty;
  uint64_t ref;
  Page *page;

  PageDescriptor(const std::string &tableName, uint64_t pgid, Page *page)
      : tableName(tableName), pgid(pgid), dirty(false), ref(0), page(page) {}
};

class BufferPool {
public:
  BufferPool() { lru = new Lru<uint64_t, PageDescriptor *>(1000); }

  ~BufferPool() {
    delete lru;
    for (const auto &entry : btree) {
      delete entry.second;
    }
  }

  uint64_t toPgid(uint64_t tid) { return tid / TupleNumber; }

  void pinPage(PageDescriptor *pg) { pg->ref++; }

  void unpinPage(PageDescriptor *pg) { pg->ref--; }

  Page *readPage(const std::string &tableName, uint64_t tid) {
    uint64_t pgid = toPgid(tid);
    BufferTag bt(tableName, pgid);
    uint64_t hash = bt.hash();
    void *p = lru->Get(hash);

    if (p == nullptr) {
      return nullptr;
    }

    PageDescriptor *pd = static_cast<PageDescriptor *>(p);
    return pd->page;
  }

  bool appendTuple(const std::string &tableName, storage::Tuple *t) {
    // TODO: Implement appendTuple logic
    // uint64_t latestTid = 0;
    // uint64_t pgid = toPgid(latestTid);

    BufferTag bt(tableName, NewPgid(tableName));
    uint64_t hash = bt.hash();
    PageDescriptor *pd = lru->Get(hash);

    if (pd == nullptr) {
      std::cout << "pd is null" << std::endl;
      return false;
    }

    pd->dirty = true;

    for (int i = 0; i < TupleNumber; i++) {
      if (TupleIsUnused(&pd->page->Tuples[i])) {
        std::cout << "insert append!!" << std::endl;
        pd->page->Tuples[i] = *t;
        break;
      }
    }

    return true;
  }

  std::pair<bool, Page *> putPage(const std::string &tableName, uint64_t pgid,
                                  Page *p) {
    BufferTag bt(tableName, pgid);
    uint64_t hash = bt.hash();
    PageDescriptor *pd = new PageDescriptor(tableName, pgid, p);

    std::pair<bool, PageDescriptor *> res = lru->Insert(hash, pd);
    bool is_evicted = res.first;
    PageDescriptor *victimPage = res.second;

    if (!is_evicted) {
      return std::make_pair(false, nullptr);
    }

    return std::make_pair(victimPage->dirty, victimPage->page);
  }

  std::pair<bool, BTree<int> *> readIndex(const std::string &indexName) {
    auto it = btree.find(indexName);

    if (it != btree.end()) {
      return std::make_pair(true, it->second);
    }

    return std::make_pair(false, nullptr);
  }

  Lru<uint64_t, PageDescriptor *> *lru;
  std::map<std::string, BTree<int> *> btree;
};
