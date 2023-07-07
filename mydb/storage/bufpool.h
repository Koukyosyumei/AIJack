#pragma once
#include "../utils/bptree.h"
#include "../utils/lru.h"
#include "../utils/meta.h"
#include "page.h"
#include <cstdint>
#include <cstring>
#include <iostream>
#include <unordered_map>
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
    for (const auto &entry : btree_map) {
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

  std::pair<bool, std::pair<int, int>> appendTuple(const std::string &tableName,
                                                   storage::Tuple *t) {
    // TODO: Implement appendTuple logic
    // uint64_t latestTid = 0;
    // uint64_t pgid = toPgid(latestTid);
    int pgid = NewPgid(tableName);
    BufferTag bt(tableName, pgid);
    uint64_t hash = bt.hash();
    PageDescriptor *pd = lru->Get(hash);

    if (pd == nullptr) {
      return {false, {-1, -1}};
    }

    if (pd->page->front >= TupleNumber) {
      return {false, {-1, -1}};
    }

    pd->dirty = true;
    pd->page->Tuples[pd->page->front] = *t;
    pd->page->front++;
    return {true, {pgid, pd->page->front - 1}};
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

  std::pair<bool, BPlusTreeMap<int, TID> *>
  readIndex(const std::string &indexName) {
    auto it = btree_map.find(indexName);

    if (it != btree_map.end()) {
      return std::make_pair(true, it->second);
    }

    return std::make_pair(false, nullptr);
  }

  Lru<uint64_t, PageDescriptor *> *lru;
  std::unordered_map<std::string, BPlusTreeMap<int, TID> *> btree_map;
};
