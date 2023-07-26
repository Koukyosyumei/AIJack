#pragma once
#include "../storage/base.h"
#include "../utils/bptree.h"
#include "../utils/lru.h"
#include "page.h"
#include <cstdint>
#include <cstring>
#include <iostream>
#include <unordered_map>
#include <vector>

const int page_cache_size = 1024;

typedef enum { NotFound, Full, Success } PageStatus;

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
  BufferPool() {
    lru = new Lru<uint64_t, PageDescriptor *>(page_cache_size);
    cache_frontpage = new Lru<std::string, int>(page_cache_size, 0);
  }

  ~BufferPool() {
    delete lru;
    for (const auto &entry : btree_map) {
      delete entry.second;
    }
  }

  int FrontPgid(const std::string &tableName) {
    if (cache_frontpage->Has(tableName)) {
      return cache_frontpage->Get(tableName);
    } else {
      return 0;
    }
  }

  int NewPgid(const std::string &tableName) {
    if (cache_frontpage->Has(tableName)) {
      int frontid = cache_frontpage->Get(tableName);
      cache_frontpage->Insert(tableName, frontid + 1);
      return frontid;
    } else {
      cache_frontpage->Insert(tableName, 0);
      return 0;
    }
  }
  uint64_t toPgid(uint64_t tid) { return tid / TupleNumber; }

  void pinPage(PageDescriptor *pg) { pg->ref++; }

  void unpinPage(PageDescriptor *pg) { pg->ref--; }

  Page *readPage(const std::string &tableName, uint64_t pgid) {
    // uint64_t pgid = toPgid(tid);
    BufferTag bt(tableName, pgid);
    uint64_t hash = bt.hash();
    void *p = lru->Get(hash);

    if (p == nullptr) {
      return nullptr;
    }

    PageDescriptor *pd = static_cast<PageDescriptor *>(p);
    return pd->page;
  }

  std::pair<PageStatus, std::pair<int, int>>
  appendTuple(const std::string &tableName, storage::Tuple *t) {
    // TODO: Implement appendTuple logic
    // uint64_t latestTid = 0;
    // uint64_t pgid = toPgid(latestTid);
    int pgid = FrontPgid(tableName);
    BufferTag bt(tableName, pgid);
    uint64_t hash = bt.hash();
    PageDescriptor *pd = lru->Get(hash);

    if (pd == nullptr) {
      return {NotFound, {-1, -1}};
    }

    if (pd->page->front >= TupleNumber) {
      return {Full, {-1, -1}};
    }

    pd->dirty = true;
    pd->page->Tuples[pd->page->front] = *t;
    pd->page->front++;
    return {Success, {pgid, pd->page->front - 1}};
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

  Lru<std::string, int> *cache_frontpage;
  Lru<uint64_t, PageDescriptor *> *lru;
  std::unordered_map<std::string, BPlusTreeMap<int, TID> *> btree_map;
};
