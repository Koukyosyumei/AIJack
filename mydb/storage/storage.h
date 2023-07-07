#pragma once
#include "../meta/meta.h"
#include "bufpool.h"
#include "disk.h"
#include "page.h"
#include <iostream>
#include <unordered_map>

class Storage {
public:
  Storage(const std::string &home) : prefix(home) {
    buffer = new BufferPool();
    disk = new DiskManager();
  }

  ~Storage() {
    delete buffer;
    delete disk;
  }

  void insertPage(const std::string &tableName) {
    Page *pg = NewPage();
    uint64_t pgid = NewPgid(tableName);
    bool isNeedPersist;
    std::pair<bool, Page *> flag_victim = buffer->putPage(tableName, pgid, pg);
    isNeedPersist = flag_victim.first;
    Page *victim = flag_victim.second;

    if (isNeedPersist) {
      // if a victim is dirty, its data must be persisted on the disk now.
      if (victim != nullptr) {
        disk->persist(prefix, tableName, pgid, victim);
      }
    }
  }

  void InsertTuple(const std::string &tablename, storage::Tuple *t) {
    while (!buffer->appendTuple(tablename, t)) {
      // if not exist in buffer, put a page to lru-cache
      insertPage(tablename);
    }
  }

  BTree<int> *CreateIndex(const std::string &indexName) {
    BTree<int> *btree = new BTree<int>();
    buffer->btree[indexName] = btree;
    return btree;
  }

  void InsertIndex(const std::string &indexName, int item) {
    BTree<int> *btree = ReadIndex(indexName);

    if (btree != nullptr) {
      btree->Insert(item);
    }
  }

  BTree<int> *ReadIndex(const std::string &indexName) {
    auto it = buffer->btree.find(indexName);
    if (it != buffer->btree.end()) {
      return it->second;
    }

    BTree<int> *btree = disk->readIndex(indexName);

    if (btree == nullptr) {
      btree = CreateIndex(indexName);
    }

    return btree;
  }

  storage::Tuple *ReadTuple(const std::string &tableName, uint64_t tid) {
    uint64_t pgid = buffer->toPgid(tid);

    Page *pg = readPage(tableName, pgid);

    if (pg == nullptr) {
      std::cerr << "Failed to read page (id=" << pgid << ")\n";
      return nullptr;
    }

    return &pg->Tuples[tid % TupleNumber];
  }

  void Terminate() {
    std::vector<PageDescriptor *> lru = buffer->lru->GetAll();

    for (const PageDescriptor *pd : lru) {
      if (pd->dirty) {
        disk->persist(prefix, pd->tableName, pd->pgid, pd->page);
      }
    }

    for (const auto &entry : buffer->btree) {
      const std::string &key = entry.first;
      BTree<int> *val = entry.second;
      disk->writeIndex(prefix, key, val);
    }
  }

private:
  BufferPool *buffer;
  DiskManager *disk;
  std::string prefix;

  Page *readPage(const std::string &tableName, uint64_t pgid) {
    Page *pg = buffer->readPage(tableName, pgid);

    if (pg != nullptr) {
      return pg;
    }
    pg = disk->fetchPage(prefix, tableName, pgid);

    if (pg != nullptr) {
      buffer->putPage(tableName, pgid, pg);
    }

    return pg;
  }
};
