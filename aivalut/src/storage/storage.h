#pragma once
#include "../ml/logisticregression.h"
#include "../storage/base.h"
#include "bufpool.h"
#include "disk.h"
#include "page.h"
#include <iostream>
#include <stdexcept>
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

  void insertPage(const std::string &tableName, PageStatus pstatus) {
    Page *pg = NewPage();
    uint64_t pgid = buffer->NewPgid(tableName);
    if (pstatus == Full) {
      pgid++;
    }
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

  TID InsertTuple(const std::string &tablename, storage::Tuple *t) {
    while (true) {
      // if not exist in buffer, put a page to lru-cache
      std::pair<PageStatus, TID> append_result =
          buffer->appendTuple(tablename, t);
      if (append_result.first == Success) {
        return append_result.second;
      }
      insertPage(tablename, append_result.first);
    }
  }

  BPlusTreeMap<int, TID> *CreateIndex(const std::string &indexName) {
    BPlusTreeMap<int, TID> *btree = new BPlusTreeMap<int, TID>();
    buffer->btree_map.insert({indexName, btree});
    return btree;
  }

  void InsertIndex(const std::string &indexName, int item, TID &tid) {
    BPlusTreeMap<int, TID> *btree = ReadIndex(indexName);

    if (btree != nullptr) {
      btree->Insert(item, tid);
    }
  }

  BPlusTreeMap<int, TID> *ReadIndex(const std::string &indexName) {
    std::pair<bool, BPlusTreeMap<int, TID> *> res =
        buffer->readIndex(indexName);
    if (res.first) {
      return res.second;
    }

    BPlusTreeMap<int, TID> *btree = disk->readIndex(prefix + "/" + indexName);

    if (btree == nullptr) {
      btree = CreateIndex(indexName);
    } else {
      buffer->btree_map.insert({indexName, btree});
    }

    return btree;
  }

  storage::Tuple *ReadTuple(const std::string &tableName, uint64_t seqtid) {
    uint64_t pgid = buffer->toPgid(seqtid);
    Page *pg = readPage(tableName, pgid);

    if (pg == nullptr) {
      std::cerr << "Failed to read page (id=" << pgid << ")\n";
      return nullptr;
    }

    return &pg->Tuples[seqtid % TupleNumber];
  }

  storage::Tuple *ReadTuple(const std::string &tableName, TID tid) {
    uint64_t pgid = tid.first;

    Page *pg = readPage(tableName, pgid);

    if (pg == nullptr) {
      std::cerr << "Failed to read page (id=" << pgid << ")\n";
      return nullptr;
    }

    return &pg->Tuples[tid.second];
  }

  void loadMLModel(LogisticRegression &clf, const std::string &model_name) {
    std::string path = prefix + "/" + model_name;
    clf.load(path);
  }

  void saveMLModel(LogisticRegression &clf, const std::string &model_name) {
    std::string path = prefix + "/" + model_name;
    clf.save(path);
  }

  void Terminate() {
    std::vector<PageDescriptor *> lru = buffer->lru->GetAll();

    for (const PageDescriptor *pd : lru) {
      if (pd == nullptr) {
        try {
          throw std::runtime_error("null page descriptor found");
        } catch (std::runtime_error &e) {
          std::cerr << "runtime_error: " << e.what() << std::endl;
          continue;
        }
      }
      if (pd->dirty) {
        disk->persist(prefix, pd->tableName, pd->pgid, pd->page);
      }
    }

    for (const auto &entry : buffer->btree_map) {
      const std::string &key = entry.first;
      BPlusTreeMap<int, TID> *val = entry.second;
      if (val == nullptr) {
        try {
          throw std::runtime_error("null index found in buffer");
        } catch (std::runtime_error &e) {
          std::cerr << "runtime_error: " << e.what() << std::endl;
          continue;
        }
      }
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
