#pragma once
#include "../storage/storage.h"
#include "../storage/tuple.h"
#include "analyze.h"
#include "ast.h"
#include "token.h"
#include <memory>
#include <string>
#include <vector>

// Forward declarations
struct Planner;
struct Plan;
struct Scanner;
struct SeqScan;
struct IndexScan;

// Planner
struct Planner {
  Query *q;

  Planner(Query *query) : q(query) {}

  Plan *planSelect(SelectQuery *q);
  Plan *planUpdate(UpdateQuery *q);
  Plan *planMain();
};

// Plan
struct Plan {
  Scanner *scanners;
};

// Scanner interface
struct Scanner {
  virtual std::vector<storage::Tuple *> Scan(Storage *store) = 0;
};

// SeqScan
struct SeqScan : public Scanner {
  std::string tblName;

  SeqScan(const std::string &tableName) : tblName(tableName) {}

  std::vector<storage::Tuple *> Scan(Storage *store) override {
    std::vector<storage::Tuple *> result;

    for (uint64_t i = 0;; i++) {
      std::cout << "s " << i << std::endl;
      storage::Tuple *t = store->ReadTuple(tblName, i);
      if (!t)
        break;

      if (TupleIsUnused(t))
        break;

      result.push_back(t);
    }
    return result;
  }
};

// IndexScan
struct IndexScan : public Scanner {
  std::string tblName;
  std::string index;
  std::string value;
  TokenKind op;

  IndexScan(const std::string &tableName, const std::string &idx,
            const std::string &val, TokenKind op)
      : tblName(tableName), index(idx), value(val), op(op) {}

  std::vector<storage::Tuple *> Scan(Storage *store) override {
    std::vector<storage::Tuple *> result;
    BPlusTreeMap<int, TID> *btree = store->ReadIndex(index);
    int i = std::stoi(value);
    if (op == EQ) {
      std::pair<bool, TID> find_result = btree->Find(i);
      if (find_result.first) {
        storage::Tuple *t = store->ReadTuple(tblName, find_result.second);
        if (t) {
          result.push_back(t);
        }
      }
    } else if (op == GEQ) {
      std::vector<TID> geq_results = btree->FindGreaterEq(i);
      for (const TID &tid : geq_results) {
        storage::Tuple *t = store->ReadTuple(tblName, tid);
        if (t) {
          result.push_back(t);
        }
      }
    }
    return result;
  }
};

inline Plan *Planner::planSelect(SelectQuery *q) {
  // if where contains a primary key, use index scan.
  for (Expr *eq : q->Where) {
    if (!eq)
      continue;

    Expr *col = eq->left;
    if (!col)
      continue;

    if (col->is_primary) {
      std::cout << "/prepare IndexScan\n";
      return new Plan{
          .scanners =
              new IndexScan(q->From[0]->Name, q->From[0]->Name + "_" + col->v,
                            eq->right->v, eq->op),
      };
    }
  }
  // use seqscan
  std::cout << "/prepare SeqScan\n";
  return new Plan{
      .scanners = new SeqScan(q->From[0]->Name),
  };
}

inline Plan *Planner::planUpdate(UpdateQuery *q) {
  // if where contains a primary key, use index scan.
  for (Expr *eq : q->Where) {
    if (!eq)
      continue;
    Expr *col = eq->left;
    if (!col)
      continue;
    if (col->is_primary) {
      return new Plan{
          .scanners = new IndexScan(q->table->Name, col->v, "", eq->op),
      };
    }
  }

  // use seqscan
  return new Plan{
      .scanners = new SeqScan(q->table->Name),
  };
}

inline Plan *Planner::planMain() {
  if (auto selectQuery = dynamic_cast<SelectQuery *>(q))
    return planSelect(selectQuery);
  if (auto updateQuery = dynamic_cast<UpdateQuery *>(q))
    return planUpdate(updateQuery);

  return nullptr;
}
