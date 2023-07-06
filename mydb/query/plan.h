#pragma once
#include "../storage/storage.h"
#include "../storage/tuple.h"
#include "analyze.h"
#include "ast.h"
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

  static std::unique_ptr<Planner> NewPlanner(Query *query) {
    return std::make_unique<Planner>(query);
  }
};

// Plan
struct Plan {
  Scanner *scanners;
};

// Scanner interface
struct Scanner {
  virtual std::vector<storage::Tuple *> Scan(Storage *storage) = 0;
};

// SeqScan
struct SeqScan : public Scanner {
  std::string tblName;

  SeqScan(const std::string &tableName) : tblName(tableName) {}

  std::vector<storage::Tuple *> Scan(Storage *storage) override {
    // Perform sequential scan logic
    return {}; // Return scanned tuples
  }
};

// IndexScan
struct IndexScan : public Scanner {
  std::string tblName;
  std::string index;
  std::string value;

  IndexScan(const std::string &tableName, const std::string &idx,
            const std::string &val)
      : tblName(tableName), index(idx), value(val) {}

  std::vector<storage::Tuple *> Scan(Storage *storage) override {
    // Perform index scan logic
    return {}; // Return scanned tuples
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

    for (auto &c : q->Cols) {
      if (col->v == c->Name && c->Primary) {
        return new Plan{
            .scanners =
                new IndexScan(q->From[0]->Name, q->From[0]->Name + "_" + col->v,
                              eq->right->v),
        };
      }
    }
  }

  // use seqscan
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
    for (auto &c : q->Cols) {
      if (col->v == c->Name && c->Primary) {
        return new Plan{
            .scanners = new IndexScan(q->table->Name, col->v, ""),
        };
      }
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
