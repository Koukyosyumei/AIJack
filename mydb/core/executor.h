#pragma once
#include "../compiler/analyze.h"
#include "../compiler/ast.h"
#include "../storage/base.h"
#include "../storage/catalog.h"
#include "../storage/storage.h"
#include "../storage/tran.h"
#include "../storage/tuple.h"
#include "../utils/bptree.h"
#include "plan.h"
#include <cstdio>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declarations
struct Executor;

// Executor
struct Executor {
  Storage *storage;
  Catalog *catalog;
  TransactionManager *tranManager;

  Executor(Storage *storage, Catalog *catalog, TransactionManager *tranManager)
      : storage(storage), catalog(catalog), tranManager(tranManager) {}

  static Executor *NewExecutor(Storage *storage, Catalog *catalog,
                               TransactionManager *tranManager) {
    return new Executor(storage, catalog, tranManager);
  }

  std::vector<storage::Tuple *> where(std::vector<storage::Tuple *> &tuples,
                                      const std::string &tableName,
                                      const std::vector<Expr *> &where);
  ResultSet *selectTable(SelectQuery *q, Plan *p, Transaction *tran);
  ResultSet *insertTable(InsertQuery *q, Transaction *tran);
  void updateTable(UpdateQuery *q, Plan *p, Transaction *tran);
  ResultSet *createTable(CreateTableQuery *q);
  Transaction *beginTransaction();
  void commitTransaction(Transaction *tran);
  void abortTransaction(Transaction *tran);
  ResultSet *executeMain(Query *q, Plan *p, Transaction *tran);
};

inline std::vector<storage::Tuple *>
Executor::where(std::vector<storage::Tuple *> &tuples,
                const std::string &tableName,
                const std::vector<Expr *> &where) {
  std::vector<storage::Tuple *> filtered;
  Scheme *s = catalog->FetchScheme(tableName);
  for (auto &w : where) {
    std::string left = w->left->v;
    std::string right = w->right->v;
    for (auto &t : tuples) {
      int colid = s->get_ColID(left);
      bool flag = false;
      if (w->op == EQ) {
        if (s->ColTypes[colid] == ColType::Int) {
          flag = TupleEqual(t, colid, std::stoi(right));
        } else if (s->ColTypes[colid] == ColType::Varchar) {
          flag = TupleEqual(t, colid, right);
        }
      } else if (w->op == GEQ) {
        if (s->ColTypes[colid] == ColType::Int) {
          flag = TupleGreaterEq(t, colid, std::stoi(right));
        } else if (s->ColTypes[colid] == ColType::Varchar) {
          flag = TupleGreaterEq(t, colid, right);
        }
      }
      if (flag) {
        filtered.emplace_back(t);
      }
    }
  }

  return filtered;
}

inline ResultSet *Executor::selectTable(SelectQuery *q, Plan *p,
                                        Transaction *tran) {
  std::vector<storage::Tuple *> tuples = p->scanners->Scan(storage);
  if (!q->Where.empty()) {
    tuples = where(tuples, q->From[0]->Name, q->Where);
  }

  Scheme *scheme = catalog->FetchScheme(q->From[0]->Name);

  std::vector<std::string> colNames;
  for (auto &c : q->Cols) {
    colNames.emplace_back(c->Name);
  }

  std::vector<std::string> values;
  for (storage::Tuple *t : tuples) {
    if (!tran || TupleCanSee(t, tran)) {
      for (std::string &cname : colNames) {
        const storage::TupleData td = t->data(scheme->get_ColID(cname));
        std::string s;
        if (td.type() == storage::TupleData_Type_INT) {
          s = std::to_string(td.number());
        } else if (td.type() == storage::TupleData_Type_STRING) {
          s = td.string();
        }
        values.emplace_back(s);
      }
    }
  }

  ResultSet *resultset = new ResultSet();
  resultset->Message = "";
  resultset->ColNames = colNames;
  resultset->Values = values;
  return resultset;
}

inline ResultSet *Executor::insertTable(InsertQuery *q, Transaction *tran) {
  bool inTransaction = tran != nullptr;

  if (!inTransaction) {
    tran = beginTransaction();
  }
  storage::Tuple *t = NewTuple(tran->Txid(), q->Values);
  std::pair<int, int> tid = storage->InsertTuple(q->table->Name, t);
  storage->InsertIndex(q->Index, t->data(0).number(), tid);
  if (!inTransaction) {
    commitTransaction(tran);
  }

  ResultSet *resultset = new ResultSet();
  resultset->Message = "A row was inserted";
  return resultset;
}

inline void Executor::updateTable(UpdateQuery *q, Plan *p, Transaction *tran) {
  // Implementation for updating the table goes here
  // Not implemented in this conversion
}

inline ResultSet *Executor::createTable(CreateTableQuery *q) {
  catalog->Add(q->scheme);

  bool created =
      storage->CreateIndex(q->scheme->TblName + "_" + q->scheme->PrimaryKey);
  if (!created) {
    return nullptr;
  }

  ResultSet *resultset = new ResultSet();
  resultset->Message = q->scheme->TblName + " was created as Table";
  return resultset;
}

inline Transaction *Executor::beginTransaction() {
  return tranManager->BeginTransaction();
}

inline void Executor::commitTransaction(Transaction *tran) {
  tranManager->Commit(tran);
}

inline void Executor::abortTransaction(Transaction *tran) {
  tranManager->Abort(tran);
}

inline ResultSet *Executor::executeMain(Query *q, Plan *p, Transaction *tran) {
  if (auto beginQuery = dynamic_cast<BeginQuery *>(q)) {
    beginTransaction();
    ResultSet *resultset = new ResultSet();
    resultset->Message = "Transaction begins.";
  }
  if (auto commitQuery = dynamic_cast<CommitQuery *>(q)) {
    commitTransaction(tran);
    ResultSet *resultset = new ResultSet();
    resultset->Message = "Transaction was committed.";
  }
  if (auto abortQuery = dynamic_cast<AbortQuery *>(q)) {
    abortTransaction(tran);
    ResultSet *resultset = new ResultSet();
    resultset->Message = "Transaction was aborted.";
  }
  if (auto createTableQuery = dynamic_cast<CreateTableQuery *>(q)) {
    return createTable(createTableQuery);
  }
  if (auto insertQuery = dynamic_cast<InsertQuery *>(q)) {
    return insertTable(insertQuery, tran);
  }
  if (auto selectQuery = dynamic_cast<SelectQuery *>(q)) {
    return selectTable(selectQuery, p, tran);
  }
  if (auto updateQuery = dynamic_cast<UpdateQuery *>(q)) {
    updateTable(updateQuery, p, tran);
    return nullptr; // Update query doesn't return a result set
  }

  try {
    throw std::runtime_error("Failed to execute query");
  } catch (std::runtime_error e) {
    std::cerr << "runtime_error: " << e.what() << std::endl;
  }

  return nullptr;
}
