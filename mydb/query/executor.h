#include "../meta/btree.h"
#include "../meta/meta.h" // Assuming meta.h contains the declaration of the meta classes
#include "../storage/catalog.h" //
#include "../storage/storage.h" // Assuming storage.h contains the declaration of the storage classes
#include "../storage/tran.h"
#include "../storage/tuple.h"
#include "analyze.h"
#include "ast.h" // Assuming expr.h contains the declaration of the Expr, Eq, and Lit classes
#include "plan.h"
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declarations
struct Executor;
struct SeqScan;
struct IndexScan;
struct Plan;
struct Query;
struct SelectQuery;
struct UpdateQuery;
struct CreateTableQuery;
struct InsertQuery;
struct BeginQuery;
struct CommitQuery;
struct AbortQuery;

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

  // SeqScan
  std::vector<storage::Tuple *> scan(SeqScan *s, Storage *store) {
    std::vector<storage::Tuple *> result;

    for (uint64_t i = 0;; i++) {
      storage::Tuple *t = store->ReadTuple(s->tblName, i);
      if (!t)
        break;

      if (TupleIsUnused(t))
        break;

      result.push_back(t);
    }
    return result;
  }

  // IndexScan
  std::vector<storage::Tuple *> scan(IndexScan *s, Storage *store) {
    std::vector<storage::Tuple *> result;
    BTree *btree = store->ReadIndex(s->index);

    int i = std::stoi(s->value);
    storage::Tuple *item = btree->Get(new IntItem(i));

    if (item)
      result.push_back(item);
    return result;
  }
};

std::vector<storage::Tuple *>
Executor::where(std::vector<storage::Tuple *> &tuples,
                const std::string &tableName,
                const std::vector<Expr *> &where) {
  std::vector<storage::Tuple *> filtered;

  for (auto &w : where) {
    std::string left = w->left->v;
    std::string right = w->right->v;
    for (auto &t : tuples) {
      auto s = catalog->FetchScheme(tableName);
      int order = 0;
      for (auto &c : s->ColNames) {
        if (c == left)
          break;
        order++;
      }

      int n = std::stoi(right);
      if (TupleEqual(t, order, right, n)) {
        filtered.push_back(t);
      }
    }
  }

  return filtered;
}

ResultSet *Executor::selectTable(SelectQuery *q, Plan *p, Transaction *tran) {
  std::vector<storage::Tuple *> tuples = p->scanners->Scan(storage);
  if (!q->Where.empty()) {
    tuples = where(tuples, q->From[0].Name, q->Where);
  }

  std::vector<std::string> values;
  for (auto &t : tuples) {
    if (!tran || t->CanSee(tran)) {
      for (int i = 0; i < q->Cols.size(); ++i) {
        std::string s = fmt::sprintf(q->Cols[i].Name, t->Data[i].String());
        values.push_back(s);
      }
    }
  }

  std::vector<std::string> colNames;
  for (auto &c : q->Cols) {
    colNames.push_back(c.Name);
  }

  return new ResultSet("", colNames, values);
}

ResultSet *Executor::insertTable(InsertQuery *q, Transaction *tran) {
  bool inTransaction = tran != nullptr;

  if (!inTransaction) {
    tran = beginTransaction();
  }

  storage::Tuple *t = new storage::Tuple(tran->Txid(), q->Values);
  storage->InsertTuple(q->Table.Name, t);
  storage->InsertIndex(q->Index, t);

  if (!inTransaction) {
    commitTransaction(tran);
  }

  return new ResultSet("A row was inserted");
}

void Executor::updateTable(UpdateQuery *q, Plan *p, Transaction *tran) {
  // Implementation for updating the table goes here
  // Not implemented in this conversion
}

ResultSet *Executor::createTable(CreateTableQuery *q) {
  bool added = catalog->Add(q->Scheme);
  if (!added) {
    return nullptr;
  }

  bool created =
      storage->CreateIndex(q->Scheme.TblName + "_" + q->Scheme.PrimaryKey);
  if (!created) {
    return nullptr;
  }

  return new ResultSet(q->Scheme.TblName + " was created as Table");
}

Transaction *Executor::beginTransaction() {
  return tranManager->BeginTransaction();
}

void Executor::commitTransaction(Transaction *tran) {
  tranManager->Commit(tran);
}

void Executor::abortTransaction(Transaction *tran) { tranManager->Abort(tran); }

ResultSet *Executor::executeMain(Query *q, Plan *p, Transaction *tran) {
  if (auto beginQuery = dynamic_cast<BeginQuery *>(q)) {
    beginTransaction();
    return new ResultSet("Transaction begins.");
  }
  if (auto commitQuery = dynamic_cast<CommitQuery *>(q)) {
    commitTransaction(tran);
    return new ResultSet("Transaction was committed.");
  }
  if (auto abortQuery = dynamic_cast<AbortQuery *>(q)) {
    abortTransaction(tran);
    return new ResultSet("Transaction was aborted.");
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

  throw std::runtime_error("Failed to execute query");
}
