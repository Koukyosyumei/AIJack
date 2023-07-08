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
  std::vector<std::pair<storage::Tuple *, storage::Tuple *>>
  join(std::vector<storage::Tuple *> &left_tuples,
       const std::string &left_table_name, const std::string &rigt_table_name,
       const std::pair<std::string, std::string> joinkeys);
  ResultSet *selectTable(SelectQuery *q, Plan *p, Transaction *tran);
  ResultSet *insertTable(InsertQuery *q, Transaction *tran);
  void updateTable(UpdateQuery *q, Plan *p, Transaction *tran);
  ResultSet *createTable(CreateTableQuery *q);
  Transaction *beginTransaction();
  void commitTransaction(Transaction *tran);
  void abortTransaction(Transaction *tran);
  ResultSet *executeMain(Query *q, Plan *p, Transaction *tran);
};

inline std::vector<std::pair<storage::Tuple *, storage::Tuple *>>
Executor::join(std::vector<storage::Tuple *> &left_tuples,
               const std::string &left_table_name,
               const std::string &right_table_name,
               const std::pair<std::string, std::string> joinkey) {
  std::vector<std::pair<storage::Tuple *, storage::Tuple *>> joined;
  Scheme *s_left = catalog->FetchScheme(left_table_name);
  Scheme *s_right = catalog->FetchScheme(right_table_name);

  std::string left_index = joinkey.first;
  std::string right_index = joinkey.second;
  int left_colid = s_left->get_ColID(left_index);
  BPlusTreeMap<int, TID> *right_btree =
      storage->ReadIndex(right_table_name + "_" + right_index);
  for (auto &lt : left_tuples) {
    std::pair<bool, TID> find_result =
        right_btree->Find(lt->data(left_colid).toi());
    if (find_result.first) {
      storage::Tuple *rt =
          storage->ReadTuple(right_table_name, find_result.second);
      if (rt) {
        joined.emplace_back(std::make_pair(lt, rt));
      }
    }
  }

  return joined;
}

inline std::vector<storage::Tuple *>
Executor::where(std::vector<storage::Tuple *> &tuples,
                const std::string &tableName,
                const std::vector<Expr *> &where) {
  std::vector<storage::Tuple *> filtered;
  Scheme *s = catalog->FetchScheme(tableName);
  for (auto &w : where) {
    std::string left = w->left->v;
    std::string right = w->right->v;
    int colid = s->get_ColID(left);
    for (auto &t : tuples) {
      bool flag = false;
      if (w->op == EQ) {
        if (s->ColTypes[colid] == ColType::Int) {
          flag = TupleEqual(t, colid, std::stoi(right));
        } else if (s->ColTypes[colid] == ColType::Float) {
          flag = TupleEqual(t, colid, std::stof(right));
        } else if (s->ColTypes[colid] == ColType::Varchar) {
          flag = TupleEqual(t, colid, right);
        }
      } else if (w->op == GEQ) {
        if (s->ColTypes[colid] == ColType::Int) {
          flag = TupleGreaterEq(t, colid, std::stoi(right));
        } else if (s->ColTypes[colid] == ColType::Float) {
          flag = TupleGreaterEq(t, colid, std::stof(right));
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

inline std::vector<std::string>
extract_values(std::vector<storage::Tuple *> &tuples, Transaction *tran,
               Scheme *scheme, std::vector<std::string> &colNames) {
  std::vector<std::string> values;
  for (storage::Tuple *t : tuples) {
    if (!tran || TupleCanSee(t, tran)) {
      for (std::string &cname : colNames) {
        const storage::TupleData td = t->data(scheme->get_ColID(cname));
        std::string s;
        if (td.type() == storage::TupleData_Type_INT) {
          s = std::to_string(td.toi());
        } else if (td.type() == storage::TupleData_Type_FLOAT) {
          s = std::to_string(td.tof());
        } else if (td.type() == storage::TupleData_Type_STRING) {
          s = td.tos();
        }
        values.emplace_back(s);
      }
    }
  }
  return values;
}

inline std::vector<std::string> extract_values(
    std::vector<std::pair<storage::Tuple *, storage::Tuple *>> &tuples,
    Transaction *tran, Scheme *scheme_left, Scheme *scheme_right,
    std::vector<std::string> &colNames) {
  std::vector<std::string> values;
  int num_tuples = tuples.size();

  for (int i = 0; i < num_tuples; i++) {
    storage::Tuple *t_left = tuples[i].first;
    storage::Tuple *t_right = tuples[i].second;
    if (!tran || (TupleCanSee(t_left, tran) && TupleCanSee(t_right, tran))) {
      for (std::string &cname : colNames) {
        std::string s;
        storage::TupleData td;
        if (scheme_left->has_ColID(cname)) {
          td = t_left->data(scheme_left->get_ColID(cname));
        } else if (scheme_right->has_ColID(cname)) {
          td = t_right->data(scheme_right->get_ColID(cname));
        } else {
          continue;
        }

        if (td.type() == storage::TupleData_Type_INT) {
          s = std::to_string(td.toi());
        } else if (td.type() == storage::TupleData_Type_FLOAT) {
          s = std::to_string(td.tof());
        } else if (td.type() == storage::TupleData_Type_STRING) {
          s = td.tos();
        }
        values.emplace_back(s);
      }
    }
  }
  return values;
}

inline ResultSet *Executor::selectTable(SelectQuery *q, Plan *p,
                                        Transaction *tran) {

  Scheme *scheme = catalog->FetchScheme(q->From[0]->Name);

  std::vector<std::string> colNames;
  for (auto &c : q->Cols) {
    colNames.emplace_back(c->Name);
  }

  std::vector<storage::Tuple *> tuples = p->scanners->Scan(storage);

  if (!q->Where.empty()) {
    tuples = where(tuples, q->From[0]->Name, q->Where);
  }

  std::vector<std::string> values;
  if (!q->Join.empty()) {
    std::vector<std::pair<storage::Tuple *, storage::Tuple *>> joined_tuples =
        join(tuples, q->From[0]->Name, q->Join[0].first, q->Join[0].second);
    Scheme *scheme_right = catalog->FetchScheme(q->Join[0].first);
    values =
        extract_values(joined_tuples, tran, scheme, scheme_right, colNames);
  } else {
    values = extract_values(tuples, tran, scheme, colNames);
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
  storage->InsertIndex(q->Index, t->data(0).toi(), tid);
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
