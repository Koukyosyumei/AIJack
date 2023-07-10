#pragma once
#include "../compiler/analyze.h"
#include "../compiler/ast.h"
#include "../ml/logisticregression.h"
#include "../ml/rain.h"
#include "../storage/base.h"
#include "../storage/catalog.h"
#include "../storage/storage.h"
#include "../storage/tran.h"
#include "../storage/tuple.h"
#include "../utils/bptree.h"
#include "plan.h"
#include <cstdio>
#include <iostream>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

template <typename T>
std::vector<size_t> kArgmax(const std::vector<T> &vec, size_t k) {
  std::vector<size_t> indices(vec.size());
  std::iota(indices.begin(), indices.end(), 0);

  // Create a min heap of size k+1
  auto comp = [&](size_t i, size_t j) { return vec[i] > vec[j]; };
  std::priority_queue<size_t, std::vector<size_t>, decltype(comp)> pq(comp);

  for (size_t i = 0; i < vec.size(); ++i) {
    pq.push(indices[i]);

    if (pq.size() > k)
      pq.pop();
  }

  std::vector<size_t> result;
  while (!pq.empty()) {
    result.push_back(pq.top());
    pq.pop();
  }

  std::reverse(result.begin(), result.end());
  return result;
}

template <typename T>
void removeIndices(std::vector<T> &vec, const std::vector<size_t> &indices) {
  // Sort the indices in descending order
  std::vector<size_t> sortedIndices(indices);
  std::sort(sortedIndices.rbegin(), sortedIndices.rend());

  // Erase the elements at the specified indices
  for (const auto &index : sortedIndices) {
    if (index < vec.size()) {
      vec.erase(vec.begin() + index);
    }
  }
}

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
  std::vector<int> whereidx(std::vector<storage::Tuple *> &tuples,
                            const std::string &tableName,
                            const std::vector<Expr *> &where);
  std::vector<std::pair<storage::Tuple *, storage::Tuple *>>
  join(std::vector<storage::Tuple *> &left_tuples,
       const std::string &left_table_name, const std::string &rigt_table_name,
       const std::pair<std::string, std::string> joinkeys);
  ResultSet *selectTable(SelectQuery *q, Plan *p, Transaction *tran);
  ResultSet *logregTable(LogregQuery *q, Plan *p, Transaction *tran);
  ResultSet *insertTable(InsertQuery *q, Transaction *tran);
  ResultSet *complaintTable(ComplaintQuery *q, Plan *p, Transaction *tran);
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
  if (s == nullptr) {
    try {
      throw std::runtime_error(
          "Failed to fetch scheme when processing `where`");
    } catch (std::runtime_error &e) {
      std::cerr << "runtime_error: " << e.what() << std::endl;
    }
  }
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
      } else if (w->op == LEQ) {
        if (s->ColTypes[colid] == ColType::Int) {
          flag = TupleLessEq(t, colid, std::stoi(right));
        } else if (s->ColTypes[colid] == ColType::Float) {
          flag = TupleLessEq(t, colid, std::stof(right));
        } else if (s->ColTypes[colid] == ColType::Varchar) {
          flag = TupleLessEq(t, colid, right);
        }
      }

      if (flag) {
        filtered.emplace_back(t);
      }
    }
  }

  return filtered;
}

inline std::vector<int>
Executor::whereidx(std::vector<storage::Tuple *> &tuples,
                   const std::string &tableName,
                   const std::vector<Expr *> &where) {
  std::vector<int> filtered;
  Scheme *s = catalog->FetchScheme(tableName);
  if (s == nullptr) {
    try {
      throw std::runtime_error(
          "Failed to fetch scheme when processing `where`");
    } catch (std::runtime_error &e) {
      std::cerr << "runtime_error: " << e.what() << std::endl;
    }
  }
  for (auto &w : where) {
    std::string left = w->left->v;
    std::string right = w->right->v;
    int colid = s->get_ColID(left);
    for (int i = 0; i < tuples.size(); i++) {
      storage::Tuple *t = tuples[i];
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
      } else if (w->op == LEQ) {
        if (s->ColTypes[colid] == ColType::Int) {
          flag = TupleLessEq(t, colid, std::stoi(right));
        } else if (s->ColTypes[colid] == ColType::Float) {
          flag = TupleLessEq(t, colid, std::stof(right));
        } else if (s->ColTypes[colid] == ColType::Varchar) {
          flag = TupleLessEq(t, colid, right);
        }
      }

      if (flag) {
        filtered.emplace_back(i);
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

inline std::pair<std::vector<int>,
                 std::pair<std::vector<std::vector<float>>, std::vector<float>>>
extract_training_dataset(std::string index_col, std::string target_col,
                         std::vector<storage::Tuple *> &tuples,
                         Transaction *tran, Scheme *scheme,
                         std::vector<std::string> &colNames) {
  std::vector<std::string> values;
  int num_tuples = tuples.size();
  std::vector<std::vector<float>> X;
  std::vector<float> y;
  std::vector<int> index;
  X.reserve(num_tuples);
  y.reserve(num_tuples);
  index.reserve(num_tuples);

  for (storage::Tuple *t : tuples) {
    int id_val;
    float y_val;
    std::vector<float> x_vec;

    if (!tran || TupleCanSee(t, tran)) {
      for (std::string &cname : colNames) {
        const storage::TupleData td = t->data(scheme->get_ColID(cname));
        if (index_col == cname) {
          id_val = td.toi();
        } else if (target_col == cname) {
          if (td.type() == storage::TupleData_Type_INT) {
            y_val = (float)td.toi();
          } else if (td.type() == storage::TupleData_Type_FLOAT) {
            y_val = td.tof();
          }
        } else {
          if (td.type() == storage::TupleData_Type_INT) {
            x_vec.emplace_back((float)td.toi());
          } else if (td.type() == storage::TupleData_Type_FLOAT) {
            x_vec.emplace_back(td.tof());
          }
        }
      }
      index.emplace_back(id_val);
      X.emplace_back(x_vec);
      y.emplace_back(y_val);
    }
  }
  return std::make_pair(index, std::make_pair(X, y));
}

inline std::pair<std::vector<int>,
                 std::pair<std::vector<std::vector<float>>, std::vector<float>>>
extract_training_dataset(
    std::string index_col, std::string target_col,
    std::vector<std::pair<storage::Tuple *, storage::Tuple *>> &tuples,
    Transaction *tran, Scheme *scheme_left, Scheme *scheme_right,
    std::vector<std::string> &colNames) {
  int num_tuples = tuples.size();
  std::vector<std::vector<float>> X;
  std::vector<float> y;
  std::vector<int> index;
  X.reserve(num_tuples);
  y.reserve(num_tuples);
  index.reserve(num_tuples);

  for (int i = 0; i < num_tuples; i++) {
    storage::Tuple *t_left = tuples[i].first;
    storage::Tuple *t_right = tuples[i].second;
    int id_val;
    float y_val;
    std::vector<float> x_vec;

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

        if (index_col == cname) {
          id_val = td.toi();
        } else if (target_col == cname) {
          if (td.type() == storage::TupleData_Type_INT) {
            y_val = (float)td.toi();
          } else if (td.type() == storage::TupleData_Type_FLOAT) {
            y_val = td.tof();
          }
        } else {
          if (td.type() == storage::TupleData_Type_INT) {
            x_vec.emplace_back((float)td.toi());
          } else if (td.type() == storage::TupleData_Type_FLOAT) {
            x_vec.emplace_back(td.tof());
          }
        }
      }
      index.emplace_back(id_val);
      X.emplace_back(x_vec);
      y.emplace_back(y_val);
    }
  }
  return std::make_pair(index, std::make_pair(X, y));
}

inline ResultSet *Executor::logregTable(LogregQuery *q, Plan *p,
                                        Transaction *tran) {
  Scheme *scheme = catalog->FetchScheme(q->selectQuery->From[0]->Name);

  std::vector<std::string> colNames;
  for (auto &c : q->selectQuery->Cols) {
    colNames.emplace_back(c->Name);
  }

  std::vector<storage::Tuple *> tuples = p->scanners->Scan(storage);

  if (!q->selectQuery->Where.empty()) {
    tuples =
        where(tuples, q->selectQuery->From[0]->Name, q->selectQuery->Where);
  }

  std::pair<std::vector<int>,
            std::pair<std::vector<std::vector<float>>, std::vector<float>>>
      training_dataset;
  if (!q->selectQuery->Join.empty()) {
    std::vector<std::pair<storage::Tuple *, storage::Tuple *>> joined_tuples =
        join(tuples, q->selectQuery->From[0]->Name,
             q->selectQuery->Join[0].first, q->selectQuery->Join[0].second);
    Scheme *scheme_right = catalog->FetchScheme(q->selectQuery->Join[0].first);
    training_dataset =
        extract_training_dataset(q->index_col, q->target_col, joined_tuples,
                                 tran, scheme, scheme_right, colNames);
  } else {
    training_dataset = extract_training_dataset(q->index_col, q->target_col,
                                                tuples, tran, scheme, colNames);
  }

  LogisticRegression clf(q->num_iterations, q->lr);
  if (!clf.fit(training_dataset.second.first, training_dataset.second.second)) {
    ResultSet *resultset = new ResultSet();
    resultset->Message =
        "Failed to train a logistic regression model on the given dataset. "
        "Please check the hyper-parameters and the training dataset";
    return resultset;
  }

  storage->saveMLModel(clf, q->model_name + ".logreg");

  std::vector<int> index = training_dataset.first;
  std::vector<std::vector<float>> y_proba =
      clf.predict_proba(training_dataset.second.first);
  std::vector<float> y_pred = clf.predict(training_dataset.second.first);
  float accuracy = clf.score(training_dataset.second.second, y_pred);

  std::string result_table_name =
      "prediction_on_training_data_" + q->model_name;
  std::string primary_key_id = "id_" + q->model_name;
  Scheme *pred_training_result = new Scheme();
  pred_training_result->TblName = result_table_name;
  pred_training_result->ColNames = {primary_key_id, "y_pred_" + q->model_name,
                                    "y_pred_neg_" + q->model_name,
                                    "y_pred_pos_" + q->model_name};
  pred_training_result->ColTypes = {ColType::Int, ColType::Float,
                                    ColType::Float, ColType::Float};
  pred_training_result->PrimaryKey = primary_key_id;
  catalog->Add(pred_training_result);

  bool created = storage->CreateIndex(result_table_name + "_" + primary_key_id);
  if (!created) {
    return nullptr;
  }

  bool inTransaction = tran != nullptr;
  if (!inTransaction) {
    tran = beginTransaction();
  }
  for (int i = 0; i < index.size(); i++) {
    std::vector<Item> row;
    row.emplace_back(Item(index[i]));
    row.emplace_back(Item(y_pred[i]));
    row.emplace_back(Item(y_proba[i][0]));
    row.emplace_back(Item(y_proba[i][1]));
    storage::Tuple *t = NewTuple(tran->Txid(), row);
    std::pair<int, int> tid = storage->InsertTuple(result_table_name, t);
    storage->InsertIndex(result_table_name + "_" + primary_key_id,
                         t->data(0).toi(), tid);
  }
  if (!inTransaction) {
    commitTransaction(tran);
  }

  std::string result_summary = "";
  result_summary += "Trained Parameters:\n";
  for (int i = 0; i < clf.params.size(); i++) {
    result_summary +=
        (" (" + std::to_string(i) + ") : " + std::to_string(clf.params[i][0])) +
        "\n";
  }
  result_summary += "Accuracy (%): " + std::to_string(accuracy * 100) + "\n";
  result_summary +=
      "Predition on the trainig data is stored at `" + result_table_name + "`";

  ResultSet *resultset = new ResultSet();
  resultset->Message = result_summary;
  return resultset;
}

inline ResultSet *Executor::complaintTable(ComplaintQuery *q, Plan *p,
                                           Transaction *tran) {
  Scheme *scheme =
      catalog->FetchScheme(q->logregQuery->selectQuery->From[0]->Name);
  if (scheme == nullptr) {
    return nullptr;
  }

  std::vector<std::string> colNames;
  for (auto &c : q->logregQuery->selectQuery->Cols) {
    colNames.emplace_back(c->Name);
  }

  std::vector<storage::Tuple *> tuples = p->scanners->Scan(storage);

  std::vector<int> filitered_idxs(tuples.size());
  std::iota(filitered_idxs.begin(), filitered_idxs.end(), 0);
  if (!q->logregQuery->selectQuery->Where.empty()) {
    filitered_idxs =
        whereidx(tuples, q->logregQuery->selectQuery->From[0]->Name,
                 q->logregQuery->selectQuery->Where);
  }

  std::pair<std::vector<int>,
            std::pair<std::vector<std::vector<float>>, std::vector<float>>>
      training_dataset;
  if (!q->logregQuery->selectQuery->Join.empty()) {
    std::vector<std::pair<storage::Tuple *, storage::Tuple *>> joined_tuples =
        join(tuples, q->logregQuery->selectQuery->From[0]->Name,
             q->logregQuery->selectQuery->Join[0].first,
             q->logregQuery->selectQuery->Join[0].second);
    Scheme *scheme_right =
        catalog->FetchScheme(q->logregQuery->selectQuery->Join[0].first);
    training_dataset = extract_training_dataset(
        q->logregQuery->index_col, q->logregQuery->target_col, joined_tuples,
        tran, scheme, scheme_right, colNames);
  } else {
    training_dataset = extract_training_dataset(q->logregQuery->index_col,
                                                q->logregQuery->target_col,
                                                tuples, tran, scheme, colNames);
  }

  LogisticRegression clf(q->logregQuery->num_iterations, q->logregQuery->lr);
  storage->loadMLModel(clf, q->logregQuery->model_name + ".logreg");
  std::vector<std::vector<float>> y_proba =
      clf.predict_proba(training_dataset.second.first);
  Rain rain(&clf);
  std::vector<float> influence = rain.getInfluence(
      q->minclass, filitered_idxs, training_dataset.second.first,
      training_dataset.second.second, y_proba);
  std::vector<size_t> topk_influencer = kArgmax(influence, q->k);
  // std::vector<size_t> topk_influencer;
  // for (size_t i : topk_influencer_within) {
  //  topk_influencer.push_back(filitered_idxs[i]);
  //}

  removeIndices(training_dataset.first, topk_influencer);
  removeIndices(training_dataset.second.first, topk_influencer);
  removeIndices(training_dataset.second.second, topk_influencer);

  clf.clear();
  if (!clf.fit(training_dataset.second.first, training_dataset.second.second)) {
    ResultSet *resultset = new ResultSet();
    resultset->Message =
        "Failed to train a logistic regression model on the given dataset. "
        "Please check the hyper-parameters and the training dataset";
    return resultset;
  }

  storage->saveMLModel(clf, q->complaint_name + "_" +
                                q->logregQuery->model_name + ".logreg");

  std::vector<int> index = training_dataset.first;
  std::vector<float> y_pred = clf.predict(training_dataset.second.first);
  std::vector<std::vector<float>> y_proba_fixed =
      clf.predict_proba(training_dataset.second.first);
  float accuracy = clf.score(training_dataset.second.second, y_pred);

  std::string result_table_name = "prediction_on_training_data_" +
                                  q->complaint_name + "_" +
                                  q->logregQuery->model_name;
  std::string primary_key_id =
      "id_" + q->complaint_name + "_" + q->logregQuery->model_name;
  Scheme *pred_training_result = new Scheme();
  pred_training_result->TblName = result_table_name;
  pred_training_result->ColNames = {primary_key_id,
                                    "y_pred_" + q->complaint_name + "_" +
                                        q->logregQuery->model_name};
  pred_training_result->ColTypes = {ColType::Int, ColType::Float};
  pred_training_result->PrimaryKey = primary_key_id;
  catalog->Add(pred_training_result);

  bool created = storage->CreateIndex(result_table_name + "_" + primary_key_id);
  if (!created) {
    return nullptr;
  }

  bool inTransaction = tran != nullptr;
  if (!inTransaction) {
    tran = beginTransaction();
  }
  for (int i = 0; i < index.size(); i++) {
    std::vector<Item> row;
    row.emplace_back(Item(index[i]));
    row.emplace_back(Item(y_pred[i]));
    storage::Tuple *t = NewTuple(tran->Txid(), row);
    std::pair<int, int> tid = storage->InsertTuple(result_table_name, t);
    storage->InsertIndex(result_table_name + "_" + primary_key_id,
                         t->data(0).toi(), tid);
  }
  if (!inTransaction) {
    commitTransaction(tran);
  }

  std::string result_summary = "";
  result_summary += "Fixed Parameters:\n";
  for (int i = 0; i < clf.params.size(); i++) {
    result_summary +=
        (" (" + std::to_string(i) + ") : " + std::to_string(clf.params[i][0])) +
        "\n";
  }
  result_summary += "Accuracy (%): " + std::to_string(accuracy * 100) + "\n";
  result_summary += "Predition on the fixed trainig data is stored at `" +
                    result_table_name + "`";

  ResultSet *resultset = new ResultSet();
  resultset->Message = result_summary;
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
    return resultset;
  }
  if (auto commitQuery = dynamic_cast<CommitQuery *>(q)) {
    commitTransaction(tran);
    ResultSet *resultset = new ResultSet();
    resultset->Message = "Transaction was committed.";
    return resultset;
  }
  if (auto abortQuery = dynamic_cast<AbortQuery *>(q)) {
    abortTransaction(tran);
    ResultSet *resultset = new ResultSet();
    resultset->Message = "Transaction was aborted.";
    return resultset;
  }
  if (auto createTableQuery = dynamic_cast<CreateTableQuery *>(q)) {
    return createTable(createTableQuery);
  }
  if (auto insertQuery = dynamic_cast<InsertQuery *>(q)) {
    return insertTable(insertQuery, tran);
  }
  if (auto logregQuery = dynamic_cast<LogregQuery *>(q)) {
    if (p == nullptr) {
      return nullptr;
    }
    return logregTable(logregQuery, p, tran);
  }
  if (auto complaintQuery = dynamic_cast<ComplaintQuery *>(q)) {
    if (p == nullptr) {
      return nullptr;
    }
    return complaintTable(complaintQuery, p, tran);
  }
  if (auto selectQuery = dynamic_cast<SelectQuery *>(q)) {
    if (p == nullptr) {
      return nullptr;
    }
    return selectTable(selectQuery, p, tran);
  }
  if (auto updateQuery = dynamic_cast<UpdateQuery *>(q)) {
    if (p == nullptr) {
      return nullptr;
    }
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
