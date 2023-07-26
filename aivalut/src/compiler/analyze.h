#pragma once
#include "../storage/base.h"
#include "../storage/catalog.h"
#include "../storage/storage.h"
#include "../storage/tuple.h"
#include "ast.h"
#include "token.h"
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

class Query {
public:
  virtual void evalQuery() = 0;
  virtual ~Query() {}
};

class SelectQuery : public Query {
public:
  vector<Column *> Cols;
  vector<Table *> From;
  vector<Expr *> Where;
  std::vector<std::pair<std::string, std::pair<std::string, std::string>>> Join;

  void evalQuery() override {}
};

class LogregQuery : public Query {
public:
  std::string model_name;
  std::string index_col;
  std::string target_col;
  SelectQuery *selectQuery;
  int num_iterations;
  float lr;
  void evalQuery() override {}
};

class ComplaintQuery : public Query {
public:
  std::string complaint_name;
  int desired_class;
  int k;
  LogregQuery *logregQuery;
  void evalQuery() override {}
};

class CreateTableQuery : public Query {
public:
  Scheme *scheme;

  void evalQuery() override {}
};

class UpdateQuery : public Query {
public:
  Table *table;
  vector<Column *> Cols;
  vector<Item> Set;
  vector<Expr *> Where;

  void evalQuery() override {}
};

class InsertQuery : public Query {
public:
  Table *table;
  vector<Item> Values;
  string Index;

  void evalQuery() override {}
};

class BeginQuery : public Query {
public:
  void evalQuery() override {}
};

class CommitQuery : public Query {
public:
  void evalQuery() override {}
};

class AbortQuery : public Query {
public:
  void evalQuery() override {}
};

class Analyzer {
  Catalog *catalog;

public:
  Analyzer(Catalog *catalog) : catalog(catalog) {}

  Query *analyzeInsert(InsertStmt *n) {
    InsertQuery *q = new InsertQuery();

    if (!catalog->HasScheme(n->TableName)) {
      try {
        throw runtime_error("insert failed: '" + n->TableName +
                            "' doesn't exist");
      } catch (runtime_error e) {
        std::cerr << "runtime_error: " << e.what() << std::endl;
        return nullptr;
      }
    }
    Scheme *scheme = catalog->FetchScheme(n->TableName);

    Table *t = new Table();
    t->Name = n->TableName;

    if (n->Values.size() != scheme->ColNames.size()) {
      try {
        throw runtime_error(
            "insert failed: 'values' should be the same length");
      } catch (runtime_error e) {
        std::cerr << "runtime_error: " << e.what() << std::endl;
        return nullptr;
      }
    }

    vector<string> lits;
    for (auto l : n->Values) {
      lits.push_back(l->v);
    }

    for (int i = 0; i < lits.size(); i++) {
      if (scheme->ColTypes[i] == ColType::Int) {
        int value = stoi(lits[i]);
        q->Values.emplace_back(Item(value));
      } else if (scheme->ColTypes[i] == ColType::Varchar) {
        q->Values.emplace_back(Item(lits[i]));
      } else if (scheme->ColTypes[i] == ColType::Float) {
        float value = stof(lits[i]);
        q->Values.emplace_back(value);
      } else {
        try {
          throw runtime_error("insert failed: unexpected types parsed");
        } catch (runtime_error e) {
          std::cerr << "runtime_error: " << e.what() << std::endl;
          return nullptr;
        }
      }
    }

    for (auto c : scheme->ColNames) {
      if (scheme->PrimaryKey == c) {
        q->Index = t->Name + "_" + c;
      }
    }

    q->table = t;
    return q;
  }

  Query *analyzeSelect(SelectStmt *n) {
    SelectQuery *q = new SelectQuery();
    vector<Scheme *> schemes;
    for (const std::string &name : n->From) {
      Scheme *scheme = catalog->FetchScheme(name);
      if (!scheme) {
        try {
          throw runtime_error("select failed: table '" + name +
                              "' doesn't exist");
        } catch (runtime_error e) {
          std::cerr << "runtime_error: " << e.what() << std::endl;
          return nullptr;
        }
      }
      schemes.push_back(scheme);
    }

    if (!n->Joins.empty()) {
      for (const std::pair<std::string, std::pair<std::string, std::string>>
               &j : n->Joins) {
        Scheme *scheme = catalog->FetchScheme(j.first);
        if (!scheme) {
          try {
            throw runtime_error("select failed: table '" + j.first +
                                "' doesn't exist");
          } catch (runtime_error e) {
            std::cerr << "runtime_error: " << e.what() << std::endl;
            return nullptr;
          }
        }
        schemes.push_back(scheme);
      }
    }

    vector<Column *> cols;
    for (const std::string &colName : n->ColNames) {

      if (colName == TokenKindToString(STAR)) {
        cols = {};
        for (Scheme *scheme : schemes) {
          for (const std::string &col : scheme->ColNames) {
            cols.emplace_back(new Column(col));
          }
        }
        break;
      }

      bool found = false;
      for (Scheme *scheme : schemes) {
        for (const std::string &col : scheme->ColNames) {
          if (col == colName) {
            found = true;
            cols.emplace_back(new Column(colName));
          }
        }
      }
      if (!found) {
        try {
          throw runtime_error("select failed: column '" + colName +
                              "' doesn't exist");
        } catch (runtime_error e) {
          std::cerr << "runtime_error: " << e.what() << std::endl;
          return nullptr;
        }
      }
    }

    for (Expr *e : n->Wheres) {
      if ((e != nullptr) && (e->left != nullptr)) {
        if (e->left->v == schemes[0]->PrimaryKey) {
          e->left->is_primary = true;
        }
      }
    }

    vector<Table *> tables;
    for (const Scheme *s : schemes) {
      Table *table = new Table;
      SchemeConverter sc(*s);
      *table = sc.ConvertTable();
      tables.push_back(table);
    }

    q->From = tables;
    q->Cols = cols;
    q->Where = n->Wheres;
    q->Join = n->Joins;
    return q;
  }

  Query *analyzeLogreg(LogregStmt *n) {
    LogregQuery *q = new LogregQuery();
    q->model_name = n->model_name;
    q->index_col = n->index_col;
    q->target_col = n->target_col;
    q->num_iterations = n->num_iterations;
    q->lr = n->lr;
    q->selectQuery = dynamic_cast<SelectQuery *>(analyzeSelect(n->selectstmt));
    return q;
  }

  Query *analyzeComplaint(ComplaintStmt *n) {
    ComplaintQuery *q = new ComplaintQuery();
    q->complaint_name = n->complaint_name;
    q->desired_class = n->desired_class;
    q->k = n->k;
    q->logregQuery = dynamic_cast<LogregQuery *>(analyzeLogreg(n->logregstmt));
    return q;
  }

  Query *analyzeUpdate(UpdateStmt *n) {
    UpdateQuery *q = new UpdateQuery();

    if (!catalog->HasScheme(n->TableName)) {
      try {
        throw runtime_error("update failed: '" + n->TableName +
                            "' doesn't exist");
      } catch (runtime_error e) {
        std::cerr << "runtime_error: " << e.what() << std::endl;
        return nullptr;
      }
    }
    Scheme *scheme = catalog->FetchScheme(n->TableName);

    Table *t = new Table();
    t->Name = n->TableName;

    vector<std::string> lits;
    for (auto l : n->Set) {
      lits.push_back(*l);
    }

    for (int i = 0; i < lits.size(); i++) {
      if (scheme->ColTypes[i] == ColType::Int) {
        int value = stoi(lits[i]);
        q->Set.emplace_back(Item(value));
      } else if (scheme->ColTypes[i] == ColType::Varchar) {
        q->Set.emplace_back(Item(lits[i]));
      } else if (scheme->ColTypes[i] == ColType::Float) {
        float value = stof(lits[i]);
        q->Set.emplace_back(Item(value));
      } else {
        try {
          throw runtime_error("update failed: unexpected types parsed");
        } catch (runtime_error e) {
          std::cerr << "runtime_error: " << e.what() << std::endl;
          return nullptr;
        }
      }
    }

    q->table = t;
    q->Where = n->Where;

    return q;
  }

  Query *analyzeCreateTable(CreateTableStmt *n) {
    CreateTableQuery *q = new CreateTableQuery();
    if (n->PrimaryKey.empty()) {
      try {
        throw runtime_error("create table failed: primary key is needed");
      } catch (runtime_error e) {
        std::cerr << "runtime_error: " << e.what() << std::endl;
        return nullptr;
      }
    }

    if (catalog->HasScheme(n->TableName)) {
      try {
        throw runtime_error("create table failed: table name '" + n->TableName +
                            "' already exists");
      } catch (runtime_error e) {
        std::cerr << "runtime_error: " << e.what() << std::endl;
        return nullptr;
      }
    }

    vector<ColType> types;
    for (auto typ : n->ColTypes) {
      if (typ == "int") {
        types.emplace_back(ColType::Int);
      } else if (typ == "float") {
        types.emplace_back(ColType::Float);
      } else if (typ == "varchar") {
        types.emplace_back(ColType::Varchar);
      }
    }

    q->scheme = new Scheme();
    q->scheme->TblName = n->TableName;
    q->scheme->ColNames = n->ColNames;
    q->scheme->ColTypes = types;
    q->scheme->PrimaryKey = n->PrimaryKey;

    return q;
  }

  Query *AnalyzeMain(Stmt *stmt) {
    if (auto concrete = dynamic_cast<SelectStmt *>(stmt)) {
      return analyzeSelect(concrete);
    }
    if (auto concrete = dynamic_cast<LogregStmt *>(stmt)) {
      return analyzeLogreg(concrete);
    }
    if (auto concrete = dynamic_cast<ComplaintStmt *>(stmt)) {
      return analyzeComplaint(concrete);
    }
    if (auto concrete = dynamic_cast<CreateTableStmt *>(stmt)) {
      return analyzeCreateTable(concrete);
    }
    if (auto concrete = dynamic_cast<InsertStmt *>(stmt)) {
      return analyzeInsert(concrete);
    }
    if (auto concrete = dynamic_cast<UpdateStmt *>(stmt)) {
      return analyzeUpdate(concrete);
    }
    if (dynamic_cast<BeginStmt *>(stmt)) {
      return new BeginQuery();
    }
    if (dynamic_cast<CommitStmt *>(stmt)) {
      return new CommitQuery();
    }
    if (dynamic_cast<AbortStmt *>(stmt)) {
      return new AbortQuery();
    }

    try {
      throw runtime_error("failed to analyze query");
    } catch (runtime_error e) {
      std::cerr << "runtime_error: " << e.what() << std::endl;
      return nullptr;
    }
  }
};

inline Analyzer *NewAnalyzer(Catalog *catalog) { return new Analyzer(catalog); }
