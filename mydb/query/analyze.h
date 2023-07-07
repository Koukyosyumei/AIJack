#pragma once
#include "../meta/meta.h"
#include "../storage/catalog.h"
#include "../storage/storage.h"
#include "../storage/tuple.h"
#include "ast.h"
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

class Query {
public:
  virtual void evalQuery() = 0;
};

class SelectQuery : public Query {
public:
  vector<Column *> Cols;
  vector<Table *> From;
  vector<Expr *> Where;

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
    for (auto name : n->From) {
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

    vector<Column *> cols;
    for (auto colName : n->ColNames) {
      bool found = false;
      for (auto scheme : schemes) {
        for (auto col : scheme->ColNames) {
          if (col == colName) {
            found = true;
            cols.push_back(new Column(colName));
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

    for (auto c : cols) {
      if (c->Name == schemes[0]->PrimaryKey) {
        c->Primary = true;
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
        types.push_back(ColType::Int);
      } else if (typ == "varchar") {
        types.push_back(ColType::Varchar);
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
