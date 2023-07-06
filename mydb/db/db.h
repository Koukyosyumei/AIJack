#pragma once
#include <csignal>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "../meta/meta.h"
#include "../query/analyze.h"
#include "../query/ast.h"
#include "../query/executor.h"
#include "../query/parse.h"
#include "../query/plan.h"
#include "../query/token.h"
#include "../storage/storage.h"

class dbSession {
public:
  Transaction *tran;
};

class MyDb {
public:
  std::unordered_map<std::string, dbSession *> contexts;
  Storage *storage;
  Catalog *catalog;
  TransactionManager *tranManager;
  std::string home;
  int exit;

  static MyDb *dbinstance;

  MyDb() {
    exit = 0;
    contexts = std::unordered_map<std::string, dbSession *>();
    home = "";
  }

  ~MyDb() {
    delete storage;
    delete catalog;
    delete tranManager;
  }

  static void SignalHandler(int signal) {
    std::cout << "ctrl+c detected, shutdown soon...." << std::endl;
    dbinstance->Terminate();
  }

  void Init() {
    // set up the signal handler
    dbinstance = this;
    std::signal(SIGINT, SignalHandler);
  }

  void Execute(const std::string &q, const std::string &userAgent,
               std::string &result, std::string &error) {
    Transaction *trn = nullptr;
    auto it = contexts.find(userAgent);
    if (it != contexts.end()) {
      trn = it->second->tran;
    }

    Tokenizer tokenizer(q);
    std::vector<Token *> tokens = tokenizer.Tokenize();

    Parser parser(tokens);
    std::vector<std::string> errs;
    Stmt *node = parser.Parse(errs);
    if (!errs.empty()) {
      error = errs[0];
      return;
    }

    Analyzer analyzer(catalog);
    Query *analyzedQuery = analyzer.AnalyzeMain(node);

    Planner planner(analyzedQuery);
    Plan *plan = planner.planMain();

    Executor executor(storage, catalog, tranManager);
    ResultSet *resultSet = executor.executeMain(analyzedQuery, plan, trn);

    result = StringifyResultSet(resultSet);
  }

  std::string StringifyResultSet(const ResultSet *resultSet) {
    std::ostringstream oss;
    oss << resultSet->Message;
    for (const auto &col : resultSet->ColNames) {
      oss << col;
    }
    oss << "\n";

    for (size_t i = 0; i < resultSet->Values.size(); i++) {
      oss << resultSet->Values[i];
      if ((i + 1) % resultSet->ColNames.size() == 0) {
        oss << "\n";
      }
    }

    return oss.str();
  }

  void Terminate() {
    if (access(home.c_str(), F_OK) != 0) {
      mkdir(home.c_str(), 0777);
    }

    SaveCatalog(home, catalog);
    std::cout << "`catalog.db` has completely saved in " << home << std::endl;

    storage->Terminate();
    std::cout << "data files have completely saved in " << home << std::endl;

    exit = 0;
  }
};

MyDb *MyDb::dbinstance;

inline MyDb *NewMyDb() {
  std::string home;
  char *bogoHome = getenv("MYDB_HOME");
  if (bogoHome == nullptr) {
    // default
    home = ".mydb/";
    if (access(home.c_str(), F_OK) != 0) {
      mkdir(home.c_str(), 0777);
    }
  } else {
    home = bogoHome;
  }

  Catalog *catalog = LoadCatalog(home);
  if (catalog == nullptr) {
    return nullptr;
  }

  MyDb *db = new MyDb();
  db->home = home;
  db->storage = new Storage(home);
  db->catalog = catalog;
  db->tranManager = new TransactionManager();

  return db;
}
