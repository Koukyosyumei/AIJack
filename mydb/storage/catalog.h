#pragma once
#include "../storage/base.h"
#include "../thirdparty/json.hpp"
#include <fstream>
#include <iostream>
#include <mutex>
#include <vector>

using json = nlohmann::json;

const std::string catalogName = "catalog.db";

class Catalog {
public:
  std::vector<Scheme *> Schemes;
  std::mutex mutex;

  Catalog() : mutex() {}

  void Add(Scheme *scheme) {
    std::lock_guard<std::mutex> lock(mutex);
    Schemes.push_back(scheme);
  }

  bool HasScheme(const std::string &tblName) {
    return FetchScheme(tblName) != nullptr;
  }

  Scheme *FetchScheme(const std::string &tblName) {
    std::lock_guard<std::mutex> lock(mutex);
    for (auto scheme : Schemes) {
      if (scheme->TblName == tblName) {
        return scheme;
      }
    }
    return nullptr;
  }
};

inline Catalog *NewEmptyCatalog() { return new Catalog(); }

inline Catalog *LoadCatalog(const std::string &catalogPath) {
  std::ifstream file(catalogPath + "/" + catalogName);
  if (!file.is_open()) {
    return NewEmptyCatalog();
  }

  json root;
  file >> root;
  file.close();
  Catalog *catalog = new Catalog();
  for (const auto &schemeJson : root["Schemes"]) {
    Scheme *scheme = new Scheme();
    scheme->TblName = schemeJson["TblName"];
    scheme->ColNames = schemeJson["ColNames"].get<std::vector<std::string>>();
    scheme->ColTypes = schemeJson["ColTypes"].get<std::vector<ColType>>();
    scheme->PrimaryKey = schemeJson["PrimaryKey"];
    catalog->Schemes.push_back(scheme);
  }

  return catalog;
}

inline void SaveCatalog(const std::string &dirPath, Catalog *c) {
  json root;
  for (const auto &scheme : c->Schemes) {
    json schemeJson;
    schemeJson["TblName"] = scheme->TblName;
    schemeJson["ColNames"] = scheme->ColNames;
    schemeJson["ColTypes"] = scheme->ColTypes;
    schemeJson["PrimaryKey"] = scheme->PrimaryKey;
    root["Schemes"].emplace_back(schemeJson);
  }

  std::ofstream file(dirPath + "/" + catalogName);
  file << root;
  file.close();
}
