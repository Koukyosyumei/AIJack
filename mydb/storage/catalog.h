#include "../json/json.hpp"
#include "../meta/meta.h"
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
    // Add other scheme properties as needed
    catalog->Schemes.push_back(scheme);
  }

  return catalog;
}

inline void SaveCatalog(const std::string &dirPath, Catalog *c) {
  json root;
  for (const auto &scheme : c->Schemes) {
    json schemeJson;
    schemeJson["TblName"] = scheme->TblName;
    // Add other scheme properties as needed
    root["Schemes"].emplace_back(schemeJson);
  }

  std::ofstream file(dirPath + "/" + catalogName);
  file << root;
  file.close();
}
