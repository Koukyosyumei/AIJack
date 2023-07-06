#pragma once
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

enum class ColType : uint8_t { Int, Varchar };

struct Column {
  std::string Name;
  std::string Type;
  bool Primary;

  Column() {}
  Column(std::string name) : Name(name) {}
};

struct Scheme {
  std::string TblName;
  std::vector<std::string> ColNames;
  std::vector<ColType> ColTypes;
  std::string PrimaryKey;
  std::unordered_map<std::string, int> ColNamesMap;
  bool ColNamesMap_not_initialized = false;

  int get_ColID(std::string colname) {
    if (!ColNamesMap_not_initialized) {
      for (int i = 0; i < ColNames.size(); i++) {
        ColNamesMap.insert({ColNames[i], i});
      }
      ColNamesMap_not_initialized = true;
    }
    return ColNamesMap[colname];
  }
};

class Table {
public:
  std::string Name;
  std::vector<Column> Columns;
};

class ResultSet {
public:
  std::string Message;
  std::vector<std::string> ColNames;
  std::vector<std::string> Values;
};

inline std::string colTypeToString(ColType c) {
  if (c == ColType::Int) {
    return "int";
  }
  if (c == ColType::Varchar) {
    return "varchar";
  }
  return "undefined";
}

class SchemeConverter {
public:
  SchemeConverter(const Scheme &s) : scheme(s) {}

  Table ConvertTable() {
    Table t;
    t.Name = scheme.TblName;

    for (size_t i = 0; i < scheme.ColNames.size(); ++i) {
      Column col;
      col.Name = scheme.ColNames[i];
      col.Type = colTypeToString(scheme.ColTypes[i]);
      t.Columns.push_back(col);
    }

    return t;
  }

private:
  const Scheme &scheme;
};
