#pragma once
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>
#include <vector>

#include "../meta/bptree.h"
#include "data.pb.h"
#include "page.h"

class DiskManager {
public:
  uint64_t toPid(uint64_t tid) { return tid / TupleNumber; }

  Page *fetchPage(const std::string &dirPath, const std::string &tableName,
                  uint64_t pgid) {
    std::string fileName = std::to_string(pgid);
    std::string pagePath = dirPath + "/" + tableName + "/" + fileName;

    std::ifstream file(pagePath, std::ios::binary);
    if (!file) {
      throw std::runtime_error("Failed to fetch page: " + pagePath);
    }

    std::array<char, PageSize> buffer;
    file.read(buffer.data(), buffer.size());
    if (!file) {
      throw std::runtime_error("Failed to read page: " + pagePath);
    }

    Page *page = DeserializePage(buffer);
    if (!page) {
      delete page;
      throw std::runtime_error("Failed to parse page: " + pagePath);
    }

    return page;
  }

  void persist(const std::string &dirName, const std::string &tableName,
               uint64_t pgid, const Page *page) {
    std::string filePath = dirName + "/" + tableName;
    if (mkdir(filePath.c_str(), 0777) != 0 && errno != EEXIST) {
      throw std::runtime_error("Failed to create directory: " + filePath);
    }

    std::string fileName = std::to_string(pgid);
    std::string savePath = filePath + "/" + fileName;

    std::ofstream file(savePath, std::ios::binary);
    if (!file) {
      throw std::runtime_error("Failed to open file for writing: " + savePath);
    }

    std::array<char, PageSize> serializedPage = SerializePage(page);

    file.write(serializedPage.data(), serializedPage.size());
    if (!file) {
      throw std::runtime_error("Failed to write page: " + savePath);
    }
  }

  BTree<int> *readIndex(const std::string &indexName) {
    std::ifstream file(indexName, std::ios::binary);
    if (!file) {
      throw std::runtime_error("Failed to read index file: " + indexName);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string data = buffer.str();

    BTree<int> *btree = new BTree<int>;
    btree->ParseFromString(data);

    return btree;
  }

  void writeIndex(const std::string &dirPath, const std::string &indexName,
                  BTree<int> *tree) {
    std::string savePath = dirPath + "/" + indexName;

    std::ofstream file(savePath, std::ios::binary);
    if (!file) {
      throw std::runtime_error("Failed to open file for writing: " + savePath);
    }

    std::string serializedTree;
    tree->SerializeToString(serializedTree);
    file.write(serializedTree.c_str(), serializedTree.size());
    if (!file) {
      throw std::runtime_error("Failed to write index file: " + savePath);
    }
  }
};
