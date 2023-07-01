#pragma once
#include <iostream>
#include <list>
#include <mutex>
#include <unordered_map>
#include <vector>

template <typename K, typename V> class Lru {
public:
  Lru(int cap) : cap(cap) {}

  V Insert(const K &key, const V &value) {
    std::lock_guard<std::mutex> lock(mutex);

    V victim;
    auto it = items.find(key);
    if (it != items.end()) {
      evictList.erase(it->second);
    }

    if (needEvict()) {
      victim = evictList.back().second;
      items.erase(evictList.back().first);
      evictList.pop_back();
    }

    evictList.push_front({key, value});
    items[key] = evictList.begin();

    return victim;
  }

  V Get(const K &key) {
    std::lock_guard<std::mutex> lock(mutex);

    auto it = items.find(key);
    if (it != items.end()) {
      evictList.splice(evictList.begin(), evictList, it->second);
      return it->second->second;
    }

    return V();
  }

  int Len() {
    std::lock_guard<std::mutex> lock(mutex);
    return evictList.size();
  }

  std::vector<V> GetAll() {
    std::lock_guard<std::mutex> lock(mutex);
    std::vector<V> result;
    result.reserve(items.size());

    for (const auto &pair : items) {
      result.push_back(pair.second->second);
    }

    return result;
  }

private:
  bool needEvict() { return evictList.size() > cap; }

  int cap;
  std::list<std::pair<K, V>> evictList;
  std::unordered_map<K, typename std::list<std::pair<K, V>>::iterator> items;
  std::mutex mutex;
};
