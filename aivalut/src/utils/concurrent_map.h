#include <iostream>
#include <memory>
#include <mutex>
#include <unordered_map>

template <typename K, typename V> class ConcurrentMap {
public:
  void Put(const K &key, const V &value) {
    std::lock_guard<std::mutex> lock(mutex);
    map[key] = std::make_shared<V>(value);
  }

  std::pair<V, bool> Get(const K &key) {
    std::lock_guard<std::mutex> lock(mutex);
    auto it = map.find(key);
    if (it != map.end()) {
      return {*(it->second), true};
    }
    return {V(), false};
  }

private:
  std::unordered_map<K, std::shared_ptr<V>> map;
  std::mutex mutex;
};
