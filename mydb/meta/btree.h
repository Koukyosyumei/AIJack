#pragma once
#include "../json/json.hpp"
#include <iostream>
#include <utility>
#include <vector>

const int maxDegree = 3;

using json = nlohmann::json;

class IntItem {
public:
  IntItem(int32_t value) : value(value) {}

  bool operator==(const IntItem &other) const { return value == other.value; }
  bool Less(const IntItem &other) const { return value < other.value; }

  int32_t value;
};

template <typename V>
std::pair<bool, int> FindAtItems(const std::vector<V> &Items, const V &item) {
  for (size_t index = 0; index < Items.size(); ++index) {
    if (!Items[index].Less(item)) {
      if (!item.Less(Items[index])) {
        return std::make_pair(true, index);
      }
      return std::make_pair(false, index);
    }
  }
  return std::make_pair(false, (int)Items.size());
}

template <typename V>
void insertAt(std::vector<V> &items, int index, const V &item) {
  if (index == -1) {
    // items.insert(items.begin(), item);
    index = items.size();
  }
  items.push_back(V(-99)); // Append a nil item

  if (index < items.size() - 1) {
    std::copy_backward(items.begin() + index, items.end() - 1, items.end());
  }
  items[index] = item;
}

template <typename V> class Node {
public:
  std::vector<V> Items;
  std::vector<Node<V> *> Children;

  Node() = default;
  ~Node() {
    for (Node<V> *child : Children) {
      delete child;
    }
  }

  void Insert(const V &item) {
    bool found;
    int index;
    std::tie(found, index) = Find(item);
    if (found) {
      return;
    }

    if (Children.size() == 0) {
      insertAt<V>(Items, index, item);
      // Items.push_back(item);
      // Items.insert(Items.begin() + index, item);

      if (Items.size() == maxDegree) {
        SplitMe();
      }

      return;
    }

    if (index == -1) {
      index = Children.size() - 1;
    }
    if (Children[index]->Items.size() == maxDegree - 1) {
      SplitChild(index, item);

      if (Items.size() == maxDegree) {
        SplitMe();
      }

      return;
    }

    Children[index]->Insert(item);
  }

  std::pair<bool, int> Find(const V &item) const {
    bool found;
    int index;
    std::pair<bool, int> result = FindAtItems<V>(Items, item);
    found = result.first;
    index = result.second;
    if (found) {
      return std::make_pair(found, index);
    }

    if (Children.size() == 0) {
      return std::make_pair(false, -1);
    }

    return Children[index]->Find(item);
  }

  void SplitMe() {
    Node<V> *left = new Node<V>();
    insertAt<V>(left->Items, 0, Items[maxDegree / 2 - 1]);
    // left->Items.insert(left->Items.begin(), Items[maxDegree / 2 - 1]);

    Node<V> *right = new Node<V>();
    insertAt<V>(right->Items, 0, Items[maxDegree / 2 + 1]);
    // right->Items.insert(right->Items.begin(), Items[maxDegree / 2 + 1]);

    V mid = Items[maxDegree / 2];
    Items.assign(1, mid);

    if (!Children.empty()) {
      if (Children.size() == maxDegree + 1) {
        left->Children.push_back(Children[0]);
        left->Children.push_back(Children[1]);

        right->Children.push_back(Children[2]);
        right->Children.push_back(Children[3]);

        Children.clear();
        Children.push_back(left);
        Children.push_back(right);
      } else {
        Children.push_back(left);
        Children.push_back(right);
      }
    } else {
      Children.push_back(left);
      Children.push_back(right);
    }
  }

  void SplitChild(int index, const V &item) {
    bool found;
    int innerIndex;
    std::tie(found, innerIndex) = Children[index]->Find(item);
    insertAt(Children[index]->Items, innerIndex, item);
    // Children[index]->Items.insert(Children[index]->Items.begin() +
    // innerIndex,
    //                               item);

    V leftItem = Children[index]->Items[maxDegree / 2 - 1];
    V midItem = Children[index]->Items[maxDegree / 2];
    V rightItem = Children[index]->Items[maxDegree / 2 + 1];

    Children[index]->Items.erase(
        Children[index]->Items.begin() + maxDegree / 2 - 1,
        Children[index]->Items.begin() + maxDegree / 2 + 1);

    int midIndex;
    std::tie(found, midIndex) = FindAtItems<V>(Items, midItem);
    insertAt(Items, midIndex, midItem);
    // Items.insert(Items.begin() + index, midItem);

    Node<V> *left = new Node<V>();
    left->Items.push_back(leftItem);

    Node<V> *right = new Node<V>();
    right->Items.push_back(rightItem);

    Children.insert(Children.begin() + index, left);
    Children.insert(Children.begin() + index + 1, right);

    std::sort(Children.begin(), Children.end(),
              [](const Node<V> *a, const Node<V> *b) {
                return a->Items[0].Less(b->Items[0]);
              });
  }

  V *Get(const V &key) {
    bool found;
    int index;
    std::tie(found, index) = Find(key);
    if (found) {
      return &Items[index];
    } else if (!Children.empty()) {
      return Children[index]->Get(key);
    }
    return nullptr;
  }
};

template <typename V> class BTree {
public:
  BTree() : top(nullptr), length(0) {}

  void Insert(const V &item) {
    length++;
    if (top == nullptr) {
      top = new Node<V>();
      top->Items.insert(top->Items.begin(), item);
      return;
    }

    top->Insert(item);
  }

  std::pair<bool, int> Find(const V &item) const {
    if (top == nullptr) {
      return std::make_pair(false, -1);
    }

    return top->Find(item);
  }

  V *Get(const V &key) const {
    if (top == nullptr) {
      return nullptr;
    }

    return top->Get(key);
  }

  Node<V> *GetTop() { return top; }

  int Len() const { return length; }

  void SerializeBTree() {
    // json j = *this;
    // std::cout << j.dump() << std::endl;
  }

  static BTree DeserializeBTree(const json &j) {
    BTree bTree;
    bTree = j;
    return bTree;
  }

private:
  Node<V> *top;
  int length;

  // friend void to_json(json &j, const BTree &bTree);
  // friend void from_json(const json &j, BTree &bTree);
};

/*
void to_json(json &j, const BTree &bTree) {
  if (bTree.top != nullptr) {
    j["top"] = *(bTree.top);
  }
  j["length"] = bTree.length;
}

void from_json(const json &j, BTree &bTree) {
  bTree.length = j["length"];
  if (!j["top"].is_null()) {
    bTree.top = new BTree::Node();
    *(bTree.top) = j["top"];
  } else {
    bTree.top = nullptr;
  }
}
*/
