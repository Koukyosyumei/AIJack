#pragma once
#include "../json/json.hpp"
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

const int maxDegree = 3;

using json = nlohmann::json;

template <typename V> class ItemType {
public:
  ItemType() {}
  ItemType(V value) : value(value) {}

  bool operator==(const ItemType<V> &other) const {
    return value == other.value;
  }
  bool operator<(const ItemType<V> &other) const { return value < other.value; }

  V value;
};

template <typename V>
std::pair<bool, int> FindAtItems(const std::vector<ItemType<V>> &Items,
                                 const ItemType<V> &item) {
  for (size_t index = 0; index < Items.size(); ++index) {
    if (!(Items[index] < item)) {
      if (!(item < Items[index])) {
        // the item already exists
        return std::make_pair(true, index);
      }
      return std::make_pair(false, index);
    }
  }
  // item is smaller than any element of items
  return std::make_pair(false, (int)Items.size());
}

template <typename V>
void insertAt(std::vector<ItemType<V>> &items, int index,
              const ItemType<V> &item) {
  if (index == -1) {
    items.push_back(item);
  } else {
    items.resize(items.size() + 1);

    if (index < items.size() - 1) {
      std::copy_backward(items.begin() + index, items.end() - 1, items.end());
    }
    items[index] = item;
  }
}

template <typename V>
void insertAt(std::vector<ItemType<V>> &items, int index, V val) {
  ItemType<V> item(val);
  insertAt(items, index, item);
}

template <typename V> class Node {
public:
  std::vector<ItemType<V>> Items;
  std::vector<Node<V> *> Children;

  Node() = default;
  ~Node() {
    for (Node<V> *child : Children) {
      delete child;
    }
  }

  void Insert(const ItemType<V> &item) {
    bool found;
    int index;
    std::tie(found, index) = Find(item);
    if (found) {
      // item is already within the node
      return;
    }

    if (Children.size() == 0) {
      insertAt<V>(Items, index, item);

      if (Items.size() == maxDegree) {
        SplitMe();
      }

      return;
    }

    if (index == -1) {
      throw std::runtime_error("Something wrong happened within Node.Insert.");
    }

    if (Children[index]->Items.size() == maxDegree - 1) {
      SplitChild(index, item.value);

      if (Items.size() == maxDegree) {
        SplitMe();
      }

      return;
    }

    Children[index]->Insert(item);
  }

  std::pair<bool, int> Find(const V val) const {
    ItemType<V> item = ItemType<V>(val);
    return Find(item);
  }

  std::pair<bool, int> Find(const ItemType<V> &item) const {
    bool found;
    int index;
    std::tie(found, index) = FindAtItems<V>(Items, item);
    if (found) {
      return std::make_pair(found, index);
    }

    // item is not found in the subtree
    if (Children.size() == 0) {
      return std::make_pair(false, -1);
    }

    if (index == -1) {
      throw std::runtime_error("something wrong happened within Node.Find");
    }

    return Children[index]->Find(item);
  }

  void SplitMe() {
    Node<V> *left = new Node<V>();
    insertAt<V>(left->Items, 0, Items[maxDegree / 2 - 1]);

    Node<V> *right = new Node<V>();
    insertAt<V>(right->Items, 0, Items[maxDegree / 2 + 1]);

    ItemType<V> mid = Items[maxDegree / 2];
    Items.push_back(mid);

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
  }

  void SplitChild(int index, const V &item) {
    bool found;
    int innerIndex;
    std::tie(found, innerIndex) = Children[index]->Find(item);
    insertAt(Children[index]->Items, innerIndex, item);

    ItemType<V> leftItem = Children[index]->Items[maxDegree / 2 - 1];
    ItemType<V> midItem = Children[index]->Items[maxDegree / 2];
    ItemType<V> rightItem = Children[index]->Items[maxDegree / 2 + 1];

    Children.erase(Children.begin() + index);

    int midIndex;
    std::tie(found, midIndex) = FindAtItems<V>(Items, midItem);
    insertAt(Items, midIndex, midItem);

    Node<V> *left = new Node<V>();
    left->Items.push_back(leftItem);

    Node<V> *right = new Node<V>();
    right->Items.push_back(rightItem);

    Children.push_back(left);
    Children.push_back(right);

    std::sort(Children.begin(), Children.end(),
              [](const Node<V> *a, const Node<V> *b) {
                return a->Items[0] < (b->Items[0]);
              });
  }

  ItemType<V> *Get(const V &key) {
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

  void Insert(ItemType<V> item) {
    length++;
    if (top == nullptr) {
      top = new Node<V>();
      insertAt(top->Items, 0, item);
      return;
    }
    top->Insert(item);
  }

  std::pair<bool, int> Find(ItemType<V> item) const {
    if (top == nullptr) {
      return std::make_pair(false, -1);
    }

    return top->Find(item);
  }

  std::pair<bool, int> Find(const V &item) const {
    if (top == nullptr) {
      return std::make_pair(false, -1);
    }

    return top->Find(item);
  }

  ItemType<V> *Get(const V &key) const {
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
