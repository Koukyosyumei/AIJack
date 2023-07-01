#pragma once
#include "../json/json.hpp"
#include <iostream>
#include <vector>

const int maxDegree = 3;

using json = nlohmann::json;

class IntItem {
public:
  IntItem(int32_t value) : value(value) {}

  bool Less(const IntItem &other) const { return value < other.value; }

  int32_t value;
};

class BTree {
public:
  BTree() : top(nullptr), length(0) {}

  void Insert(const IntItem &item) {
    length++;
    if (top == nullptr) {
      top = new Node();
      top->Items.insert(top->Items.begin(), item);
      return;
    }

    top->Insert(item);
  }

  std::pair<bool, int> Find(const IntItem &item) const {
    if (top == nullptr) {
      return std::make_pair(false, -1);
    }

    return top->Find(item);
  }

  IntItem *Get(const IntItem &key) const {
    if (top == nullptr) {
      return nullptr;
    }

    return top->Get(key);
  }

  int Len() const { return length; }

  void SerializeBTree() {
    json j = *this;
    std::cout << j.dump() << std::endl;
  }

  static BTree DeserializeBTree(const json &j) {
    BTree bTree;
    bTree = j;
    return bTree;
  }

private:
  class Node {
  public:
    std::vector<IntItem> Items;
    std::vector<Node *> Children;

    Node() = default;
    ~Node() {
      for (Node *child : Children) {
        delete child;
      }
    }

    void Insert(const IntItem &item) {
      bool found;
      int index;
      std::tie(found, index) = Find(item);
      if (found) {
        return;
      }

      if (Children.empty()) {
        Items.insert(Items.begin() + index, item);

        if (Items.size() == maxDegree) {
          SplitMe();
        }

        return;
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

    std::pair<bool, int> Find(const IntItem &item) const {
      for (size_t index = 0; index < Items.size(); ++index) {
        if (!Items[index].Less(item)) {
          if (!item.Less(Items[index])) {
            return std::make_pair(true, index);
          }
          return std::make_pair(false, index);
        }
      }
      return std::make_pair(false, Items.size());
    }

    void SplitMe() {
      Node *left = new Node();
      left->Items.insert(left->Items.begin(), Items[maxDegree / 2 - 1]);

      Node *right = new Node();
      right->Items.insert(right->Items.begin(), Items[maxDegree / 2 + 1]);

      IntItem mid = Items[maxDegree / 2];
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

    void SplitChild(int index, const IntItem &item) {
      bool found;
      int innerIndex;
      std::tie(found, innerIndex) = Children[index]->Find(item);
      Children[index]->Items.insert(Children[index]->Items.begin() + innerIndex,
                                    item);

      IntItem leftItem = Children[index]->Items[maxDegree / 2 - 1];
      IntItem midItem = Children[index]->Items[maxDegree / 2];
      IntItem rightItem = Children[index]->Items[maxDegree / 2 + 1];

      Children[index]->Items.erase(
          Children[index]->Items.begin() + maxDegree / 2 - 1,
          Children[index]->Items.begin() + maxDegree / 2 + 1);

      Items.insert(Items.begin() + index, midItem);

      Node *left = new Node();
      left->Items.push_back(leftItem);

      Node *right = new Node();
      right->Items.push_back(rightItem);

      Children.insert(Children.begin() + index, left);
      Children.insert(Children.begin() + index + 1, right);

      std::sort(Children.begin(), Children.end(),
                [](const Node *a, const Node *b) {
                  return a->Items[0].Less(b->Items[0]);
                });
    }

    IntItem *Get(const IntItem &key) {
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

  Node *top;
  int length;

  friend void to_json(json &j, const BTree &bTree);
  friend void from_json(const json &j, BTree &bTree);
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
