#pragma once
#include "../thirdparty/json.hpp"
#include <exception>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

using json = nlohmann::json;

const int m = 3;
const int hm = 2;

template <typename K, typename V> struct BPlusTreeMap {
  BPlusTreeMap() {}

  struct BPlusNode {
    bool is_active = false;
    bool is_bottom = false;
    std::vector<K> ks;
    std::vector<V> vs;
    std::vector<BPlusNode *> children;
    BPlusNode *next = nullptr;

    BPlusNode() {}
    BPlusNode(K key, BPlusNode *left, BPlusNode *right, bool is_active,
              bool is_bottom)
        : is_active(is_active), is_bottom(is_bottom) {
      ks.push_back(key);
      children.push_back(left);
      children.push_back(right);
    }
    BPlusNode(K key, V val) {
      ks.push_back(key);
      vs.push_back(val);
      is_active = false;
      is_bottom = true;
    }

    json tojson() {
      if ((!is_bottom) && nullptr != next) {
        throw std::runtime_error("next should be null");
      }
      json node;
      if (is_bottom) {
        node["vs"] = vs;
        node["ks"] = ks;
        node["is_bottom"] = true;
      } else {
        for (BPlusNode *n : children) {
          node["children"].emplace_back(n->tojson());
        }
        node["ks"] = ks;
        node["is_bottom"] = false;
      }
      return node;
    }

    BPlusNode *trim() { return is_bottom ? trim_bottom() : trim_inner(); }
    BPlusNode *insert(K key, V val) {
      return is_bottom ? insert_bottom(key, val) : insert_inner(key, val);
    }
    BPlusNode *del(K key) { is_bottom ? del_bottom(key) : del_inner(key); }
    BPlusNode *split() { return is_bottom ? split_bottom() : split_inner(); }
    K deleteMin() { return is_bottom ? deleteMin_bottom() : deleteMin_inner(); }
    void balanceL(BPlusNode *t, int i) {
      return is_bottom ? balanceL_bottom(t, i) : balanceL_inner(t, i);
    }
    void balanceR(BPlusNode *t, int i) {
      return is_bottom ? balanceR_bottom(t, i) : balanceR_inner(t, i);
    }
    K moveRL(BPlusNode *left, BPlusNode *right) {
      return is_bottom ? moveRL_bottom(left, right) : moveRL_inner(left, right);
    }
    K moveLR(BPlusNode *left, BPlusNode *right) {
      return is_bottom ? moveLR_bottom(left, right) : moveLR_inner(left, right);
    }

    BPlusNode *trim_inner() {
      if (children.size() == 1) {
        return children[0];
      } else {
        return this;
      }
    }

    BPlusNode *trim_bottom() {
      if (ks.size() == 0) {
        return nullptr;
      } else {
        return this;
      }
    }

    BPlusNode *insert_inner(K key, V val) {
      int i;
      for (i = 0; i < ks.size(); i++) {
        if (key < ks[i]) {
          children[i] = children[i]->insert(key, val);
          return balance(i);
        } else if (key == ks[i]) {
          children[i + 1] = children[i + 1]->insert(key, val);
          return balance(i + 1);
        }
      }
      children[i] = children[i]->insert(key, val);
      return balance(i);
    }

    // Ichildrenert a key-value pair (key, x) into the tree rooted at 'this'
    BPlusNode *insert_bottom(K key, V val) {
      if (this == nullptr)
        return new BPlusNode(key, val);

      int i;
      for (i = 0; i < ks.size(); i++) {
        if (key < ks[i])
          return balance(i, key, val);
        else if (key == ks[i]) {
          vs[i] = val;
          return this;
        }
      }
      return balance(i, key, val);
    }

    BPlusNode *balance(int i) {
      BPlusNode *ni = children[i];
      if (!ni->is_active)
        return this;

      ks.insert(ks.begin() + i, ni->ks[0]);
      children[i] = ni->children[1];
      children.insert(children.begin() + i, ni->children[0]);

      return ks.size() < m ? this : split();
    }

    // Balance adjustment during insertion
    BPlusNode *balance(int i, K key, V val) {
      ks.insert(ks.begin() + i, key);
      vs.insert(vs.begin() + i, val);
      return (ks.size() < m) ? this : split();
    }

    BPlusNode *split_inner() {
      int j = hm;
      int i = j - 1;

      BPlusNode *left = this;
      BPlusNode *right = new BPlusNode();
      right->is_bottom = false;

      right->ks.insert(right->ks.end(), left->ks.begin() + j,
                       left->ks.begin() + m);
      right->children.insert(right->children.end(), left->children.begin() + j,
                             left->children.begin() + m + 1);
      left->ks.erase(left->ks.begin() + j, left->ks.begin() + m);
      left->children.erase(left->children.begin() + j,
                           left->children.begin() + m + 1);
      BPlusNode *new_node =
          new BPlusNode(left->ks[i], left, right, true, false);
      left->ks.erase(left->ks.begin() + i);
      return new_node;
    }

    // Split a node with m elements into an active node
    BPlusNode *split_bottom() {
      int j = hm - 1;
      BPlusNode *left = this;
      BPlusNode *right = new BPlusNode();
      right->is_bottom = true;

      right->ks.insert(right->ks.begin(), left->ks.begin() + j, left->ks.end());
      right->vs.insert(right->vs.begin(), left->vs.begin() + j, left->vs.end());
      left->ks.erase(left->ks.begin() + j, left->ks.end());
      left->vs.erase(left->vs.begin() + j, left->vs.end());
      right->next = left->next;
      left->next = right;
      BPlusNode *new_node =
          new BPlusNode(right->ks[0], left, right, true, false);
      return new_node;
    }

    void del_inner(K key) {
      int i;
      for (i = 0; i < ks.size(); i++) {
        if (key < ks[i]) {
          children[i].del(key);
          children[i].balanceL(this, i);
          return;
        } else if (key == ks[i]) {
          ks[i] = children[i + 1].delMin();
          children[i + 1].balanceR(this, i + 1);
          return;
        }
      }
      children[i].del(key);
      children[i].balanceR(this, i);
    }

    // Delete the node with key 'key' from the tree rooted at 'this'
    void del_bottom(K key) {
      for (int i = 0; i < ks.size(); i++) {
        if (key < ks[i])
          return;
        else if (key == ks[i]) {
          ks.erase(ks.begin() + i);
          vs.erase(vs.begin() + i);
          return;
        }
      }
    }

    // Delete the minimum key from the subtree rooted at 'this'
    // Returchildren the new minimum key in the subtree rooted at 'this'
    K deleteMin_inner() {
      K nmin = children[0].deleteMin();
      K spare = ks[0];
      children[0].balanceL(this, 0);
      return (nmin != nullptr) ? nmin : spare;
    }

    K deleteMin_bottom() {
      ks.erase(ks.begin());
      vs.erase(vs.begin());
      return (!ks.empty()) ? ks[0] : nullptr;
    }

    // Balance adjustment during deletion in the left subtree
    void balanceL_inner(BPlusNode *t, int i) {
      BPlusNode *ni = this;
      if (ni->children.size() >= hm)
        return;

      // ni is active
      int j = i + 1;
      K key = t->ks[i];
      BPlusNode *nj = t->children[j];
      nj->is_bottom = false;

      if (nj->children.size() == hm) {
        // nj does not have enough space (merge)
        ni->ks.push_back(key);
        ni->ks.insert(ni->ks.end(), nj->ks.begin(), nj->ks.end());
        ni->children.insert(ni->children.end(), nj->children.begin(),
                            nj->children.end());
        t->ks.erase(t->ks.begin() + i);
        t->children.erase(t->children.begin() + j);
      } else {
        t->ks[i] = moveRL(key, ni, nj); // nj has enough space
      }
    }

    // Balance adjustment during deletion in the left subtree
    void balanceL_bottom(BPlusNode *t, int i) {
      BPlusNode *ni = this;
      if (ni->ks.size() >= hm - 1)
        return;

      // ni is active
      int j = i + 1;
      BPlusNode *nj = t->children[j];
      nj->is_bottom = true;

      if (nj->ks.size() == hm - 1) {
        // nj does not have enough space (merge)
        ni->ks.insert(ni->ks.end(), nj->ks.begin(), nj->ks.end());
        ni->vs.insert(ni->vs.end(), nj->vs.begin(), nj->vs.end());
        t->ks.erase(t->ks.begin() + i);
        t->children.erase(t->children.begin() + j);
        ni->next = nj->next;
      } else {
        t->ks[i] = moveRL(ni, nj); // nj has enough space
      }
    }

    // Balance adjustment during deletion in the right subtree
    void balanceR_inner(BPlusNode *t, int j) {
      BPlusNode *nj = this;
      if (nj->children.size() >= hm)
        return;

      // nj is active
      int i = j - 1;
      K key = t->ks[i];
      BPlusNode *ni = t->children[i];
      ni->is_bottom = false;

      if (ni->children.size() == hm) {
        // ni does not have enough space (merge)
        ni->ks.push_back(key);
        ni->ks.insert(ni->ks.end(), nj->ks.begin(), nj->ks.end());
        ni->children.insert(ni->children.end(), nj->children.begin(),
                            nj->children.end());
        t->ks.erase(t->ks.begin() + i);
        t->children.erase(t->children.begin() + j);
      } else {
        t->ks[i] = moveLR(key, ni, nj); // ni has enough space
      }
    }

    // Balance adjustment during deletion in the right subtree
    void balanceR_bottom(BPlusNode *t, int j) {
      BPlusNode *nj = this;
      if (nj->ks.size() >= hm - 1)
        return;

      // nj is active
      int i = j - 1;
      BPlusNode *ni = t->children[i];
      ni->is_bottom = true;

      if (ni->ks.size() == hm - 1) {
        // ni does not have enough space (merge)
        ni->ks.insert(ni->ks.end(), nj->ks.begin(), nj->ks.end());
        ni->vs.insert(ni->vs.end(), nj->vs.begin(), nj->vs.end());
        t->ks.erase(t->ks.begin() + i);
        t->children.erase(t->children.begin() + j);
        ni->next = nj->next;
      } else {
        t->ks[i] = moveLR(ni, nj); // ni has enough space
      }
    }

    // Take a branch from the right node with enough space
    K moveRL_inner(K key, BPlusNode *left, BPlusNode *right) {
      left->ks.push_back(key);
      left->children.push_back(right->children[0]);
      return right->ks.erase(right->ks.begin());
    }

    // Take a branch from the right node with enough space
    K moveRL_bottom(BPlusNode *l, BPlusNode *r) {
      l->ks.push_back(r->ks[0]);
      l->vs.push_back(r->vs[0]);
      return r->ks[0];
    }

    // Take a branch from the left node with enough space
    K moveLR_inner(K key, BPlusNode *l, BPlusNode *r) {
      int j = l->ks.size();
      int i = j - 1;
      r->ks.insert(r->ks.begin(), key);
      r->children.insert(r->children.begin(), l->children[j]);
      return l->ks.erase(l->ks.begin() + i);
    }

    // Take a branch from the left node with enough space
    K moveLR_bottom(BPlusNode *l, BPlusNode *r) {
      int i = l->ks.size() - 1;
      r->ks.insert(r->ks.begin(), l->ks[i]);
      r->vs.insert(r->vs.begin(), l->vs[i]);
      return r->ks[0];
    }

    BPlusNode *deactivate() {
      if (!is_active) {
        return this;
      }
      return new BPlusNode(ks[0], children[0], children[1], false, false);
    }
  };

  void Insert(K key, V val) {
    if (root == nullptr) {
      root = new BPlusNode(key, val);
    } else {
      root = root->insert(key, val)->deactivate();
    }
  }
  void Delete(K key) {
    root->del(key);
    root = root->trim();
  }

  std::pair<bool, V> Find(K key) {
    if (root == nullptr) {
      std::pair<bool, V> res;
      res.first = false;
      return res;
    }

    BPlusNode *t = root;
    while (!t->is_bottom) {
      int i;
      for (i = 0; i < t->ks.size(); i++) {
        if (key < t->ks[i]) {
          break;
        } else if (key == t->ks[i]) {
          i++;
          break;
        }
      }
      t = t->children[i];
    }
    BPlusNode *u = t;

    for (int i = 0; i < u->ks.size(); i++) {
      if (key == u->ks[i]) {
        return std::make_pair(true, u->vs[i]);
      }
    }
    std::pair<bool, V> res;
    res.first = false;
    return res;
  }

  std::vector<V> FindGreaterEq(K key) {
    if (root == nullptr) {
      std::vector<V> res;
      return res;
    }

    BPlusNode *t = root;
    while (!t->is_bottom) {
      int i;
      for (i = 0; i < t->ks.size(); i++) {
        if (key < t->ks[i]) {
          break;
        } else if (key == t->ks[i]) {
          i++;
          break;
        }
      }
      t = t->children[i];
    }
    BPlusNode *u = t;

    std::vector<V> result;
    for (int i = 0; i < u->ks.size(); i++) {
      if (key <= u->ks[i]) {
        result.push_back(u->vs[i]);
      }
    }
    u = u->next;
    while (u != nullptr) {
      for (V v : u->vs) {
        result.push_back(v);
      }
      u = u->next;
    }
    return result;
  }

  std::vector<K> GetKeys() {
    if (root == nullptr) {
      return {};
    }
    BPlusNode *t = root;
    while (!t->is_bottom) {
      t = t->children[0];
    }
    std::vector<K> keys;
    BPlusNode *u = t;
    while (u != nullptr) {
      keys.insert(keys.end(), u->ks.begin(), u->ks.end());
      u = u->next;
    }

    return keys;
  }

  BPlusNode *fromjson(json &j) {
    BPlusNode *node = new BPlusNode();
    node->is_bottom = j["is_bottom"];
    node->ks = j["ks"].get<std::vector<K>>();

    if (node->is_bottom) {
      node->vs = j["vs"].get<std::vector<V>>();
    } else {
      for (json &jc : j["children"]) {
        node->children.push_back(fromjson(jc));
      }
    }
    return node;
  }

  int Len() { return GetKeys().size(); }

  void SerializeToString(std::string &buffer) {
    buffer = nlohmann::to_string(root->tojson());
  }
  void ParseFromString(std::string &buffer) {
    json j = json::parse(buffer);
    root = fromjson(j);
  }

  BPlusNode *root = nullptr;
};

/*
template <typename V> struct BTree {
  BPlusTreeMap<V, V> bpmap;
  BTree() {}
  void Insert(V val) { bpmap.Insert(val, val); }
  std::pair<bool, V> Find(V key) { return bpmap.Find(key); }
  std::vector<V> FindGreaterEq(V key) { return bpmap.FindGreaterEq(key); }
  int Len() { return bpmap.GetKeys().size(); }

  void SerializeToString(std::string &buffer) {
    buffer = nlohmann::to_string(bpmap.Serialize());
  }
  void ParseFromString(std::string buffer) {
    json j = json::parse(buffer);
    bpmap.Deserialize(j);
  }
  // BPlusNode *GetTop() { return bpmap.root; }
};*/
