#pragma once
#include <exception>
#include <iostream>
#include <type_traits>
#include <utility>
#include <vector>

const int m = 3;
const int hm = 2;

template <typename K, typename V> struct BPlusTreeMap {
  BPlusTreeMap() {}

  struct BPlusNode {
    bool is_active;
    bool is_bottom;
    std::vector<K> ks;
    std::vector<V> vs;
    std::vector<BPlusNode *> ns;
    BPlusNode *next = nullptr;

    BPlusNode() {}
    BPlusNode(K key, BPlusNode *left, BPlusNode *right, bool is_active,
              bool is_bottom)
        : is_active(is_active), is_bottom(is_bottom) {
      ks.push_back(key);
      ns.push_back(left);
      ns.push_back(right);
    }
    BPlusNode(K key, V val) {
      ks.push_back(key);
      vs.push_back(val);
      is_active = false;
      is_bottom = true;
    }

    void print() {
      if (is_bottom) {
        std::cout << "Leaf: ";
        for (K key : ks) {
          std::cout << key << ",";
        }
        std::cout << "\n      ";
        for (V val : vs) {
          std::cout << val << ",";
        }
      } else {
        std::cout << "Node: ";
        for (K key : ks) {
          std::cout << key << ",";
        }
        std::cout << "\n";
        for (BPlusNode *n : ns) {
          n->print();
        }
      }
    }

    BPlusNode *trim() { return is_bottom ? trim_bottom() : trim_inner(); }
    BPlusNode *insert(K key, V val) {
      return is_bottom ? insert_bottom(key, val) : insert_inner(key, val);
    }
    BPlusNode *del(K key) { is_bottom ? del_bottom(key) : del_inner(key); }
    // BPlusNode *balance(int i) {
    //  return is_bottom ? balance_bottom(i) : balanceL_inner(i);
    //  }
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
      if (ns.size() == 1) {
        return ns[0];
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
          ns[i] = ns[i]->insert(key, val);
        } else if (key == ks[i]) {
          ns[i + 1] = ns[i + 1]->insert(key, val);
        }
      }
      ns[i] = ns[i]->insert(key, val);
      return balance(i);
    }

    // Insert a key-value pair (key, x) into the tree rooted at 'this'
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
      BPlusNode *ni = ns[i];
      if (!ni->is_active)
        return this;

      ks.insert(ks.begin() + i, ni->ks[0]);
      ns[i] = ni->ns[1];
      ns.insert(ns.begin() + i, ni->ns[0]);

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
      right->ns.insert(right->ns.end(), left->ns.begin() + j,
                       left->ns.begin() + m + 1);
      left->ks.erase(left->ks.begin() + j, left->ks.begin() + m);
      left->ns.erase(left->ns.begin() + j, left->ns.begin() + m + 1);
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
      return new BPlusNode(right->ks[0], left, right, true, false);
    }

    void del_inner(K key) {
      int i;
      for (i = 0; i < ks.size(); i++) {
        if (key < ks[i]) {
          ns[i].del(key);
          ns[i].balanceL(this, i);
          return;
        } else if (key == ks[i]) {
          ks[i] = ns[i + 1].delMin();
          ns[i + 1].balanceR(this, i + 1);
          return;
        }
      }
      ns[i].del(key);
      ns[i].balanceR(this, i);
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
    // Returns the new minimum key in the subtree rooted at 'this'
    K deleteMin_inner() {
      K nmin = ns[0].deleteMin();
      K spare = ks[0];
      ns[0].balanceL(this, 0);
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
      if (ni->ns.size() >= hm)
        return;

      // ni is active
      int j = i + 1;
      K key = t->ks[i];
      BPlusNode *nj = t->ns[j];
      nj->is_bottom = false;

      if (nj->ns.size() == hm) {
        // nj does not have enough space (merge)
        ni->ks.push_back(key);
        ni->ks.insert(ni->ks.end(), nj->ks.begin(), nj->ks.end());
        ni->ns.insert(ni->ns.end(), nj->ns.begin(), nj->ns.end());
        t->ks.erase(t->ks.begin() + i);
        t->ns.erase(t->ns.begin() + j);
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
      BPlusNode *nj = t->ns[j];
      nj->is_bottom = true;

      if (nj->ks.size() == hm - 1) {
        // nj does not have enough space (merge)
        ni->ks.insert(ni->ks.end(), nj->ks.begin(), nj->ks.end());
        ni->vs.insert(ni->vs.end(), nj->vs.begin(), nj->vs.end());
        t->ks.erase(t->ks.begin() + i);
        t->ns.erase(t->ns.begin() + j);
        ni->next = nj->next;
      } else {
        t->ks[i] = moveRL(ni, nj); // nj has enough space
      }
    }

    // Balance adjustment during deletion in the right subtree
    void balanceR_inner(BPlusNode *t, int j) {
      BPlusNode *nj = this;
      if (nj->ns.size() >= hm)
        return;

      // nj is active
      int i = j - 1;
      K key = t->ks[i];
      BPlusNode *ni = t->ns[i];
      ni->is_bottom = false;

      if (ni->ns.size() == hm) {
        // ni does not have enough space (merge)
        ni->ks.push_back(key);
        ni->ks.insert(ni->ks.end(), nj->ks.begin(), nj->ks.end());
        ni->ns.insert(ni->ns.end(), nj->ns.begin(), nj->ns.end());
        t->ks.erase(t->ks.begin() + i);
        t->ns.erase(t->ns.begin() + j);
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
      BPlusNode *ni = t->ns[i];
      ni->is_bottom = true;

      if (ni->ks.size() == hm - 1) {
        // ni does not have enough space (merge)
        ni->ks.insert(ni->ks.end(), nj->ks.begin(), nj->ks.end());
        ni->vs.insert(ni->vs.end(), nj->vs.begin(), nj->vs.end());
        t->ks.erase(t->ks.begin() + i);
        t->ns.erase(t->ns.begin() + j);
        ni->next = nj->next;
      } else {
        t->ks[i] = moveLR(ni, nj); // ni has enough space
      }
    }

    // Take a branch from the right node with enough space
    K moveRL_inner(K key, BPlusNode *left, BPlusNode *right) {
      left->ks.push_back(key);
      left->ns.push_back(right->ns[0]);
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
      r->ns.insert(r->ns.begin(), l->ns[j]);
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
      return new BPlusNode(ks[0], ns[0], ns[1], false, false);
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
      t = t->ns[i];
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

  BPlusNode *root = nullptr;
};
