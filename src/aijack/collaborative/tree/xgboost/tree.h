#pragma once
#include "../core/tree.h"
#include "node.h"
#include <iostream>
#include <iterator>
#include <limits>
#include <vector>

inline XGBoostNode
make_root_node(vector<XGBoostParty> *parties, vector<float> &y, int num_classes,
               vector<vector<float>> &gradient, vector<vector<float>> &hessian,
               float min_child_weight, float lam, float gamma, float eps,
               int min_leaf, int depth, int active_party_id = -1,
               bool use_only_active_party = false, int n_job = 1) {
  vector<int> idxs(y.size());
  iota(idxs.begin(), idxs.end(), 0);
  for (int i = 0; i < parties->size(); i++) {
    parties->at(i).subsample_columns();
  }

  return XGBoostNode(parties, y, num_classes, gradient, hessian, idxs,
                     min_child_weight, lam, gamma, eps, depth, active_party_id,
                     use_only_active_party, n_job);
}

struct XGBoostTree : public Tree<XGBoostNode> {
  XGBoostTree(vector<XGBoostParty> *parties, vector<float> &y, int num_classes,
              vector<vector<float>> &gradient, vector<vector<float>> &hessian,
              float min_child_weight, float lam, float gamma, float eps,
              int min_leaf, int depth, int active_party_id = -1,
              bool use_only_active_party = false, int n_job = 1)
      : Tree<XGBoostNode>(make_root_node(parties, y, num_classes, gradient,
                                         hessian, min_child_weight, lam, gamma,
                                         eps, min_leaf, depth, active_party_id,
                                         use_only_active_party, n_job)) {}

  XGBoostNode &get_root_xgboost_node() { return dtree; }

  // vector<vector<float>> predict(vector<vector<float>> X)
  //{
  //     return dtree.predict(X);
  // }
};
