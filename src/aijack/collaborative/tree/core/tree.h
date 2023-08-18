#pragma once
#include "../core/nodeapi.h"
#include <iostream>
#include <iterator>
#include <limits>
#include <vector>
using namespace std;

template <typename NodeType> struct Tree {
  NodeType *dtree;
  NodeAPI<NodeType> nodeapi;

  Tree() {}

  /**
   * @brief Get the root node object
   *
   * @return NodeType&
   */
  NodeType &get_root_node() { return *dtree; }

  /**
   * @brief Return the predicted value of the give new sample X
   *
   * @param X the new sample to be predicted
   * @return vector<vector<float>>
   */
  vector<vector<float>> predict(vector<vector<float>> &X) {
    return nodeapi.predict(dtree, X);
  }

  /**
   * @brief Recursively extract the vector of predictions of the training data
   * from the specified node
   *
   * @param node target node
   * @return vector<pair<vector<int>, vector<vector<float>>>>
   */
  vector<pair<vector<int>, vector<vector<float>>>>
  extract_train_prediction_from_node(NodeType *node) {
    if (node->is_leaf()) {
      vector<pair<vector<int>, vector<vector<float>>>> result;
      result.push_back(make_pair(
          node->idxs, vector<vector<float>>(node->idxs.size(), node->val)));
      return result;
    } else {
      vector<pair<vector<int>, vector<vector<float>>>> left_result =
          extract_train_prediction_from_node(node->left);
      vector<pair<vector<int>, vector<vector<float>>>> right_result =
          extract_train_prediction_from_node(node->right);
      left_result.insert(left_result.end(), right_result.begin(),
                         right_result.end());
      return left_result;
    }
  }

  /**
   * @brief Recursively extract the vector of predictions of the training data
   *
   * @return vector<vector<float>>
   */
  vector<vector<float>> get_train_prediction() {
    vector<pair<vector<int>, vector<vector<float>>>> result =
        extract_train_prediction_from_node(dtree);
    vector<vector<float>> y_train_pred(dtree->y.size());
    for (int i = 0; i < result.size(); i++) {
      for (int j = 0; j < result[i].first.size(); j++) {
        y_train_pred[result[i].first[j]] = result[i].second[j];
      }
    }

    return y_train_pred;
  }

  /**
   * @brief Printout the structure of this tree
   *
   * @param show_purity Show leaf purity of each leaf node, if true
   * @param binary_color Color the leaf purity (red: >= 0.8, yellow: 0.8 ~ 0.7,
   * green: 0.7 >)
   * @param target_party_id The id of the active party
   * @return string
   */
  string print(bool show_purity = false, bool binary_color = true,
               int target_party_id = -1) {
    return nodeapi.print(dtree, show_purity, binary_color, target_party_id);
  }

  /**
   * @brief Get the average of leaf purity
   *
   * @return float
   */
  float get_leaf_purity() {
    return nodeapi.get_leaf_purity(dtree, dtree->idxs.size());
  }
};
