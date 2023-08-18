#pragma once
#include <algorithm>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <set>
#include <tuple>
#include <vector>
using namespace std;

/**
 * @brief Bast structure for a node.
 *
 * @tparam PartyType Type of party.
 */
template <typename PartyType> struct Node {
  vector<PartyType> *parties;
  vector<float> &y;
  vector<int> idxs;

  int num_classes;
  int depth;
  int active_party_id;
  int n_job;

  int party_id, record_id;
  int row_count, num_parties;
  float score;
  vector<float> val;

  int best_party_id = -1;
  int best_col_id = -1;
  int best_threshold_id = -1;

  float best_score = -1 * numeric_limits<float>::infinity();
  int is_leaf_flag = -1; // -1:not calculated yer, 0: is not leaf, 1: is leaf

  // Node(){};
  Node(vector<PartyType> *parties_, vector<int> &idxs_, vector<float> &y_)
      : parties(parties_), idxs(idxs_), y(y_) {}

  /**
   * @brief Get the idxs object
   *
   * @return vector<int>
   */
  virtual vector<int> get_idxs() = 0;

  /**
   * @brief Get the party id object
   *
   * @return int
   */
  virtual int get_party_id() = 0;

  /**
   * @brief Get the record id object
   *
   * @return int
   */
  virtual int get_record_id() = 0;

  /**
   * @brief Get the value assigned to this node.
   *
   * @return float
   */
  virtual vector<float> get_val() = 0;

  /**
   * @brief Get the evaluation score of this node.
   *
   * @return float
   */
  virtual float get_score() = 0;

  /**
   * @brief Get the num of parties used for this node.
   *
   * @return int
   */
  virtual int get_num_parties() = 0;

  /**
   * @brief Compute the weight (val) of this node.
   *
   * @return vector<float>
   */
  virtual vector<float> compute_weight() = 0;

  /**
   * @brief Find the best split which gives the best score (gain).
   *
   * @return tuple<int, int, int>
   */
  virtual tuple<int, int, int> find_split() = 0;

  /**
   * @brief Generate the children nodes.
   *
   * @param best_party_id The index of the best party.
   * @param best_col_id The index of the best feature.
   * @param best_threshold_id The index of the best threshold.
   */
  virtual void make_children_nodes(int best_party_id, int best_col_id,
                                   int best_threshold_id) = 0;

  /**
   * @brief Return true if this node is a leaf.
   *
   * @return true
   * @return false
   */
  virtual bool is_leaf() = 0;

  /**
   * @brief Return true if the node is pure; the assigned labels to this node
   * consist of a unique label.
   *
   * @return true
   * @return false
   */
  virtual bool is_pure() = 0;
};
