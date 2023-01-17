#pragma once
#include "dataframe.h"
#include <algorithm>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <string>
#include <vector>
using namespace std;

/**
 * @brief Aggregates categorical column
 *
 * @param df
 * @param indices
 * @param column
 * @return std::string
 */
std::string aggregate_categorical_column(DataFrame &df,
                                         std::vector<int> &indices,
                                         std::string column) {
  std::set<string> s;
  int n = indices.size();
  for (int i = 0; i < n; i++) {
    s.insert(df.data_categorical[column][indices[i]]);
  }

  std::set<string>::iterator it;
  std::string result = "";
  for (it = s.begin(); it != s.end(); ++it) {
    result += "_" + *it;
  }
  return result.substr(1);
}

/**
 * @brief Aggregates continuous column
 *
 * @param df
 * @param indices
 * @param column
 * @return float
 */
float aggregte_continuous_column(DataFrame &df, std::vector<int> &indices,
                                 std::string column) {
  int n = indices.size();
  float sum_val = 0;
  for (int i = 0; i < n; i++) {
    sum_val += df.data_continuous[column][indices[i]];
  }
  return sum_val / float(n);
}
