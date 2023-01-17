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
 * @brief Returns sorted arguments by the values (descending order)
 *
 * @tparam T
 * @param array
 * @return std::vector<size_t>
 */
template <typename T>
std::vector<size_t> argsort(const std::vector<T> &array)
{
  std::vector<size_t> indices(array.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(),
            [&array](int left, int right) -> bool
            {
              // sort indices according to corresponding array element
              return array[left] < array[right];
            });

  return indices;
}

/**
 * @brief Returns the sliced vector
 *
 * @tparam T
 * @param v target vector
 * @param m start position
 * @param n end position
 * @return std::vector<T>
 */
template <typename T>
std::vector<T> slice(std::vector<T> &v, int m, int n)
{
  std::vector<T> vec(n - m + 1);
  std::copy(v.begin() + m, v.begin() + n + 1, vec.begin());
  return vec;
}

/**
 * @brief Gets the number of unique object
 *
 * @param xs
 * @return size_t
 */
size_t get_nunique(std::vector<string> &xs)
{
  std::vector<string> ys;
  std::copy(xs.begin(), xs.end(), std::back_inserter(ys));
  std::sort(ys.begin(), ys.end());
  std::vector<string>::iterator result = std::unique(ys.begin(), ys.end());
  ys.erase(result, ys.end());

  return ys.size();
}

/**
 * @brief Prints out the elements of map
 *
 * @tparam K
 * @tparam V
 * @param m
 */
template <typename K, typename V>
void print_map(std::map<K, V> &m)
{
  for (auto const &pair : m)
  {
    std::cout << "{" << pair.first << ": " << pair.second << "}\n";
  }
}

/**
 * @brief Gets the span of partitioned dataframe
 *
 * @param df
 * @param partition
 * @param scale_map
 * @return std::map<std::string, float>
 */
std::map<std::string, float> get_spans(DataFrame &df,
                                       std::vector<int> &partition,
                                       std::map<string, float> &scale_map)
{
  std::map<std::string, float> span;

  for (std::string col : df.columns)
  {
    std::vector<std::string> tmp_col = {col};
    std::pair<std::vector<string>, std::vector<int>> tmp_pair =
        std::make_pair(tmp_col, partition);
    DataFrame dfp = df[tmp_pair];
    if (df.is_continuous[col])
    {
      span[col] =
          *max_element(dfp.data_continuous[col].begin(), dfp.data_continuous[col].end()) -
          *min_element(dfp.data_continuous[col].begin(), dfp.data_continuous[col].end());
    }
    else
    {
      span[col] = (float)get_nunique(dfp.data_categorical[col]);
    }

    span[col] /= scale_map[col];
  }

  return span;
}

/**
 * @brief Split the dataframe along with continuous number column
 *
 * @param df
 * @param partition
 * @param column
 * @return std::pair<std::vector<int>, std::vector<int>>
 */
std::pair<std::vector<int>, std::vector<int>>
split_dataframe_continuous_column(DataFrame &df, std::vector<int> &partition,
                                  std::string column)
{

  vector<string> tmp_columns;
  tmp_columns.push_back(column);
  std::pair<std::vector<string>, std::vector<int>> indices =
      std::make_pair(tmp_columns, partition);
  DataFrame dfp = df[indices];

  std::vector<size_t> argsorted_indices = argsort(dfp.data_continuous[column]);
  size_t n = argsorted_indices.size();

  std::vector<int> left_partition = std::vector<int>(n / 2);
  std::vector<int> right_partition = std::vector<int>(n - n / 2);
  for (size_t i = 0; i < n / 2; i++)
  {
    left_partition[i] = partition[argsorted_indices[i]];
  }
  for (size_t i = n / 2; i < n; i++)
  {
    right_partition[i - n / 2] = partition[argsorted_indices[i]];
  }

  return std::make_pair(left_partition, right_partition);
}

/**
 * @brief Splits the dataframe along with categorical column
 *
 * @param df
 * @param partition
 * @param column
 * @return std::pair<std::vector<int>, std::vector<int>>
 */
std::pair<std::vector<int>, std::vector<int>>
split_dataframe_categorical_column(DataFrame &df, std::vector<int> &partition,
                                   std::string column)
{

  vector<string> tmp_columns;
  tmp_columns.push_back(column);
  std::pair<std::vector<string>, std::vector<int>> indices =
      std::make_pair(tmp_columns, partition);
  DataFrame dfp = df[indices];

  std::vector<string> unique_values;
  std::copy(dfp.data_categorical[column].begin(),
            dfp.data_categorical[column].end(),
            std::back_inserter(unique_values));
  std::sort(unique_values.begin(), unique_values.end());
  std::vector<string>::iterator tmp_unique_result =
      std::unique(unique_values.begin(), unique_values.end());
  unique_values.erase(tmp_unique_result, unique_values.end());

  size_t n_unique = unique_values.size();
  std::set<string> left_set;
  std::set<string> right_set;
  for (size_t i = 0; i < n_unique / 2; i++)
  {
    left_set.insert(unique_values[i]);
  }
  for (size_t i = n_unique / 2; i < n_unique; i++)
  {
    right_set.insert(unique_values[i]);
  }

  size_t n = dfp.data_categorical[column].size();
  std::vector<int> left_partition;
  left_partition.reserve(n);
  std::vector<int> right_partition;
  right_partition.reserve(n);
  size_t left_size = 0;
  size_t right_size = 0;
  for (size_t i = 0; i < n; i++)
  {
    std::set<string>::iterator it =
        left_set.find(dfp.data_categorical[column][i]);
    if (it != left_set.end())
    {
      left_partition.push_back(partition[i]);
      left_size++;
    }
    else
    {
      right_partition.push_back(partition[i]);
      right_size++;
    }
  }
  left_partition.resize(left_size);
  right_partition.resize(right_size);

  return std::make_pair(left_partition, right_partition);
}

/**
 * @brief Split the dataframe along with the specified column
 *
 * @param df
 * @param partition
 * @param column
 * @return std::pair<std::vector<int>, std::vector<int>>
 */
std::pair<std::vector<int>, std::vector<int>>
split_dataframe(DataFrame &df, std::vector<int> &partition,
                std::string column)
{
  if (df.is_continuous[column])
  {
    return split_dataframe_continuous_column(df, partition, column);
  }
  else
  {
    return split_dataframe_categorical_column(df, partition, column);
  }
}

/**
 * @brief Sorts a map by value (descending order)
 *
 * @tparam K
 * @tparam V
 * @param m
 * @return std::vector<std::pair<K, V>>
 */
template <typename K, typename V>
std::vector<std::pair<K, V>> sort_map_by_value(std::map<K, V> &m)
{
  std::vector<std::pair<K, V>> arr;
  arr.reserve(m.size());
  for (const auto &item : m)
  {
    arr.push_back(item);
  }

  std::sort(arr.begin(), arr.end(),
            [](const auto &x, const auto &y)
            { return x.second > y.second; });

  return arr;
}
