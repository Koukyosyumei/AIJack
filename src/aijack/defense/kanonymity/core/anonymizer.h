#pragma once
#include "agg.h"
#include "utils.h"
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
 * @brief Checks whether the dataframe and partition meet k-anonymous property
 *
 * @param df
 * @param partition
 * @param sensitive_column
 * @param k
 * @return true
 * @return false
 */
bool is_k_anonymous(DataFrame &df, std::vector<int> &partition,
                    string sensitive_column, int k = 3) {
  return partition.size() >= k;
}

/**
 * @brief Anonymizes the subset of the dataframe
 *
 * @param df
 * @param anonymized_df
 * @param indices
 * @param column
 * @param is_continuous_flag
 */
void insert_anonymized_quasiid(DataFrame &df, DataFrame &anonymized_df,
                               std::vector<int> &indices, std::string column,
                               bool is_continuous_flag) {
  int tmp_partition_size = indices.size();
  if (is_continuous_flag) {
    float agg_val = aggregte_continuous_column(df, indices, column);
    for (int k = 0; k < tmp_partition_size; k++) {
      anonymized_df.insert_continuous(column, agg_val);
    }
  } else {
    std::string agg_val = aggregate_categorical_column(df, indices, column);
    for (int k = 0; k < tmp_partition_size; k++) {
      anonymized_df.insert_categorical(column, agg_val);
    }
  }
}

struct Mondrian {
  int k;
  std::vector<std::vector<int>> final_partitions;

  Mondrian(int k = 3) { this->k = k; }

  std::vector<std::vector<int>> get_final_partitions() {
    return this->final_partitions;
  }

  /**
   * @brief Splits partitions and insert them to que if they meet k-anonymous
   *
   * @param df
   * @param sorted_spans
   * @param partition
   * @param que
   * @param sensitive_column
   * @return true
   * @return false
   */
  bool insert_partitions_to_que(
      DataFrame &df, std::vector<std::pair<string, float>> &sorted_spans,
      std::vector<int> &partition, std::queue<std::vector<int>> &que,
      std::string sensitive_column) {
    int tmp_num = sorted_spans.size();
    std::pair<std::vector<int>, std::vector<int>> tmp_partition_result;
    bool update_flag = false;
    for (int i = 0; i < tmp_num; i++) {
      tmp_partition_result =
          split_dataframe(df, partition, sorted_spans[i].first);

      if (!is_k_anonymous(df, tmp_partition_result.first, sensitive_column,
                          this->k) ||
          !is_k_anonymous(df, tmp_partition_result.second, sensitive_column,
                          this->k)) {
        continue;
      }

      que.push(tmp_partition_result.first);
      que.push(tmp_partition_result.second);
      update_flag = true;
      break;
    }
    return update_flag;
  }

  /**
   * @brief Repeatedly partitions the specified dataframe until it satisfies
   * k-anonymous
   *
   * @param df
   * @param quasiid_columns
   * @param sensitive_column
   * @param scale_map
   * @return std::vector<std::vector<int>>
   */
  std::vector<std::vector<int>>
  partition_dataframe(DataFrame &df, std::vector<string> &quasiid_columns,
                      std::string sensitive_column,
                      std::map<string, float> scale_map) {
    std::vector<std::vector<int>> final_partitions;
    std::queue<std::vector<int>> que;
    std::vector<int> init_idx(df.get_num_row());
    std::iota(init_idx.begin(), init_idx.end(), 0);
    que.push(init_idx);

    while (!que.empty()) {
      std::vector<int> partition = que.front();
      que.pop();
      std::map<std::string, float> spans = get_spans(df, partition, scale_map);

      std::vector<std::pair<string, float>> sorted_spans =
          sort_map_by_value<string, float>(spans);

      bool update_flag = insert_partitions_to_que(df, sorted_spans, partition,
                                                  que, sensitive_column);

      if (!update_flag) {
        final_partitions.push_back(partition);
      }
    }

    return final_partitions;
  }

  /**
   * @brief Anonymizes the dataframe
   *
   * @param df
   * @param partition
   * @param quasiid_columns
   * @param sensitive_column
   * @return DataFrame
   */
  DataFrame anonymize_dataframe_with_partition(
      DataFrame &df, std::vector<std::vector<int>> &partition,
      std::vector<string> &quasiid_columns, std::string sensitive_column) {
    int num_partition = partition.size();
    int num_quasiids = quasiid_columns.size();
    std::vector<string> new_columns(num_quasiids + 1);
    std::map<std::string, bool> new_is_continuous;
    for (int j = 0; j < num_quasiids; j++) {
      new_columns[j] = quasiid_columns[j];
      new_is_continuous.insert(std::make_pair(
          quasiid_columns[j], df.is_continuous[quasiid_columns[j]]));
    }
    new_columns[num_quasiids] = sensitive_column;
    new_is_continuous.insert(std::make_pair(sensitive_column, false));

    DataFrame anonymized_df =
        DataFrame(new_columns, new_is_continuous, df.get_num_row());
    for (int i = 0; i < num_partition; i++) {
      int tmp_partition_size = partition[i].size();
      for (int j = 0; j < num_quasiids; j++) {
        insert_anonymized_quasiid(df, anonymized_df, partition[i],
                                  quasiid_columns[j],
                                  new_is_continuous[quasiid_columns[j]]);
      }

      for (int k = 0; k < tmp_partition_size; k++) {
        anonymized_df.insert_categorical(
            sensitive_column,
            df.data_categorical[sensitive_column][partition[i][k]]);
      }
    }
    return anonymized_df;
  }

  DataFrame anonymize(DataFrame &df, std::vector<string> &quasiid_columns,
                      std::string sensitive_column) {
    std::map<std::string, float> init_scale;
    for (std::string col : quasiid_columns) {
      init_scale.insert(std::make_pair(col, 1.0));
    }
    std::vector<int> init_idx(df.get_num_row());
    std::iota(init_idx.begin(), init_idx.end(), 0);

    std::map<std::string, float> init_spans =
        get_spans(df, init_idx, init_scale);
    this->final_partitions =
        partition_dataframe(df, quasiid_columns, sensitive_column, init_spans);
    DataFrame anonymized_df = anonymize_dataframe_with_partition(
        df, final_partitions, quasiid_columns, sensitive_column);
    return anonymized_df;
  }
};
