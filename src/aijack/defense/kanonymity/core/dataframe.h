#pragma once
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
 * @brief DataFrame Class
 *
 */
class DataFrame
{
public:
  std::vector<std::string> columns;
  std::map<std::string, bool> is_continuous;
  std::map<std::string, std::vector<std::string>> data_categorical;
  std::map<std::string, std::vector<float>> data_continuous;

  int num_col = 0;

  DataFrame() {}

  DataFrame(std::vector<std::string> columns,
            std::map<std::string, bool> is_continuous, int n_rows = 0)
  {
    this->columns = columns;
    this->is_continuous = is_continuous;
    this->num_col = columns.size();

    for (std::string col : this->columns)
    {
      if (is_continuous[col])
      {
        this->data_continuous[col] = std::vector<float>();
        this->data_continuous[col].reserve(n_rows);
      }
      else
      {
        this->data_categorical[col] = std::vector<string>();
        this->data_categorical[col].reserve(n_rows);
      }
    }
  }

  std::map<std::string, std::vector<std::string>> get_data_categorical()
  {
    return data_categorical;
  }

  std::map<std::string, std::vector<float>> get_data_continuous()
  {
    return data_continuous;
  }

  /**
   * @brief Returns the sliced dataframe
   *
   * @param indices
   * @return DataFrame
   */
  DataFrame
  operator[](std::pair<std::vector<std::string>, std::vector<int>> &indices)
  {
    DataFrame df_slice;

    for (std::string col : indices.first)
    {
      df_slice.columns.push_back(col);
      df_slice.is_continuous.insert(std::make_pair(col, is_continuous[col]));

      if (is_continuous[col])
      {
        df_slice.data_continuous[col].reserve(indices.second.size());
        for (int index : indices.second)
        {
          df_slice.insert_continuous(col, this->data_continuous[col][index]);
        }
      }
      else
      {
        df_slice.data_categorical[col].reserve(indices.second.size());
        for (int index : indices.second)
        {
          df_slice.insert_categorical(col, this->data_categorical[col][index]);
        }
      }
    }
    df_slice.num_col = num_col;

    return df_slice;
  }

  /**
   * @brief Inserts a continuous value
   *
   * @param column
   * @param value
   */
  void insert_continuous(std::string column, float value)
  {
    this->data_continuous[column].push_back(value);
  }

  /**
   * @brief Insert a continuous column
   *
   * @param column
   * @param values
   */
  void insert_continuous_column(std::string column, std::vector<float> values)
  {
    this->data_continuous[column] = values;
  }

  /**
   * @brief Inserts a categorical value
   *
   * @param column
   * @param value
   */
  void insert_categorical(std::string column, std::string value)
  {
    this->data_categorical[column].push_back(value);
  }

  /**
   * @brief Insert a categorical column
   *
   * @param column
   * @param values
   */
  void insert_categorical_column(std::string column,
                                 std::vector<std::string> values)
  {
    this->data_categorical[column] = values;
  }

  /**
   * @brief Get the number of rows
   *
   * @return size_t
   */
  size_t get_num_row()
  {
    if (this->is_continuous[this->columns[0]])
    {
      return this->data_continuous[this->columns[0]].size();
    }
    else
    {
      return this->data_categorical[this->columns[0]].size();
    }
  }

  /**
   * @brief Get the minmum number of rows
   *
   * @param max_row
   * @return size_t
   */
  size_t get_min_num_row(size_t max_row = 1000)
  {
    size_t num_row = max_row;
    for (std::string col : this->columns)
    {
      if (this->is_continuous[col])
      {
        num_row = min(num_row, this->data_continuous[col].size());
      }
      else
      {
        num_row = min(num_row, this->data_categorical[col].size());
      }
    }
    return num_row;
  }

  /**
   * @brief Print out thie oject
   *
   * @param max_row
   */
  void print(size_t max_row = 1000)
  {
    for (std::string col : this->columns)
    {
      std::cout << col << " ";
    }
    std::cout << std::endl;

    size_t num_row = this->get_min_num_row(max_row);

    for (int i = 0; i < num_row; i++)
    {
      for (std::string col : this->columns)
      {
        if (this->is_continuous[col])
        {
          std::cout << this->data_continuous[col][i] << " ";
        }
        else
        {
          std::cout << this->data_categorical[col][i] << " ";
        }
      }
      std::cout << std::endl;
    }
  }
};
