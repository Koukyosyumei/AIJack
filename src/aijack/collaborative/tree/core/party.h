#pragma once
#include <cmath>
#include <numeric>
#include <vector>
#include <iostream>
#include <iterator>
#include <limits>
#include <algorithm>
#include <thread>
#include <set>
#include <tuple>
#include <random>
#include <ctime>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include "../utils/utils.h"
using namespace std;

struct Party
{
    vector<vector<float>> x; // a feature vector of this party
    vector<int> feature_id;  // id of the features
    int party_id;            // id of this party
    int min_leaf;
    float subsample_cols; // ratio of subsampled columuns
    bool use_missing_value;
    int seed;

    int col_count; // the number of columns
    int subsample_col_count;

    int num_classes;

    unordered_map<int, tuple<int, float, int>> lookup_table; // record_id: (feature_id, threshold, missing_value_dir)
    vector<int> temp_column_subsample;
    vector<vector<float>> temp_thresholds; // feature_id->threshold

    Party() {}
    Party(vector<vector<float>> x_, int num_classes_, vector<int> feature_id_, int party_id_,
          int min_leaf_, float subsample_cols_,
          bool use_missing_value_ = false, int seed_ = 0)
    {
        validate_arguments(x_, feature_id_, party_id_, min_leaf_, subsample_cols_);
        x = x_;
        num_classes = num_classes_;
        feature_id = feature_id_;
        party_id = party_id_;
        min_leaf = min_leaf_;
        subsample_cols = subsample_cols_;
        use_missing_value = use_missing_value_;
        seed = seed_;

        col_count = x.at(0).size();
        subsample_col_count = max(1, int(subsample_cols * float(col_count)));
    }

    void validate_arguments(vector<vector<float>> &x_, vector<int> &feature_id_, int &party_id_,
                            int min_leaf_, float subsample_cols_)
    {
        try
        {
            if (x_.size() == 0)
            {
                throw invalid_argument("x is empty");
            }
        }
        catch (std::exception &e)
        {
            std::cerr << e.what() << std::endl;
        }

        try
        {
            if (x_[0].size() != feature_id_.size())
            {
                throw invalid_argument("the number of columns of x is different from the size of feature_id");
            }
        }
        catch (std::exception &e)
        {
            std::cerr << e.what() << std::endl;
        }

        try
        {
            if (subsample_cols_ > 1 || subsample_cols_ < 0)
            {
                throw out_of_range("subsample_cols should be in [1, 0]");
            }
        }
        catch (std::exception &e)
        {
            std::cerr << e.what() << std::endl;
        }
    }

    unordered_map<int, tuple<int, float, int>> get_lookup_table()
    {
        return lookup_table;
    }

    vector<float> get_threshold_candidates(vector<float> &x_col)
    {
        vector<float> x_col_wo_duplicates = remove_duplicates<float>(x_col);
        vector<float> thresholds(x_col_wo_duplicates.size());
        copy(x_col_wo_duplicates.begin(), x_col_wo_duplicates.end(), thresholds.begin());
        sort(thresholds.begin(), thresholds.end());
        return thresholds;
    }

    bool is_left(int record_id, vector<float> &xi)
    {
        bool flag;
        float x_criterion = xi[feature_id[get<0>(lookup_table[record_id])]];
        if (isnan(x_criterion))
        {
            try
            {
                if (!use_missing_value)
                {
                    throw std::runtime_error("given data contains NaN, but use_missing_value is false");
                }
                else
                {
                    flag = get<2>(lookup_table[record_id]) == 0;
                }
            }
            catch (std::exception &e)
            {
                std::cout << e.what() << std::endl;
            }
        }
        else
        {
            flag = x_criterion <= get<1>(lookup_table[record_id]);
        }
        return flag;
    }

    void subsample_columns()
    {
        temp_column_subsample.resize(col_count);
        iota(temp_column_subsample.begin(), temp_column_subsample.end(), 0);
        mt19937 engine(seed);
        seed += 1;
        shuffle(temp_column_subsample.begin(), temp_column_subsample.end(), engine);
    }

    vector<int> split_rows(vector<int> &idxs, int feature_opt_pos, int threshold_opt_pos)
    {
        // feature_opt_idがthreshold_opt_id以下のindexを返す
        int feature_opt_id, missing_dir;
        feature_opt_id = temp_column_subsample[feature_opt_pos % subsample_col_count];
        if (feature_opt_pos > subsample_col_count)
        {
            missing_dir = 1;
        }
        else
        {
            missing_dir = 0;
        }
        int row_count = idxs.size();
        vector<float> x_col(row_count);
        for (int r = 0; r < row_count; r++)
            x_col[r] = x[idxs[r]][feature_opt_id];

        vector<int> left_idxs;
        float threshold = temp_thresholds[feature_opt_pos][threshold_opt_pos];
        for (int r = 0; r < row_count; r++)
            if (((!isnan(x_col[r])) && (x_col[r] <= threshold)) ||
                ((isnan(x_col[r])) && (missing_dir == 1)))
                left_idxs.push_back(idxs[r]);

        return left_idxs;
    }

    int insert_lookup_table(int feature_opt_pos, int threshold_opt_pos)
    {
        int feature_opt_id, missing_dir;
        float threshold_opt;
        feature_opt_id = temp_column_subsample[feature_opt_pos % subsample_col_count];
        threshold_opt = temp_thresholds[feature_opt_pos][threshold_opt_pos];

        if (use_missing_value)
        {
            if (feature_opt_pos > subsample_col_count)
            {
                missing_dir = 1;
            }
            else
            {
                missing_dir = 0;
            }
        }
        else
        {
            missing_dir = -1;
        }

        lookup_table.emplace(lookup_table.size(),
                             make_tuple(feature_opt_id, threshold_opt, missing_dir));
        return lookup_table.size() - 1;
    }
};
