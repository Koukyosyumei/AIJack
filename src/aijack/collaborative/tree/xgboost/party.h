#pragma once
#include "../../../defense/paillier/src/paillier.h"
#include "../core/party.h"
#include <tuple>
using namespace std;

struct XGBoostParty : Party {
  int num_percentile_bin;

  // XGBoostParty() {}
  XGBoostParty(vector<vector<float>> x_, int num_classes_,
               vector<int> feature_id_, int party_id_, int min_leaf_,
               float subsample_cols_, int num_precentile_bin_ = 256,
               bool use_missing_value_ = false, int seed_ = 0)
      : Party(x_, num_classes_, feature_id_, party_id_, min_leaf_,
              subsample_cols_, use_missing_value_, seed_) {
    num_percentile_bin = num_precentile_bin_;
  }

  vector<float> get_threshold_candidates(vector<float> &x_col) {
    if (x_col.size() > num_percentile_bin) {
      vector<float> probs(num_percentile_bin);
      for (int i = 1; i <= num_percentile_bin; i++)
        probs[i] = float(i) / float(num_percentile_bin);
      vector<float> percentiles_candidate = Quantile<float>(x_col, probs);
      vector<float> percentiles =
          remove_duplicates<float>(percentiles_candidate);
      return percentiles;
    } else {
      vector<float> x_col_wo_duplicates = remove_duplicates<float>(x_col);
      vector<float> percentiles(x_col_wo_duplicates.size());
      copy(x_col_wo_duplicates.begin(), x_col_wo_duplicates.end(),
           percentiles.begin());
      sort(percentiles.begin(), percentiles.end());
      return percentiles;
    }
  }

  vector<vector<tuple<vector<float>, vector<float>, float, vector<float>>>>
  greedy_search_split(const vector<vector<float>> &gradient,
                      const vector<vector<float>> &hessian, vector<float> &y,
                      vector<int> &idxs) {
    // feature_id -> [(grad hess)]
    // the threshold of split_candidates_grad_hess[i][j] = temp_thresholds[i][j]
    int num_thresholds;
    if (use_missing_value)
      num_thresholds = subsample_col_count * 2;
    else
      num_thresholds = subsample_col_count;
    vector<vector<tuple<vector<float>, vector<float>, float, vector<float>>>>
        split_candidates_grad_hess(num_thresholds);
    temp_thresholds = vector<vector<float>>(num_thresholds);

    int row_count = idxs.size();
    int recoed_id = 0;

    int grad_dim = gradient[0].size();

    for (int i = 0; i < subsample_col_count; i++) {
      // extract the necessary data
      int k = temp_column_subsample[i];
      vector<float> x_col(row_count);

      int not_missing_values_count = 0;
      int missing_values_count = 0;
      for (int r = 0; r < row_count; r++) {
        if (!isnan(x[idxs[r]][k])) {
          x_col[not_missing_values_count] = x[idxs[r]][k];
          not_missing_values_count += 1;
        } else {
          missing_values_count += 1;
        }
      }
      x_col.resize(not_missing_values_count);

      vector<int> x_col_idxs(not_missing_values_count);
      iota(x_col_idxs.begin(), x_col_idxs.end(), 0);
      sort(x_col_idxs.begin(), x_col_idxs.end(),
           [&x_col](size_t i1, size_t i2) { return x_col[i1] < x_col[i2]; });

      sort(x_col.begin(), x_col.end());

      // get percentiles of x_col
      vector<float> percentiles = get_threshold_candidates(x_col);

      // enumerate all threshold value (missing value goto right)
      int current_min_idx = 0;
      int cumulative_left_size = 0;
      for (int p = 0; p < percentiles.size(); p++) {
        vector<float> temp_grad(grad_dim, 0);
        vector<float> temp_hess(grad_dim, 0);
        float temp_left_size = 0;
        vector<float> temp_left_y_class_cnt(num_classes, 0);

        for (int r = current_min_idx; r < not_missing_values_count; r++) {
          if (x_col[r] <= percentiles[p]) {
            for (int c = 0; c < grad_dim; c++) {
              temp_grad[c] += gradient[idxs[x_col_idxs[r]]][c];
              temp_hess[c] += hessian[idxs[x_col_idxs[r]]][c];
            }
            temp_left_size += 1.0;
            temp_left_y_class_cnt[int(y[idxs[x_col_idxs[r]]])] += 1.0;
            cumulative_left_size += 1;
          } else {
            current_min_idx = r;
            break;
          }
        }

        if (cumulative_left_size >= min_leaf &&
            row_count - cumulative_left_size >= min_leaf) {
          split_candidates_grad_hess[i].push_back(make_tuple(
              temp_grad, temp_hess, temp_left_size, temp_left_y_class_cnt));
          temp_thresholds[i].push_back(percentiles[p]);
        }
      }

      // enumerate missing value goto left
      if (use_missing_value) {
        int current_max_idx = not_missing_values_count - 1;
        int cumulative_right_size = 0;
        for (int p = percentiles.size() - 1; p >= 0; p--) {
          vector<float> temp_grad(grad_dim, 0);
          vector<float> temp_hess(grad_dim, 0);
          float temp_left_size = 0;
          vector<float> temp_left_y_class_cnt(num_classes, 0);

          for (int r = current_max_idx; r >= 0; r--) {
            if (x_col[r] >= percentiles[p]) {
              for (int c = 0; c < grad_dim; c++) {
                temp_grad[c] += gradient[idxs[x_col_idxs[r]]][c];
                temp_hess[c] += hessian[idxs[x_col_idxs[r]]][c];
              }
              temp_left_size += 1.0;
              temp_left_y_class_cnt[int(y[idxs[x_col_idxs[r]]])] += 1.0;
              cumulative_right_size += 1;
            } else {
              current_max_idx = r;
              break;
            }
          }

          if (cumulative_right_size >= min_leaf &&
              row_count - cumulative_right_size >= min_leaf) {
            split_candidates_grad_hess[i + subsample_col_count].push_back(
                make_tuple(temp_grad, temp_hess, temp_left_size,
                           temp_left_y_class_cnt));
            temp_thresholds[i + subsample_col_count].push_back(percentiles[p]);
          }
        }
      }
    }

    return split_candidates_grad_hess;
  }
};
