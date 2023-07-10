#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numeric>
#include <set>
#include <string>
#include <vector>
using namespace std;

inline float trapz(vector<float> x, vector<float> y) {
  float res = 0;
  int num_elements = x.size();
  for (int i = 1; i < num_elements; i++) {
    res += (x[i] - x[i - 1]) * (y[i] + y[i - 1]) / 2;
  }
  return res;
}

inline vector<float> get_thresholds_idxs(vector<float> y_pred) {
  vector<float> thresholds_idxs;
  set<float> s{};
  reverse(y_pred.begin(), y_pred.end());
  int y_pred_size = y_pred.size();
  for (int i = 0; i < y_pred_size; i++) {
    if (s.insert(y_pred[i]).second) {
      thresholds_idxs.push_back(y_pred_size - i);
    }
  }
  return thresholds_idxs;
}

inline float roc_auc_score(vector<float> y_pred, vector<int> y_true) {
  int num_elements = y_pred.size();
  vector<int> temp_idxs(num_elements);
  iota(temp_idxs.begin(), temp_idxs.end(), 0);
  sort(temp_idxs.begin(), temp_idxs.end(),
       [&y_pred](size_t i, size_t j) { return y_pred[i] < y_pred[j]; });
  vector<int> temp_y_true(y_true.size());
  copy(y_true.begin(), y_true.end(), temp_y_true.begin());
  for (int i = 0; i < num_elements; i++) {
    y_true[i] = temp_y_true[temp_idxs[i]];
  }
  sort(y_pred.begin(), y_pred.end());

  vector<float> thresholds_idxs = get_thresholds_idxs(y_pred);

  vector<float> tps = {0};
  for (int i = 1; i < num_elements; i++) {
    tps.push_back(y_true[i] + tps[i - 1]);
  }
  for (int i = 0; i < tps.size(); i++) {
    tps[i] = tps[i] / tps[tps.size() - 1];
  }

  vector<float> fps = {1};
  for (int i = 1; i < num_elements; i++) {
    fps.push_back(1 - y_true[i] + fps[i - 1]);
  }
  for (int i = 0; i < fps.size(); i++) {
    fps[i] = fps[i] / fps[fps.size() - 1];
  }

  return trapz(tps, fps);
}

inline float ovr_roc_auc_score(vector<vector<float>> y_pred,
                               vector<int> y_true) {
  int num_elements = y_pred.size();
  int num_classes = y_pred[0].size();

  if (num_classes == 2) {
    vector<float> y_pred_pos(num_elements, 0);
    for (int i = 0; i < num_elements; i++) {
      y_pred_pos[i] = y_pred[i][1];
    }
    return roc_auc_score(y_pred_pos, y_true);
  } else {
    float ovr_average_score = 0;
    float tmp_roc_auc_score = 0;
    float count_classes = 0;
    for (int c = 0; c < num_classes; c++) {
      vector<float> y_pred_c(num_elements, 0);
      vector<int> y_true_c(num_elements, 0);
      for (int i = 0; i < num_elements; i++) {
        y_pred_c[i] = y_pred[i][c];
        if (c == y_true[i]) {
          y_true_c[i] = 1;
        }
      }

      tmp_roc_auc_score = roc_auc_score(y_pred_c, y_true_c);
      if (!isnan(tmp_roc_auc_score)) {
        count_classes++;
        ovr_average_score += tmp_roc_auc_score;
      }
    }
    return ovr_average_score / count_classes;
  }
}

inline float ovr_roc_auc_score(vector<vector<float>> y_pred,
                               vector<float> y_true) {
  std::vector<int> out;
  out.reserve(y_pred.size());
  std::transform(y_true.begin(), y_true.end(), std::back_inserter(out),
                 [](float n) { return (int)(n); });
  return ovr_roc_auc_score(y_pred, out);
}
