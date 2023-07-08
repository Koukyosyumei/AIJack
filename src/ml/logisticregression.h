#pragma once
#include "loss.h"
#include <vector>

std::vector<std::vector<float>>
minMaxNormalize(const std::vector<std::vector<float>> &data) {
  // Find the minimum and maximum values in the data
  float minVal = data[0][0];
  float maxVal = data[0][0];
  for (const auto &row : data) {
    for (float value : row) {
      minVal = std::min(minVal, value);
      maxVal = std::max(maxVal, value);
    }
  }

  // Perform min-max normalization
  std::vector<std::vector<float>> normalizedData;
  for (const auto &row : data) {
    std::vector<float> normalizedRow;
    for (float value : row) {
      float normalizedValue = (value - minVal) / (maxVal - minVal);
      normalizedRow.push_back(normalizedValue);
    }
    normalizedData.push_back(normalizedRow);
  }

  return normalizedData;
}

struct LogisticRegression {
  int epochs;
  float lr;
  std::vector<std::vector<float>> params;
  BCELoss lossfn;

  LogisticRegression(int epochs = 50, float lr = 0.3)
      : epochs(epochs), lr(lr), lossfn(BCELoss()) {}

  std::vector<std::vector<float>>
  preprocess(const std::vector<std::vector<float>> &xs) {
    std::vector<std::vector<float>> xs_processed(xs);
    for (int i = 0; i < xs_processed.size(); i++) {
      xs_processed[i].push_back(1.0);
    }

    xs_processed = minMaxNormalize(xs_processed);
    return xs_processed;
  }

  void fit(const std::vector<std::vector<float>> &xs,
           const std::vector<float> &ys) {
    int n = xs.size();
    std::vector<std::vector<float>> xs_normalized = preprocess(xs);
    int m = xs_normalized[0].size();

    params = std::vector<std::vector<float>>(m, std::vector<float>(1, 0));
    std::vector<std::vector<float>> grads =
        std::vector<std::vector<float>>(m, std::vector<float>(1, 0));

    std::vector<std::vector<float>> x_batch;
    std::vector<float> y_batch;
    std::vector<std::vector<float>> y_pred_batch;

    for (int e = 0; e < epochs; e++) {
      for (int i = 0; i < n; i++) {
        float pred = 0;
        for (int j = 0; j < m; j++) {
          pred += xs_normalized[i][j] * params[j][0];
        }
        pred = sigmoid(pred);

        x_batch = {xs_normalized[i]};
        y_batch = {ys[i]};
        y_pred_batch = {{pred}};
        grads = lossfn.get_grad_w(x_batch, y_pred_batch, y_batch);

        for (int j = 0; j < m; j++) {
          params[j][0] -= grads[j][0];
        }
      }
    }
  }
};
