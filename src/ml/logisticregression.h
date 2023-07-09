#pragma once
#include "loss.h"
#include <fstream>
#include <vector>

inline std::vector<std::vector<float>>
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
  bool isinitialized;

  LogisticRegression(int epochs = 50, float lr = 0.3)
      : epochs(epochs), lr(lr), lossfn(BCELoss()), isinitialized(false) {}

  std::vector<std::vector<float>>
  preprocess(const std::vector<std::vector<float>> &xs) {
    std::vector<std::vector<float>> xs_processed(xs);
    for (int i = 0; i < xs_processed.size(); i++) {
      xs_processed[i].push_back(1.0);
    }

    xs_processed = minMaxNormalize(xs_processed);
    return xs_processed;
  }

  void save(const std::string &path) {
    std::ofstream param_file;
    param_file.open(path, std::ios::out);
    param_file << params.size() << "\n";
    for (std::vector<float> w : params) {
      param_file << w[0] << " ";
    }
    param_file.close();
  }

  void load(const std::string &path) {
    std::ifstream param_file;
    param_file.open(path, std::ios::in);
    int m;
    param_file >> m;
    params.clear();
    for (int i = 0; i < m; i++) {
      float w;
      param_file >> w;
      params.push_back({w});
    }
    param_file.close();
    isinitialized = true;
  }

  bool fit(const std::vector<std::vector<float>> &xs,
           const std::vector<float> &ys) {

    for (float y : ys) {
      if ((y != 0) && (y != 1)) {
        try {
          throw std::runtime_error("Labels should be 0 or 1");
        } catch (std::range_error &e) {
          std::cerr << "range_error: " << e.what() << std::endl;
          return false;
        }
      }
    }

    int n = xs.size();
    std::vector<std::vector<float>> xs_normalized = preprocess(xs);
    int m = xs_normalized[0].size();

    if (!isinitialized) {
      params = std::vector<std::vector<float>>(m, std::vector<float>(1, 1));
      isinitialized = true;
    }

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

    return true;
  }

  std::vector<std::vector<float>>
  predict_proba(const std::vector<std::vector<float>> &xs) {
    int n = xs.size();
    std::vector<std::vector<float>> xs_normalized = preprocess(xs);
    int m = xs_normalized[0].size();
    std::vector<std::vector<float>> y_probas(n);
    for (int i = 0; i < n; i++) {
      float pred = 0;
      for (int j = 0; j < m; j++) {
        pred += xs_normalized[i][j] * params[j][0];
      }
      pred = sigmoid(pred);
      y_probas[i] = {1 - pred, pred};
    }
    return y_probas;
  }

  std::vector<float> predict(const std::vector<std::vector<float>> &xs) {
    std::vector<std::vector<float>> y_probas = predict_proba(xs);
    std::vector<float> y_preds(y_probas.size());
    for (int i = 0; i < y_probas.size(); i++) {
      y_preds[i] = (int)(y_probas[i][0] < y_probas[i][1]);
    }
    return y_preds;
  }

  float score(const std::vector<float> &y, const std::vector<float> &y_pred) {
    float acc = 0;
    float n = y.size();
    for (int i = 0; i < n; i++) {
      acc += (float)(y[i] == y_pred[i]);
    }
    return acc / n;
  }
};
