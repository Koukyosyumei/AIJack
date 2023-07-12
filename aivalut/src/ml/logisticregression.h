#pragma once
#include "loss.h"
#include <fstream>
#include <tuple>
#include <vector>

inline std::pair<std::vector<std::vector<double>>,
                 std::vector<std::pair<double, double>>>
minMaxNormalize(const std::vector<std::vector<double>> &data) {
  // Find the minimum and maximum values in the data
  int n = data.size();
  int m = data[0].size();
  std::vector<std::pair<double, double>> normalizer(m);

  for (int j = 0; j < m; j++) {
    double minVal = data[0][j];
    double maxVal = data[0][j];
    for (int i = 0; i < n; i++) {
      minVal = std::min(minVal, data[i][j]);
      maxVal = std::max(maxVal, data[i][j]);
    }
    if (minVal != maxVal) {
      normalizer[j] = {minVal, maxVal};
    } else {
      normalizer[j] = {0, maxVal};
    }
  }

  // Perform min-max normalization
  std::vector<std::vector<double>> normalizedData(n, std::vector<double>(m));
  for (int i = 0; i < n; i++) {
    std::vector<double> normalizedRow;
    for (int j = 0; j < m; j++) {
      normalizedData[i][j] = (data[i][j] - normalizer[j].first) /
                             (normalizer[j].second - normalizer[j].first);
    }
  }

  return std::make_pair(normalizedData, normalizer);
}

struct LogisticRegression {
  int epochs;
  double lr;
  std::vector<std::vector<double>> params;
  std::vector<std::pair<double, double>> normalizer;
  BCELoss lossfn;
  bool isinitialized;

  LogisticRegression(int epochs = 50, double lr = 0.3)
      : epochs(epochs), lr(lr), lossfn(BCELoss()), isinitialized(false) {}

  void clear() {
    params.clear();
    isinitialized = false;
  }

  std::vector<std::vector<double>>
  preprocess(const std::vector<std::vector<double>> &xs) {
    std::vector<std::vector<double>> xs_processed(xs);
    for (int i = 0; i < xs_processed.size(); i++) {
      xs_processed[i].push_back(1.0);
    }
    std::pair<std::vector<std::vector<double>>,
              std::vector<std::pair<double, double>>>
        result = minMaxNormalize(xs_processed);
    xs_processed = result.first;
    normalizer = result.second;
    return xs_processed;
  }

  std::vector<std::vector<double>>
  normalize(const std::vector<std::vector<double>> &xs) {
    std::vector<std::vector<double>> xs_processed(xs);
    for (int i = 0; i < xs_processed.size(); i++) {
      xs_processed[i].push_back(1.0);
    }

    int n = xs_processed.size();
    int m = xs_processed[0].size();
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        xs_processed[i][j] = (xs[i][j] - normalizer[j].first) /
                             (normalizer[j].second - normalizer[j].first);
      }
    }
    return xs_processed;
  }

  void save(const std::string &path) {
    std::ofstream param_file;
    param_file.open(path, std::ios::out);
    param_file << params.size() << "\n";
    for (const std::vector<double> &w : params) {
      param_file << w[0] << " ";
    }
    param_file << "\n";
    for (auto const &n : normalizer) {
      param_file << n.first << " " << n.second << "\n";
    }
    param_file.close();
  }

  void load(const std::string &path) {
    std::ifstream param_file;
    param_file.open(path, std::ios::in);
    int m;
    param_file >> m;
    params.clear();
    normalizer.clear();
    for (int i = 0; i < m; i++) {
      double w;
      param_file >> w;
      params.push_back({w});
    }
    for (int i = 0; i < m; i++) {
      double minv, maxv;
      param_file >> minv >> maxv;
      normalizer.push_back({minv, maxv});
    }
    param_file.close();
    isinitialized = true;
  }

  bool fit(const std::vector<std::vector<double>> &xs,
           const std::vector<double> &ys) {

    for (double y : ys) {
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
    std::vector<std::vector<double>> xs_normalized = preprocess(xs);
    int m = xs_normalized[0].size();

    if (!isinitialized) {
      params = std::vector<std::vector<double>>(m, std::vector<double>(1, 1));
      isinitialized = true;
    }

    std::vector<std::vector<double>> grads =
        std::vector<std::vector<double>>(m, std::vector<double>(1, 0));

    std::vector<std::vector<double>> x_batch;
    std::vector<double> y_batch;
    std::vector<std::vector<double>> y_pred_batch;

    for (int e = 0; e < epochs; e++) {
      for (int i = 0; i < n; i++) {
        double pred = 0;
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

  std::vector<std::vector<double>>
  predict_proba(const std::vector<std::vector<double>> &xs) {
    int n = xs.size();
    std::vector<std::vector<double>> xs_normalized = normalize(xs);
    int m = xs_normalized[0].size();
    std::vector<std::vector<double>> y_probas(n);
    for (int i = 0; i < n; i++) {
      double pred = 0;
      for (int j = 0; j < m; j++) {
        pred += xs_normalized[i][j] * params[j][0];
      }
      pred = sigmoid(pred);
      y_probas[i] = {1 - pred, pred};
    }
    return y_probas;
  }

  std::vector<double> predict(const std::vector<std::vector<double>> &xs) {
    std::vector<std::vector<double>> y_probas = predict_proba(xs);
    std::vector<double> y_preds(y_probas.size());
    for (int i = 0; i < y_probas.size(); i++) {
      y_preds[i] = (int)(y_probas[i][0] < y_probas[i][1]);
    }
    return y_preds;
  }

  double score(const std::vector<double> &y,
               const std::vector<double> &y_pred) {
    double acc = 0;
    double n = y.size();
    for (int i = 0; i < n; i++) {
      acc += (double)(y[i] == y_pred[i]);
    }
    return acc / n;
  }
};
