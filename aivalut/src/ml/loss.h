#pragma once
#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>
using namespace std;

inline float sigmoid(float x) {
  float sigmoid_range = 34.538776394910684;
  if (x <= -1 * sigmoid_range)
    return 1e-15;
  else if (x >= sigmoid_range)
    return 1.0 - 1e-15;
  else
    return 1.0 / (1.0 + exp(-1 * x));
}

inline std::vector<std::vector<float>>
sigmoid(std::vector<std::vector<float>> xs) {
  std::vector<std::vector<float>> ys(xs.size(),
                                     std::vector<float>(xs[0].size()));
  for (int i = 0; i < xs.size(); i++) {
    for (int j = 0; j < xs[i].size(); j++) {
      ys[i][j] = sigmoid(xs[i][j]);
    }
  }
  return ys;
}

inline vector<float> softmax(vector<float> x) {
  int n = x.size();
  float max_x = *max_element(x.begin(), x.end());
  vector<float> numerator(n, 0);
  vector<float> output(n, 0);
  float denominator = 0;

  for (int i = 0; i < n; i++) {
    numerator[i] = exp(x[i] - max_x);
    denominator += numerator[i];
  }

  for (int i = 0; i < n; i++) {
    output[i] = numerator[i] / denominator;
  }

  return output;
}

struct LossFunc {
  LossFunc(){};

  virtual float get_loss(vector<vector<float>> &y_pred, vector<float> &y) = 0;
  virtual vector<vector<float>> get_grad_o(vector<vector<float>> &y_pred,
                                           vector<float> &y) = 0;
  virtual vector<vector<float>> get_hess_o(vector<vector<float>> &y_pred,
                                           vector<float> &y) = 0;
};

struct BCELoss : LossFunc {
  BCELoss(){};

  float get_loss(vector<vector<float>> &y_pred, vector<float> &y) {
    float loss = 0;
    float n = y_pred.size();
    for (int i = 0; i < n; i++) {
      if (y[i] == 1) {
        loss += log(1 + exp(-1 * y_pred[i][0])) / n;
      } else {
        loss += log(1 + exp(y_pred[i][0])) / n;
      }
    }
    return loss;
  }

  vector<vector<float>> get_grad_w_ewise(vector<vector<float>> &x,
                                         vector<vector<float>> &y_pred,
                                         vector<float> &y) {
    int n = x.size();
    if (n <= 0) {
      try {
        throw std::runtime_error("the number of rows should be positive");
      } catch (std::runtime_error &e) {
        std::cerr << "runtime_error: " << e.what() << std::endl;
      }
    }
    int m = x[0].size();
    if (m <= 0) {
      try {
        throw std::runtime_error("the number of columns should be positive");
      } catch (std::runtime_error &e) {
        std::cerr << "runtime_error: " << e.what() << std::endl;
      }
    }

    std::vector<std::vector<float>> grad(n, std::vector<float>(m, 0.0));
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        grad[i][j] = (y_pred[i][0] - y[i]) * x[i][j];
      }
    }

    return grad;
  }

  vector<vector<float>> get_hess_w(vector<vector<float>> &x,
                                   vector<vector<float>> &y_pred,
                                   vector<float> &y) {
    int n = x.size();
    if (n <= 0) {
      try {
        throw std::runtime_error("the number of rows should be positive");
      } catch (std::runtime_error &e) {
        std::cerr << "runtime_error: " << e.what() << std::endl;
      }
    }
    int m = x[0].size();
    if (m <= 0) {
      try {
        throw std::runtime_error("the number of columns should be positive");
      } catch (std::runtime_error &e) {
        std::cerr << "runtime_error: " << e.what() << std::endl;
      }
    }

    std::vector<std::vector<float>> hess(m, std::vector<float>(m, 0.0));
    for (int i = 0; i < n; i++) {
      float tmp_sq = y_pred[i][0] * (1.0 - y_pred[i][0]);
      for (int j = 0; j < m; j++) {
        for (int k = 0; k < m; k++) {
          hess[j][k] += tmp_sq * x[i][j] * x[i][k] / (float)(n);
        }
      }
    }
    return hess;
  }

  vector<vector<float>> get_grad_w(vector<vector<float>> &x,
                                   vector<vector<float>> &y_pred,
                                   vector<float> &y) {
    int n = x.size();
    if (n <= 0) {
      try {
        throw std::runtime_error("the number of rows should be positive");
      } catch (std::runtime_error &e) {
        std::cerr << "runtime_error: " << e.what() << std::endl;
      }
    }
    int m = x[0].size();
    if (m <= 0) {
      try {
        throw std::runtime_error("the number of columns should be positive");
      } catch (std::runtime_error &e) {
        std::cerr << "runtime_error: " << e.what() << std::endl;
      }
    }

    std::vector<std::vector<float>> grad(m, std::vector<float>(1, 0.0));
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        grad[j][0] += (y_pred[i][0] - y[i]) * x[i][j] / (float)(n);
      }
    }

    return grad;
  }

  vector<vector<float>> get_grad_o(vector<vector<float>> &y_pred,
                                   vector<float> &y) {
    int element_num = y_pred.size();
    vector<vector<float>> grad(element_num);
    for (int i = 0; i < element_num; i++)
      grad[i] = {y_pred[i][0] - y[i]};
    return grad;
  }

  vector<vector<float>> get_hess_o(vector<vector<float>> &y_pred,
                                   vector<float> &y) {
    int element_num = y_pred.size();
    vector<vector<float>> hess(element_num);
    for (int i = 0; i < element_num; i++) {
      float temp_proba = y_pred[i][0];
      hess[i] = {temp_proba * (1 - temp_proba)};
    }
    return hess;
  }
};
