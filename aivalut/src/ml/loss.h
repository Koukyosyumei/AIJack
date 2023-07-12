#pragma once
#include <algorithm>
#include <cmath>
#include <exception>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>
using namespace std;

inline double sigmoid(double x) {
  double sigmoid_range = 34.538776394910684;
  if (x <= -1 * sigmoid_range)
    return 1e-15;
  else if (x >= sigmoid_range)
    return 1.0 - 1e-15;
  else
    return 1.0 / (1.0 + exp(-1 * x));
}

inline std::vector<std::vector<double>>
sigmoid(std::vector<std::vector<double>> xs) {
  std::vector<std::vector<double>> ys(xs.size(),
                                      std::vector<double>(xs[0].size()));
  for (int i = 0; i < xs.size(); i++) {
    for (int j = 0; j < xs[i].size(); j++) {
      ys[i][j] = sigmoid(xs[i][j]);
    }
  }
  return ys;
}

inline vector<double> softmax(vector<double> x) {
  int n = x.size();
  double max_x = *max_element(x.begin(), x.end());
  vector<double> numerator(n, 0);
  vector<double> output(n, 0);
  double denominator = 0;

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

  virtual double get_loss(vector<vector<double>> &y_pred,
                          vector<double> &y) = 0;
  virtual vector<vector<double>> get_grad_o(vector<vector<double>> &y_pred,
                                            vector<double> &y) = 0;
  virtual vector<vector<double>> get_hess_o(vector<vector<double>> &y_pred,
                                            vector<double> &y) = 0;
};

struct BCELoss : LossFunc {
  BCELoss(){};

  double get_loss(vector<vector<double>> &y_pred, vector<double> &y) {
    double loss = 0;
    double n = y_pred.size();
    for (int i = 0; i < n; i++) {
      if (y[i] == 1) {
        loss += log(1 + exp(-1 * y_pred[i][0])) / n;
      } else {
        loss += log(1 + exp(y_pred[i][0])) / n;
      }
    }
    return loss;
  }

  vector<vector<double>> get_grad_w_ewise(vector<vector<double>> &x,
                                          vector<vector<double>> &y_pred,
                                          vector<double> &y) {
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

    std::vector<std::vector<double>> grad(n, std::vector<double>(m, 0.0));
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        grad[i][j] = (y_pred[i][0] - y[i]) * x[i][j];
      }
    }

    return grad;
  }

  vector<vector<double>> get_hess_w(vector<vector<double>> &x,
                                    vector<vector<double>> &y_pred,
                                    vector<double> &y) {
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

    std::vector<std::vector<double>> hess(m, std::vector<double>(m, 0.0));
    for (int i = 0; i < n; i++) {
      double tmp_sq = y_pred[i][0] * (1.0 - y_pred[i][0]);
      for (int j = 0; j < m; j++) {
        for (int k = 0; k < m; k++) {
          hess[j][k] += tmp_sq * x[i][j] * x[i][k] / (double)(n);
        }
      }
    }
    return hess;
  }

  vector<vector<double>> get_grad_w(vector<vector<double>> &x,
                                    vector<vector<double>> &y_pred,
                                    vector<double> &y) {
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

    std::vector<std::vector<double>> grad(m, std::vector<double>(1, 0.0));
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < m; j++) {
        grad[j][0] += (y_pred[i][0] - y[i]) * x[i][j] / (double)(n);
      }
    }

    return grad;
  }

  vector<vector<double>> get_grad_o(vector<vector<double>> &y_pred,
                                    vector<double> &y) {
    int element_num = y_pred.size();
    vector<vector<double>> grad(element_num);
    for (int i = 0; i < element_num; i++)
      grad[i] = {y_pred[i][0] - y[i]};
    return grad;
  }

  vector<vector<double>> get_hess_o(vector<vector<double>> &y_pred,
                                    vector<double> &y) {
    int element_num = y_pred.size();
    vector<vector<double>> hess(element_num);
    for (int i = 0; i < element_num; i++) {
      double temp_proba = y_pred[i][0];
      hess[i] = {temp_proba * (1 - temp_proba)};
    }
    return hess;
  }
};
