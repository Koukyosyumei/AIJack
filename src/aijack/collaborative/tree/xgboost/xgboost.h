#pragma once
#include "../core/model.h"
#include "loss.h"
#include "party.h"
#include "tree.h"
#include <cmath>
#include <iostream>
#include <iterator>
#include <limits>
#include <vector>
using namespace std;

struct XGBoostBase : TreeModelBase<XGBoostParty> {
  float subsample_cols;
  float min_child_weight;
  int depth;
  int min_leaf;
  float learning_rate;
  int boosting_rounds;
  float lam;
  float gamma;
  float eps;
  int active_party_id;
  int completelly_secure_round;
  float init_value;
  int n_job;
  bool save_loss;
  int num_classes;

  float upsilon_Y;

  LossFunc *lossfunc_obj;

  vector<vector<float>> init_pred;
  vector<XGBoostTree> estimators;
  vector<float> logging_loss;
  vector<XGBoostParty> parties_cp;

  XGBoostBase(int num_classes_, float subsample_cols_ = 0.8,
              float min_child_weight_ = -1 * numeric_limits<float>::infinity(),
              int depth_ = 5, int min_leaf_ = 5, float learning_rate_ = 0.4,
              int boosting_rounds_ = 5, float lam_ = 1.5, float gamma_ = 1,
              float eps_ = 0.1, int active_party_id_ = -1,
              int completelly_secure_round_ = 0, float init_value_ = 1.0,
              int n_job_ = 1, bool save_loss_ = true) {
    num_classes = num_classes_;
    subsample_cols = subsample_cols_;
    min_child_weight = min_child_weight_;
    depth = depth_;
    min_leaf = min_leaf_;
    learning_rate = learning_rate_;
    boosting_rounds = boosting_rounds_;
    lam = lam_;
    gamma = gamma_;
    eps = eps_;
    active_party_id = active_party_id_;
    completelly_secure_round = completelly_secure_round_;
    init_value = init_value_;
    n_job = n_job_;
    save_loss = save_loss_;

    if (num_classes == 2) {
      lossfunc_obj = new BCELoss();
    } else {
      lossfunc_obj = new CELoss(num_classes);
    }
  }

  virtual vector<vector<float>> get_init_pred(vector<float> &y) = 0;

  void load_estimators(vector<XGBoostTree> &_estimators) {
    estimators = _estimators;
  }

  void clear() {
    estimators.clear();
    logging_loss.clear();
  }

  vector<XGBoostTree> get_estimators() { return estimators; }

  void fit(vector<XGBoostParty> &parties, vector<float> &y) {
    int row_count = y.size();

    parties_cp = parties;

    vector<vector<float>> base_pred;
    if (estimators.size() == 0) {
      init_pred = get_init_pred(y);
      copy(init_pred.begin(), init_pred.end(), back_inserter(base_pred));
    } else {
      base_pred.resize(row_count);
      for (int j = 0; j < row_count; j++)
        for (int c = 0; c < num_classes; c++)
          base_pred[j][c] = 0;

      for (int i = 0; i < estimators.size(); i++) {
        vector<vector<float>> pred_temp = estimators[i].get_train_prediction();
        for (int j = 0; j < row_count; j++)
          for (int c = 0; c < num_classes; c++)
            base_pred[j][c] += learning_rate * pred_temp[j][c];
      }
    }

    for (int i = 0; i < boosting_rounds; i++) {
      vector<vector<float>> grad = lossfunc_obj->get_grad(base_pred, y);
      vector<vector<float>> hess = lossfunc_obj->get_hess(base_pred, y);

      XGBoostTree boosting_tree(parties_cp, y, num_classes, grad, hess,
                                min_child_weight, lam, gamma, eps, min_leaf,
                                depth, active_party_id,
                                (completelly_secure_round > i), n_job);
      vector<vector<float>> pred_temp = boosting_tree.get_train_prediction();
      for (int j = 0; j < row_count; j++)
        for (int c = 0; c < num_classes; c++)
          base_pred[j][c] += learning_rate * pred_temp[j][c];

      estimators.push_back(boosting_tree);

      if (save_loss) {
        logging_loss.push_back(lossfunc_obj->get_loss(base_pred, y));
      }
    }
  }

  vector<vector<float>> predict_raw(vector<vector<float>> &X) {
    int pred_dim;
    if (num_classes == 2) {
      pred_dim = 1;
    } else {
      pred_dim = num_classes;
    }

    int row_count = X.size();
    vector<vector<float>> y_pred(row_count,
                                 vector<float>(pred_dim, init_value));
    // copy(init_pred.begin(), init_pred.end(), back_inserter(y_pred));
    int estimators_num = estimators.size();
    for (int i = 0; i < estimators_num; i++) {
      vector<vector<float>> y_pred_temp = estimators[i].predict(X);
      for (int j = 0; j < row_count; j++) {
        for (int c = 0; c < pred_dim; c++) {
          y_pred[j][c] += learning_rate * y_pred_temp[j][c];
        }
      }
    }

    return y_pred;
  }
};

struct XGBoostClassifier : public XGBoostBase {
  using XGBoostBase::XGBoostBase;

  vector<vector<float>> get_init_pred(vector<float> &y) {
    vector<vector<float>> init_pred(y.size(),
                                    vector<float>(num_classes, init_value));
    return init_pred;
  }

  vector<vector<float>> predict_proba(vector<vector<float>> &x) {
    vector<vector<float>> raw_score = predict_raw(x);
    int row_count = x.size();
    vector<vector<float>> predicted_probas(row_count,
                                           vector<float>(num_classes, 0));
    for (int i = 0; i < row_count; i++) {
      if (num_classes == 2) {
        predicted_probas[i][1] = sigmoid(raw_score[i][0]);
        predicted_probas[i][0] = 1 - predicted_probas[i][1];
      } else {
        predicted_probas[i] = softmax(raw_score[i]);
      }
    }
    return predicted_probas;
  }
};
