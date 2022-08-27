#pragma once
#include <vector>
#include <iterator>
#include <limits>
#include <iostream>
#include <cmath>
#include "../core/model.h"
#include "../utils/utils.h"
#include "tree.h"
using namespace std;

struct LossFunc
{
    LossFunc(){};
    virtual float get_loss(vector<vector<float>> &y_pred, vector<float> &y) = 0;
    virtual vector<vector<float>> get_grad(vector<vector<float>> &y_pred, vector<float> &y) = 0;
    virtual vector<vector<float>> get_hess(vector<vector<float>> &y_pred, vector<float> &y) = 0;
};

struct BCELoss : LossFunc
{
    BCELoss(){};

    float get_loss(vector<vector<float>> &y_pred, vector<float> &y)
    {
        float loss = 0;
        float n = y_pred.size();
        for (int i = 0; i < n; i++)
        {
            if (y[i] == 1)
            {
                loss += log(1 + exp(-1 * sigmoid(y_pred[i][0]))) / n;
            }
            else
            {
                loss += log(1 + exp(sigmoid(y_pred[i][0]))) / n;
            }
        }
        return loss;
    }

    vector<vector<float>> get_grad(vector<vector<float>> &y_pred, vector<float> &y)
    {
        int element_num = y_pred.size();
        vector<vector<float>> grad(element_num);
        for (int i = 0; i < element_num; i++)
            grad[i] = {sigmoid(y_pred[i][0]) - y[i]};
        return grad;
    }

    vector<vector<float>> get_hess(vector<vector<float>> &y_pred, vector<float> &y)
    {
        int element_num = y_pred.size();
        vector<vector<float>> hess(element_num);
        for (int i = 0; i < element_num; i++)
        {
            float temp_proba = sigmoid(y_pred[i][0]);
            hess[i] = {temp_proba * (1 - temp_proba)};
        }
        return hess;
    }
};

struct CELoss : LossFunc
{
    int num_classes;

    CELoss(){};
    CELoss(int num_classes_) { num_classes = num_classes_; }

    float get_loss(vector<vector<float>> &y_pred, vector<float> &y)
    {
        int n = y_pred.size();

        vector<vector<float>> y_pred_proba(n);
        for (int i = 0; i < n; i++)
        {
            y_pred_proba[i] = softmax(y_pred[i]);
        }

        float loss = 0;
        for (int i = 0; i < n; i++)
        {
            for (int c = 0; c < num_classes; c++)
            {
                if (y[i] == c)
                {
                    loss -= log(y_pred_proba[i][c]);
                }
            }
        }
        return loss;
    }

    vector<vector<float>> get_grad(vector<vector<float>> &y_pred, vector<float> &y)
    {
        int n = y_pred.size();

        vector<vector<float>> y_pred_proba(n);
        for (int i = 0; i < n; i++)
        {
            y_pred_proba[i] = softmax(y_pred[i]);
        }

        vector<vector<float>> grad(n, vector<float>(num_classes, 0));

        for (int i = 0; i < n; i++)
        {
            for (int c = 0; c < num_classes; c++)
            {
                grad[i][c] = y_pred_proba[i][c];
                if (y[i] == c)
                {
                    grad[i][c] -= 1;
                }
            }
        }
        return grad;
    }

    vector<vector<float>> get_hess(vector<vector<float>> &y_pred, vector<float> &y)
    {
        int n = y_pred.size();

        vector<vector<float>> y_pred_proba(n);
        for (int i = 0; i < n; i++)
        {
            y_pred_proba[i] = softmax(y_pred[i]);
        }

        vector<vector<float>> hess(n, vector<float>(num_classes, 0));

        for (int i = 0; i < n; i++)
        {
            for (int c = 0; c < num_classes; c++)
            {
                hess[i][c] = y_pred_proba[i][c] * (1 - y_pred_proba[i][c]);
            }
        }
        return hess;
    }
};
