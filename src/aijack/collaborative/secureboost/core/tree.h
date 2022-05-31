#include <vector>
#include <iterator>
#include <limits>
#include <iostream>
#include "node.h"

struct XGBoostTree
{
    Node dtree;

    XGBoostTree() {}

    void fit(vector<Party> &parties, vector<double> y,
             vector<double> gradient, vector<double> hessian,
             double min_child_weight, double lam, double gamma, double eps,
             int min_leaf, int depth)
    {
        vector<int> idxs(y.size());
        iota(idxs.begin(), idxs.end(), 0);
        dtree = Node(parties, y, gradient, hessian, idxs,
                     min_child_weight, lam, gamma, eps, depth);
    }

    Node get_root_node()
    {
        return dtree;
    }

    vector<double> predict(vector<vector<double>> X)
    {
        return dtree.predict(X);
    }

    vector<pair<vector<int>, vector<double>>> extract_train_prediction_from_node(Node node)
    {
        if (node.is_leaf())
        {
            vector<pair<vector<int>, vector<double>>> result;
            result.push_back(make_pair(node.idxs,
                                       vector<double>(node.idxs.size(),
                                                      node.val)));
            return result;
        }
        else
        {
            vector<pair<vector<int>, vector<double>>> left_result =
                extract_train_prediction_from_node(*node.left);
            vector<pair<vector<int>, vector<double>>> right_result =
                extract_train_prediction_from_node(*node.right);
            left_result.insert(left_result.end(), right_result.begin(), right_result.end());
            return left_result;
        }
    }

    vector<double> get_train_prediction()
    {
        vector<pair<vector<int>, vector<double>>> result = extract_train_prediction_from_node(dtree);
        vector<double> y_train_pred(dtree.y.size());
        for (int i = 0; i < result.size(); i++)
            for (int j = 0; j < result[i].first.size(); j++)
                y_train_pred[result[i].first[j]] = result[i].second[j];

        return y_train_pred;
    }
};
