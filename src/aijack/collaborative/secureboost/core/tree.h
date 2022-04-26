#include <vector>
#include <iterator>
#include <limits>
#include <iostream>
#include "node.h"
#include "utils.h"

struct XGBoostTree
{
    Node dtree;

    XGBoostTree() {}

    void fit(vector<Party> parties, vector<double> y, vector<double> gradient,
             vector<double> hessian, vector<int> idxs,
             double min_child_weight, double lam, double gamma, double eps,
             int min_leaf, int depth)
    {
        vector<int> idxs(y.size());
        iota(idxs.begin(), idxs.end(), 0);
        dtree = Node(parties, y, gradient, hessian, idxs,
                     min_child_weight, lam, gamma, eps, depth);
    }

    vector<double> predict(vector<vector<double>> X)
    {
        return dtree.predict(X);
    }
};
