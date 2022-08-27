#pragma once
#include <vector>
#include <iterator>
#include <limits>
#include <iostream>
#include "../core/tree.h"
#include "node.h"

struct SecureBoostTree : Tree<SecureBoostNode>
{
    SecureBoostTree() {}
    void fit(vector<SecureBoostParty> &parties, vector<float> y,
             int num_classes,
             vector<vector<PaillierCipherText>> &gradient,
             vector<vector<PaillierCipherText>> &hessian,
             vector<vector<float>> &vanila_gradient, vector<vector<float>> &vanila_hessian,
             float min_child_weight, float lam, float gamma, float eps,
             int min_leaf, int depth, int active_party_id = 0,
             bool use_only_active_party = false, int n_job = 1)
    {
        vector<int> idxs(y.size());
        iota(idxs.begin(), idxs.end(), 0);
        for (int i = 0; i < parties.size(); i++)
        {
            parties[i].subsample_columns();
        }
        dtree = SecureBoostNode(parties, y, num_classes, gradient, hessian, vanila_gradient,
                                vanila_hessian, idxs, min_child_weight,
                                lam, gamma, eps, depth, active_party_id,
                                use_only_active_party, n_job);
    }
};
