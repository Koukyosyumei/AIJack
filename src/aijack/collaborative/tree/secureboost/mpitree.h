#pragma once
#include <vector>
#include <iterator>
#include <limits>
#include <iostream>
#include "../core/tree.h"
#include "mpinode.h"

struct MPISecureBoostTree : Tree<MPISecureBoostNode>
{
    MPISecureBoostTree() {}
    void fit(MPISecureBoostParty *active_party, int parties_num,
             vector<float> &y, float min_child_weight, float lam,
             float gamma, float eps, int min_leaf, int depth,
             int active_party_id = 0,
             bool use_only_active_party = false)
    {
        vector<int> idxs(y.size());
        iota(idxs.begin(), idxs.end(), 0);
        active_party->set_instance_space(idxs);
        active_party->subsample_columns();
        dtree = MPISecureBoostNode(active_party, parties_num, idxs,
                                   depth, min_child_weight,
                                   lam, gamma, eps, depth,
                                   active_party_id, use_only_active_party);
    }
};
