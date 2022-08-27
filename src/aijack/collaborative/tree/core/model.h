#pragma once
#include <vector>
#include <iterator>
#include <limits>
#include <iostream>
using namespace std;

/**
 * @brief Base structure for the tree-based model
 *
 * @tparam PartyName
 */
template <typename PartyName>
struct TreeModelBase
{
    TreeModelBase(){};

    /**
     * @brief Function to train the model given the parties and ground-truth labels.
     *
     * @param parties The vector of parties.
     * @param y The vector of ground-truth vectors
     */
    virtual void fit(vector<PartyName> &parties, vector<float> &y) = 0;

    /**
     * @brief Function to return the predicted scores of the given data.
     *
     * @param X The feature matrix.
     * @return vector<float> The vector of predicted raw scores.
     */
    virtual vector<vector<float>> predict_raw(vector<vector<float>> &X) = 0;
};
