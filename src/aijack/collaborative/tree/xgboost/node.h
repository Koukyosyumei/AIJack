#pragma once
#include <cmath>
#include <numeric>
#include <vector>
#include <iterator>
#include <limits>
#include <algorithm>
#include <thread>
#include <set>
#include <tuple>
#include <random>
#include <ctime>
#include <string>
#include <queue>
#include <unordered_map>
#include <stdexcept>
#include "party.h"
#include "utils.h"
#include "../core/node.h"
#include "../utils/metric.h"
#include "../utils/utils.h"
using namespace std;

struct XGBoostNode : Node<XGBoostParty>
{
    vector<XGBoostParty> *parties;
    vector<vector<float>> gradient, hessian;
    float min_child_weight, lam, gamma, eps;
    float best_entropy;
    bool use_only_active_party;
    XGBoostNode *left, *right;

    int num_classes;

    float entire_datasetsize = 0;
    vector<float> entire_class_cnt;

    XGBoostNode() {}
    XGBoostNode(vector<XGBoostParty> *parties_, vector<float> &y_, int num_classes_,
                vector<vector<float>> &gradient_,
                vector<vector<float>> &hessian_, vector<int> &idxs_,
                float min_child_weight_, float lam_, float gamma_, float eps_, int depth_,
                int active_party_id_ = -1, bool use_only_active_party_ = false, int n_job_ = 1)
    {
        parties = parties_;
        y = y_;
        num_classes = num_classes_;
        gradient = gradient_;
        hessian = hessian_;
        idxs = idxs_;
        min_child_weight = min_child_weight_;
        lam = lam_;
        gamma = gamma_;
        eps = eps_;
        depth = depth_;
        active_party_id = active_party_id_;
        use_only_active_party = use_only_active_party_;
        n_job = n_job_;

        row_count = idxs.size();
        num_parties = parties->size();

        entire_class_cnt.resize(num_classes, 0);
        entire_datasetsize = y.size();
        for (int i = 0; i < entire_datasetsize; i++)
        {
            entire_class_cnt[int(y[i])] += 1.0;
        }

        try
        {
            if (use_only_active_party && active_party_id > parties->size())
            {
                throw invalid_argument("invalid active_party_id");
            }
        }
        catch (std::exception &e)
        {
            std::cerr << e.what() << std::endl;
        }

        val = compute_weight();

        if (is_leaf())
        {
            is_leaf_flag = 1;
        }
        else
        {
            is_leaf_flag = 0;
        }

        if (is_leaf_flag == 0)
        {
            tuple<int, int, int> best_split = find_split();
            party_id = get<0>(best_split);
            if (party_id != -1)
            {
                record_id = parties->at(party_id).insert_lookup_table(get<1>(best_split), get<2>(best_split));
                make_children_nodes(get<0>(best_split), get<1>(best_split), get<2>(best_split));
            }
            else
            {
                is_leaf_flag = 1;
            }
        }
    }

    vector<int> get_idxs()
    {
        return idxs;
    }

    int get_party_id()
    {
        return party_id;
    }

    int get_record_id()
    {
        return record_id;
    }

    vector<float> get_val()
    {
        return val;
    }

    float get_score()
    {
        return score;
    }

    XGBoostNode get_left()
    {
        return *left;
    }

    XGBoostNode get_right()
    {
        return *right;
    }

    int get_num_parties()
    {
        return parties->size();
    }

    vector<float> compute_weight()
    {
        return xgboost_compute_weight(row_count, gradient, hessian, idxs, lam);
    }

    float compute_gain(vector<float> &left_grad, vector<float> &right_grad, vector<float> &left_hess, vector<float> &right_hess)
    {
        return xgboost_compute_gain(left_grad, right_grad, left_hess, right_hess, gamma, lam);
    }

    void find_split_per_party(int party_id_start, int temp_num_parties, vector<float> &sum_grad, vector<float> &sum_hess, float tot_cnt, vector<float> &temp_y_class_cnt)
    {

        vector<float> temp_left_class_cnt, temp_right_class_cnt;
        temp_left_class_cnt.resize(num_classes, 0);
        temp_right_class_cnt.resize(num_classes, 0);

        int grad_dim = sum_grad.size();

        for (int temp_party_id = party_id_start; temp_party_id < party_id_start + temp_num_parties; temp_party_id++)
        {

            vector<vector<tuple<vector<float>, vector<float>, float, vector<float>>>> search_results =
                parties->at(temp_party_id).greedy_search_split(gradient, hessian, y, idxs);

            float temp_score, temp_entropy;
            vector<float> temp_left_grad(grad_dim, 0);
            vector<float> temp_left_hess(grad_dim, 0);
            vector<float> temp_right_grad(grad_dim, 0);
            vector<float> temp_right_hess(grad_dim, 0);
            float temp_left_size, temp_right_size;
            bool skip_flag = false;

            for (int j = 0; j < search_results.size(); j++)
            {
                temp_score = 0;
                temp_entropy = 0;
                temp_left_size = 0;
                temp_right_size = 0;

                for (int c = 0; c < grad_dim; c++)
                {
                    temp_left_grad[c] = 0;
                    temp_left_hess[c] = 0;
                }

                for (int c = 0; c < num_classes; c++)
                {
                    temp_left_class_cnt[c] = 0;
                    temp_right_class_cnt[c] = 0;
                }

                for (int k = 0; k < search_results[j].size(); k++)
                {
                    for (int c = 0; c < grad_dim; c++)
                    {
                        temp_left_grad[c] += get<0>(search_results[j][k])[c];
                        temp_left_hess[c] += get<1>(search_results[j][k])[c];
                    }
                    temp_left_size += get<2>(search_results[j][k]);
                    temp_right_size = tot_cnt - temp_left_size;

                    for (int c = 0; c < num_classes; c++)
                    {
                        temp_left_class_cnt[c] += get<3>(search_results[j][k])[c];
                        temp_right_class_cnt[c] = temp_y_class_cnt[c] - temp_left_class_cnt[c];
                    }

                    skip_flag = false;
                    for (int c = 0; c < grad_dim; c++)
                    {
                        if (temp_left_hess[c] < min_child_weight ||
                            sum_hess[c] - temp_left_hess[c] < min_child_weight)
                        {
                            skip_flag = true;
                        }
                    }
                    if (skip_flag)
                    {
                        continue;
                    }

                    for (int c = 0; c < grad_dim; c++)
                    {
                        temp_right_grad[c] = sum_grad[c] - temp_left_grad[c];
                        temp_right_hess[c] = sum_hess[c] - temp_left_hess[c];
                    }

                    temp_score = compute_gain(temp_left_grad, temp_right_grad,
                                              temp_left_hess, temp_right_hess);

                    if (temp_score > best_score)
                    {
                        best_score = temp_score;
                        best_entropy = temp_entropy;
                        best_party_id = temp_party_id;
                        best_col_id = j;
                        best_threshold_id = k;
                    }
                }
            }
        }
    }

    tuple<int, int, int> find_split()
    {
        vector<float> sum_grad(gradient[0].size(), 0);
        vector<float> sum_hess(hessian[0].size(), 0);
        for (int i = 0; i < row_count; i++)
        {
            for (int c = 0; c < sum_grad.size(); c++)
            {
                sum_grad[c] += gradient[idxs[i]][c];
                sum_hess[c] += hessian[idxs[i]][c];
            }
        }

        float tot_cnt = row_count;
        vector<float> temp_y_class_cnt(num_classes, 0);
        for (int r = 0; r < row_count; r++)
        {
            temp_y_class_cnt[int(y[idxs[r]])] += 1;
        }

        float temp_score, temp_left_grad, temp_left_hess;

        if (use_only_active_party)
        {
            find_split_per_party(active_party_id, 1, sum_grad, sum_hess, tot_cnt, temp_y_class_cnt);
        }
        else
        {
            if (n_job == 1)
            {
                find_split_per_party(0, num_parties, sum_grad, sum_hess, tot_cnt, temp_y_class_cnt);
            }
            else
            {
                vector<int> num_parties_per_thread = get_num_parties_per_process(n_job, num_parties);

                int cnt_parties = 0;
                vector<thread> threads_parties;
                for (int i = 0; i < n_job; i++)
                {
                    int local_num_parties = num_parties_per_thread[i];
                    thread temp_th([this, cnt_parties, local_num_parties, &sum_grad, &sum_hess, tot_cnt, &temp_y_class_cnt]
                                   { this->find_split_per_party(cnt_parties, local_num_parties, sum_grad, sum_hess, tot_cnt, temp_y_class_cnt); });
                    threads_parties.push_back(move(temp_th));
                    cnt_parties += num_parties_per_thread[i];
                }
                for (int i = 0; i < num_parties; i++)
                {
                    threads_parties[i].join();
                }
            }
        }

        score = best_score;
        return make_tuple(best_party_id, best_col_id, best_threshold_id);
    }

    void make_children_nodes(int best_party_id, int best_col_id, int best_threshold_id)
    {
        // TODO: remove idx with nan values from right_idxs;
        vector<int> left_idxs = parties->at(best_party_id).split_rows(idxs, best_col_id, best_threshold_id);
        vector<int> right_idxs;
        for (int i = 0; i < row_count; i++)
            if (!any_of(left_idxs.begin(), left_idxs.end(), [&](float x)
                        { return x == idxs[i]; }))
                right_idxs.push_back(idxs[i]);

        left = new XGBoostNode(parties, y, num_classes, gradient, hessian, left_idxs, min_child_weight,
                               lam, gamma, eps, depth - 1, active_party_id, use_only_active_party, n_job);
        if (left->is_leaf_flag == 1)
        {
            left->party_id = party_id;
        }
        right = new XGBoostNode(parties, y, num_classes, gradient, hessian, right_idxs, min_child_weight,
                                lam, gamma, eps, depth - 1, active_party_id, use_only_active_party, n_job);
        if (right->is_leaf_flag == 1)
        {
            right->party_id = party_id;
        }
    }

    vector<vector<float>> predict(vector<vector<float>> x_new)
    {
        int x_new_size = x_new.size();
        vector<vector<float>> y_pred(x_new_size);
        for (int i = 0; i < x_new_size; i++)
            y_pred[i] = predict_row(x_new[i]);
        return y_pred;
    }

    vector<float> predict_row(vector<float> xi)
    {
        if (is_leaf())
            return val;
        else
        {
            if (parties[party_id].is_left(record_id, xi))
                return left->predict_row(xi);
            else
                return right->predict_row(xi);
        }
    }

    bool is_leaf()
    {
        if (is_leaf_flag == -1)
        {
            return is_pure() || std::isinf(score) || depth <= 0;
        }
        else
        {
            return is_leaf_flag;
        }
    }

    bool is_pure()
    {
        set<float> s{};
        for (int i = 0; i < row_count; i++)
        {
            if (s.insert(y[idxs[i]]).second)
            {
                if (s.size() == 2)
                    return false;
            }
        }
        return true;
    }
};
