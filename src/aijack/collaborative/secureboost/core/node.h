#include <cmath>
#include <numeric>
#include <vector>
#include <iterator>
#include <limits>
#include <algorithm>
#include <set>
#include <tuple>
#include <unordered_map>
using namespace std;

struct Party
{
    vector<vector<double>> x; // partyが持つ特徴量
    vector<int> feature_id;   // partyが持つ特徴量のidのベクトル
    int party_id;             // partyのid
    int min_leaf;
    double subsample_cols; // サンプリングされる特徴量の割合

    int col_count; // partyが持つ特徴量の種類の数

    unordered_map<int, pair<int, double>> lookup_table; // record_id: (feature_id, threshold)
    vector<vector<double>> temp_thresholds;             // feature_id->threshold

    Party() {}
    Party(vector<vector<double>> x_, vector<int> feaure_id_, int party_id_,
          int min_leaf_, double subsample_cols_ = 1.0)
    {
        x = x_;
        feature_id = feaure_id_;
        party_id = party_id_;
        min_leaf = min_leaf_;
        subsample_cols = subsample_cols_;

        col_count = x.at(0).size();
    }

    vector<double> get_percentiles(vector<double> x_col)
    {
        vector<double> percentiles;
        copy(x_col.begin(), x_col.end(), back_inserter(percentiles));
        sort(percentiles.begin(), percentiles.end(),
             [&percentiles](size_t i1, size_t i2)
             { return percentiles[i1] < percentiles[i2]; });
        return percentiles;
    }

    bool is_left(int record_id, vector<double> xi)
    {
        return xi[lookup_table[record_id].first] <= lookup_table[record_id].second;
    }

    vector<vector<pair<double, double>>> greedy_search_split(vector<double> gradient,
                                                             vector<double> hessian,
                                                             vector<int> idxs)
    {
        vector<int> column_subsample;
        column_subsample.resize(col_count);
        iota(column_subsample.begin(), column_subsample.end(), 0);

        // feature_id -> [(grad hess)]
        // the threshold of split_candidates_grad_hess[i][j] = temp_thresholds[i][j]
        vector<vector<pair<double, double>>> split_candidates_grad_hess(column_subsample.size());
        temp_thresholds = vector<vector<double>>(column_subsample.size());

        int row_count = idxs.size();
        int recoed_id = 0;

        for (int i = 0; i < column_subsample.size(); i++)
        {
            int k = column_subsample[i];
            vector<double> x_col(row_count);
            for (int r = 0; r < row_count; r++)
                x_col[r] = x[idxs[r]][k];

            vector<double> percentiles = get_percentiles(x_col);

            for (int p = 0; p < percentiles.size(); p++)
            {
                double temp_grad = 0;
                double temp_hess = 0;
                int temp_left_size = 0;
                for (int r = 0; r < row_count; r++)
                {
                    if (x_col[r] <= percentiles[p])
                    {
                        temp_grad += gradient[idxs[r]];
                        temp_hess += hessian[idxs[r]];
                        temp_left_size += 1;
                    }
                }

                if (temp_left_size >= min_leaf &&
                    row_count - temp_left_size >= min_leaf)
                {
                    split_candidates_grad_hess[i].push_back(make_pair(temp_grad, temp_hess));
                    temp_thresholds[i].push_back(percentiles[p]);
                }
            }
        }

        return split_candidates_grad_hess;
    }

    vector<int> split_rows(vector<int> idxs, int feature_opt_id, int threshold_opt_id)
    {
        // feature_opt_idがthreshold_opt_id以下のindexを返す
        int row_count = idxs.size();
        vector<double> x_col(row_count);
        for (int r = 0; r < row_count; r++)
            x_col[r] = x[idxs[r]][feature_opt_id];

        vector<int> left_idxs;
        double threshold = temp_thresholds[feature_opt_id][threshold_opt_id];
        for (int r = 0; r < row_count; r++)
            if (x_col[r] <= threshold)
                left_idxs.push_back(idxs[r]);

        return left_idxs;
    }

    int insert_lookup_table(int feature_opt_id, int threshold_opt_id)
    {
        lookup_table.emplace(lookup_table.size(),
                             make_pair(feature_opt_id,
                                       temp_thresholds[feature_opt_id][threshold_opt_id]));
        return lookup_table.size() - 1;
    }
};

struct Node
{
    vector<Party> parties;
    vector<double> y, gradient, hessian;
    vector<int> idxs;
    double min_child_weight, lam, gamma, eps;
    int depth;
    bool use_ispure;

    int party_id, record_id;
    int row_count, num_parties;
    double val, score;
    Node *left, *right;

    Node() {}
    Node(vector<Party> parties_, vector<double> y_, vector<double> gradient_,
         vector<double> hessian_, vector<int> idxs_,
         double min_child_weight_, double lam_, double gamma_, double eps_,
         int depth_)
    {
        parties = parties_;
        y = y_;
        gradient = gradient_;
        hessian = hessian_;
        idxs = idxs_;
        min_child_weight = min_child_weight_;
        lam = lam_;
        gamma = gamma_;
        eps = eps_;
        depth = depth_;

        row_count = idxs.size();
        num_parties = parties.size();

        val = compute_weight();
        tuple<int, int, int> best_split = find_split();

        if (!is_leaf())
        {
            party_id = get<0>(best_split);
            record_id = parties[party_id].insert_lookup_table(get<1>(best_split), get<2>(best_split));
            make_children_nodes(get<0>(best_split), get<1>(best_split), get<2>(best_split));
        }
    }

    double compute_weight()
    {
        double sum_grad = 0;
        double sum_hess = 0;
        for (int i = 0; i < row_count; i++)
        {
            sum_grad += gradient[idxs[i]];
            sum_hess += hessian[idxs[i]];
        }
        return -1 * (sum_grad / (sum_hess + lam));
    }

    double compute_gain(int left_grad, int right_grad, int left_hess, int right_hess)
    {
        return 0.5 * ((left_grad * left_grad) / (left_hess + lam) +
                      (right_grad * right_grad) / (right_hess + lam) -
                      ((left_grad + right_grad) *
                       (left_grad + right_grad) / (left_hess + right_hess + lam))) -
               gamma;
    }

    tuple<int, int, int> find_split()
    {
        double sum_grad = 0;
        double sum_hess = 0;
        for (int i = 0; i < row_count; i++)
        {
            sum_grad += gradient[idxs[i]];
            sum_hess += hessian[idxs[i]];
        }

        double best_score = -1 * numeric_limits<double>::infinity();
        double temp_score, temp_left_grad, temp_left_hess;
        int best_party_id, best_col_id, best_threshold_id;
        for (int i = 0; i < num_parties; i++)
        {
            vector<vector<pair<double, double>>> search_results =
                parties[i].greedy_search_split(gradient, hessian, idxs);

            for (int j = 0; j < search_results.size(); j++)
            {
                for (int k = 0; k < search_results[j].size(); k++)
                {
                    temp_left_grad = search_results[j][k].first;
                    temp_left_hess = search_results[j][k].second;

                    if (temp_left_hess < min_child_weight ||
                        sum_hess - temp_left_hess < min_child_weight)
                        continue;

                    temp_score = compute_gain(temp_left_grad, sum_grad - temp_left_grad,
                                              temp_left_hess, sum_hess - temp_left_hess);

                    if (temp_score > best_score)
                    {
                        best_score = temp_score;
                        best_party_id = i;
                        best_col_id = j;
                        best_threshold_id = k;
                    }
                }
            }
        }
        score = best_score;
        return make_tuple(best_party_id, best_col_id, best_threshold_id);
    }

    void make_children_nodes(int best_party_id, int best_col_id, int best_threshold_id)
    {
        vector<int> left_idxs = parties[best_party_id].split_rows(idxs, best_col_id, best_threshold_id);
        vector<int> right_idxs;
        for (int i = 0; i < idxs.size(); i++)
            if (!any_of(left_idxs.begin(), left_idxs.end(), [&](double x)
                        { return x == idxs[i]; }))
                right_idxs.push_back(idxs[i]);

        left = new Node(parties, y, gradient, hessian, left_idxs, min_child_weight,
                        lam, gamma, eps, depth - 1);
        right = new Node(parties, y, gradient, hessian, right_idxs, min_child_weight,
                         lam, gamma, eps, depth - 1);
    }

    bool is_leaf()
    {
        return is_pure() || std::isinf(score) || depth <= 0;
    }

    bool is_pure()
    {
        vector<int> y_temp(row_count);
        for (int i = 0; i < row_count; i++)
            y_temp[i] = y[idxs[i]];
        set<int> y_set_temp(y_temp.begin(), y_temp.end());
        return use_ispure && y_set_temp.size() == 1;
    }

    vector<double> predict(vector<vector<double>> x_new)
    {
        int x_new_size = x_new.size();
        vector<double> y_pred(x_new_size);
        for (int i = 0; i < x_new_size; i++)
            y_pred[i] = predict_row(x_new[i]);
        return y_pred;
    }

    double predict_row(vector<double> xi)
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
};
