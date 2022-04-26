#include <cmath>
#include <numeric>
#include <vector>
#include <iterator>
#include <limits>
#include <algorithm>
#include <set>
#include <unordered_map>
using namespace std;

struct Party
{
    vector<vector<double>> x; // partyが持つ特徴量
    vector<int> feature_id;   // partyが持つ特徴量のidのベクトル
    int party_id;             // partyのid
    double subsample_cols;    // サンプリングされる特徴量の割合

    int col_count; // partyが持つ特徴量の種類の数

    unordered_map<int, pair<int, double>> lookup_table; // record_id: (feature_id, threshold)
    vector<vector<double>> temp_thresholds;             // feature_id->threshold

    Party(vector<vector<double>> x_, vector<int> feaure_id_, int party_id_, double subsample_cols_ = 1.0)
    {
        x = x_;
        feature_id = feaure_id_;
        party_id = party_id_;
        subsample_cols = subsample_cols_;

        col_count = x.at(0).size();
    }

    vector<double> get_percentiles(vector<double> x_col)
    {
        vector<double> percentiles;
        copy(percentiles.begin(), percentiles.end(), back_inserter(x_col));
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
                int temp_grad = 0;
                int temp_hess = 0;
                for (int r = 0; r < row_count; r++)
                {
                    if (x_col[r] <= percentiles[p])
                    {
                        temp_grad += gradient[idxs[r]];
                        temp_hess += hessian[idxs[r]];
                    }
                }

                split_candidates_grad_hess[i].push_back(make_pair(temp_grad, temp_hess));
                temp_thresholds[i].push_back(percentiles[p]);
            }
        }

        return split_candidates_grad_hess;
    }

    vector<double> split_rows(vector<int> idxs, int feature_opt_id, int threshold_opt_id)
    {
        // feature_opt_idがthreshold_opt_id以下のindexを返す
        int row_count = idxs.size();
        vector<double> x_col(row_count);
        for (int r = 0; r < row_count; r++)
            x_col[r] = x[idxs[r]][feature_opt_id];

        vector<double> left_idxs;
        double threshold = temp_thresholds[feature_opt_id][threshold_opt_id];
        for (int r = 0; r < row_count; r++)
            if (x_col[r] <= threshold)
                left_idxs.push_back(idxs[r]);

        return left_idxs;
    }
};

struct Node
{
    vector<Party> parties;
    vector<double> y, gradient, hessian;
    vector<int> idxs;
    double subsample_cols, min_child_weight, lam, gamma, eps;
    int min_leaf, depth;
    bool use_ispure;

    int party_id, record_id;
    int row_count, num_parties;
    double val, score;
    Node *left, *right;

    Node() {}
    Node(vector<Party> parties_, vector<double> y_, vector<double> gradient_,
         vector<double> hessian_, vector<int> idxs_)
    {
        parties = parties_;
        y = y_;
        gradient = gradient_;
        hessian = hessian_;
        idxs = idxs_;

        row_count = y.size();
        num_parties = parties.size();

        val = compute_weight();
        find_split();
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

    void find_split()
    {
        double best_score = -1 * numeric_limits<double>::infinity();
        int best_party_id, best_col_id, best_threshold_id;
        for (int i = 0; i < num_parties; i++)
        {
            vector<vector<pair<double, double>>> search_results =
                parties[i].greedy_search_split(gradient, hessian, idxs);
        }
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
