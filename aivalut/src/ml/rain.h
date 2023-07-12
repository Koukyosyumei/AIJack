#include "conjgrad.h"
#include "logisticregression.h"
#include "loss.h"
#include <vector>

struct Rain {
  BCELoss loss;
  LogisticRegression *clf;
  Rain(LogisticRegression *clf) : clf(clf) {}
  std::vector<double> getdQ(bool shoudbepos, std::vector<int> idxs,
                            std::vector<std::vector<double>> &xs_normalized) {
    if (xs_normalized.size() == 0) {
      return {};
    }
    int m = xs_normalized[0].size();

    std::vector<double> dQ(m, 0);

    for (int i : idxs) {
      double pred = 0;
      for (int j = 0; j < m; j++) {
        pred += clf->params[j][0] * xs_normalized[i][j];
      }
      for (int j = 0; j < m; j++) {
        double tmp =
            xs_normalized[i][j] / (2 + std::exp(pred) + std::exp(-pred));
        if (shoudbepos) {
          dQ[j] += tmp;
        } else {
          dQ[j] -= tmp;
        }
      }
    }

    return dQ;
  }

  std::vector<double> getInfluence(bool shoudbepos, std::vector<int> &idxs,
                                   std::vector<std::vector<double>> &x,
                                   std::vector<double> &y,
                                   std::vector<std::vector<double>> &y_proba) {
    std::vector<std::vector<double>> xs_normalized = clf->normalize(x);
    std::vector<double> dQ = getdQ(shoudbepos, idxs, xs_normalized); // m
    std::vector<std::vector<double>> H =
        loss.get_hess_w(xs_normalized, y_proba, y); // m \times m
    std::vector<std::vector<double>> E =
        loss.get_grad_w_ewise(xs_normalized, y_proba, y);          // n \times m
    std::vector<double> HinvdQ = conjugateGradient<double>(H, dQ); // m
    std::vector<double> influence = matrixVectorMultiply<double>(E, HinvdQ);
    return influence;
  }
};
