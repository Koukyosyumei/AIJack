#include "conjgrad.h"
#include "logisticregression.h"
#include "loss.h"
#include <vector>

struct Rain {
  BCELoss loss;
  LogisticRegression *clf;
  Rain(LogisticRegression *clf) : clf(clf) {}
  std::vector<float> getdQ(bool minpos, std::vector<int> idxs,
                           std::vector<std::vector<float>> &x) {
    if (x.size() == 0) {
      return {};
    }
    int m = x[0].size();

    std::vector<std::vector<float>> xs_normalized = clf->preprocess(x);
    std::vector<float> dQ(m, 0);

    for (int i : idxs) {
      float pred = 0;
      for (int j = 0; j < m; j++) {
        pred += clf->params[j][0] * xs_normalized[i][j];
      }
      for (int j = 0; j < m; j++) {
        float tmp =
            xs_normalized[i][j] / (2 + std::exp(pred) + std::exp(-pred));
        if (minpos) {
          dQ[j] -= tmp;
        } else {
          dQ[j] += tmp;
        }
      }
    }

    return dQ;
  }

  std::vector<float> getInfluence(bool minpos, std::vector<int> &idxs,
                                  std::vector<std::vector<float>> &x,
                                  std::vector<float> &y,
                                  std::vector<std::vector<float>> &y_proba) {
    std::vector<float> dQ = getdQ(minpos, idxs, x); // m
    std::vector<std::vector<float>> H =
        loss.get_hess_w(x, y_proba, y); // m \times m
    std::vector<std::vector<float>> E =
        loss.get_grad_w_ewise(x, y_proba, y);                    // n \times m
    std::vector<float> HinvdQ = conjugateGradient<float>(H, dQ); // m
    std::vector<float> influence = matrixVectorMultiply<float>(E, HinvdQ);
    return influence;
  }
};
