#include <gtest/gtest.h>
#include <vector>

#include "../src/ml/logisticregression.h"
#include "../src/ml/metrics.h"

TEST(LogisticRegressionTest, Pipeline) {
  std::vector<std::vector<double>> xs = {{8, 4},  {0, 1}, {1, 1}, {5, 15},
                                         {5, 19}, {0, 0}, {1, 0}, {18, 6}};
  std::vector<double> ys = {1, 0, 0, 1, 1, 0, 0, 1};

  LogisticRegression clf(100, 1.0);
  clf.fit(xs, ys);

  std::vector<std::pair<double, double>> test_normalizer = {
      {0, 18}, {0, 19}, {0, 1}};
  for (int i = 0; i < 3; i++) {
    ASSERT_EQ(clf.normalizer[i].first, test_normalizer[i].first);
    ASSERT_EQ(clf.normalizer[i].second, test_normalizer[i].second);
  }

  std::vector<std::vector<double>> y_proba = clf.predict_proba(xs);
  ASSERT_EQ(ovr_roc_auc_score(y_proba, ys), 1.0);
}
