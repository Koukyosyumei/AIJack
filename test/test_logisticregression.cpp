#include <gtest/gtest.h>
#include <vector>

#include "../src/ml/logisticregression.h"

TEST(LogisticRegressionTest, Pipeline) {
  std::vector<std::vector<float>> xs = {{8, 4},  {0, 1}, {1, 1}, {5, 15},
                                        {5, 19}, {0, 0}, {1, 0}, {18, 6}};
  std::vector<float> ys = {1, 0, 0, 1, 1, 0, 0, 1};

  LogisticRegression clf(100, 1.0);
  clf.fit(xs, ys);

  std::vector<float> y_preds = clf.predict(xs);
  for (int i = 0; i < y_preds.size(); i++) {
    ASSERT_EQ(ys[i], y_preds[i]);
  }
}
