#include <cmath>
using namespace std;

double sigmoid(double x)
{
    double sigmoid_range = 34.538776394910684;
    if (x <= -1 * sigmoid_range)
        return 1e-15;
    else if (x >= sigmoid_range)
        return 1.0 - 1e-15;
    else
        return 1.0 / (1.0 + exp(-1 * x));
}
