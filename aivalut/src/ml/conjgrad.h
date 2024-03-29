#include <cmath>
#include <iostream>
#include <vector>

using namespace std;

const double eps = 1e-8;

// Function to calculate the dot product of two vectors
template <typename V>
inline V dotProduct(const vector<V> &v1, const vector<V> &v2) {
  V result = 0.0;
  for (size_t i = 0; i < v1.size(); i++) {
    result += v1[i] * v2[i];
  }
  return result;
}

// Function to perform matrix-vector multiplication
template <typename V>
inline vector<V> matrixVectorMultiply(vector<vector<V>> &A, vector<V> &x) {
  size_t n = A.size();
  vector<V> result(n, 0.0);
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      result[i] += A[i][j] * x[j];
    }
  }
  return result;
}

// Function to perform Conjugate Gradient method
template <typename V>
inline vector<V> conjugateGradient(vector<vector<V>> &A, vector<V> &b) {
  size_t n = A.size();
  vector<V> x(n, 0.0); // Initial guess for solution x
  vector<V> r = b;     // Residual vector r
  vector<V> p = r;     // Search direction vector p
  V alpha, beta, residualNorm, prevResidualNorm;

  residualNorm = sqrt(dotProduct(r, r));
  prevResidualNorm = residualNorm;

  // Maximum number of iterations
  size_t maxIterations = n;

  for (size_t k = 0; k < maxIterations; k++) {
    vector<V> Ap = matrixVectorMultiply<V>(A, p);
    V pAp = dotProduct<V>(p, Ap);

    alpha = prevResidualNorm * prevResidualNorm / pAp;

    for (size_t i = 0; i < n; i++) {
      x[i] = x[i] + alpha * p[i];
      r[i] = r[i] - alpha * Ap[i];
    }

    residualNorm = sqrt(dotProduct(r, r));
    if (residualNorm < (V)eps) { // Convergence condition
      break;
    }

    beta = residualNorm * residualNorm / (prevResidualNorm * prevResidualNorm);

    for (size_t i = 0; i < n; i++) {
      p[i] = r[i] + beta * p[i];
    }

    prevResidualNorm = residualNorm;
  }

  return x;
}
