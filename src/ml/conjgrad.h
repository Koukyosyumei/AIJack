#include <cmath>
#include <iostream>
#include <vector>

using namespace std;

const double eps = 1e-8;

// Function to calculate the dot product of two vectors
inline double dotProduct(const vector<double> &v1, const vector<double> &v2) {
  double result = 0.0;
  for (size_t i = 0; i < v1.size(); i++) {
    result += v1[i] * v2[i];
  }
  return result;
}

// Function to perform matrix-vector multiplication
inline vector<double> matrixVectorMultiply(const vector<vector<double>> &A,
                                           const vector<double> &x) {
  size_t n = A.size();
  vector<double> result(n, 0.0);
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      result[i] += A[i][j] * x[j];
    }
  }
  return result;
}

// Function to perform Conjugate Gradient method
inline vector<double> conjugateGradient(const vector<vector<double>> &A,
                                        const vector<double> &b) {
  size_t n = A.size();
  vector<double> x(n, 0.0); // Initial guess for solution x
  vector<double> r = b;     // Residual vector r
  vector<double> p = r;     // Search direction vector p
  double alpha, beta, residualNorm, prevResidualNorm;

  residualNorm = sqrt(dotProduct(r, r));
  prevResidualNorm = residualNorm;

  // Maximum number of iterations
  size_t maxIterations = n;

  for (size_t k = 0; k < maxIterations; k++) {
    vector<double> Ap = matrixVectorMultiply(A, p);
    double pAp = dotProduct(p, Ap);

    alpha = prevResidualNorm * prevResidualNorm / pAp;

    for (size_t i = 0; i < n; i++) {
      x[i] = x[i] + alpha * p[i];
      r[i] = r[i] - alpha * Ap[i];
    }

    residualNorm = sqrt(dotProduct(r, r));
    if (residualNorm < eps) { // Convergence condition
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
