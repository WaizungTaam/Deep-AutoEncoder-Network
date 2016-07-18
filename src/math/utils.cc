#include <cmath>
#include <string>
#include <random>
#include "vector.h"
#include "matrix.h"
#include "utils.h"

namespace nn {

Vector _forall(const Vector & v, double (*pf)(double)) {
  Vector res = v;
  int idx;
  for (idx = 0; idx < res.shape()[0]; ++idx) {
    res[idx] = (*pf)(res[idx]);
  }
  return res;
}
Matrix _forall(const Matrix & m, double (*pf)(double)) {
  Matrix res = m;
  int idx_row, idx_col;
  for (idx_row = 0; idx_row < res.shape()[0]; ++idx_row) {
    for (idx_col = 0; idx_col < res.shape()[1]; ++idx_col) {
      res[idx_row][idx_col] = (*pf)(res[idx_row][idx_col]);
    }
  }
  return res;
}
bool approx(double x_1, double x_2, double error=1e-8) {
  if (x_1 - x_2 <= error &&
      x_2 - x_1 <= error) {
    return true;
  } else {
    return false;
  }
}
double exp(double x) {
  return std::exp(x);
}
Vector exp(const Vector & x) {
  return _forall(x, nn::exp);
}
Matrix exp(const Matrix & x) {
  return _forall(x, nn::exp);
}
double log(double x) {
  return std::log(x);
}
Vector log(const Vector & x) {
  return _forall(x, nn::log);
}
Matrix log(const Matrix & x) {
  return _forall(x, nn::log);
}
double pow(double x, double exp) {
  return std::pow(x, exp);
}
Vector pow(const Vector & x, double exp) {
  Vector res(x.shape()[0]);
  int idx;
  for (idx = 0; idx < x.shape()[0]; ++idx) {
    res[idx] = nn::pow(x[idx], exp);
  }
  return res;
}
Matrix pow(const Matrix & x, double exp) {
  Matrix res(x.shape()[0], x.shape()[1]);
  int idx_row, idx_col;
  for (idx_row = 0; idx_row < x.shape()[0]; ++idx_row) {
    for (idx_col = 0; idx_col < x.shape()[1]; ++idx_col) {
      res[idx_row][idx_col] = nn::pow(x[idx_row][idx_col], exp);
    }
  }
  return res;
}
double sqrt(double x) {
  return std::sqrt(x);
}
Vector sqrt(const Vector & x) {
  return _forall(x, nn::sqrt);
}
Matrix sqrt(const Matrix & x) {
  return _forall(x, nn::sqrt);
}
double mean(const Vector & vec) {
  return vec.sum() / static_cast<double>(vec.shape()[0]);
}
double mean(const Matrix & mat) {
  return mat.sum() / mat.shape()[0] / mat.shape()[1];
}
Vector mean(const Matrix & mat, int dim) {
  return mat.sum(dim) / static_cast<double>(mat.shape()[dim]);
}
double variance(const Vector & vec) {
  return (nn::pow(vec - nn::mean(vec), 2)).sum() / vec.shape()[0];
}
double variance(const Matrix & mat) {
  return (nn::pow(mat, 2)).sum() / mat.shape()[0] / mat.shape()[1];
}
Vector variance(const Matrix & mat, int dim) {
  if (dim == 0) {
    int idx_row;
    Vector res(mat.shape()[1], 0);
    Vector mean_vec = nn::mean(mat, 0);
    for (idx_row = 0; idx_row < mat.shape()[0]; ++idx_row) {
      res += nn::pow(mat[idx_row] - mean_vec, 2);
    }
    return res / mat.shape()[0];
  } else if (dim == 1) {
    Matrix mat_mean = nn::mean(mat, 1);
    return (nn::pow(mat - mat_mean, 2)).sum(1) / mat.shape()[1];
  }
}
double stddev(const Vector & vec) {
  return nn::sqrt(nn::variance(vec));
}
double stddev(const Matrix & mat) {
  return nn::sqrt(nn::variance(mat));
}
Vector stddev(const Matrix & mat, int dim) {
  return nn::sqrt(nn::variance(mat, dim));
}
Vector activ_func(const Vector & x, std::string func_name) {
  if (func_name == "identity") {
    return x;
  } else if (func_name == "relu") {
    return nn::relu(x);
  } else if (func_name == "logistic") {
    return nn::logistic(x);
  } else if (func_name == "tanh") {
    return nn::tanh(x);
  } else if (func_name == "softmax") {
    return nn::softmax(x);
  } else {
    throw "Unsupported activation function";
  }
}
Matrix activ_func(const Matrix & x, std::string func_name) {
  if (func_name == "identity") {
    return x;
  } else if (func_name == "relu") {
    return nn::relu(x);
  } else if (func_name == "logistic") {
    return nn::logistic(x);
  } else if (func_name == "tanh") {
    return nn::tanh(x);
  } else if (func_name == "softmax") {
    return nn::softmax(x);
  } else {
    throw "Unsupported activation function";
  }
}
Vector d_activ_func(const Vector & x, std::string func_name) {
  if (func_name == "identity") {
    Vector res(x.shape()[0], 1.0);
    return res;
  } else if (func_name == "relu") {
    return nn::d_relu(x);
  } else if (func_name == "logistic") {
    return nn::d_logistic(x);
  } else if (func_name == "tanh") {
    return nn::d_tanh(x);
  } else if (func_name == "softmax") {
    return nn::d_softmax(x);
  } else {
    throw "Unsupported derivative of activation function";
  }
}
Matrix d_activ_func(const Matrix & x, std::string func_name) {
  if (func_name == "identity") {
    Matrix res(x.shape()[0], x.shape()[1], 1.0);
    return res;
  } else if (func_name == "relu") {
    return nn::d_relu(x);
  } else if (func_name == "logistic") {
    return nn::d_logistic(x);
  } else if (func_name == "tanh") {
    return nn::d_tanh(x);
  } else if (func_name == "softmax") {
    return nn::d_softmax(x);
  } else {
    throw "Unsupported derivative of activation function";
  }
}
double relu(double x) {
  return x >= 0 ? x : 0;
}
Vector relu(const Vector & x) {
  return _forall(x, nn::relu);
}
Matrix relu(const Matrix & x) {
  return _forall(x, nn::relu);
}
double d_relu(double x) {
  if (x >= 0) {
    return 1;
  } else {
    return 0;
  }
}
Vector d_relu(const Vector & x) {
  return _forall(x, nn::d_relu);
}
Matrix d_relu(const Matrix & x) {
  return _forall(x, nn::d_relu);
}
double logistic(double x) {
  return 1.0 / (1.0 + std::exp(-x));
}
Vector logistic(const Vector & x) {
  return _forall(x, nn::logistic);
}
Matrix logistic(const Matrix & x) {
  return _forall(x, nn::logistic);
}
double d_logistic(double x) {
  return nn::logistic(x) * (1.0 - nn::logistic(x));
}
Vector d_logistic(const Vector & x) {
  return _forall(x, nn::d_logistic);
}
Matrix d_logistic(const Matrix & x) {
  return _forall(x, nn::d_logistic);
}
double tanh(double x) {
  return std::tanh(x);
}
Vector tanh(const Vector & x) {
  return _forall(x, nn::tanh);
}
Matrix tanh(const Matrix & x) {
  return _forall(x, nn::tanh);
}
double d_tanh(double x) {
  return nn::tanh(x) * (1 - nn::tanh(x));
}
Vector d_tanh(const Vector & x) {
  return _forall(x, nn::d_tanh);
}
Matrix d_tanh(const Matrix & x) {
  return _forall(x, nn::d_tanh);
}
Vector softmax(const Vector & x) {
  return nn::exp(x - x.max()) / nn::exp(x - x.max()).sum();
}
Matrix softmax(const Matrix & x) {
  return nn::exp(x - x.max()) / nn::exp(x - x.max()).sum();
}
Vector d_softmax(const Vector & x) {
  return nn::softmax(x) * (1 - nn::softmax(x));
}
Matrix d_softmax(const Matrix & x) {
  return nn::softmax(x).cross(1 - nn::softmax(x));
}
Vector convolve(const Vector & x, const Vector & y) {
  if (x.shape()[0] < y.shape()[0]) {
    throw "Invalid shape for convolution";
  }
  Vector ans = Vector(x.shape()[0] - y.shape()[0] + 1);
  int idx;
  for (idx = 0; idx < ans.shape()[0]; ++idx) {
    ans[idx] = (x(idx, idx + y.shape()[0]) * y).sum();
  }
  return ans;
}
Matrix convolve(const Matrix & x, const Matrix & y) {
  if (!(x.shape()[0] >= y.shape()[0] && 
        x.shape()[1] >= y.shape()[1])) {
    throw "Invalid shapes for convolution";
  }
  Matrix ans = Matrix(x.shape()[0] - y.shape()[0] + 1,
                      x.shape()[1] - y.shape()[1] + 1);
  int idx_row, idx_col;
  for (idx_row = 0; idx_row < ans.shape()[0]; ++idx_row) {
    for (idx_col = 0; idx_col < ans.shape()[1]; ++idx_col) {
      ans[idx_row][idx_col] = 
        (x(idx_row, idx_row + y.shape()[0], 
           idx_col, idx_col + y.shape()[1]).cross(y)).sum();
    }
  }
  return ans;
}
double binomial_sample(int upper_bound, double prob) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::binomial_distribution<int> bin_dis(upper_bound, prob);
  double res_sample = static_cast<double>(bin_dis(gen));
  return res_sample;
}
Vector binomial_sample(int upper_bound, const Vector & vec_prob) {
  int idx;
  Vector vec_sampled(vec_prob.shape()[0]);
  for (idx = 0; idx < vec_sampled.shape()[0]; ++idx) {
    vec_sampled[idx] = binomial_sample(upper_bound, vec_prob[idx]);
  }
  return vec_sampled;
}
Matrix binomial_sample(int upper_bound, const Matrix & mat_prob) {
  int idx_row;
  Matrix mat_sampled(mat_prob.shape());
  for (idx_row = 0; idx_row < mat_sampled.shape()[0]; ++idx_row) {
    mat_sampled[idx_row] = binomial_sample(upper_bound, mat_prob[idx_row]);
  }
  return mat_sampled;
}

double normal_sample(double mean, double stddev) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<double> nor_dis(mean, stddev);
  double res_sample = nor_dis(gen);
  return res_sample;
}
Vector normal_sample(const Vector & vec_mean, double stddev) {
  int idx;
  Vector vec_sampled(vec_mean.shape()[0]);
  for (idx = 0; idx < vec_sampled.shape()[0]; ++idx) {
    vec_sampled[idx] = normal_sample(vec_mean[idx], stddev);
  }
  return vec_sampled;
}
Matrix normal_sample(const Matrix & mat_mean, double stddev) {
  int idx_row;
  Matrix mat_sampled(mat_mean.shape());
  for (idx_row = 0; idx_row < mat_sampled.shape()[0]; ++idx_row) {
    mat_sampled[idx_row] = normal_sample(mat_mean[idx_row], stddev);
  }
  return mat_sampled;
}
Matrix normal_sample(const Matrix & mat_mean, 
                     const Vector & vec_stddev) {
  if (mat_mean.shape()[1] != vec_stddev.shape()[0]) {
    throw "Inconsistent shape for normal sample.";
  }
  int idx_row, idx_col;
  Matrix mat_sampled(mat_mean.shape());
  for (idx_row = 0; idx_row < mat_sampled.shape()[0]; ++idx_row) {
    for (idx_col = 0; idx_col < mat_sampled.shape()[1]; ++idx_col) {
      mat_sampled[idx_row][idx_col] = normal_sample(mat_mean[idx_row][idx_col],
                                                    vec_stddev[idx_col]);
    }
  }
  return mat_sampled;
}

}  // namespace nn
