/*
Copyright 2016 WaizungTaam.  All rights reserved.

License: Apache License 2.0
Email:   waizungtaam@gmail.com

*/

#include <cmath>
#include <string>
#include "vector.h"
#include "matrix.h"

namespace nn {

Vector _forall(const Vector &, double (*pf)(double));
Matrix _forall(const Matrix &, double (*pf)(double));

bool approx(double, double, double);

double exp(double);
Vector exp(const Vector &);
Matrix exp(const Matrix &);

double log(double);
Vector log(const Vector &);
Matrix log(const Matrix &);

double pow(double, double);
Vector pow(const Vector &, double);
Matrix pow(const Matrix &, double);

double sqrt(double);
Vector sqrt(const Vector &);
Matrix sqrt(const Matrix &);

double mean(const Vector &);
double mean(const Matrix &);
Vector mean(const Matrix &, int);

double variance(const Vector &);
double variance(const Matrix &);
Vector variance(const Matrix &, int);

double stddev(const Vector &);
double stddev(const Matrix &);
Vector stddev(const Matrix &, int);

Vector activ_func(const Vector &, std::string);
Matrix activ_func(const Matrix &, std::string);
Vector d_activ_func(const Vector &, std::string);
Matrix d_activ_func(const Matrix &, std::string);

double relu(double);
Vector relu(const Vector &);
Matrix relu(const Matrix &);
double d_relu(double);
Vector d_relu(const Vector &);
Matrix d_relu(const Matrix &);

double logistic(double);
Vector logistic(const Vector &);
Matrix logistic(const Matrix &);
double d_logistic(double);
Vector d_logistic(const Vector &);
Matrix d_logistic(const Matrix &);

double tanh(double);
Vector tanh(const Vector &);
Matrix tanh(const Matrix &);
double d_tanh(double);
Vector d_tanh(const Vector &);
Matrix d_tanh(const Matrix &);

Vector softmax(const Vector &);
Matrix softmax(const Matrix &);
Vector d_softmax(const Vector &);
Matrix d_softmax(const Matrix &);

Vector convolve(const Vector &, const Vector &);
Matrix convolve(const Matrix &, const Matrix &);

double binomial_sample(int, double);
Vector binomial_sample(int, const Vector &);
Matrix binomial_sample(int, const Matrix &);

double normal_sample(double, double);
Vector normal_sample(const Vector &, double);
Matrix normal_sample(const Matrix &, double);
Matrix normal_sample(const Matrix &, const Vector &);

#ifndef PARAM_LIST
#define PARAM_LIST

class param_list {
public:
  int num_epochs;
  int batch_size;
  double learning_rate;
  double momentum;
  std::string activ_func;
  std::string output_func;

  param_list() = default;
  param_list(const param_list &) = default;
  param_list(param_list &&) = default;
  param_list & operator=(const param_list &) = default;
  param_list & operator=(param_list &&) = default;
  ~param_list() = default;
  param_list(int n, int b, double l, double m, 
             std::string a, std::string o) :
             num_epochs(n), batch_size(b),
             learning_rate(l), momentum(m),
             activ_func(a), output_func(o) {}
};

#endif  // class param_lisst

}  // namespace nn
