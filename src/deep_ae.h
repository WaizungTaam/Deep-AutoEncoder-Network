/*
Copyright 2016 WaizungTaam.  All rights reserved.

License:       Apache License 2.0
Email:         waizungtaam@gmail.com
Creation time: 2016-07-16
Last modified: 2016-07-17
Reference: Hinton, G. E., & Salakhutdinov, R. R. (2006). 
           Reducing the dimensionality of data with neural networks. 
           Science, 313(5786), 504-507.

*/

// Deep AutoEncoder Network

#ifndef DEEP_AE_H
#define DEEP_AE_H

#include <vector>
#include <string>
#include "./math/vector.h"
#include "./math/matrix.h"
#include "./math/utils.h"

class MLP {
public:
  MLP() = default;
  MLP(const std::vector<int> &, 
      const std::string & activ_func = "logistic",
      const std::string & output_func = "binary_step");
  MLP(const std::initializer_list<int> &,
      const std::string & activ_func = "logistic",
      const std::string & output_func = "binary_step");
  MLP(const MLP &) = default;
  MLP(MLP &&) = default;
  MLP & operator=(const MLP &) = default;
  MLP & operator=(MLP &&) = default;
  ~MLP() = default;
  void train(const Matrix &, const Matrix &, 
             int, int, double, double);
  Matrix output(const Matrix &);
  Matrix predict(const Matrix &);
  Matrix & share_weight(int);  // share weight with rbm
  Matrix & share_w_bias(int);  // share w_bias with rbm
  bool debug;
private:
  std::string activ_func_name;
  std::string output_func_name;
  std::vector<Matrix> weights;
  std::vector<Matrix> ws_bias;
  std::vector<Matrix> delta_weights;
  std::vector<Matrix> delta_ws_bias;
  std::vector<Matrix> data_forward;
  std::vector<Matrix> local_fields;
  std::vector<Matrix> local_gradients;
  void forward(const Matrix &);
  void backward(const Matrix &);
  void update(double, double);
};

class RBM {
public:
  RBM() = default;
  RBM(Matrix &, const Matrix &, Matrix &,
      const std::string & activ_func = "logistic",
      const std::string & type_vis = "binary",
      const std::string & type_hid = "binary");
  RBM(const RBM &) = default;
  RBM(RBM &&) = default;
  RBM & operator=(const RBM &) = default;
  RBM & operator=(RBM &&)  =default;
  ~RBM() = default;
  void train(const Matrix &, int, int, double, double);
  Matrix output(const Matrix &);
  Matrix reconstruct(const Matrix &);
  Matrix reconstruct(const Matrix &, const Vector &);  // for linear vis
  void set_type(const std::string &, const std::string &);  // set the layer type: binary, linear
  bool debug;
private:
  std::string activ_func_name;
  std::string type_vis;
  std::string type_hid;
  Matrix & weight;
  Matrix w_bias_vis;
  Matrix & w_bias_hid;
  Matrix delta_weight;
  Matrix delta_w_bias_vis;
  Matrix delta_w_bias_hid;
  Matrix prop_vh(const Matrix &);
  Matrix prop_vh(const Matrix &, const Vector &);  // for linear vis
  Matrix prop_hv(const Matrix &);
  Matrix sample_vh(const Matrix &);
  Matrix sample_hv(const Matrix &);
  Matrix sample_hv(const Matrix &, const Vector &);  // for linear vis
};

class DeepAE {
public:
  DeepAE() = default;
  // continuous: true  -> Input being continuous data
  //             false -> Input being distrete (binary) data
  // debug:      true  -> Print the error variations during training
  //             false -> No printing
  DeepAE(const std::vector<int> &, 
         bool debug = false, bool continuous = false);
  DeepAE(const std::initializer_list<int> &,
         bool debug = false, bool continuous = false);
  DeepAE(const DeepAE &) = default;
  DeepAE(DeepAE &&) = default;
  DeepAE & operator=(const DeepAE &) = default;
  DeepAE & operator=(DeepAE &&)  =default;
  ~DeepAE() = default;
  void train(const Matrix &, const Matrix &,
             int, int, double, double,
             int, int, double, double);
  Matrix predict(const Matrix &);
private:
  MLP mlp;
  std::vector<RBM> rbms;  
  void pre_train(const Matrix &, int, int, double, double);
  void fine_tune(const Matrix &, const Matrix &, 
                 int, int, double, double);
};

#endif  // deep_ae.h