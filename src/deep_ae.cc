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

#include <vector>
#include <string>
#include "deep_ae.h"
#include "./math/vector.h"
#include "./math/matrix.h"
#include "./math/utils.h"

// <-- class MLP  

MLP::MLP(const std::vector<int> & dim_layers,
         const std::string & activ_func,
         const std::string & output_func) {
  int idx;
  for (idx = 0; idx < dim_layers.size() - 1; ++idx) {
    weights.push_back(Matrix(dim_layers[idx], dim_layers[idx + 1], 
                             "uniform", -1.0, 1.0));
    ws_bias.push_back(Matrix(1, dim_layers[idx + 1],
                             "uniform", -1.0, 1.0));
    delta_weights.push_back(Matrix(weights[idx].shape(), 0));
    delta_ws_bias.push_back(Matrix(ws_bias[idx].shape(), 0));
  }
  activ_func_name = activ_func;
  output_func_name = output_func;
  debug = false;
}
MLP::MLP(const std::initializer_list<int> & dim_layers,
         const std::string & activ_func,
         const std::string & output_func) : 
  MLP(std::vector<int>(dim_layers), activ_func, output_func){
}
void MLP::train(const Matrix & data_in, const Matrix & data_out,
                int num_epochs, int batch_size, 
                double learning_rate, double momentum) {
  data_forward.resize(weights.size());
  local_fields.resize(weights.size());
  local_gradients.resize(weights.size());
  int num_samples = data_in.shape()[0],
      num_batches = static_cast<int>(num_samples / batch_size), 
      idx_epoch, idx_batch, idx_batch_begin, idx_batch_end;
  if (num_batches * batch_size != num_samples) {
    ++num_batches;
  }
  for (idx_epoch = 0; idx_epoch < num_epochs; ++idx_epoch) {
    for (idx_batch = 0; idx_batch < num_batches; ++idx_batch) {
      idx_batch_begin = idx_batch * batch_size;
      idx_batch_end = (idx_batch + 1) * batch_size;
      if (idx_batch == num_batches - 1) {
        idx_batch_end = num_samples;
      }
      forward(data_in(idx_batch_begin, idx_batch_end));
      backward(data_out(idx_batch_begin, idx_batch_end));
      update(learning_rate, momentum);

      if (debug) {
        std::cout << "mlp\t"                                              
                  << idx_epoch << "\t" << idx_batch << "\t"               
                  << (nn::pow(output(data_in) - data_out, 2)).sum()       
                  / data_in.shape()[0] / data_in.shape()[1] << std::endl; 
      }      
    }
  }
}
Matrix MLP::output(const Matrix & data_in) {
  int idx_layer;
  Matrix data_out = data_in;
  for (idx_layer = 0; idx_layer < weights.size(); ++idx_layer) {
    data_out = nn::activ_func(data_out * weights[idx_layer] + ws_bias[idx_layer], 
                              activ_func_name);
  }
  return data_out;
}
Matrix MLP::predict(const Matrix & data_in) {
  if (output_func_name == "binary_step") {
    return output(data_in) >= 0.5;
  } else if (output_func_name == "identity") {
    return output(data_in);
  } else {
    throw "Unsupported output function.";
  }
}
void MLP::forward(const Matrix & mat_in) {
  int idx_layer;
  Matrix mat_forward = mat_in;
  for (idx_layer = 0; idx_layer < weights.size(); ++idx_layer) {
    data_forward[idx_layer] = mat_forward;
    local_fields[idx_layer] = data_forward[idx_layer] * weights[idx_layer] + 
                              ws_bias[idx_layer];
    mat_forward = nn::activ_func(local_fields[idx_layer], activ_func_name);
  }
}
void MLP::backward(const Matrix & mat_out) {
  int num_layers = weights.size(), idx_layer;
  Matrix mat_pred = nn::activ_func(local_fields[num_layers - 1], 
                                   activ_func_name);
  local_gradients[num_layers - 1] = (mat_out - mat_pred).cross(
    nn::d_logistic(local_fields[num_layers - 1]));
  for (idx_layer = num_layers - 2; idx_layer >= 0; --idx_layer) {
    local_gradients[idx_layer] = nn::d_logistic(local_fields[idx_layer]).cross(
      local_gradients[idx_layer + 1] * weights[idx_layer + 1].T());
  }
  // local_fields.clear();
}
void MLP::update(double learning_rate, double momentum) {
  int idx_layer;
  Matrix bias(data_forward[0].shape()[0], 1, 1.0);
  for (idx_layer = 0; idx_layer < weights.size(); ++idx_layer) {
    delta_weights[idx_layer] = momentum * delta_weights[idx_layer] + 
      learning_rate * data_forward[idx_layer].T() * local_gradients[idx_layer];
    delta_ws_bias[idx_layer] = momentum * delta_ws_bias[idx_layer] +
      learning_rate * bias.T() * local_gradients[idx_layer];
    weights[idx_layer] += delta_weights[idx_layer];
    ws_bias[idx_layer] += delta_ws_bias[idx_layer];
  }
  // data_forward.clear();
  // local_gradients.clear();
}
Matrix & MLP::share_weight(int idx_layer) {
  return weights.at(idx_layer);
}
Matrix & MLP::share_w_bias(int idx_layer) {
  return ws_bias.at(idx_layer);
}

// class MLP -->

// <-- class RBM 

RBM::RBM(Matrix & weight_init, 
         const Matrix & w_bias_vis_init, 
         Matrix & w_bias_hid_init,
         const std::string & activ_func_init,
         const std::string & type_vis_init,
         const std::string & type_hid_init) :
  weight(weight_init),
  w_bias_vis(w_bias_vis_init),
  w_bias_hid(w_bias_hid_init),
  activ_func_name(activ_func_init),
  type_vis(type_vis_init),
  type_hid(type_hid_init) {
  delta_weight = Matrix(weight.shape());
  delta_w_bias_vis = Matrix(w_bias_vis.shape());
  delta_w_bias_hid = Matrix(w_bias_hid.shape());
  debug = false;
}
void RBM::train(const Matrix & data_in, 
                int num_epochs, int batch_size, 
                double learnig_rate, double momentum) {
  int num_samples = data_in.shape()[0],
      num_batches = static_cast<int>(num_samples / batch_size),
      idx_epoch, idx_batch, idx_batch_begin, idx_batch_end;
  if (num_batches * batch_size != num_samples) {
    ++num_batches;
  }      
  for (idx_epoch = 0; idx_epoch < num_epochs; ++idx_epoch) {
    for (idx_batch = 0; idx_batch < num_batches; ++idx_batch) {
      idx_batch_begin = idx_batch * batch_size;
      idx_batch_end = (idx_batch + 1) * batch_size;
      if (idx_batch == num_batches - 1) {
        idx_batch_end = num_samples;
      }
      if (type_vis == "binary" && type_hid == "binary") {
        Matrix state_vis = data_in(idx_batch_begin, idx_batch_end);
        Matrix state_hid = sample_vh(prop_vh(state_vis));
        Matrix state_vis_re = reconstruct(state_vis);
        Matrix state_hid_re = sample_vh(prop_vh(state_vis_re));

        Matrix bias(idx_batch_end - idx_batch_begin, 1, 1.0);

        delta_weight = momentum * delta_weight +
                       (state_vis.T() * state_hid - 
                       state_vis_re.T() * state_hid_re);
        delta_w_bias_vis = momentum * delta_w_bias_vis + 
                           bias.T() * (state_vis - state_vis_re);
        delta_w_bias_hid = momentum * delta_w_bias_hid +
                           bias.T() * (state_hid - state_hid_re);
        weight += learnig_rate * delta_weight;
        w_bias_vis += learnig_rate * delta_w_bias_vis;
        w_bias_hid += learnig_rate * delta_w_bias_hid;

        if (debug) {
          std::cout << "rbm\t"                                                          
                    << idx_epoch << "\t"                                                
                    << idx_batch << "\t"                                                
                    << nn::pow(data_in - prop_hv(sample_vh(prop_vh(
                       data_in))), 2).sum() 
                       / data_in.shape()[0] / data_in.shape()[1] << "\n";                          
        }          
      } else if (type_vis == "linear" && type_hid == "binary") {
        Vector stddev = nn::stddev(data_in, 0);
        Matrix state_vis = data_in(idx_batch_begin, idx_batch_end);
        Matrix state_hid = sample_vh(prop_vh(state_vis, stddev));
        Matrix state_vis_re = reconstruct(state_vis, stddev);
        Matrix state_hid_re = sample_vh(prop_vh(state_vis_re, stddev));


        Matrix bias(idx_batch_end - idx_batch_begin, 1, 1.0);

        delta_weight = momentum * delta_weight +
                       (state_vis.T() * state_hid - 
                       state_vis_re.T() * state_hid_re);
        delta_w_bias_vis = momentum * delta_w_bias_vis + 
                           bias.T() * (state_vis - state_vis_re);
        delta_w_bias_hid = momentum * delta_w_bias_hid +
                           bias.T() * (state_hid - state_hid_re);
        weight += learnig_rate * delta_weight;
        w_bias_vis += learnig_rate * delta_w_bias_vis;
        w_bias_hid += learnig_rate * delta_w_bias_hid;     

        if (debug) {
          std::cout << "rbm\t"                                                          
                    << idx_epoch << "\t"                                                
                    << idx_batch << "\t"                                                
                    << nn::pow(data_in - reconstruct(
                       data_in, stddev), 2).sum() 
                       / data_in.shape()[0] / data_in.shape()[1] << "\n";                          
        }       
      } else {
        throw "Unsupported type for RBM.";
      }    
    }
  }
}
Matrix RBM::output(const Matrix & data_in) {
  return sample_vh(prop_vh(data_in));
}
Matrix RBM::reconstruct(const Matrix & data_in) {
  return sample_hv(prop_hv(sample_vh(prop_vh(data_in))));
}
Matrix RBM::reconstruct(const Matrix & data_in, const Vector & stddev) {
  // Note: sample_hv receive data directly from sample_vh WITHOUT prop_hv
  return sample_hv(sample_vh(prop_vh(data_in, stddev)), stddev);  
}
Matrix RBM::prop_vh(const Matrix & state_vis) {
  return nn::activ_func(state_vis * weight + w_bias_hid, activ_func_name);
}
Matrix RBM::prop_vh(const Matrix & state_vis, const Vector & stddev) {
  Matrix  state_vis_proc = state_vis;
  int idx_row;
  for (idx_row = 0; idx_row < state_vis_proc.shape()[0]; ++idx_row) {
    state_vis_proc[idx_row] /= stddev;
  }
  return nn::activ_func(state_vis_proc * weight + w_bias_hid, activ_func_name);
}
Matrix RBM::prop_hv(const Matrix & state_hid) {
  return nn::activ_func(state_hid * weight.T() + w_bias_vis, activ_func_name);
}
Matrix RBM::sample_vh(const Matrix & prob_vis) {
  return nn::binomial_sample(1, prob_vis);
}
Matrix RBM::sample_hv(const Matrix & prob_hid) {
  return nn::binomial_sample(1, prob_hid);
}
Matrix RBM::sample_hv(const Matrix & state_hid, const Vector & stddev) {  
// Note: the input matrix here is the state of hidden layer and is NOT the probability
  Matrix mat_mean = state_hid * weight.T();
  int idx_row;
  for (idx_row = 0; idx_row < mat_mean.shape()[0]; ++idx_row) {
    mat_mean[idx_row] = mat_mean[idx_row] * stddev + w_bias_vis[0];
  }
  return nn::normal_sample(mat_mean, stddev);
}
void RBM::set_type(const std::string & vis, const std::string & hid) {
  type_vis = vis;
  type_hid = hid;
}

// class RBM -->

// < -- DeepAE

DeepAE::DeepAE(const std::vector<int> & dim_layers,
               bool debug, bool continuous) {
  mlp = MLP(dim_layers);
  int idx_layer;
  for (idx_layer = 0; idx_layer < dim_layers.size() - 2; ++idx_layer) {
    rbms.push_back(RBM(mlp.share_weight(idx_layer),
                       Matrix(1, dim_layers[idx_layer], "uniform", -1.0, 1.0),
                       mlp.share_w_bias(idx_layer)));    
  }
  if (debug) {
    mlp.debug = true;
    for (idx_layer = 0; idx_layer < rbms.size(); ++idx_layer) {
      rbms[idx_layer].debug = true;
    }    
  }
  if (continuous) {
    rbms[0].set_type("linear", "binary");
  }  
}
DeepAE::DeepAE(const std::initializer_list<int> & init_list,
               bool debug, bool continuous) :
  DeepAE(std::vector<int>(init_list), debug, continuous) {
}
void DeepAE::train(const Matrix & data_in, const Matrix & data_out,
                   int pre_num_epochs, int pre_batch_size,
                   double pre_learnig_rate, double pre_momentum,
                   int tune_num_epochs, int tune_batch_size,
                   double tune_learning_rate, double tune_momentum) {
  pre_train(data_in, pre_num_epochs, pre_batch_size,
            pre_learnig_rate, pre_momentum);
  fine_tune(data_in, data_out, tune_num_epochs, tune_batch_size,
            tune_learning_rate, tune_momentum);
}
Matrix DeepAE::predict(const Matrix & data_in) {
  return mlp.predict(data_in);
}
void DeepAE::pre_train(const Matrix & mat_in, 
                       int num_epochs, int batch_size, 
                       double learnig_rate, double momentum) {
  int idx_layer;
  Matrix mat_forward = mat_in;
  for (idx_layer = 0; idx_layer < rbms.size(); ++idx_layer) {
    rbms[idx_layer].train(mat_forward, num_epochs, batch_size, 
                          learnig_rate, momentum);
    mat_forward = rbms[idx_layer].output(mat_forward);
  }
}
void DeepAE::fine_tune(const Matrix & data_in, const Matrix & data_out,
                       int num_epochs, int batch_size,
                       double learnig_rate, double momentum) {
  mlp.train(data_in, data_out, num_epochs, batch_size, learnig_rate, momentum);
}

// DeepAE -- >
