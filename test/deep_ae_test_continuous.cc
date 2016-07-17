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

#include <iostream>
#include "../src/deep_ae.h"
#include "../src/math/matrix.h"

int main() {
  Matrix x_train = {{0.1, 0.3}, {0.1, 0.7}, {0.3, 0.4}, {0.2, 0.2}},
         y_train = {{0}, {1}, {1}, {0}},
         x_test = {{0.2, 0.2}, {0.5, 0.6}, {0.1, 0.3}, {0.2, 0.1}, {0.4, 0.6}, {0.3, 0.6}},
         y_test = {{0}, {1}, {0}, {0}, {1}, {1}};
  DeepAE network({2, 16, 8, 8, 4, 1}, true, true);  // debug = true, continuous = true
  network.train(x_train, y_train, 500, 4, 3e-3, 6e-1, 1000, 4, 2e-0, 4e-1); 
  std::cout << y_test << std::endl;
  std::cout << network.predict(x_test) << std::endl;
  return 0;
}
