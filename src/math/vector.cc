/*
Copyright 2016 WaizungTaam.  All rights reserved.

License: Apache License 2.0
Email:   waizungtaam@gmail.com

*/

#include <vector>
#include <iostream>
#include <iomanip>
#include <random>
#include <algorithm>
#include "vector.h"

Vector::Vector(int size) {
  vec = std::vector<double>(size, 0);
}
Vector::Vector(int size, double value) {
	vec = std::vector<double>(size, value);
}
Vector::Vector(int size, const std::string & mode, 
               double param_1, double param_2) {
  bool is_uniform = (mode == "uniform") || (mode == "Uniform") || 
                    (mode == "UNIFORM") || (mode == "u") || (mode == "U");
  bool is_normal = (mode == "normal") || (mode == "Normal") || 
                   (mode == "NORMAL") || (mode == "n") || (mode == "N");
  bool is_binomial = (mode == "binomial") || (mode == "Binomial") ||
                     (mode == "BINOMIAL") || (mode == "b") || (mode == "B");
  if (!(is_uniform || is_normal || is_binomial)) {
    throw "Unsupported mode";
  }
  vec = std::vector<double>(size, 0); 
  std::random_device rd;
  std::mt19937 gen(rd());
  if (is_uniform) {
    std::uniform_real_distribution<> uni_dis(param_1, param_2);
    for (double & element : vec) {
      element = uni_dis(gen);
    }
  } else if (is_normal) {
    std::normal_distribution<> nor_dis(param_1, param_2);
    for (double & element : vec) {
      element = nor_dis(gen);
    }
  } else if (is_binomial) {
    std::binomial_distribution<int> bin_dis(param_1, param_2);
    for (double & element : vec) {
      element = static_cast<double>(bin_dis(gen));
    }
  }
}
Vector::Vector(const std::vector<double> & vec_init) {
  vec = vec_init;
}
Vector::Vector(const std::initializer_list<double> & ls) : vec(ls) {
}
Vector & Vector::operator=(double value) {
	for (double & element : vec) {
		element = value;
	}
	return *this;
}
Vector & Vector::operator=(const std::vector<double> & vec_copy) {
  vec = vec_copy;
  return *this;
}
Vector & Vector::operator=(const std::initializer_list<double> & ls) {
  if (vec.size() != ls.size()) {
    throw "Inconsistent shape";
  }
  vec = ls;
  return *this;
}
std::vector<std::size_t> Vector::shape() const {
	std::vector<std::size_t> shape_vec;
	shape_vec.push_back(vec.size());
	return shape_vec;
}
Vector Vector::insert(double value, int index) const {
	if (index > vec.size()) {
		throw "Out-of-range";
	}	
	Vector vec_inserted = *this;
	vec_inserted.vec.insert(vec_inserted.vec.begin() + index, value);
	return vec_inserted;
}
Vector Vector::insert(const Vector & vec_to_insert, int index) const {
	if (index > vec.size()) {
		throw "Out-of-range";
	}
	Vector vec_inserted = *this;
	for (const double element : vec_to_insert.vec) {
		vec_inserted.vec.insert(vec_inserted.vec.begin() + index, element);
		++index;
	}
	return vec_inserted;
}
Vector Vector::remove(int index) const {
	if (index >= vec.size()) {
		throw "Out-of-range";
	}
	Vector vec_removed = *this;
	vec_removed.vec.erase(vec_removed.vec.begin() + index);
	return vec_removed;
} 
Vector Vector::remove(int idx_begin, int idx_end) const {
	if (idx_begin > idx_end) {
		int tmp_swap = idx_begin;
		idx_begin = idx_end;
		idx_end = tmp_swap;
	}
	if (idx_end >= vec.size()) {
		throw "Out-of-range";
	}
	Vector vec_removed = *this;
	vec_removed.vec.erase(vec_removed.vec.begin() + idx_begin, 
                        vec_removed.vec.begin() + idx_end);
	return vec_removed;
}
Vector Vector::replace(const Vector & vec_to_replace, int index) const {
  if (index >= vec.size()) {
    return *this;
  }
  Vector vec_replaced = *this;
  int idx_rep;
  for (idx_rep = 0; idx_rep < vec_to_replace.vec.size() &&
       idx_rep + index < vec_replaced.vec.size(); ++idx_rep) {
    vec_replaced[index + idx_rep] = vec_to_replace.vec[idx_rep];
  }
  return vec_replaced;
}
Vector Vector::shuffle() const {
  std::random_device rd;
  std::mt19937 gen(rd());
  Vector vec_shuffled = *this;
  std::shuffle(vec_shuffled.vec.begin(), vec_shuffled.vec.end(), gen);
  return vec_shuffled;
}
void Vector::clear() {
  vec.clear();
}
bool Vector::empty() {
  if (vec.size() == 0) {
    return true;
  } else {
    return false;
  }
}
double Vector::sum() const {
	double sum = 0;
	for (const double element : vec) {
		sum += element;
	}
	return sum;
}
double Vector::max() const {
  double max = vec[0];
  int idx;
  for (idx = 1; idx < shape()[0]; ++idx) {
    if (vec[idx] > max) {
      max = vec[idx];
    }
  }
  return max;
}
double Vector::min() const {
  double min = vec[0];
  int idx;
  for (idx = 1; idx < shape()[0]; ++idx) {
    if (vec[idx] < min) {
      min = vec[idx];
    }
  }
  return min;
}
Vector Vector::approx(const Vector & vec_to_compare, double error) const {
  Vector vec_is_approx(shape()[0], 0);
  int idx;
  for (idx = 0; idx < vec_is_approx.shape()[0]; ++idx) {
    if (vec[idx] - vec_to_compare.vec[idx] <= error && 
        vec_to_compare.vec[idx] - vec[idx] <= error) {
      vec_is_approx.vec[idx] = 1.0;
    }
  } 
  return vec_is_approx;
}
Vector Vector::approx(double value, double error) const {
  Vector vec_val = Vector(shape()[0], value);
  return approx(vec_val, error);
}
Vector operator+(const Vector & vec_lhs, const Vector & vec_rhs) {
	if (vec_lhs.vec.size() != vec_rhs.vec.size()) {
		throw "Inconsistent shape";
	}	
	Vector vec_sum = vec_lhs;
	int idx;
  for (idx = 0; idx < vec_sum.vec.size(); ++idx) {
  	vec_sum.vec[idx] += vec_rhs.vec[idx];
  }
  return vec_sum;
}
Vector operator+(const Vector & vec_lhs, double value) {
	Vector vec_sum = vec_lhs;
	for (double & element : vec_sum.vec) {
		element += value;
	} 
	return vec_sum;
}
Vector operator+(double value, const Vector & vec_rhs) {
  return operator+(vec_rhs, value);
}
Vector operator-(const Vector & vec_lhs, const Vector & vec_rhs) {
  if (vec_lhs.vec.size() != vec_rhs.vec.size()) {
    throw "Inconsistent shape";
  } 
  Vector vec_diff = vec_lhs;
  int idx;
  for (idx = 0; idx < vec_diff.vec.size(); ++idx) {
    vec_diff.vec[idx] -= vec_rhs.vec[idx];
  }
  return vec_diff;
}
Vector operator-(const Vector & vec_lhs, double value) {
	Vector vec_diff = vec_lhs;
	for (double & element : vec_diff.vec) {
		element -= value;
	} 
	return vec_lhs;
}
Vector operator-(double value, const Vector & vec_rhs) {
  Vector vec_diff = operator-(vec_rhs, value);
  vec_diff *= -1.0;
  return vec_diff;
}
Vector operator*(const Vector & vec_lhs, const Vector & vec_rhs) {
  if (vec_lhs.vec.size() != vec_rhs.vec.size()) {
    throw "Inconsistent shape";
  } 
  Vector vec_prod = vec_lhs;
  int idx;
  for (idx = 0; idx < vec_prod.vec.size(); ++idx) {
    vec_prod.vec[idx] *= vec_rhs.vec[idx];
  }
  return vec_prod;
}
Vector operator*(const Vector & vec_lhs, double value) {
	Vector vec_prod = vec_lhs;
	for (double & element : vec_prod.vec) {
		element *= value;
	} 
	return vec_prod;
}
Vector operator*(double value, const Vector & vec_rhs) {
  return operator*(vec_rhs, value);
}
Vector operator/(const Vector & vec_lhs, const Vector & vec_rhs) {
  if (vec_lhs.vec.size() != vec_rhs.vec.size()) {
    throw "Inconsistent shape";
  } 
  Vector vec_quot = vec_lhs;
  int idx;
  for (idx = 0; idx < vec_quot.vec.size(); ++idx) {
    vec_quot.vec[idx] /= vec_rhs.vec[idx];
  }
  return vec_quot;
}
Vector operator/(const Vector & vec_lhs, double value) {
	Vector vec_quot = vec_lhs;
	for (double & element : vec_quot.vec) {
		element /= value;
	} 
	return vec_quot;
}
Vector operator/(double value, const Vector & vec_rhs) {
  Vector vec_quot = vec_rhs;
  for (double & element : vec_quot.vec) {
    element = value / element;
  } 
  return vec_quot;  
}
void Vector::operator+=(double value) {
	(*this) = (*this) + value;
}
void Vector::operator+=(const Vector & vec_to_add) {
  (*this) = (*this) + vec_to_add;
}
void Vector::operator-=(double value) {
	(*this) = (*this) - value;
}
void Vector::operator-=(const Vector & vec_to_sub) {
  (*this) = (*this) - vec_to_sub;
}
void Vector::operator*=(double value) {
	(*this) = (*this) * value;
}
void Vector::operator*=(const Vector & vec_to_mul) {
  (*this) = (*this) * vec_to_mul;
}
void Vector::operator/=(double value) {
	(*this) = (*this) / value;
}
void Vector::operator/=(const Vector & vec_to_div) {
  (*this) = (*this) / vec_to_div;
}
bool operator==(const Vector & vec_lhs, const Vector & vec_rhs) {
	if (vec_lhs.vec.size() != vec_rhs.vec.size()) {
		return false;
	}
	int idx;
	for (idx = 0; idx < vec_lhs.vec.size(); ++idx) {
		if (vec_lhs.vec[idx] != vec_rhs.vec[idx]) {
			return false;
		}
	}
	return true;
}
Vector operator==(const Vector & vec_lhs, double value) {
  Vector vec_is_equ(vec_lhs.shape()[0]);
  int idx;
  for (idx = 0; idx < vec_is_equ.shape()[0]; ++idx) {
    if (value == vec_lhs[idx]) {
      vec_is_equ[idx] = 1.0;
    }
  }
  return vec_is_equ;
}
Vector operator==(double value, const Vector & vec_rhs) {
  return vec_rhs == value;
}
bool operator!=(const Vector & vec_lhs, const Vector & vec_rhs) {
  return !(vec_lhs == vec_rhs);
}
Vector operator!=(const Vector & vec_lhs, double value) {
  Vector vec_is_equ(vec_lhs.shape()[0]);
  int idx;
  for (idx = 0; idx < vec_is_equ.shape()[0]; ++idx) {
    if (value != vec_lhs[idx]) {
      vec_is_equ[idx] = 1.0;
    }
  }
  return vec_is_equ;
}
Vector operator!=(double value, const Vector & vec_rhs) {
  return vec_rhs != value;
}
Vector operator<(const Vector & vec_lhs, const Vector & vec_rhs) {
  if (vec_lhs.shape()[0] != vec_rhs.shape()[0]) {
    throw "Inconsistent shape";
  }
  Vector vec_is_les(vec_lhs.shape()[0]);
  int idx;
  for (idx = 0; idx < vec_lhs.shape()[0]; ++idx) {
    if (vec_lhs.vec[idx] < vec_rhs.vec[idx]) {
      vec_is_les.vec[idx] = 1.0;
    }
  }
  return vec_is_les;
}
Vector operator<(const Vector & vec_lhs, double value) {
  Vector vec_is_les(vec_lhs.shape()[0]);
  int idx;
  for (idx = 0; idx < vec_lhs.shape()[0]; ++idx) {
    if (vec_lhs.vec[idx] < value) {
      vec_is_les.vec[idx] = 1.0;
    }
  }
  return vec_is_les;
}
Vector operator<(double value, const Vector & vec_rhs) {
  Vector vec_is_les(vec_rhs.shape()[0]);
  int idx;
  for (idx = 0; idx < vec_rhs.shape()[0]; ++idx) {
    if (value < vec_rhs.vec[idx]) {
      vec_is_les.vec[idx] = 1.0;
    }
  }
  return vec_is_les;  
}
Vector operator<=(const Vector & vec_lhs, const Vector & vec_rhs) {
  if (vec_lhs.shape()[0] != vec_rhs.shape()[0]) {
    throw "Inconsistent shape";
  }
  Vector vec_is_leq(vec_lhs.shape()[0]);
  int idx;
  for (idx = 0; idx < vec_lhs.shape()[0]; ++idx) {
    if (vec_lhs.vec[idx] <= vec_rhs.vec[idx]) {
      vec_is_leq.vec[idx] = 1.0;
    }
  }
  return vec_is_leq;
}
Vector operator<=(const Vector & vec_lhs, double value) {
  Vector vec_is_leq(vec_lhs.shape()[0]);
  int idx;
  for (idx = 0; idx < vec_lhs.shape()[0]; ++idx) {
    if (vec_lhs.vec[idx] <= value) {
      vec_is_leq.vec[idx] = 1.0;
    }
  }
  return vec_is_leq;
}
Vector operator<=(double value, const Vector & vec_rhs) {
  Vector vec_is_leq(vec_rhs.shape()[0]);
  int idx;
  for (idx = 0; idx < vec_rhs.shape()[0]; ++idx) {
    if (value <= vec_rhs.vec[idx]) {
      vec_is_leq.vec[idx] = 1.0;
    }
  }
  return vec_is_leq;  
}
Vector operator>(const Vector & vec_lhs, const Vector & vec_rhs) {
  if (vec_lhs.shape()[0] != vec_rhs.shape()[0]) {
    throw "Inconsistent shape";
  }
  Vector vec_is_gre(vec_lhs.shape()[0]);
  int idx;
  for (idx = 0; idx < vec_lhs.shape()[0]; ++idx) {
    if (vec_lhs.vec[idx] > vec_rhs.vec[idx]) {
      vec_is_gre.vec[idx] = 1.0;
    }
  }
  return vec_is_gre;
}
Vector operator>(const Vector & vec_lhs, double value) {
  Vector vec_is_gre(vec_lhs.shape()[0]);
  int idx;
  for (idx = 0; idx < vec_lhs.shape()[0]; ++idx) {
    if (vec_lhs.vec[idx] > value) {
      vec_is_gre.vec[idx] = 1.0;
    }
  }
  return vec_is_gre;
}
Vector operator>(double value, const Vector & vec_rhs) {
  Vector vec_is_gre(vec_rhs.shape()[0]);
  int idx;
  for (idx = 0; idx < vec_rhs.shape()[0]; ++idx) {
    if (value > vec_rhs.vec[idx]) {
      vec_is_gre.vec[idx] = 1.0;
    }
  }
  return vec_is_gre;  
}
Vector operator>=(const Vector & vec_lhs, const Vector & vec_rhs) {
  if (vec_lhs.shape()[0] != vec_rhs.shape()[0]) {
    throw "Inconsistent shape";
  }
  Vector vec_is_geq(vec_lhs.shape()[0]);
  int idx;
  for (idx = 0; idx < vec_lhs.shape()[0]; ++idx) {
    if (vec_lhs.vec[idx] >= vec_rhs.vec[idx]) {
      vec_is_geq.vec[idx] = 1.0;
    }
  }
  return vec_is_geq;
}
Vector operator>=(const Vector & vec_lhs, double value) {
  Vector vec_is_geq(vec_lhs.shape()[0]);
  int idx;
  for (idx = 0; idx < vec_lhs.shape()[0]; ++idx) {
    if (vec_lhs.vec[idx] >= value) {
      vec_is_geq.vec[idx] = 1.0;
    }
  }
  return vec_is_geq;
}
Vector operator>=(double value, const Vector & vec_rhs) {
  Vector vec_is_geq(vec_rhs.shape()[0]);
  int idx;
  for (idx = 0; idx < vec_rhs.shape()[0]; ++idx) {
    if (value >= vec_rhs.vec[idx]) {
      vec_is_geq.vec[idx] = 1;
    }
  }
  return vec_is_geq;  
}
double & Vector::operator[](int index) {
  if (index >= 0) {
    if (index >= vec.size()) {
      throw "Out-of-range";
    }
    return vec.at(index);
  } else if (index < 0) {
    if ((-1) * index > vec.size()) {
      throw "Out-of-range";
    }
    return vec.at(vec.size() - index);
  }
}
const double & Vector::operator[](int index) const {
  if (index >= 0) {
    if (index >= vec.size()) {
      throw "Out-of-range";
    }
    return vec.at(index);
  } else if (index < 0) {
    if ((-1) * index > vec.size()) {
      throw "Out-of-range";
    }
    return vec.at(vec.size() - index);
  }
}
Vector Vector::operator()(int idx_begin, int idx_end) const {
  if (idx_begin > idx_end) {
  	int tmp_swap = idx_begin;
  	idx_begin = idx_end;
  	idx_end = tmp_swap;
  }
  if (idx_end > vec.size()) {
  	throw "Out-of-range";
  }
  Vector vec_partial(idx_end - idx_begin);
  vec_partial.vec = std::vector<double>(vec.begin() + idx_begin, 
                                        vec.begin() + idx_end);
  return vec_partial;
}
std::ostream & operator<<(std::ostream & os, const Vector & vec_os) {
  os << "[";
  int idx = 0;
  for (const double element : vec_os.vec) {
    os << std::setw(15) << std::setprecision(8) << std::setfill(' ')
       << std::scientific << std::left << std::showpos
       << element;
    if (idx != vec_os.vec.size() - 1) {
      os << " ";
    }
    ++idx;
  }
  os << std::resetiosflags(std::ios_base::scientific) 
     << std::resetiosflags(std::ios_base::right)
     << std::resetiosflags(std::ios_base::showpos);
  os << "]";
  return os;
}
std::istream & operator>>(std::istream & is, Vector & vec_is) {
	if (vec_is.vec.size() == 0) {
		double element;
		while (is >> element) {
			vec_is.vec.push_back(element);
		}
	} else {
		for (double & element : vec_is.vec) {
			is >> element;
		}
	}
	return is;
}
