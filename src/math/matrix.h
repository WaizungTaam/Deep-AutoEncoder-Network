/*
Copyright 2016 WaizungTaam.  All rights reserved.

License: Apache License 2.0
Email:   waizungtaam@gmail.com

*/

#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include "vector.h"

class Matrix {
public:
  Matrix() = default;
  Matrix(const Matrix &) = default;
  Matrix(Matrix &&) = default;
  Matrix & operator=(const Matrix &) = default;
  Matrix & operator=(Matrix &&) = default;
  ~Matrix() = default;
  Matrix(int, int);
  Matrix(const std::vector<std::size_t> &);
  Matrix(int, int, double);
  Matrix(const std::vector<std::size_t> &, double); 
  Matrix(int, int, const std::string &, double, double);
  Matrix(const std::vector<std::size_t> &, const std::string &, double, double);
  Matrix(const Vector &);
  Matrix(const std::vector<std::vector<double>> &);
  Matrix(const std::initializer_list< 
         std::initializer_list<double>> &);
  Matrix & operator=(const Vector &);
  Matrix & operator=(double);
  Matrix & operator=(const std::initializer_list< 
                     std::initializer_list<double>> &);
  std::vector<std::size_t> shape() const;
  void clear();
  bool empty() const;
  Matrix insert(const Vector &, int, int) const;
  Matrix insert(const Matrix &, int, int) const;
  Matrix remove(int, int) const;
  Matrix remove(int, int, int) const;
  Matrix replace(const Vector &, int, int) const;
  Matrix replace(const Matrix &, int, int) const;
  Matrix shuffle() const;
  Matrix T() const;
  Matrix reshape(int, int) const;
  Matrix inverse() const;  // TODO
  double sum() const;
  Vector sum(int) const;
  double max() const;
  Vector max(int) const;
  double min() const;
  Vector min(int) const;
  Matrix cross(const Matrix &) const;
  bool approx(const Matrix &, double);
  Matrix approx(double, double);
  friend Matrix operator+(const Matrix &, const Matrix &);
  friend Matrix operator+(const Matrix &, double);
  friend Matrix operator+(double, const Matrix &);
  friend Matrix operator-(const Matrix &, const Matrix &);
  friend Matrix operator-(const Matrix &, double);
  friend Matrix operator-(double, const Matrix &);
  friend Matrix operator*(const Matrix &, const Matrix &);
  friend Matrix operator*(const Matrix &, double);  
  friend Matrix operator*(double, const Matrix &);
  friend Matrix operator*(const Matrix &, const Vector &);
  friend Matrix operator/(const Matrix &, const Matrix &);
  friend Matrix operator/(const Matrix &, double);
  friend Matrix operator/(double, const Matrix &);
  void operator+=(double);
  void operator+=(const Matrix &);
  void operator-=(double);
  void operator-=(const Matrix &);
  void operator*=(double);
  void operator*=(const Matrix &);
  void operator/=(double);  
  void operator/=(const Matrix &);
  friend bool operator==(const Matrix &, const Matrix &);
  friend Matrix operator==(const Matrix &, double);
  friend Matrix operator==(double, const Matrix &);
  friend bool operator!=(const Matrix &, const Matrix &);
  friend Matrix operator!=(const Matrix &, double);
  friend Matrix operator!=(double, const Matrix &);
  friend Matrix operator<(const Matrix &, const Matrix &);  
  friend Matrix operator<(const Matrix &, double);
  friend Matrix operator<(double, const Matrix &);
  friend Matrix operator<=(const Matrix &, const Matrix &);
  friend Matrix operator<=(const Matrix &, double);
  friend Matrix operator<=(double, const Matrix &);
  friend Matrix operator>(const Matrix &, const Matrix &);
  friend Matrix operator>(const Matrix &, double);
  friend Matrix operator>(double, const Matrix &);  
  friend Matrix operator>=(const Matrix &, const Matrix &);
  friend Matrix operator>=(const Matrix &, double);
  friend Matrix operator>=(double, const Matrix &);  
  Vector & operator[](int);
  const Vector & operator[](int) const;
  Matrix operator()(int) const;
  Matrix operator()(int, int) const;
  Matrix operator()(int, int, int, int) const;
  friend std::ostream & operator<<(std::ostream &, const Matrix &);
  friend std::istream & operator>>(std::istream &, Matrix &);
private:
  std::vector<Vector> mat;
};

#endif  // matrix.h