/*
Copyright 2016 WaizungTaam.  All rights reserved.

License: Apache License 2.0
Email:   waizungtaam@gmail.com

*/

#ifndef VECTOR_H
#define VECTOR_H

#include <vector>
#include <iostream>
#include <fstream>
#include <string>

class Vector {
public:
  Vector() = default;
  Vector(const Vector &) = default;
  Vector(Vector &&) = default;
  Vector & operator=(const Vector &) = default;
  Vector & operator=(Vector &&) = default;
  ~Vector() = default;
  Vector(int);
  Vector(int, double);
  Vector(int, const std::string &, double, double);
  Vector(const std::vector<double> &);
  Vector(const std::initializer_list<double> &);
  Vector & operator=(double);
  Vector & operator=(const std::vector<double> &);
  Vector & operator=(const std::initializer_list<double> &);
  std::vector<std::size_t> shape() const;
  Vector insert(double, int) const;
  Vector insert(const Vector &, int) const;
  Vector remove(int) const;
  Vector remove(int, int) const;
  Vector replace(const Vector &, int) const;
  Vector shuffle() const;
  void clear();
  bool empty();
  double sum() const;
  double max() const;
  double min() const;
  Vector approx(const Vector &, double) const;
  Vector approx(double, double) const;
  friend Vector operator+(const Vector &, const Vector &);
  friend Vector operator+(const Vector &, double);
  friend Vector operator+(double, const Vector &);
  friend Vector operator-(const Vector &, const Vector &);
  friend Vector operator-(const Vector &, double);
  friend Vector operator-(double, const Vector &);
  friend Vector operator*(const Vector &, const Vector &);
  friend Vector operator*(const Vector &, double);
  friend Vector operator*(double, const Vector &);
  friend Vector operator/(const Vector &, const Vector &);
  friend Vector operator/(const Vector &, double);
  friend Vector operator/(double, const Vector &);
  void operator+=(double);
  void operator+=(const Vector &);
  void operator-=(double);
  void operator-=(const Vector &);
  void operator*=(double);
  void operator*=(const Vector &);
  void operator/=(double);
  void operator/=(const Vector &);
  friend bool operator==(const Vector &, const Vector &);
  friend Vector operator==(const Vector &, double);
  friend Vector operator==(double, const Vector &);
  friend bool operator!=(const Vector &, const Vector &);
  friend Vector operator!=(const Vector &, double);
  friend Vector operator!=(double, const Vector &);
  friend Vector operator<(const Vector &, const Vector &);
  friend Vector operator<(const Vector &, double);
  friend Vector operator<(double, const Vector &);
  friend Vector operator<=(const Vector &, const Vector &);
  friend Vector operator<=(const Vector &, double);
  friend Vector operator<=(double, const Vector &);
  friend Vector operator>(const Vector &, const Vector &);
  friend Vector operator>(const Vector &, double);
  friend Vector operator>(double, const Vector &);
  friend Vector operator>=(const Vector &, const Vector &);
  friend Vector operator>=(const Vector &, double);
  friend Vector operator>=(double, const Vector &);  
  double & operator[](int);
  const double & operator[](int) const;
  Vector operator()(int, int) const;
  friend std::ostream & operator<<(std::ostream & out, const Vector &);
  friend std::istream & operator>>(std::istream & in, Vector &);
private:
  std::vector<double> vec;
};

#endif // vector.h
