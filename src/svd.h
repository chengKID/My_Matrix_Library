#pragma once

#include "matrix.h"

#ifndef FOAM_SVD

// Hier we only implement th cast for double and double data type
// Return the sigular values
Vector<double> SVD(Matrix<double> &m);

int dsvd(Matrix<double> &a, int m, int n, std::vector<double>& w, Matrix<double> &v);

#endif