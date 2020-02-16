#pragma once

#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <sstream>
#include <string>
#include <vector>
#include <tuple>
#include <cmath>
#include <math.h>

template<typename T>
class Matrix;
template<typename T>
class Vector;

template<typename T>
class Matrix {
    private:
    int row_size;                                           // For a "mxn" matrix, row_size = m; col_size = n
    int col_size;                                           // Useunsigned data type to save some memory, because the dimention of matrix can't be negative
    std::vector<std::vector<T> > my_matrix;                 // Use the std::vecotr as container
    std::vector<T> tmp_mat;                                 // For temporate storage

    public:
    // ***********************************************************************************************
    // * Various constructors for Matrix:
    // *    
    // Default constructor: use dimentions & initial values for each element
    Matrix(int m, int n, T init_elem = 0.0);

    // Copy constructor: copy one instance to another & deep copy
    Matrix(const Matrix<T>& other_m);

    ~Matrix() {};

    // ***********************************************************************************************
    // * Basic operations for Matrix:
    // * Hier we implement the basic operations like: access, read-in, print
    // *
    // Refer to a specific cell in the matrix: row = m; column = n
    T& operator() (const int& m, const int& n);

    // Read data into matrix class; the same way as Eigen library
    template<typename insT, typename insI>
    friend Matrix<insT>& operator<< (Matrix<insT>& A, insI input);
    template<typename intT>
    friend Matrix<T>& operator, (Matrix<T>& A, intT n) { return A << n; };

    // Print data of the matrix class; the same way as Eigen library
    template<typename insT>
    friend std::ostream& operator<< (std::ostream& out, Matrix<insT>& A);

    // Return row dimention
    int getRows() const { return row_size; };         

    // Return column dimention               
    int getCols() const { return col_size; };     

    // Return the stored data
    std::vector<std::vector<T> >& GetData() { return my_matrix; };     

    // Insert the data into temporate storage
    void Insert(T input) { this->tmp_mat.push_back(input); };

    // Return the temporate storage
    std::vector<T>& GetTmpData() { return this->tmp_mat; };

    // ***********************************************************************************************
    // * Linear Matrix operations:
    // * Hier we implement the linear operations like: add, subtract, multiply, transpose
    // *
    // * Note: we assume that these matrices also vectors have the same data type,
    // *       e.g. for addition, both matrices are double
    // *
    // Addition: matrix A + matrix B
    Matrix<T> operator+ (Matrix<T>& A_);     

    // Addition: scalar + matrix B
    template<typename insT>
    friend Matrix<insT> operator+ (insT scalar, Matrix<insT>& B);               

    // Subtraction: matrix A - matrix B
    template<typename insT>
    friend Matrix<insT> operator- (Matrix<insT>& A, Matrix<insT>& B);      

    // Subtraction: matrix B - scalar
    template<typename insT>
    friend Matrix<insT> operator- (Matrix<insT>& B, insT scalar);               

    // Multiplication: matrix A(mxn) * matrix B(nxl)
    template<typename insT>
    friend Matrix<insT> operator* (Matrix<insT>& A, Matrix<insT>& B);                 

    // Multiplication: scalar * matrix B
    template<typename insT>
    friend Matrix<insT> operator* (insT scalar, Matrix<insT>& B);               

    // Transpose of the matrix
    Matrix<T> Transpose();   

    // A matrix times a vector
    template<typename insT>
    friend Vector<insT> operator* (Matrix<insT> M, Vector<insT> V);   

    // ***********************************************************************************************
    // * Non linear algebra operations:
    // * Hier we implement the basic operations like: inverse, 
    // *
    // * Note: we only implement the case that these matrices also vectors have the same data type,
    // *       e.g. for matrix times vector, both matrix and vector are double
    // *
    // Matrix determinat
    template<typename insT>
    friend insT Determinat(Matrix<insT> A, int size);

    // Matrix inverse
    Matrix<T> Inverse();
};

template<typename T>
class Vector {
    private:
    int vec_size;                                            // The dimention of a vector
    bool row_vector;                                         // Determine whether it's a row vector or column vector
    std::vector<T> my_vector;                                // Use the std::vector as container
    std::vector<T> tmp_vec;                                  // For temporate storage

    public:
    // ***********************************************************************************************
    // * Various constructors for vector
    // * Hier we implement both the row vector & column vector
    // *
    // Default constructor: use dimentions & initial values for each element
    Vector(int m, int n, T init_elem = 0.0);

    // Copy constructor: copy one instance to another & deep copy
    Vector(const Vector<T>& other_v);

    ~Vector() {};

    // ***********************************************************************************************
    // * Basic methods for Vector:
    // *
    // Refer to a specific element in the vector: position m
    T& operator() (const int& m);

    // Read data into Vector class; the same way as Eigen library
    template<typename insT, typename insI>
    friend Vector<insT>& operator<< (Vector<insT>& A, insI input);
    template<typename intT>
    friend Vector<T>& operator, (Vector<T>& A, intT n) { return A << n; };

    // Print data of the Vector class; the same way as Eigen library
    template<typename insT>
    friend std::ostream& operator<< (std::ostream& out, Vector<insT>& A);

    // Determine ob it's a row vector: if it's a row vector, return true
    const int isRowvector() const { return row_vector; };

    // Return vector's size
    int getSize() const { return vec_size; };  

    // Return the stored data
    std::vector<T>& GetData() { return my_vector; };

    // Insert the data into temporate storage
    void Insert(T input) { this->tmp_vec.push_back(input); };

    // Return the temporate storage
    std::vector<T>& GetTmpData() { return this->tmp_vec; };

    // ***********************************************************************************************
    // * Linear algebra operations:
    // * Hier we implement the basic operations like: vector times matrix 
    // *
    // * Note: we only implement the case that these matrices also vectors have the same data type,
    // *       e.g. for vector times matrix, both matrix and vector are double
    // *
    // A vector times a matrix
    template<typename insT>
    friend Vector<insT> operator* (Vector<insT> V, Matrix<insT> M);
};