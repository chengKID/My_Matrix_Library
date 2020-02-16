#include "svd.h"
// using templates for Matrix & Vector so also include .cpp to help linker
#include "matrix.cpp"

// ***********************************************************************************************
// * Linear algebra operations:
// * Hier we implement the basic operations like: Linear solving: A * x = b 
// *
// * Note: we only implement the case that these matrices also vectors have the same data type,
// *       e.g. for matrix times vector, both matrix and vector are double
// *
// Linear solving: A * x = b
template<typename T>
Vector<T> LinearSolving(Matrix<T> A, Vector<T> b) {
    int size = b.getSize();
    Vector<T> x(size, 1);

    x = A.Inverse() * b;
    return x;
};


int main(int argc, char* argv[]) {
    // Initialise some matrices and vectors:
    // identity matrix 3x3
    Matrix<double> A(3, 3);
    A << 1.0, 0.0, 0.0,
         0.0, 1.0, 0.0,
         0.0, 0.0, 1.0;

    Matrix<double> B(3, 3);
    B << 1.1, 2.1, 3.1,
         4.1, 5.1, 6.1,
         7.1, 8.1, 9.1;

    Matrix<double> D(3, 3);
    D << 1.0, 2.0, 3.0,
         4.0, 5.0, 6.0,
         7.0, 8.0, 10.0;

    // column vector 3x1
    Vector<double> col_b(3, 1);
    col_b << 1.0, 2.0, 3.0;

    // row vector 1x3
    Vector<double> row_b(1, 3);
    row_b << 1.0, 2.0, 3.0;

    // column vector 3x1
    Vector<double> b(3, 1);
    b << 3.0, 3.0, 4.0;

    // -------------------------------------------------------
    // ------   Test the Matrix and Vector operations   ------
    // -------------------------------------------------------
    // 1. A matrix times a vector & a vector times a matrix
    Vector<double> m_time_v(3, 1);
    m_time_v = A * col_b;
    std::cout << "[TEST] The result of a matrix times a column vector is:" << std::endl << m_time_v << std::endl;

    Vector<double> v_time_m(1, 3);
    v_time_m = row_b * A;
    std::cout << "[TEST] The result of a row vector times a matrix is:" << std::endl << v_time_m << std::endl;

    // 2. A matrix addition & a scalar multiplication
    Matrix<double> C(3, 3);
    C = A + B;
    std::cout << "[TEST] The result of matrix addition is:" << std::endl << C << std::endl;
    C = 2.0 * B;
    std::cout << "[TEST] The result of matrix scalar multiplication is:" << std::endl << C << std::endl;
    
    // 3. Matrix tranpose
    Matrix<double> B_tran(3, 3);
    B_tran = B.Transpose();
    std::cout << "[TEST] The result of matrix transpose is:" << std::endl << B_tran << std::endl;

    // 4. Matrix inverse
    Matrix<double> D_inv(3, 3);
    D_inv = D.Inverse();
    std::cout << "[TEST] The result of matrix inverse is:" << std::endl << D_inv << std::endl;

    // 5. Linear solving
    Vector<double> x(3, 1);
    x = LinearSolving(D, b);
    std::cout << "[TEST] The result of linear solving is:" << std::endl << x << std::endl;

    // 6. SVD
    Vector<double> sig_B = SVD(D);
    std::cout << "[TEST] The result of sigular value decomposition is:" << std::endl << sig_B << std::endl;

    return 0;
}