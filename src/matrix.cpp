#include <exception>
#include "matrix.h"

// ***********************************************************************************************
// * Various constructors for Matrix:
// *
// Default constructor: use dimentions & initial values for each element
template<typename T>
Matrix<T>::Matrix(int m, int n, T init_elem) {
    // set the dimention
    row_size = m;
    col_size = n;

    // store the data
    my_matrix.resize(row_size);
    for (int i=0; i<row_size; i++)
        my_matrix[i].resize(col_size, init_elem);
}

// Copy constructor: copy one instance to another & deep copy
template<typename T>
Matrix<T>::Matrix(const Matrix<T>& other_m) {
    // set the dimention
    row_size = other_m.getRows();
    col_size = other_m.getCols();

    // deep copy the data
    my_matrix.resize(row_size);
    for (int i=0; i<row_size; i++) {
        my_matrix[i].resize(col_size);
        for (int j=0; j<col_size; j++)
            my_matrix[i][j] = other_m.my_matrix[i][j];
    }
}


// ***********************************************************************************************
// * Basic operations for Matrix:
// * Hier we implement the basic operations like: access, read-in, print
// *
// Refer to a specific cell in the matrix: row = m; column = n
template<typename T>
T& Matrix<T>::operator() (const int& m, const int& n) {
    return this->my_matrix[m][n];
}

// Read data into matrix class; the same way as Eigen library
template<typename T, typename I>
Matrix<T>& operator<< (Matrix<T>& A, I input) {
    int rows = A.getRows();
    int cols = A.getCols();
    int size = rows * cols;
    static int mat_data_cnt = 0;

    // insert the data
    A.Insert((T) input);
    mat_data_cnt++;
    
    // when all data haven benn readed
    if (mat_data_cnt == size) {
        std::vector<T>& inputs = A.GetTmpData();

        for (int i=0; i<rows; i++) {
            for (int j=0; j<cols; j++)
                // set the values in vector into matrix
                A(i, j) = inputs[i*3+j];
        }

        // reset the counter
        mat_data_cnt = 0;     
    }

    return A;
}

// Print data of the matrix class; the same way as Eigen library
template<typename T>
std::ostream& operator<< (std::ostream& out, Matrix<T>& A) {
    int rows = A.getRows();
    int cols = A.getCols();

    for (int i=0; i<rows; i++) {
        for (int j=0; j<cols; j++) {
            out << A(i, j) << " ";
        }
        out << std::endl;
    }

    return out;
}


// ***********************************************************************************************
// * Linear Matrix operations:
// * Hier we implement the linear operations like: add, subtract, multiply, transpose
// *
// * Note: we assume that these matrices also vectors have the same data type,
// *       e.g. for addition, both matrices are double
// *
// Matrix addition: addition of two matrices
template<typename T>
Matrix<T> Matrix<T>::operator+ (Matrix<T>& A_) {
    // check the dimentions of these matrices
    if (A_.getRows() != row_size || A_.getCols() != col_size) {
        std::cout << "Warning: illegal addition of two matrices, which have different sizes!" << std::endl;
        throw std::exception();
    }

    // set the summation into result matrix
    Matrix<T> sum(row_size, col_size);

    for (int i=0; i<row_size; i++) {
        for (int j=0; j<col_size; j++)
            sum(i, j) = A_(i, j) + my_matrix[i][j];
    }

    return sum;
}

// Matrix addition: add matrix with a number
template<typename T>
Matrix<T> operator+ (T scalar, Matrix<T>& B) {
    // initial the result matrix
    int row_size = B.getRows();
    int col_size = B.getCols();
    Matrix<T> sum(row_size, col_size);

    // set the summation into result matrix
    for (int i=0; i<row_size; i++) {
        for (int j=0; j<col_size; j++)
            sum(i, j) = B(i, j) + scalar;
    }

    return sum;
}

// Matrix subtraction: subtraction of two matrices
template<typename T>
Matrix<T> operator- (Matrix<T>& A, Matrix<T>& B) {
    // check the dimentions of these matrices
    if (A.getRows() != B.getRows() || A.getCols() != B.getCols()) {
        std::cout << "Warning: illegal subtraction of two matrices, which have different sizes!" << std::endl;
        throw std::exception();
    }

    // set the subtraction into result matrix
    int row_size = A.getRows();
    int col_size = A.getCols();
    Matrix<T> diff(row_size, col_size);

    for (int i=0; i<row_size; i++) {
        for (int j=0; j<col_size; j++)
            diff.my_matrix[i][j] = A.my_matrix[i][j] - B.my_matrix[i][j];
    }

    return diff;
}

// Matrix subtraction: subtract matrix from a number
template<typename T>
Matrix<T> operator- (Matrix<T>& B, T scalar) {
    // initial the result matrix
    int row_size = B.getRows();
    int col_size = B.getCols();
    Matrix<T> diff(row_size, col_size);

    // set the summation into result matrix
    for (int i=0; i<row_size; i++) {
        for (int j=0; j<col_size; j++)
            diff.my_matrix[i][j] = B.my_matrix[i][j] - scalar;
    }

    return diff;
}

// Matrix multiplication: multipl of two matrices
template<typename T>
Matrix<T> operator* (Matrix<T>& A, Matrix<T>& B) {
    // check the dimentions of these matrices
    if (A.getCols() != B.getRows()) {
        std::cout << "Warning: illegal multiplication of two matrices and the column dimention of matrix A dosen't match the row dimention of matirx B!" << std::endl;
        throw std::exception();
    }

    // set the product into result matrix
    int row_size = A.getRows();
    int col_size = B.getCols();
    Matrix<T> prod(row_size, col_size);

    // mth row of matrix A times lth column of matrix B
    int col_A = A.getCols();
    for (int m=0; m<row_size; m++) {
        for (int l=0; l<col_size; l++) {
            T prod_elem = 0;

            // the corresponding factor from matrix B has the exactly reverse indicators as these of matrix A
            for (int j=0; j<col_A; j++) {
                T tmp = A.my_matrix[m][j] * B.my_matrix[j][l];
                prod_elem += tmp;
            }

            prod.my_matrix[m][l] = prod_elem;
        }
    }

    return prod;
}

// Matrix multiplication: matrix multiple a number
template<typename T>
Matrix<T> operator* (T scalar, Matrix<T>& B) {
    // initial the result matrix
    int row_size = B.getRows();
    int col_size = B.getCols();
    Matrix<T> prod(row_size, col_size);

    // set the product into result matrix
    for (int i=0; i<row_size; i++) {
        for (int j=0; j<col_size; j++)
            prod(i, j) = B(i, j) * scalar;
    }

    return prod;
}

// Transpose of the matrix
template<typename T>
Matrix<T> Matrix<T>::Transpose() {
    // initial the matrix
    int row_size = this->getRows();
    int col_size = this->getCols();
    Matrix<T> trans(col_size, row_size);

    // the element of tranposed matrix has reversed indicators as the original matrix
    for (int i=0; i<row_size; i++) {
        for (int j=0; j<col_size; j++)
            trans.my_matrix[j][i] = this->my_matrix[i][j];
    }

    return trans;
}

// A matrix times a vector
template<typename T>
Vector<T> operator* (Matrix<T> M, Vector<T> V) {
    // check: the vector must be a column vector
    if (V.isRowvector()) {
        std::cout << "[MATRIX * VECTOR]Warning: the vector is't a column vector" << std::endl;
        throw std::exception();
    }
    // check the dimention of the matrix and the vector
    if (M.getCols() != V.getSize()) {
        std::cout << "[MATRIX * VECTOR]Warning: Wrong dimention!" << std::endl;
        throw std::exception();
    }

    std::vector<T> V_vec = V.GetData();
    int rows = M.getCols();
    int m = M.getRows();
    int n = 1;
    Vector<T> prod_v(m, n);

    for (int i=0; i<m; i++) {
        T tmp = 0;
        for (int j=0; j<rows; j++)
            tmp = tmp + M.my_matrix[i][j] * V_vec[j];
        
        prod_v(i) = tmp;
    }

    return prod_v;
}


// ***********************************************************************************************
// * Non linear algebra operations:
// * Hier we implement the basic operations like: matrix times vector, inverse,
// *
// * Note: we only implement the case that these matrices also vectors have the same data type,
// *       e.g. for matrix times vector, both matrix and vector are double
// *
// Matrix determinat
template<typename T>
T Determinat(Matrix<T> A, int size) {
    // The matrix is a square matrix
    int rows = A.getRows();
    int cols = A.getCols();
    if (rows != cols) {
        std::cout << "[MATRIX DETERMINAT]Warning: the matrix must be a square matrix!" << std::endl;
        throw std::exception();
    }

    T det = 0;
    Matrix<T> sub_mat(size-1, size-1);
    if (size == 2) {
        det = A.my_matrix[0][0] * A.my_matrix[1][1] - A.my_matrix[0][1] * A.my_matrix[1][0];
        return det;
    }
    else {
        for (int index=0; index<size; index++) {
            int sub_i = 0;

            for (int i=1; i<size; i++) {
                int sub_j = 0;

                for (int j=0; j<size; j++) {
                    if (j == index) 
                        continue;
                    
                    sub_mat.my_matrix[sub_i][sub_j] = A.my_matrix[i][j];
                    sub_j++;
                }

                sub_i++;
            }

            det = det + std::pow(-1, index) * A.my_matrix[0][index] * Determinat(sub_mat, size-1);
        }
    }
    return det;
}

// Matrix inverse: use the "Gauss-Jordan-Method" instead of "Minors, Cofactors and Ad-jugate Method", which are inefficient
template<typename T>
Matrix<T> Matrix<T>::Inverse() {
    // The inverse of a matrix is only possible when:
    // 1. The matrix is a square matrix
    int rows = this->getRows();
    int cols = this->getCols();
    if (rows != cols) {
        std::cout << "[MATRIX INVERSE]Warning: the matrix must be a square matrix!" << std::endl;
        throw std::exception();
    }
    // 2. The matrix is a non-sigular matrix, i.e. determinant is non-zero
    int det = Determinat(*this, rows);
    if (det == 0) {
        std::cout << "[MATRIX INVERSE]Warning: the matrix must be a non-sigular matrix!" << std::endl;
        throw std::exception();
    }

    // Hier we use the "Gauss-Jordan-Method" instead of "Minors, Cofactors and Ad-jugate Method", which are inefficient
    T temp;
    Matrix<T> aug_mat(rows, 2*cols);
    // Create the augmented matrix: append the identity matrix at the end of the original matrix
    for (int i=0; i<rows; i++) {
        for (int j=0; j<2*rows; j++) {
            // Add '1' at the diagonal places of the matrix to create the identity matrix
            if (j == (i + cols))
                aug_mat(i, j) = (T) 1.0;
            
            if (j < cols)
                aug_mat(i, j) = this->my_matrix[i][j];
        }
    }

    // Interchange the row of matrix: interchanging of row will start from the last row
    for (int i=cols-1; i>0; i--) {
        // Swapping each element of the two rows
        if (aug_mat(i-1, 0) < aug_mat(i, 0)) {
            for (int j=0; j< 2*cols; j++) {
                // swapping of the row, if above condition satisfied
                temp = aug_mat(i, j);
                aug_mat(i, j) = aug_mat(i-1, j);
                aug_mat(i-1, j) = temp;
            }
        }
    }

    // Replace a row by sum of itself and a constant multiple of another row of the matrix
    for (int i=0; i<cols; i++) {
        for (int j=0; j<cols; j++) {
            if (j != i) {
                temp = aug_mat(j, i) / aug_mat(i, i);
                for (int k=0; k<2*cols; k++) 
                    aug_mat(j, k) -= aug_mat(i, k) * temp;
            }
        }
    }

    // Multiply each row by a nonzero integer;
    // Then divide row element by the diagonal element
    for (int i=0; i<cols; i++) {
        temp = aug_mat(i, i);
        for (int j=0; j<2*cols; j++)
            aug_mat(i, j) = aug_mat(i, j) / temp;
    }

    // Set the output
    Matrix<T> inv_A(rows, cols);
    for (int i=0; i<rows; i++) {
        for (int j=0; j<cols; j++)
            inv_A(i, j) = aug_mat(i, j+cols);
    }
    return inv_A;
}


// ***********************************************************************************************
// * Variaus Construct functions for vector:
// *
// Default constructor: use dimentions & initial values for each element
template<typename T>
Vector<T>::Vector(int m, int n, T init_elem) {
    // check: there must be only one dimention to be 1
    if (m != 1 && n != 1) {
        std::cout << "[INITIAL VECTOR]Warning: vector has wrong size!" << std::endl;
        throw std::exception();
    }

    if (m == 1) {
        vec_size = n;
        row_vector = true;
    } else {
        vec_size = m;
        row_vector = false;
    }
    my_vector.resize(vec_size, init_elem);
}

// Copy constructor: copy one instance to another & deep copy
template<typename T>
Vector<T>::Vector(const Vector<T>& other_v) {
    vec_size = other_v.vec_size;
    my_vector.resize(vec_size);
    row_vector = other_v.row_vector;

    for (int i=0; i<vec_size; i++)
        my_vector[i] = other_v.my_vector[i];
}


// ***********************************************************************************************
// * Besic methods for Vector:
// *
// Refer to a specific element in the vector: position m
template<typename T>
T& Vector<T>::operator() (const int& m) {
    return my_vector[m];
}

// Read data into Vector class; the same way as Eigen library
template<typename T, typename I>
Vector<T>& operator<< (Vector<T>& A, I input) {
    int size = A.getSize();
    static int vec_data_cnt = 0;

    // insert the data
    A.Insert((T) input);
    vec_data_cnt++;

    // when all data haven been readed
    if (vec_data_cnt == size) {
        std::vector<T> inputs = A.GetTmpData();

        // set the inputs into class Vector
        for (int i=0; i<size; i++)
            A(i) = inputs[i];
        
        // reset the counter
        vec_data_cnt = 0;
    }
    return A;
}

// Print data of the Vector class; the same way as Eigen library
template<typename T>
std::ostream& operator<< (std::ostream& out, Vector<T>& A) {
    int size = A.getSize();
    const bool row_vector = A.isRowvector();

    if (row_vector) {
        for (int i=0; i<size; i++)
            out << A.my_vector[i] << " ";

        out << std::endl;
    }
    else {
        for (int i=0; i<size; i++)
            out << A.my_vector[i] << std::endl;
    }

    return out;
}


// ***********************************************************************************************
// * Linear algebra operations:
// * Hier we implement the basic operations like: vector times matrix
// *
// * Note: we only implement the case that these matrices also vectors have the same data type,
// *       e.g. for vector times matrix, both matrix and vector are double
// *
// A vector times a matrix
template<typename T>
Vector<T> operator* (Vector<T>V, Matrix<T> M) {
    // check: the vector must be a row vector
    if (!V.isRowvector()) {
        std::cout << "[VECTOR * MATRIX]Warning: the vector is't a row vector!" << std::endl;
        throw std::exception();
    }
    // check the dimention of the matrix and the vector
    if (M.getCols() != V.getSize()) {
        std::cout << "[VECTOR * MATRIX]Warning: Wrong dimention!" << std::endl;
        throw std::exception();
    }

    std::vector<std::vector<T> > M_mat = M.GetData();
    std::vector<T> V_vec = V.GetData();
    int rows = M.getRows();
    int m = 1;
    int n = M.getCols();
    Vector<T> prod_v(m, n);

    for (int j=0; j<n; j++) {
        T tmp = 0;
        for (int i=0; i<rows; i++)
            tmp = tmp + M_mat[i][j] * V_vec[i];
        
        prod_v(j) = tmp;
    }

    return prod_v;
}

// ***********************************************************************************************
// *  Explicitly instantiate some of the template classes:
// *  e.g. for class Vector and Marix with the type: int, float, double
// *
/*template Matrix<double> operator<< <double>(Matrix<double>, double);
template Vector<double> operator<< <double>(Vector<double>, double);
template class Vector<int>;
template class Vector<float>;
template class Vector<double>;

template class Matrix<int>;
template class Matrix<float>;
template class Matrix<double>;*/