# Implement Matrix & Vector Class and Corresponding Operations

## Implement Matrix & Vector Class as EIGEN Library Manner

**Matrix Class**:
It is a template class. We can use it to store data with such datatype: int, float, double. We can initial the matrix with default number 0, or use operator "<<" to set the matrix like EIGEN Library. We also can use the parentheses to access the corresponding element. If you want to print the matrix, just use "std::cout << the matrix" like EIGEN Library.

**Vector Class**:
It is a template class. We can use it to store data with such datatype: int, float, double. We can initial the vector with default number 0, or use operator "<<" to set the vector like EIGEN Library. We also can use the parentheses to access the corresponding element. If you want to print the vector, just use "std::cout << the vector" like EIGEN Library.

## Implement the Linear Matrix & Vector Operations
For example, there are matrices: A, B and vectors: b, x and scalar a.
Hier we implement the linear operations like: matrix addition, matrix subtraction, matrix multiplication, matrix transpose.
* Matrix Addition: A + B;    A + a;     a + A
* Matrix Subtraction: A - B;    A - a;      a - A
* Matrix Multiplication: A * B;     A * b;      b * A;      a * A
* Matrix Transpose: A.Transpose()
* Linear Solving the equation A * x = b: x = LinaerSolving(A, b)
Note: for all of these operators the corresponding oprators must have the right diemntions.

## Implement the Non-linear Matrix Operations
For example, there are matrices: A, B and vectors: b, x and scalar a.
Hier we implement the non-linear operations like: determinant, inverse, svd.
* Matrix Determinant: det = Determinant(A, size)
* Matrix Inverse: A.Inverse()
* SVD: sigular values = SVD(A)
Note: for matrix inverse, the matrix must be square and non-sigular.

## Installation
We use Linux Ubuntu 16.04 and C++

The link here is very helpful for understanding svd function:
paper: http://www.cs.utexas.edu/users/inderjit/public_papers/HLA_SVD.pdf
code: https://padas.oden.utexas.edu/software/

## Build the Project
```bash
$> sudo apt install libpcl-dev
$> cd ~
$> git clone https://github.com/chengKID/Lidar_Obstacle_Detection.git
$> cd Lidar_Obstacle_Detection
$> mkdir build && cd build
$> cmake ..
$> make
$> ./environment
```

## Process
Test the operations. We list some examples in "matrix_operations.cpp". Hier we add matrices, subtract matrices, multiply matrices, do the matrix inverse and singular value decomposition.

Feel free to test other oprations, like add two integer matrices.