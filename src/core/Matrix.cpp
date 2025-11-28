/**
 * ============================================================================
 * Matrix Class Implementation
 * ============================================================================
 */

#include "Matrix.hpp"
#include <algorithm>

// ============================================================================
// Constructors
// ============================================================================

Matrix::Matrix() {
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            data[i][j] = 0.0;
            grad[i][j] = 0.0;
        }
    }
}

Matrix::Matrix(double a, double b, double c, double d) {
    data[0][0] = a; data[0][1] = b;
    data[1][0] = c; data[1][1] = d;
    zeroGrad();
}

Matrix::Matrix(const Matrix& other) {
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            data[i][j] = other.data[i][j];
            grad[i][j] = other.grad[i][j];
        }
    }
}

// ============================================================================
// Basic Operations
// ============================================================================

Matrix Matrix::multiply(const Matrix& other) const {
    Matrix result;
    
    // Standard matrix multiplication: result[i][j] = sum_k (this[i][k] * other[k][j])
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            result.data[i][j] = 0.0;
            for (int k = 0; k < 2; k++) {
                result.data[i][j] += data[i][k] * other.data[k][j];
            }
        }
    }
    
    return result;
}

Matrix Matrix::transpose() const {
    Matrix result;
    result.data[0][0] = data[0][0];
    result.data[0][1] = data[1][0];  // swap
    result.data[1][0] = data[0][1];  // swap
    result.data[1][1] = data[1][1];
    return result;
}

Matrix Matrix::scale(double s) const {
    Matrix result;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            result.data[i][j] = data[i][j] * s;
        }
    }
    return result;
}

Matrix Matrix::add(const Matrix& other) const {
    Matrix result;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            result.data[i][j] = data[i][j] + other.data[i][j];
        }
    }
    return result;
}

Matrix Matrix::subtract(const Matrix& other) const {
    Matrix result;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            result.data[i][j] = data[i][j] - other.data[i][j];
        }
    }
    return result;
}

Matrix Matrix::hadamard(const Matrix& other) const {
    Matrix result;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            result.data[i][j] = data[i][j] * other.data[i][j];
        }
    }
    return result;
}

// ============================================================================
// Utility Functions
// ============================================================================

void Matrix::zeroGrad() {
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            grad[i][j] = 0.0;
        }
    }
}

double Matrix::frobeniusNormSquared() const {
    double sum = 0.0;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            sum += data[i][j] * data[i][j];
        }
    }
    return sum;
}

double Matrix::sum() const {
    double s = 0.0;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            s += data[i][j];
        }
    }
    return s;
}

double Matrix::get(int i, int j) const {
    return data[i][j];
}

void Matrix::set(int i, int j, double value) {
    data[i][j] = value;
}

// ============================================================================
// Hand-Calculation Helpers
// ============================================================================

void Matrix::print(const std::string& name) const {
    std::cout << name << ":\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  [" << data[0][0] << ", " << data[0][1] << "]\n";
    std::cout << "  [" << data[1][0] << ", " << data[1][1] << "]\n";
}

void Matrix::printWithSteps(const std::string& name, const std::string& operation) const {
    std::cout << "\n" << name << " (" << operation << "):\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  [" << std::setw(10) << data[0][0] << ", " << std::setw(10) << data[0][1] << "]\n";
    std::cout << "  [" << std::setw(10) << data[1][0] << ", " << std::setw(10) << data[1][1] << "]\n";
}

void Matrix::printGrad(const std::string& name) const {
    std::cout << "d" << name << "/dL (gradient):\n";
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  [" << grad[0][0] << ", " << grad[0][1] << "]\n";
    std::cout << "  [" << grad[1][0] << ", " << grad[1][1] << "]\n";
}

// ============================================================================
// Operators
// ============================================================================

Matrix Matrix::operator+(const Matrix& other) const {
    return add(other);
}

Matrix Matrix::operator-(const Matrix& other) const {
    return subtract(other);
}

Matrix Matrix::operator*(const Matrix& other) const {
    return multiply(other);
}

Matrix Matrix::operator*(double scalar) const {
    return scale(scalar);
}

Matrix& Matrix::operator=(const Matrix& other) {
    if (this != &other) {
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                data[i][j] = other.data[i][j];
                grad[i][j] = other.grad[i][j];
            }
        }
    }
    return *this;
}

// ============================================================================
// Static Helpers
// ============================================================================

Matrix Matrix::identity() {
    return Matrix(1.0, 0.0, 0.0, 1.0);
}

Matrix Matrix::zeros() {
    return Matrix(0.0, 0.0, 0.0, 0.0);
}

Matrix Matrix::fromArray(double arr[2][2]) {
    Matrix result;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            result.data[i][j] = arr[i][j];
        }
    }
    return result;
}

