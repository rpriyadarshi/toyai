/**
 * ============================================================================
 * Matrix Class: 2x2 Matrix Operations
 * ============================================================================
 * 
 * A simple 2x2 matrix class for hand-calculable transformer operations.
 * All operations are designed to be verifiable on paper.
 * 
 * Purpose: Foundation for all transformer computations
 */

#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <iostream>
#include <iomanip>
#include <cmath>

class Matrix {
public:
    // Data storage: 2x2 matrix
    double data[2][2];
    
    // Gradient storage for backpropagation
    double grad[2][2];
    
    // ========================================================================
    // Constructors
    // ========================================================================
    
    /**
     * Default constructor: Initialize to zero
     */
    Matrix();
    
    /**
     * Constructor with values: [a b; c d]
     */
    Matrix(double a, double b, double c, double d);
    
    /**
     * Copy constructor
     */
    Matrix(const Matrix& other);
    
    // ========================================================================
    // Basic Operations
    // ========================================================================
    
    /**
     * Matrix multiplication: this × other
     */
    Matrix multiply(const Matrix& other) const;
    
    /**
     * Transpose: swap rows and columns
     */
    Matrix transpose() const;
    
    /**
     * Scalar multiplication: scale all elements
     */
    Matrix scale(double s) const;
    
    /**
     * Element-wise addition
     */
    Matrix add(const Matrix& other) const;
    
    /**
     * Element-wise subtraction
     */
    Matrix subtract(const Matrix& other) const;
    
    /**
     * Element-wise multiplication (Hadamard product)
     */
    Matrix hadamard(const Matrix& other) const;
    
    // ========================================================================
    // Utility Functions
    // ========================================================================
    
    /**
     * Zero out gradient storage
     */
    void zeroGrad();
    
    /**
     * Compute Frobenius norm squared (sum of squares)
     */
    double frobeniusNormSquared() const;
    
    /**
     * Sum of all elements
     */
    double sum() const;
    
    /**
     * Get element at position (i, j)
     */
    double get(int i, int j) const;
    
    /**
     * Set element at position (i, j)
     */
    void set(int i, int j, double value);
    
    // ========================================================================
    // Hand-Calculation Helpers
    // ========================================================================
    
    /**
     * Print matrix with label
     */
    void print(const std::string& name) const;
    
    /**
     * Print matrix with detailed computation steps
     */
    void printWithSteps(const std::string& name, const std::string& operation) const;
    
    /**
     * Print gradient
     */
    void printGrad(const std::string& name) const;
    
    // ========================================================================
    // Operators
    // ========================================================================
    
    Matrix operator+(const Matrix& other) const;
    Matrix operator-(const Matrix& other) const;
    Matrix operator*(const Matrix& other) const;  // Matrix multiplication
    Matrix operator*(double scalar) const;       // Scalar multiplication
    Matrix& operator=(const Matrix& other);
    
    // ========================================================================
    // Static Helpers
    // ========================================================================
    
    /**
     * Create identity matrix
     */
    static Matrix identity();
    
    /**
     * Create zero matrix
     */
    static Matrix zeros();
    
    /**
     * Create matrix from array
     */
    static Matrix fromArray(double arr[2][2]);
};

#endif // MATRIX_HPP

