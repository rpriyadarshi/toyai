#ifndef LINEAR_PROJECTION_HPP
#define LINEAR_PROJECTION_HPP

#include "Matrix.hpp"

/**
 * ============================================================================
 * Linear Projection Layer
 * ============================================================================
 * 
 * Performs linear transformation: output = input × weights
 * 
 * Used for Q, K, V projections in attention.
 * 
 * For 2x2 case:
 * - Input: 2x2 matrix (2 tokens, 2 dimensions each)
 * - Weights: 2x2 matrix
 * - Output: 2x2 matrix
 */
class LinearProjection {
public:
    // Weight matrix: 2x2
    Matrix weights;
    
    // Whether weights are trainable
    bool trainable;
    
    /**
     * Constructor with initial weights
     */
    LinearProjection(const Matrix& initial_weights, bool trainable = true);
    
    /**
     * Default constructor (identity weights)
     */
    LinearProjection(bool trainable = true);
    
    /**
     * Forward pass: output = input × weights
     */
    Matrix forward(const Matrix& input) const;
    
    /**
     * Backward pass: compute gradients
     * 
     * For output = input × weights:
     * - dL/dweights = input^T × dL/doutput
     * - dL/dinput = dL/doutput × weights^T
     */
    void backward(const Matrix& input, 
                  const Matrix& grad_output,
                  Matrix& grad_input,
                  Matrix& grad_weights) const;
    
    /**
     * Update weights using gradient
     */
    void updateWeights(const Matrix& grad_weights, double learning_rate);
    
    /**
     * Zero gradients
     */
    void zeroGrad();
    
    /**
     * Print weights
     */
    void print(const std::string& name) const;
};

#endif // LINEAR_PROJECTION_HPP

