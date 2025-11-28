#ifndef SOFTMAX_HPP
#define SOFTMAX_HPP

#include "Matrix.hpp"
#include <vector>

/**
 * ============================================================================
 * Softmax Activation
 * ============================================================================
 * 
 * Converts scores to probability distribution.
 * 
 * Formula: softmax(x_i) = exp(x_i) / sum(exp(x_j))
 * 
 * Applied row-wise: each row sums to 1
 */
class Softmax {
public:
    /**
     * Forward pass: apply softmax row-wise
     */
    Matrix forward(const Matrix& input) const;
    
    /**
     * Forward pass for vector (treats as single row)
     */
    std::vector<double> forward(const std::vector<double>& input) const;
    
    /**
     * Backward pass: compute gradient w.r.t. input
     * 
     * For softmax output s and gradient dL/ds:
     * dL/dx_i = s_i * (dL/ds_i - sum_j(dL/ds_j * s_j))
     */
    Matrix backward(const Matrix& softmax_output, const Matrix& grad_output) const;
    
    /**
     * Backward pass for vector
     */
    std::vector<double> backward(const std::vector<double>& softmax_output, 
                                  const std::vector<double>& grad_output) const;
    
    /**
     * Print softmax computation steps (for hand calculation)
     */
    void printSteps(const std::vector<double>& scores, 
                    const std::vector<double>& probs) const;
};

#endif // SOFTMAX_HPP

