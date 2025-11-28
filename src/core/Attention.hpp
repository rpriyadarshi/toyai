#ifndef ATTENTION_HPP
#define ATTENTION_HPP

#include "Matrix.hpp"
#include "Softmax.hpp"
#include <cmath>

/**
 * ============================================================================
 * Scaled Dot-Product Attention
 * ============================================================================
 * 
 * Implements: Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V
 * 
 * For 2x2 case:
 * - Q, K, V are 2x2 matrices (2 positions, 2 dimensions)
 * - d_k = 2 (dimension of keys)
 * - Scale factor = 1/√2
 */
class Attention {
public:
    Softmax softmax;
    double scale_factor;  // 1/√d_k
    
    /**
     * Constructor
     */
    Attention();
    
    /**
     * Forward pass
     * 
     * Returns structure with:
     * - scores: Q × K^T / √d_k
     * - weights: softmax(scores)
     * - output: weights × V
     */
    struct AttentionResult {
        Matrix scores;      // Q × K^T / √d_k
        Matrix weights;     // softmax(scores)
        Matrix output;      // weights × V
    };
    
    AttentionResult forward(const Matrix& Q, const Matrix& K, const Matrix& V) const;
    
    /**
     * Backward pass
     * 
     * Computes gradients w.r.t. Q, K, V
     */
    struct AttentionGradients {
        Matrix dQ;
        Matrix dK;
        Matrix dV;
    };
    
    AttentionGradients backward(const Matrix& Q,
                                const Matrix& K,
                                const Matrix& V,
                                const AttentionResult& forward_result,
                                const Matrix& grad_output) const;
    
    /**
     * Print attention computation steps (for hand calculation)
     */
    void printSteps(const Matrix& Q, const Matrix& K, const Matrix& V,
                    const AttentionResult& result) const;
};

#endif // ATTENTION_HPP

