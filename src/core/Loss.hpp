#ifndef LOSS_HPP
#define LOSS_HPP

#include "Matrix.hpp"
#include <vector>
#include <cmath>

/**
 * ============================================================================
 * Loss Functions
 * ============================================================================
 * 
 * Cross-entropy loss for next-token prediction
 */
class Loss {
public:
    /**
     * Cross-entropy loss
     * 
     * L = -log(P(target))
     * 
     * Where P is the predicted probability for the target token
     */
    static double crossEntropy(const std::vector<double>& probs, int target);
    
    /**
     * Gradient of cross-entropy w.r.t. logits
     * 
     * For softmax + cross-entropy:
     * dL/dlogit_i = prob_i - 1[i == target]
     */
    static std::vector<double> crossEntropyGradient(const std::vector<double>& probs, int target);
    
    /**
     * Mean squared error (for comparison)
     */
    static double mse(const Matrix& prediction, const Matrix& target);
    
    /**
     * Print loss computation
     */
    static void printLoss(const std::vector<double>& probs, int target, double loss);
};

#endif // LOSS_HPP

