#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "Matrix.hpp"

/**
 * ============================================================================
 * Optimizer: Gradient Descent
 * ============================================================================
 * 
 * Implements stochastic gradient descent (SGD)
 */
class Optimizer {
public:
    double learning_rate;
    
    /**
     * Constructor
     */
    Optimizer(double lr = 0.1);
    
    /**
     * Update weights using gradient
     * 
     * W_new = W_old - lr * gradient
     */
    void step(Matrix& weights, const Matrix& gradient) const;
    
    /**
     * Set learning rate
     */
    void setLearningRate(double lr);
    
    /**
     * Get learning rate
     */
    double getLearningRate() const;
};

#endif // OPTIMIZER_HPP

