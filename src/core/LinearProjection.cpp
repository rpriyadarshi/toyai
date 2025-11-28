#include "LinearProjection.hpp"
#include <iostream>

LinearProjection::LinearProjection(const Matrix& initial_weights, bool trainable)
    : weights(initial_weights), trainable(trainable) {
}

LinearProjection::LinearProjection(bool trainable) 
    : trainable(trainable) {
    // Default: scaled identity (0.1 * I)
    weights = Matrix(0.1, 0.0, 0.0, 0.1);
}

Matrix LinearProjection::forward(const Matrix& input) const {
    return input.multiply(weights);
}

void LinearProjection::backward(const Matrix& input,
                                const Matrix& grad_output,
                                Matrix& grad_input,
                                Matrix& grad_weights) const {
    // For output = input × weights:
    // dL/dweights = input^T × grad_output
    Matrix inputT = input.transpose();
    grad_weights = inputT.multiply(grad_output);
    
    // dL/dinput = grad_output × weights^T
    Matrix weightsT = weights.transpose();
    grad_input = grad_output.multiply(weightsT);
}

void LinearProjection::updateWeights(const Matrix& grad_weights, double learning_rate) {
    if (trainable) {
        // weights = weights - lr * grad_weights
        Matrix update = grad_weights.scale(learning_rate);
        weights = weights.subtract(update);
    }
}

void LinearProjection::zeroGrad() {
    weights.zeroGrad();
}

void LinearProjection::print(const std::string& name) const {
    weights.print(name);
}

