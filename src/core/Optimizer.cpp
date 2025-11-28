#include "Optimizer.hpp"

Optimizer::Optimizer(double lr) : learning_rate(lr) {
}

void Optimizer::step(Matrix& weights, const Matrix& gradient) const {
    // W_new = W_old - lr * gradient
    Matrix update = gradient.scale(learning_rate);
    weights = weights.subtract(update);
}

void Optimizer::setLearningRate(double lr) {
    learning_rate = lr;
}

double Optimizer::getLearningRate() const {
    return learning_rate;
}

