#include "Loss.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>

double Loss::crossEntropy(const std::vector<double>& probs, int target) {
    if (target < 0 || target >= static_cast<int>(probs.size())) {
        return 0.0;  // Invalid target
    }
    
    double prob = probs[target];
    if (prob <= 0.0) {
        // Avoid log(0)
        return 1000.0;  // Large loss
    }
    
    return -std::log(prob);
}

std::vector<double> Loss::crossEntropyGradient(const std::vector<double>& probs, int target) {
    std::vector<double> grad(probs.size());
    
    for (size_t i = 0; i < probs.size(); i++) {
        if (i == static_cast<size_t>(target)) {
            grad[i] = probs[i] - 1.0;  // Negative (we want to increase this)
        } else {
            grad[i] = probs[i] - 0.0;  // Positive (we want to decrease this)
        }
    }
    
    return grad;
}

double Loss::mse(const Matrix& prediction, const Matrix& target) {
    Matrix diff = prediction.subtract(target);
    return diff.frobeniusNormSquared() / 4.0;  // Mean over 4 elements
}

void Loss::printLoss(const std::vector<double>& probs, int target, double loss) {
    std::cout << "\nLoss Computation:\n";
    std::cout << std::fixed << std::setprecision(6);
    
    const char* tokens[] = {"A", "B", "C", "D"};
    std::cout << "  Probabilities:\n";
    for (size_t i = 0; i < probs.size(); i++) {
        std::cout << "    P(" << tokens[i] << ") = " << probs[i];
        if (i == static_cast<size_t>(target)) {
            std::cout << "  ← Target";
        }
        std::cout << "\n";
    }
    
    std::cout << "  Loss = -log(P(" << tokens[target] << ")) = -log(" 
              << probs[target] << ") = " << loss << "\n";
}

