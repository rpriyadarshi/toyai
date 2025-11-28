#include "Softmax.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>

Matrix Softmax::forward(const Matrix& input) const {
    Matrix result;
    
    // Apply softmax row-wise
    for (int i = 0; i < 2; i++) {
        // Find max for numerical stability
        double max_val = std::max(input.data[i][0], input.data[i][1]);
        
        // Compute exponentials
        double exp0 = std::exp(input.data[i][0] - max_val);
        double exp1 = std::exp(input.data[i][1] - max_val);
        
        // Normalize
        double sum = exp0 + exp1;
        result.data[i][0] = exp0 / sum;
        result.data[i][1] = exp1 / sum;
    }
    
    return result;
}

std::vector<double> Softmax::forward(const std::vector<double>& input) const {
    std::vector<double> result(input.size());
    
    // Find max for numerical stability
    double max_val = *std::max_element(input.begin(), input.end());
    
    // Compute exponentials and sum
    double sum = 0.0;
    std::vector<double> exps(input.size());
    for (size_t i = 0; i < input.size(); i++) {
        exps[i] = std::exp(input[i] - max_val);
        sum += exps[i];
    }
    
    // Normalize
    for (size_t i = 0; i < input.size(); i++) {
        result[i] = exps[i] / sum;
    }
    
    return result;
}

Matrix Softmax::backward(const Matrix& softmax_output, const Matrix& grad_output) const {
    Matrix grad_input;
    
    for (int i = 0; i < 2; i++) {
        // Compute dot product: sum_j(grad_output[j] * softmax_output[j])
        double dot = grad_output.data[i][0] * softmax_output.data[i][0] + 
                     grad_output.data[i][1] * softmax_output.data[i][1];
        
        // Softmax Jacobian: ds_i/dx_j = s_i * (delta_ij - s_j)
        // Gradient: dL/dx_i = s_i * (dL/ds_i - dot)
        for (int j = 0; j < 2; j++) {
            grad_input.data[i][j] = softmax_output.data[i][j] * 
                                     (grad_output.data[i][j] - dot);
        }
    }
    
    return grad_input;
}

std::vector<double> Softmax::backward(const std::vector<double>& softmax_output,
                                       const std::vector<double>& grad_output) const {
    std::vector<double> grad_input(softmax_output.size());
    
    // Compute dot product
    double dot = 0.0;
    for (size_t i = 0; i < softmax_output.size(); i++) {
        dot += grad_output[i] * softmax_output[i];
    }
    
    // Apply softmax Jacobian
    for (size_t i = 0; i < softmax_output.size(); i++) {
        grad_input[i] = softmax_output[i] * (grad_output[i] - dot);
    }
    
    return grad_input;
}

void Softmax::printSteps(const std::vector<double>& scores,
                         const std::vector<double>& probs) const {
    std::cout << "\nSoftmax Computation:\n";
    std::cout << std::fixed << std::setprecision(6);
    
    // Find max
    double max_val = *std::max_element(scores.begin(), scores.end());
    std::cout << "  Max value (for stability): " << max_val << "\n";
    
    // Show exponentials
    std::cout << "  Exponentials (shifted by max):\n";
    double sum = 0.0;
    for (size_t i = 0; i < scores.size(); i++) {
        double exp_val = std::exp(scores[i] - max_val);
        sum += exp_val;
        std::cout << "    exp(" << (scores[i] - max_val) << ") = " << exp_val << "\n";
    }
    
    std::cout << "  Sum: " << sum << "\n";
    std::cout << "  Probabilities:\n";
    for (size_t i = 0; i < probs.size(); i++) {
        std::cout << "    P[" << i << "] = " << probs[i] << "\n";
    }
    
    // Verify sum
    double prob_sum = 0.0;
    for (double p : probs) prob_sum += p;
    std::cout << "  Sum of probabilities: " << prob_sum << " (should be 1.0)\n";
}

