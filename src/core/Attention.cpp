#include "Attention.hpp"
#include <iostream>
#include <iomanip>

Attention::Attention() {
    // d_k = 2 for 2x2 matrices
    scale_factor = 1.0 / std::sqrt(2.0);
}

Attention::AttentionResult Attention::forward(const Matrix& Q, const Matrix& K, const Matrix& V) const {
    AttentionResult result;
    
    // Step 1: Q × K^T
    Matrix Kt = K.transpose();
    Matrix raw_scores = Q.multiply(Kt);
    
    // Step 2: Scale by 1/√d_k
    result.scores = raw_scores.scale(scale_factor);
    
    // Step 3: Softmax (row-wise)
    result.weights = softmax.forward(result.scores);
    
    // Step 4: Weighted sum of values
    result.output = result.weights.multiply(V);
    
    return result;
}

Attention::AttentionGradients Attention::backward(const Matrix& Q,
                                                   const Matrix& K,
                                                   const Matrix& V,
                                                   const AttentionResult& forward_result,
                                                   const Matrix& grad_output) const {
    AttentionGradients grads;
    
    // Step 1: Gradient through output = weights × V
    Matrix Vt = V.transpose();
    Matrix dWeights = grad_output.multiply(Vt);
    
    Matrix weightsT = forward_result.weights.transpose();
    grads.dV = weightsT.multiply(grad_output);
    
    // Step 2: Gradient through softmax
    Matrix dScores = softmax.backward(forward_result.weights, dWeights);
    
    // Step 3: Gradient through scaling
    Matrix dRawScores = dScores.scale(scale_factor);
    
    // Step 4: Gradient through Q × K^T
    // For C = A × B^T:
    // dL/dA = dL/dC × B
    // dL/dB = dL/dC^T × A
    grads.dQ = dRawScores.multiply(K);
    
    Matrix dRawScoresT = dRawScores.transpose();
    grads.dK = dRawScoresT.multiply(Q);
    
    return grads;
}

void Attention::printSteps(const Matrix& Q, const Matrix& K, const Matrix& V,
                           const AttentionResult& result) const {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "Attention Computation Steps\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    std::cout << "Step 1: Q × K^T\n";
    Q.print("Q");
    K.transpose().print("K^T");
    Matrix Kt = K.transpose();
    Matrix raw_scores = Q.multiply(Kt);
    raw_scores.print("Raw scores (Q × K^T)");
    
    std::cout << "\nStep 2: Scale by 1/√d_k = " << scale_factor << "\n";
    result.scores.print("Scaled scores");
    
    std::cout << "\nStep 3: Softmax\n";
    result.weights.print("Attention weights (softmax)");
    
    std::cout << "\nStep 4: Weighted sum of values\n";
    V.print("V");
    result.output.print("Output (weights × V)");
    std::cout << "\n";
}

