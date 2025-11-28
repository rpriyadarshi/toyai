/**
 * ============================================================================
 * Example 5: Feed-Forward Layers
 * ============================================================================
 * 
 * Goal: Add non-linearity and depth with feed-forward networks
 * 
 * This example demonstrates:
 * - Feed-forward network after attention
 * - ReLU activation
 * - Residual connections
 * - Layer composition
 */

#include "../../src/core/Matrix.hpp"
#include "../../src/core/Embedding.hpp"
#include "../../src/core/LinearProjection.hpp"
#include "../../src/core/Attention.hpp"
#include "../../src/core/Softmax.hpp"
#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>

// ReLU activation
Matrix relu(const Matrix& input) {
    Matrix result;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            result.set(i, j, std::max(0.0, input.get(i, j)));
        }
    }
    return result;
}

int main() {
    std::cout << std::string(70, '=') << "\n";
    std::cout << "Example 5: Feed-Forward Layers\n";
    std::cout << "Goal: Add non-linearity with FFN\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    // Model setup
    Embedding embedding;
    Matrix WQ(0.1, 0.0, 0.0, 0.1);
    Matrix WK(0.1, 0.0, 0.0, 0.1);
    Matrix WV(0.1, 0.0, 0.0, 0.1);
    
    // Feed-forward weights
    Matrix W1(0.2, 0.1, 0.1, 0.2);  // First linear layer
    Matrix W2(0.2, 0.1, 0.1, 0.2);  // Second linear layer
    
    LinearProjection projQ(WQ, false);
    LinearProjection projK(WK, false);
    LinearProjection projV(WV, false);
    LinearProjection ffn1(W1, false);
    LinearProjection ffn2(W2, false);
    
    Attention attention;
    
    std::vector<int> input_tokens = {0, 1};  // A, B
    
    // Forward pass
    std::cout << "FORWARD PASS\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    // Step 1: Attention
    Matrix X = embedding.forwardSequence(input_tokens);
    Matrix Q = projQ.forward(X);
    Matrix K = projK.forward(X);
    Matrix V = projV.forward(X);
    
    Attention::AttentionResult attn_result = attention.forward(Q, K, V);
    
    std::cout << "Attention output:\n";
    attn_result.output.print("Attention");
    std::cout << "\n";
    
    // Step 2: Feed-forward network
    std::cout << "Feed-Forward Network:\n";
    std::cout << std::string(70, '-') << "\n";
    
    // FFN(x) = ReLU(x × W1) × W2
    Matrix ffn_input = attn_result.output;
    Matrix ffn_hidden = ffn1.forward(ffn_input);
    Matrix ffn_activated = relu(ffn_hidden);
    Matrix ffn_output = ffn2.forward(ffn_activated);
    
    ffn_hidden.print("FFN hidden (before ReLU)");
    ffn_activated.print("FFN activated (after ReLU)");
    ffn_output.print("FFN output");
    std::cout << "\n";
    
    // Step 3: Residual connection
    std::cout << "Residual Connection:\n";
    std::cout << std::string(70, '-') << "\n";
    Matrix residual_output = attn_result.output.add(ffn_output);
    
    attn_result.output.print("Attention output");
    ffn_output.print("FFN output");
    residual_output.print("Final output (attention + FFN)");
    std::cout << "\n";
    
    std::cout << "Feed-forward adds non-linearity and capacity!\n";
    std::cout << "Residual connection enables gradient flow.\n\n";
    
    std::cout << "Next: See Example 6 for complete transformer.\n";
    std::cout << std::string(70, '=') << "\n";
    
    return 0;
}

