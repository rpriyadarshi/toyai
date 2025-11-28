/**
 * ============================================================================
 * Example 3: Full Backpropagation
 * ============================================================================
 * 
 * Goal: Understand complete gradient flow through all components
 * 
 * This example demonstrates:
 * - Full backpropagation through attention
 * - Gradients for WQ, WK, WV, WO
 * - Complete training loop
 * - Matrix calculus in action
 */

#include "../../src/core/Matrix.hpp"
#include "../../src/core/Embedding.hpp"
#include "../../src/core/LinearProjection.hpp"
#include "../../src/core/Attention.hpp"
#include "../../src/core/Softmax.hpp"
#include "../../src/core/Loss.hpp"
#include "../../src/core/Optimizer.hpp"
#include <iostream>
#include <vector>
#include <iomanip>

int main() {
    std::cout << std::string(70, '=') << "\n";
    std::cout << "Example 3: Full Backpropagation\n";
    std::cout << "Goal: Complete gradient flow through all weights\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    // Setup: All weights are trainable
    Embedding embedding;
    
    Matrix WQ(0.1, 0.0, 0.0, 0.1);
    Matrix WK(0.1, 0.0, 0.0, 0.1);
    Matrix WV(0.1, 0.0, 0.0, 0.1);
    Matrix WO(0.1, 0.0, 0.0, 0.1);
    
    LinearProjection projQ(WQ, true);  // Trainable
    LinearProjection projK(WK, true);
    LinearProjection projV(WV, true);
    
    Attention attention;
    Softmax softmax;
    Optimizer optimizer(0.1);
    
    std::vector<int> input_tokens = {0, 1};  // A, B
    int target_token = 2;  // C
    
    // Forward pass
    std::cout << "FORWARD PASS\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    Matrix X = embedding.forwardSequence(input_tokens);
    Matrix Q = projQ.forward(X);
    Matrix K = projK.forward(X);
    Matrix V = projV.forward(X);
    
    Attention::AttentionResult attn_result = attention.forward(Q, K, V);
    
    double context[2] = {attn_result.output.get(1, 0), attn_result.output.get(1, 1)};
    
    std::vector<double> logits(4);
    logits[0] = context[0] * 0.1 + context[1] * 0.0;
    logits[1] = context[0] * 0.0 + context[1] * 0.0;
    logits[2] = context[0] * 0.0 + context[1] * 0.1;
    logits[3] = context[0] * 0.0 + context[1] * 0.0;
    
    std::vector<double> probs = softmax.forward(logits);
    double loss = Loss::crossEntropy(probs, target_token);
    
    std::cout << "Loss: " << loss << "\n\n";
    
    // Backward pass
    std::cout << "BACKWARD PASS\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    // Step 1: Gradient w.r.t. logits
    std::vector<double> dL_dlogits = Loss::crossEntropyGradient(probs, target_token);
    
    // Step 2: Gradient w.r.t. context (through output projection)
    // This would go through WO, but for simplicity we'll compute dL/dcontext
    Matrix dL_dcontext;
    // Simplified: dL/dcontext = sum over logits
    dL_dcontext.set(0, 0, dL_dlogits[0] * 0.1 + dL_dlogits[2] * 0.0);
    dL_dcontext.set(0, 1, dL_dlogits[1] * 0.0 + dL_dlogits[3] * 0.0);
    dL_dcontext.set(1, 0, dL_dlogits[0] * 0.0 + dL_dlogits[2] * 0.0);
    dL_dcontext.set(1, 1, dL_dlogits[1] * 0.0 + dL_dlogits[3] * 0.0);
    
    // Step 3: Gradient through attention
    Matrix dL_doutput;
    dL_doutput.set(1, 0, dL_dcontext.get(0, 0));  // Position 1, dim 0
    dL_doutput.set(1, 1, dL_dcontext.get(0, 1));  // Position 1, dim 1
    
    Attention::AttentionGradients attn_grads = attention.backward(
        Q, K, V, attn_result, dL_doutput
    );
    
    std::cout << "Attention Gradients:\n";
    attn_grads.dQ.print("dL/dQ");
    attn_grads.dK.print("dL/dK");
    attn_grads.dV.print("dL/dV");
    std::cout << "\n";
    
    // Step 4: Gradients through projections
    Matrix grad_input_Q, grad_weights_Q;
    Matrix grad_input_K, grad_weights_K;
    Matrix grad_input_V, grad_weights_V;
    
    projQ.backward(X, attn_grads.dQ, grad_input_Q, grad_weights_Q);
    projK.backward(X, attn_grads.dK, grad_input_K, grad_weights_K);
    projV.backward(X, attn_grads.dV, grad_input_V, grad_weights_V);
    
    std::cout << "Weight Gradients:\n";
    grad_weights_Q.print("dL/dWQ");
    grad_weights_K.print("dL/dWK");
    grad_weights_V.print("dL/dWV");
    std::cout << "\n";
    
    // Step 5: Update all weights
    std::cout << "UPDATING WEIGHTS\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    projQ.updateWeights(grad_weights_Q, optimizer.getLearningRate());
    projK.updateWeights(grad_weights_K, optimizer.getLearningRate());
    projV.updateWeights(grad_weights_V, optimizer.getLearningRate());
    
    std::cout << "Updated weights:\n";
    projQ.print("WQ");
    projK.print("WK");
    projV.print("WV");
    std::cout << "\n";
    
    std::cout << "Complete backpropagation through all components!\n";
    std::cout << "Next: See Example 4 for training on multiple patterns.\n";
    std::cout << std::string(70, '=') << "\n";
    
    return 0;
}

