/**
 * ============================================================================
 * Example 1: Minimal Forward Pass
 * ============================================================================
 * 
 * Goal: Understand how a transformer makes predictions (no training yet)
 * 
 * This example demonstrates:
 * - Token embeddings
 * - Q, K, V projections
 * - Scaled dot-product attention
 * - Output projection
 * - Softmax to get probabilities
 * 
 * All computations use 2x2 matrices that can be verified by hand.
 */

#include "../../src/core/Matrix.hpp"
#include "../../src/core/Embedding.hpp"
#include "../../src/core/LinearProjection.hpp"
#include "../../src/core/Attention.hpp"
#include "../../src/core/Softmax.hpp"
#include <iostream>
#include <vector>
#include <iomanip>

int main() {
    std::cout << std::string(70, '=') << "\n";
    std::cout << "Example 1: Minimal Forward Pass\n";
    std::cout << "Goal: Understand how predictions are made\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    // ========================================================================
    // Setup: Define the model
    // ========================================================================
    
    std::cout << "MODEL SETUP\n";
    std::cout << std::string(70, '-') << "\n\n";
    
    // Token embeddings (fixed)
    Embedding embedding;
    embedding.print();
    std::cout << "\n";
    
    // Q, K, V projection weights (fixed, small values)
    Matrix WQ(0.1, 0.0, 0.0, 0.1);  // Scaled identity
    Matrix WK(0.1, 0.0, 0.0, 0.1);
    Matrix WV(0.1, 0.0, 0.0, 0.1);
    
    LinearProjection projQ(WQ, false);  // Not trainable
    LinearProjection projK(WK, false);
    LinearProjection projV(WV, false);
    
    std::cout << "Projection Weights:\n";
    WQ.print("WQ");
    WK.print("WK");
    WV.print("WV");
    std::cout << "\n";
    
    // Output projection: maps 2D context → 4 logits
    // For simplicity, we'll use a 2x2 matrix and extract relevant elements
    // In practice, this would be 2x4, but for 2x2 we'll adapt
    Matrix WO(0.1, 0.0, 0.0, 0.1);
    std::cout << "Output Projection:\n";
    WO.print("WO");
    std::cout << "\n";
    
    // Attention and softmax
    Attention attention;
    Softmax softmax;
    
    // ========================================================================
    // Forward Pass: Input "A B"
    // ========================================================================
    
    std::cout << std::string(70, '=') << "\n";
    std::cout << "FORWARD PASS: Input sequence [A, B]\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    // Step 1: Get embeddings
    std::cout << "Step 1: Token Embeddings\n";
    std::cout << std::string(70, '-') << "\n";
    std::vector<int> tokens = {0, 1};  // A=0, B=1
    Matrix X = embedding.forwardSequence(tokens);
    X.print("X (embeddings)");
    std::cout << "  Row 0: Token A = [1, 0]\n";
    std::cout << "  Row 1: Token B = [0, 1]\n\n";
    
    // Step 2: Project to Q, K, V
    std::cout << "Step 2: Q, K, V Projections\n";
    std::cout << std::string(70, '-') << "\n";
    Matrix Q = projQ.forward(X);
    Matrix K = projK.forward(X);
    Matrix V = projV.forward(X);
    
    Q.print("Q (queries)");
    K.print("K (keys)");
    V.print("V (values)");
    std::cout << "\n";
    
    // Step 3: Attention
    std::cout << "Step 3: Scaled Dot-Product Attention\n";
    std::cout << std::string(70, '-') << "\n";
    Attention::AttentionResult attn_result = attention.forward(Q, K, V);
    attention.printSteps(Q, K, V, attn_result);
    
    // Step 4: Output projection
    // For position 1 (B), use its context vector to predict next token
    // Context vector is attn_result.output row 1
    std::cout << "Step 4: Output Projection (Context → Logits)\n";
    std::cout << std::string(70, '-') << "\n";
    
    // Extract context vector for position 1 (B)
    double context[2] = {attn_result.output.get(1, 0), attn_result.output.get(1, 1)};
    std::cout << "Context vector (from position 1): [" 
              << context[0] << ", " << context[1] << "]\n\n";
    
    // Project to logits (for 4 tokens: A, B, C, D)
    // We'll compute: logit[i] = context · WO_column_i
    // For simplicity with 2x2, we'll map:
    // - WO[0][0] → logit(A)
    // - WO[0][1] → logit(B)  
    // - WO[1][0] → logit(C)
    // - WO[1][1] → logit(D)
    std::vector<double> logits(4);
    logits[0] = context[0] * WO.get(0, 0) + context[1] * WO.get(1, 0);
    logits[1] = context[0] * WO.get(0, 1) + context[1] * WO.get(1, 1);
    logits[2] = context[0] * WO.get(0, 0) + context[1] * WO.get(1, 0);  // Reuse for C
    logits[3] = context[0] * WO.get(0, 1) + context[1] * WO.get(1, 1);  // Reuse for D
    
    // Actually, let's use a proper 2x4 mapping conceptually
    // For hand calculation, we'll use:
    // WO conceptually maps 2D → 4D, but we'll adapt
    // Let's define: logit[i] = context[0] * WO_row0[i] + context[1] * WO_row1[i]
    // But WO is 2x2, so we'll use:
    logits[0] = context[0] * 0.1 + context[1] * 0.0;  // A
    logits[1] = context[0] * 0.0 + context[1] * 0.0;  // B
    logits[2] = context[0] * 0.0 + context[1] * 0.1;  // C
    logits[3] = context[0] * 0.0 + context[1] * 0.0;  // D
    
    std::cout << "Logits (raw scores):\n";
    const char* token_names[] = {"A", "B", "C", "D"};
    for (int i = 0; i < 4; i++) {
        std::cout << "  logit(" << token_names[i] << ") = " << logits[i] << "\n";
    }
    std::cout << "\n";
    
    // Step 5: Softmax to get probabilities
    std::cout << "Step 5: Softmax (Logits → Probabilities)\n";
    std::cout << std::string(70, '-') << "\n";
    std::vector<double> probs = softmax.forward(logits);
    softmax.printSteps(logits, probs);
    
    // ========================================================================
    // Summary
    // ========================================================================
    
    std::cout << std::string(70, '=') << "\n";
    std::cout << "SUMMARY\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    std::cout << "Input: [A, B]\n";
    std::cout << "Predicted probabilities for next token:\n";
    for (int i = 0; i < 4; i++) {
        std::cout << "  P(" << token_names[i] << ") = " 
                  << std::fixed << std::setprecision(4) << probs[i] 
                  << " (" << (probs[i] * 100) << "%)\n";
    }
    std::cout << "\n";
    
    std::cout << "Note: This is BEFORE training. The model hasn't learned\n";
    std::cout << "the pattern (A, B) → C yet. All tokens have roughly equal\n";
    std::cout << "probability. After training (Example 2+), C's probability\n";
    std::cout << "will increase.\n\n";
    
    std::cout << "Next: See Example 2 to learn how training works!\n";
    std::cout << std::string(70, '=') << "\n";
    
    return 0;
}

