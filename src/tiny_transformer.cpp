/**
 * tiny_transformer.cpp
 * 
 * A minimal, hand-calculable transformer implementation using 2x2 matrices.
 * This code demonstrates the complete pipeline:
 * - Forward pass (inference)
 * - Loss computation
 * - Backpropagation (training)
 * - Weight updates
 * 
 * All matrices are 2x2 so you can verify every calculation with pen and paper!
 * 
 * Compile: g++ -std=c++17 -o tiny_transformer tiny_transformer.cpp -lm
 * Run:     ./tiny_transformer
 * 
 * Author: ToyAI Project
 * Purpose: Educational - Understanding the matrix core of generative AI
 */

#include <iostream>
#include <cmath>
#include <iomanip>
#include <array>

// ============================================================================
// MATRIX CLASS: The fundamental building block of transformers
// ============================================================================

/**
 * WHY A MATRIX CLASS?
 * 
 * Transformers are fundamentally matrix operations. Every major computation:
 * - Embedding lookup → matrix indexing
 * - Q/K/V projections → matrix multiplication
 * - Attention scores → matrix multiplication
 * - Feed-forward layers → matrix multiplication
 * 
 * By building a simple Matrix class, we can trace EXACTLY what happens at each step.
 */
class Matrix2x2 {
public:
    // 2x2 matrix stored as [row][col]
    double data[2][2];
    
    // Default constructor - zero matrix
    Matrix2x2() {
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                data[i][j] = 0.0;
    }
    
    // Constructor with values
    Matrix2x2(double a00, double a01, double a10, double a11) {
        data[0][0] = a00; data[0][1] = a01;
        data[1][0] = a10; data[1][1] = a11;
    }
    
    // Access elements
    double& at(int row, int col) { return data[row][col]; }
    double at(int row, int col) const { return data[row][col]; }
    
    // ========================================================================
    // MATRIX MULTIPLICATION: The core operation of transformers
    // ========================================================================
    /**
     * WHY MATRIX MULTIPLICATION?
     * 
     * Matrix multiplication is a linear transformation that:
     * 1. Rotates and scales the input space
     * 2. Combines information across dimensions
     * 3. Is differentiable (crucial for learning!)
     * 
     * For C = A × B:
     *   C[i][j] = Σ_k A[i][k] × B[k][j]
     * 
     * This means each output element is a weighted sum of inputs,
     * where weights are learned parameters.
     */
    Matrix2x2 operator*(const Matrix2x2& other) const {
        Matrix2x2 result;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                result.data[i][j] = 0.0;
                for (int k = 0; k < 2; k++) {
                    result.data[i][j] += data[i][k] * other.data[k][j];
                }
            }
        }
        return result;
    }
    
    // Element-wise operations for gradient computation
    Matrix2x2 operator+(const Matrix2x2& other) const {
        Matrix2x2 result;
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                result.data[i][j] = data[i][j] + other.data[i][j];
        return result;
    }
    
    Matrix2x2 operator-(const Matrix2x2& other) const {
        Matrix2x2 result;
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                result.data[i][j] = data[i][j] - other.data[i][j];
        return result;
    }
    
    Matrix2x2 operator*(double scalar) const {
        Matrix2x2 result;
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                result.data[i][j] = data[i][j] * scalar;
        return result;
    }
    
    // ========================================================================
    // TRANSPOSE: Essential for attention and backpropagation
    // ========================================================================
    /**
     * WHY TRANSPOSE?
     * 
     * 1. In attention: Q × K^T computes dot products between all pairs
     *    - K^T turns column vectors into row vectors
     *    - This allows Q[i] · K[j] to be computed as matrix multiplication
     * 
     * 2. In backprop: Gradients flow through transposes
     *    - If forward is: out = A × B
     *    - Backward is: dA = dout × B^T, dB = A^T × dout
     */
    Matrix2x2 transpose() const {
        Matrix2x2 result;
        for (int i = 0; i < 2; i++)
            for (int j = 0; j < 2; j++)
                result.data[i][j] = data[j][i];
        return result;
    }
    
    // Print matrix for debugging
    void print(const std::string& name = "") const {
        if (!name.empty()) {
            std::cout << name << ":" << std::endl;
        }
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  [" << data[0][0] << "  " << data[0][1] << "]" << std::endl;
        std::cout << "  [" << data[1][0] << "  " << data[1][1] << "]" << std::endl;
    }
};

// ============================================================================
// SOFTMAX: Converting scores to attention weights
// ============================================================================

/**
 * WHY SOFTMAX?
 * 
 * Attention needs to distribute focus across all tokens.
 * We need weights that:
 * 1. Are non-negative (can't have negative attention)
 * 2. Sum to 1 (fixed "attention budget" per token)
 * 3. Preserve relative ordering (higher score → more attention)
 * 4. Are differentiable (for learning)
 * 
 * Softmax: softmax(x_i) = exp(x_i) / Σ_j exp(x_j)
 * 
 * The exponential amplifies differences and ensures positivity.
 * The division normalizes to sum to 1.
 * 
 * WHY NOT JUST NORMALIZE?
 * 
 * Raw normalization (x_i / Σx) doesn't work if scores can be negative.
 * Softmax converts any real numbers to valid probabilities.
 */
Matrix2x2 softmax_rows(const Matrix2x2& m) {
    Matrix2x2 result;
    
    for (int i = 0; i < 2; i++) {
        // Find max for numerical stability
        // WHY: exp(x) can overflow; exp(x - max) keeps values manageable
        double max_val = std::max(m.data[i][0], m.data[i][1]);
        
        // Compute exp(x - max) for each element
        double exp0 = std::exp(m.data[i][0] - max_val);
        double exp1 = std::exp(m.data[i][1] - max_val);
        
        // Sum for normalization
        double sum = exp0 + exp1;
        
        // Normalize to get probabilities
        result.data[i][0] = exp0 / sum;
        result.data[i][1] = exp1 / sum;
    }
    
    return result;
}

// ============================================================================
// TINY TRANSFORMER CLASS
// ============================================================================

/**
 * This implements a minimal self-attention layer with:
 * - 2 tokens (sequence length = 2)
 * - 2-dimensional embeddings
 * - Single attention head
 * 
 * All matrices are 2x2, making it possible to verify every step by hand!
 */
class TinyTransformer {
public:
    // Weight matrices (learned parameters)
    Matrix2x2 W_Q;  // Query weights: transforms input to "what am I looking for?"
    Matrix2x2 W_K;  // Key weights: transforms input to "what do I contain?"
    Matrix2x2 W_V;  // Value weights: transforms input to "what info do I carry?"
    Matrix2x2 W_O;  // Output projection (optional, for complete model)
    
    // Gradients for backpropagation
    Matrix2x2 dW_Q, dW_K, dW_V, dW_O;
    
    // Cached values for backprop (stored during forward pass)
    Matrix2x2 cached_X;       // Input
    Matrix2x2 cached_Q;       // Query
    Matrix2x2 cached_K;       // Key
    Matrix2x2 cached_V;       // Value
    Matrix2x2 cached_scores;  // Q × K^T (before scaling/softmax)
    Matrix2x2 cached_scaled;  // Scaled scores
    Matrix2x2 cached_attn;    // Attention weights (after softmax)
    Matrix2x2 cached_output;  // Final output
    
    double scale_factor;      // 1/√d_k for scaling attention scores
    
    // ========================================================================
    // CONSTRUCTOR: Initialize weights
    // ========================================================================
    TinyTransformer() {
        // d_k = 2 (dimension of keys/queries)
        scale_factor = 1.0 / std::sqrt(2.0);  // ≈ 0.707
        
        /**
         * Initialize weights with simple values for hand calculation.
         * 
         * In real transformers, weights are initialized randomly (e.g., Xavier/He)
         * to break symmetry and start in a "useful" region of parameter space.
         * 
         * For this example, we use values that produce non-trivial but
         * hand-calculable results.
         */
        W_Q = Matrix2x2(1.0, 0.0,   // Query weights
                        0.0, 1.0);
        
        W_K = Matrix2x2(0.0, 1.0,   // Key weights (different to create interesting attention)
                        1.0, 0.0);
        
        W_V = Matrix2x2(1.0, 1.0,   // Value weights
                        0.0, 1.0);
        
        W_O = Matrix2x2(1.0, 0.0,   // Output projection (identity for simplicity)
                        0.0, 1.0);
    }
    
    // ========================================================================
    // FORWARD PASS: The inference path
    // ========================================================================
    /**
     * The forward pass computes:
     * 
     * 1. Q = X × W_Q  (What is each token looking for?)
     * 2. K = X × W_K  (What can each token be found by?)
     * 3. V = X × W_V  (What information does each token carry?)
     * 4. scores = Q × K^T  (How compatible are all token pairs?)
     * 5. scaled = scores / √d_k  (Normalize for stable gradients)
     * 6. attn = softmax(scaled)  (Convert to probabilities)
     * 7. output = attn × V  (Weighted combination of values)
     * 
     * Each step transforms the representation, allowing the model to:
     * - Identify relationships between tokens
     * - Aggregate information from relevant positions
     * - Create context-aware representations
     */
    Matrix2x2 forward(const Matrix2x2& X, bool verbose = false) {
        // Cache input for backpropagation
        cached_X = X;
        
        // --------------------------------------------------------------------
        // STEP 1: Compute Q, K, V projections
        // --------------------------------------------------------------------
        /**
         * WHY THREE SEPARATE PROJECTIONS?
         * 
         * Q, K, V serve different roles:
         * - Q: "What am I searching for?" - defines the query at each position
         * - K: "What can I be found by?" - defines what each position offers
         * - V: "What do I actually contain?" - the content to return
         * 
         * This separation allows the model to learn:
         * - HOW to search (Q projection)
         * - HOW to be searchable (K projection)  
         * - WHAT to return (V projection)
         * 
         * These can be different! A word might be searchable by one feature
         * but return information about a different feature.
         */
        cached_Q = X * W_Q;
        cached_K = X * W_K;
        cached_V = X * W_V;
        
        if (verbose) {
            std::cout << "\n=== STEP 1: Q, K, V Projections ===" << std::endl;
            std::cout << "WHY: Transform input into three views for attention" << std::endl;
            cached_Q.print("Q = X × W_Q");
            cached_K.print("K = X × W_K");
            cached_V.print("V = X × W_V");
        }
        
        // --------------------------------------------------------------------
        // STEP 2: Compute attention scores
        // --------------------------------------------------------------------
        /**
         * WHY Q × K^T?
         * 
         * The dot product measures similarity/compatibility:
         * - Large positive: vectors point same direction (similar)
         * - Near zero: vectors are orthogonal (unrelated)
         * - Large negative: vectors point opposite (opposite meaning)
         * 
         * scores[i][j] = Q[i] · K[j] = "How much should token i attend to j?"
         * 
         * This creates an attention pattern that is:
         * - CONTENT-BASED: depends on actual token values
         * - DYNAMIC: different for every input
         * - ALL-PAIRS: every token can attend to every other token
         */
        cached_scores = cached_Q * cached_K.transpose();
        
        if (verbose) {
            std::cout << "\n=== STEP 2: Attention Scores ===" << std::endl;
            std::cout << "WHY: Measure compatibility between all token pairs" << std::endl;
            cached_scores.print("Scores = Q × K^T");
        }
        
        // --------------------------------------------------------------------
        // STEP 3: Scale scores
        // --------------------------------------------------------------------
        /**
         * WHY SCALE BY √d_k?
         * 
         * When d_k is large, dot products grow with √d_k in expectation.
         * This pushes softmax into saturated regions where:
         * - Gradients are tiny (vanishing gradient problem)
         * - One score dominates completely (less nuanced attention)
         * 
         * Scaling by 1/√d_k keeps the variance ~1, which:
         * - Maintains healthy gradients
         * - Allows learning at any model size
         * 
         * For our d_k=2: scale = 1/√2 ≈ 0.707
         */
        cached_scaled = cached_scores * scale_factor;
        
        if (verbose) {
            std::cout << "\n=== STEP 3: Scaled Scores ===" << std::endl;
            std::cout << "WHY: Prevent softmax saturation, keep gradients healthy" << std::endl;
            std::cout << "Scale factor = 1/√2 ≈ " << scale_factor << std::endl;
            cached_scaled.print("Scaled = Scores × 0.707");
        }
        
        // --------------------------------------------------------------------
        // STEP 4: Apply softmax
        // --------------------------------------------------------------------
        /**
         * WHY SOFTMAX?
         * 
         * We need to combine information from multiple positions.
         * Softmax converts scores to a probability distribution:
         * 
         * softmax(x_i) = exp(x_i) / Σ exp(x_j)
         * 
         * Properties:
         * - All outputs are positive
         * - Outputs sum to 1 (like probabilities)
         * - Higher input → higher probability
         * - Differentiable (can learn through it)
         * 
         * The result tells us: "What fraction of attention should go to each token?"
         */
        cached_attn = softmax_rows(cached_scaled);
        
        if (verbose) {
            std::cout << "\n=== STEP 4: Attention Weights ===" << std::endl;
            std::cout << "WHY: Convert scores to probability distribution" << std::endl;
            cached_attn.print("Attention = softmax(Scaled)");
            std::cout << "Each row sums to 1.0" << std::endl;
        }
        
        // --------------------------------------------------------------------
        // STEP 5: Compute output
        // --------------------------------------------------------------------
        /**
         * WHY ATTENTION × V?
         * 
         * This computes a weighted sum of values:
         * output[i] = Σ_j attention[i][j] × V[j]
         * 
         * Each output position is a mixture of all value vectors,
         * weighted by how much attention it pays to each position.
         * 
         * This is the "information aggregation" step:
         * - Position i attends to all positions
         * - Gets information from each in proportion to attention weights
         * - Creates a context-aware representation
         * 
         * WHAT MAKES THIS POWERFUL:
         * The attention weights are computed FROM the content,
         * so the model learns WHICH information is relevant for each position.
         */
        cached_output = cached_attn * cached_V;
        
        if (verbose) {
            std::cout << "\n=== STEP 5: Attention Output ===" << std::endl;
            std::cout << "WHY: Aggregate information from attended positions" << std::endl;
            cached_output.print("Output = Attention × V");
        }
        
        return cached_output;
    }
    
    // ========================================================================
    // LOSS FUNCTION: Measuring prediction error
    // ========================================================================
    /**
     * Cross-Entropy Loss: The standard loss for classification/prediction
     * 
     * Loss = -Σ target[i] × log(prediction[i])
     * 
     * WHY CROSS-ENTROPY?
     * 
     * 1. Information-theoretic meaning: "How many bits to encode the target
     *    if we use the prediction as our probability model?"
     * 
     * 2. Penalizes confident wrong predictions: If target=1 and prediction=0.01,
     *    loss = -log(0.01) = 4.6 (very high!)
     * 
     * 3. Gradient is simple: For softmax + cross-entropy,
     *    gradient = prediction - target (elegant!)
     * 
     * 4. Maximum likelihood: Minimizing cross-entropy = maximizing likelihood
     *    of the training data under the model
     */
    double cross_entropy_loss(const Matrix2x2& prediction, const Matrix2x2& target) {
        double loss = 0.0;
        
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                if (target.at(i, j) > 0) {
                    // Add small epsilon to prevent log(0)
                    loss -= target.at(i, j) * std::log(prediction.at(i, j) + 1e-10);
                }
            }
        }
        
        return loss;
    }
    
    // ========================================================================
    // BACKPROPAGATION: Learning from errors
    // ========================================================================
    /**
     * Backpropagation computes gradients using the CHAIN RULE:
     * 
     * If y = f(g(x)), then dy/dx = dy/dg × dg/dx
     * 
     * For our transformer:
     * Loss → Output → Attention → Scaled → Scores → Q,K,V → W_Q, W_K, W_V
     * 
     * We propagate gradients backward through this chain to find:
     * dLoss/dW_Q, dLoss/dW_K, dLoss/dW_V
     * 
     * These gradients tell us: "How much does the loss change if we nudge each weight?"
     * 
     * WHY BACKPROP?
     * 
     * 1. Efficient: Computes ALL gradients in one backward pass
     * 2. Exact: No approximation, uses true derivatives  
     * 3. Composable: Works for any differentiable function composition
     */
    void backward(const Matrix2x2& dL_dOutput, bool verbose = false) {
        if (verbose) {
            std::cout << "\n=== BACKPROPAGATION ===" << std::endl;
            std::cout << "WHY: Compute gradients to know how to improve weights" << std::endl;
            dL_dOutput.print("dL/dOutput (gradient from loss)");
        }
        
        // --------------------------------------------------------------------
        // STEP 1: Gradient w.r.t. V (through Output = Attn × V)
        // --------------------------------------------------------------------
        /**
         * If Output = Attention × V, then:
         * dL/dV = Attention^T × dL/dOutput
         * 
         * WHY TRANSPOSE?
         * 
         * The chain rule for matrix multiplication A × B = C:
         * - dL/dA = dL/dC × B^T
         * - dL/dB = A^T × dL/dC
         * 
         * The transpose "reroutes" the gradient flow to match dimensions.
         * Intuitively: if A affects C through B, the gradient must flow
         * through B's transpose to get to A.
         */
        Matrix2x2 dL_dV = cached_attn.transpose() * dL_dOutput;
        
        if (verbose) {
            std::cout << "\n--- Gradient w.r.t. V ---" << std::endl;
            std::cout << "WHY: V contributes to output via attention × V" << std::endl;
            dL_dV.print("dL/dV = Attention^T × dL/dOutput");
        }
        
        // --------------------------------------------------------------------
        // STEP 2: Gradient w.r.t. Attention weights
        // --------------------------------------------------------------------
        /**
         * dL/dAttention = dL/dOutput × V^T
         * 
         * This tells us how much changing each attention weight affects the loss.
         * High gradient = that attention weight strongly affects the output.
         */
        Matrix2x2 dL_dAttn = dL_dOutput * cached_V.transpose();
        
        if (verbose) {
            std::cout << "\n--- Gradient w.r.t. Attention ---" << std::endl;
            std::cout << "WHY: Need this to backprop through softmax" << std::endl;
            dL_dAttn.print("dL/dAttn = dL/dOutput × V^T");
        }
        
        // --------------------------------------------------------------------
        // STEP 3: Gradient through softmax
        // --------------------------------------------------------------------
        /**
         * WHY IS SOFTMAX GRADIENT COMPLEX?
         * 
         * Softmax outputs are interdependent:
         * softmax(x_i) depends on ALL x_j (through the denominator)
         * 
         * The Jacobian is: 
         * d(softmax_i)/d(x_j) = softmax_i × (δ_ij - softmax_j)
         * 
         * where δ_ij = 1 if i=j, else 0
         * 
         * For each row, we compute:
         * dL/d(scaled_scores) = Σ_j dL/d(attn_j) × d(attn_j)/d(scaled_i)
         * 
         * Simplified: dL/d(scaled_i) = attn_i × (dL/d(attn_i) - Σ_j attn_j × dL/d(attn_j))
         */
        Matrix2x2 dL_dScaled;
        for (int i = 0; i < 2; i++) {
            // Sum of (attention × gradient) for this row
            double sum = 0.0;
            for (int j = 0; j < 2; j++) {
                sum += cached_attn.at(i, j) * dL_dAttn.at(i, j);
            }
            
            // Apply softmax gradient formula
            for (int j = 0; j < 2; j++) {
                dL_dScaled.at(i, j) = cached_attn.at(i, j) * (dL_dAttn.at(i, j) - sum);
            }
        }
        
        if (verbose) {
            std::cout << "\n--- Gradient through Softmax ---" << std::endl;
            std::cout << "WHY: Softmax couples all inputs, gradient is non-trivial" << std::endl;
            dL_dScaled.print("dL/dScaled");
        }
        
        // --------------------------------------------------------------------
        // STEP 4: Gradient through scaling
        // --------------------------------------------------------------------
        /**
         * If Scaled = Scores × scale_factor, then:
         * dL/dScores = dL/dScaled × scale_factor
         * 
         * Simple scalar multiplication: gradient scales by the same factor.
         */
        Matrix2x2 dL_dScores = dL_dScaled * scale_factor;
        
        if (verbose) {
            std::cout << "\n--- Gradient through Scaling ---" << std::endl;
            std::cout << "WHY: Simple - multiply gradient by scale factor" << std::endl;
            dL_dScores.print("dL/dScores = dL/dScaled × scale_factor");
        }
        
        // --------------------------------------------------------------------
        // STEP 5: Gradient w.r.t. Q and K (through Scores = Q × K^T)
        // --------------------------------------------------------------------
        /**
         * For Scores = Q × K^T:
         * dL/dQ = dL/dScores × K
         * dL/dK = dL/dScores^T × Q
         * 
         * WHY THESE FORMS?
         * 
         * The chain rule for matrix products propagates gradients
         * through the "other" operand (transposed appropriately).
         * 
         * Q and K both contribute to every attention score,
         * so both receive gradient from all scores.
         */
        Matrix2x2 dL_dQ = dL_dScores * cached_K;  // Note: K, not K^T, because we're undoing the transpose
        Matrix2x2 dL_dK = dL_dScores.transpose() * cached_Q;
        
        if (verbose) {
            std::cout << "\n--- Gradient w.r.t. Q and K ---" << std::endl;
            std::cout << "WHY: Q and K both affect attention scores" << std::endl;
            dL_dQ.print("dL/dQ = dL/dScores × K");
            dL_dK.print("dL/dK = dL/dScores^T × Q");
        }
        
        // --------------------------------------------------------------------
        // STEP 6: Gradient w.r.t. weight matrices
        // --------------------------------------------------------------------
        /**
         * For Q = X × W_Q:
         * dL/dW_Q = X^T × dL/dQ
         * 
         * This tells us exactly how to adjust each weight to reduce the loss.
         * 
         * WHY X^T?
         * 
         * W_Q connects input X to output Q.
         * The gradient must account for how X routes through W_Q.
         * X^T reorients the gradient to match W_Q's shape.
         */
        dW_Q = cached_X.transpose() * dL_dQ;
        dW_K = cached_X.transpose() * dL_dK;
        dW_V = cached_X.transpose() * dL_dV;
        
        if (verbose) {
            std::cout << "\n--- Gradient w.r.t. Weight Matrices ---" << std::endl;
            std::cout << "WHY: These are what we update to learn!" << std::endl;
            dW_Q.print("dW_Q = X^T × dL/dQ");
            dW_K.print("dW_K = X^T × dL/dK");
            dW_V.print("dW_V = X^T × dL/dV");
        }
    }
    
    // ========================================================================
    // WEIGHT UPDATE: Gradient Descent
    // ========================================================================
    /**
     * Update rule: W_new = W_old - learning_rate × gradient
     * 
     * WHY SUBTRACT?
     * 
     * The gradient points in the direction of INCREASING loss.
     * We want to DECREASE loss, so we go in the OPPOSITE direction.
     * 
     * WHY LEARNING RATE?
     * 
     * 1. Stability: Large steps might overshoot the minimum
     * 2. Precision: Small steps allow fine-tuning
     * 3. Escape: Not too small or we get stuck in local minima
     * 
     * Typical values: 0.001 to 0.1 (often decayed during training)
     */
    void update_weights(double learning_rate, bool verbose = false) {
        if (verbose) {
            std::cout << "\n=== WEIGHT UPDATE ===" << std::endl;
            std::cout << "WHY: Move weights opposite to gradient to reduce loss" << std::endl;
            std::cout << "Learning rate = " << learning_rate << std::endl;
        }
        
        W_Q = W_Q - dW_Q * learning_rate;
        W_K = W_K - dW_K * learning_rate;
        W_V = W_V - dW_V * learning_rate;
        
        if (verbose) {
            std::cout << "\nUpdated weights:" << std::endl;
            W_Q.print("W_Q (new)");
            W_K.print("W_K (new)");
            W_V.print("W_V (new)");
        }
    }
    
    // ========================================================================
    // COMPLETE TRAINING STEP
    // ========================================================================
    /**
     * One training step:
     * 1. Forward pass: compute prediction
     * 2. Loss: measure error
     * 3. Backward pass: compute gradients
     * 4. Update: adjust weights
     * 
     * WHY THIS ORDER?
     * 
     * - Forward must come first to compute the prediction
     * - Loss requires the prediction
     * - Backward uses cached values from forward
     * - Update uses gradients from backward
     * 
     * This cycle repeats thousands/millions of times during training!
     */
    double train_step(const Matrix2x2& input, const Matrix2x2& target, 
                      double learning_rate, bool verbose = false) {
        // Forward pass
        Matrix2x2 output = forward(input, verbose);
        
        // Apply softmax to get predictions (for classification)
        Matrix2x2 prediction = softmax_rows(output);
        
        if (verbose) {
            std::cout << "\n=== PREDICTION ===" << std::endl;
            prediction.print("Prediction = softmax(Output)");
            target.print("Target");
        }
        
        // Compute loss
        double loss = cross_entropy_loss(prediction, target);
        
        if (verbose) {
            std::cout << "\n=== LOSS ===" << std::endl;
            std::cout << "Cross-entropy loss = " << loss << std::endl;
        }
        
        // Compute gradient of loss w.r.t. output (softmax + CE simplification)
        /**
         * WHY IS THIS SO SIMPLE?
         * 
         * For softmax + cross-entropy, the gradient simplifies to:
         * dL/dOutput = prediction - target
         * 
         * This elegant result is one reason cross-entropy is popular:
         * - Gradient is never zero (always learning signal)
         * - Larger error = larger gradient (learns faster from mistakes)
         * - Mathematically beautiful!
         */
        Matrix2x2 dL_dOutput = prediction - target;
        
        // Backward pass
        backward(dL_dOutput, verbose);
        
        // Update weights
        update_weights(learning_rate, verbose);
        
        return loss;
    }
};

// ============================================================================
// MAIN: Demonstration
// ============================================================================
int main() {
    std::cout << "================================================================" << std::endl;
    std::cout << "   TINY TRANSFORMER: 2x2 Hand-Calculable Example" << std::endl;
    std::cout << "================================================================" << std::endl;
    
    TinyTransformer transformer;
    
    // Input: 2 tokens with 2-dimensional embeddings
    // Token 0: [1, 0] - could represent "hello"
    // Token 1: [0, 1] - could represent "world"
    Matrix2x2 input(1.0, 0.0,
                    0.0, 1.0);
    
    // Target: what should come after each token?
    // After token 0, predict token 1 (probability [0, 1])
    // After token 1, predict token 0 (probability [1, 0])
    Matrix2x2 target(0.0, 1.0,   // After "hello", expect "world"
                     1.0, 0.0);   // After "world", expect "hello"
    
    std::cout << "\n================================================================" << std::endl;
    std::cout << "   PART 1: INFERENCE (Forward Pass Only)" << std::endl;
    std::cout << "================================================================" << std::endl;
    
    input.print("\nInput X (2 tokens × 2 dimensions)");
    
    std::cout << "\nInitial weights:" << std::endl;
    transformer.W_Q.print("W_Q");
    transformer.W_K.print("W_K");
    transformer.W_V.print("W_V");
    
    // Run forward pass with verbose output
    std::cout << "\n>>> Running forward pass (inference)..." << std::endl;
    Matrix2x2 output = transformer.forward(input, true);
    
    std::cout << "\n================================================================" << std::endl;
    std::cout << "   PART 2: TRAINING (Forward + Backward + Update)" << std::endl;
    std::cout << "================================================================" << std::endl;
    
    // Reset transformer for training demo
    transformer = TinyTransformer();
    
    target.print("\nTarget (what we want to predict)");
    
    double learning_rate = 0.1;
    std::cout << "\n>>> Running one training step..." << std::endl;
    double loss = transformer.train_step(input, target, learning_rate, true);
    
    std::cout << "\n================================================================" << std::endl;
    std::cout << "   PART 3: TRAINING LOOP (Multiple Steps)" << std::endl;
    std::cout << "================================================================" << std::endl;
    
    // Reset and train for multiple steps
    transformer = TinyTransformer();
    
    std::cout << "\nTraining for 100 steps with learning_rate = " << learning_rate << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    
    for (int epoch = 0; epoch < 100; epoch++) {
        loss = transformer.train_step(input, target, learning_rate, false);
        
        if (epoch % 20 == 0 || epoch == 99) {
            std::cout << "Epoch " << std::setw(3) << epoch << ": Loss = " << loss << std::endl;
        }
    }
    
    std::cout << "\nFinal weights after training:" << std::endl;
    transformer.W_Q.print("W_Q (trained)");
    transformer.W_K.print("W_K (trained)");
    transformer.W_V.print("W_V (trained)");
    
    // Final inference
    std::cout << "\n>>> Final inference with trained model:" << std::endl;
    output = transformer.forward(input, false);
    Matrix2x2 final_pred = softmax_rows(output);
    
    final_pred.print("\nFinal predictions");
    target.print("Target");
    
    std::cout << "\nObservation: After training, predictions should be closer to target!" << std::endl;
    
    std::cout << "\n================================================================" << std::endl;
    std::cout << "   SUMMARY: The Matrix Core of Generative AI" << std::endl;
    std::cout << "================================================================" << std::endl;
    
    std::cout << R"(
The transformer's "matrix core" is:

1. WEIGHTS (learned parameters)
   - W_Q, W_K, W_V: transform inputs to attention components
   - These matrices STORE the model's knowledge

2. ACTIVATIONS (computed during forward pass)
   - Q = X × W_Q: queries ("what am I looking for?")
   - K = X × W_K: keys ("what do I contain?")
   - V = X × W_V: values ("what information to return?")

3. ATTENTION (dynamic information routing)
   - Scores = Q × K^T: compatibility between all pairs
   - Weights = softmax(Scores/√d): probability distribution
   - Output = Weights × V: weighted combination of values

4. LEARNING (gradient-based optimization)
   - Loss: how wrong is the prediction?
   - Gradients: which direction improves weights?
   - Update: W -= lr × gradient

All of this is MATRIX OPERATIONS, which is why:
- GPUs excel at this (parallel matrix multiplication)
- Transformers scale to billions of parameters
- The same architecture works for text, images, audio, etc.

)" << std::endl;

    return 0;
}
