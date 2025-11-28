#include "Embedding.hpp"
#include <iostream>
#include <iomanip>

Embedding::Embedding() {
    learnable = false;
    getDefaultEmbeddings(embeddings);
    zeroGrad();
}

Embedding::Embedding(double emb[4][2]) {
    learnable = false;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 2; j++) {
            embeddings[i][j] = emb[i][j];
        }
    }
    zeroGrad();
}

Matrix Embedding::forward(int token_idx) const {
    // Return as 1x2 matrix (single row)
    // For consistency with other operations, we'll return as 2x2 with second row zero
    Matrix result;
    result.set(0, 0, embeddings[token_idx][0]);
    result.set(0, 1, embeddings[token_idx][1]);
    result.set(1, 0, 0.0);
    result.set(1, 1, 0.0);
    return result;
}

Matrix Embedding::forwardSequence(const std::vector<int>& tokens) const {
    Matrix result;
    for (size_t i = 0; i < tokens.size() && i < 2; i++) {
        result.set(i, 0, embeddings[tokens[i]][0]);
        result.set(i, 1, embeddings[tokens[i]][1]);
    }
    // If only one token, fill second row with zeros
    if (tokens.size() == 1) {
        result.set(1, 0, 0.0);
        result.set(1, 1, 0.0);
    }
    return result;
}

void Embedding::getEmbedding(int token_idx, double out[2]) const {
    out[0] = embeddings[token_idx][0];
    out[1] = embeddings[token_idx][1];
}

void Embedding::setEmbedding(int token_idx, double emb[2]) {
    embeddings[token_idx][0] = emb[0];
    embeddings[token_idx][1] = emb[1];
}

void Embedding::zeroGrad() {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 2; j++) {
            grad[i][j] = 0.0;
        }
    }
}

void Embedding::print() const {
    std::cout << "Embeddings:\n";
    std::cout << std::fixed << std::setprecision(4);
    const char* tokens[] = {"A", "B", "C", "D"};
    for (int i = 0; i < 4; i++) {
        std::cout << "  " << tokens[i] << ": [" 
                  << embeddings[i][0] << ", " 
                  << embeddings[i][1] << "]\n";
    }
}

void Embedding::getDefaultEmbeddings(double emb[4][2]) {
    // A = [1, 0]
    emb[0][0] = 1.0; emb[0][1] = 0.0;
    // B = [0, 1]
    emb[1][0] = 0.0; emb[1][1] = 1.0;
    // C = [1, 1]
    emb[2][0] = 1.0; emb[2][1] = 1.0;
    // D = [0, 0]
    emb[3][0] = 0.0; emb[3][1] = 0.0;
}

