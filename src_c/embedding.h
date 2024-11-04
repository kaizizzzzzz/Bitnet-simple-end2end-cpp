#ifndef EMBEDDING_H
#define EMBEDDING_H

#include <iostream>
#include <vector>
#include <random>
#include <stdexcept>

class Embedding {
public:
    // Constructor to initialize with provided embedding table
    Embedding(const std::vector<std::vector<float>>& embedding_table) 
        : vocab_size(embedding_table.size()), 
          embedding_dim(embedding_table[0].size()), 
          embedding_matrix(embedding_table) {

        // Validate that all rows have the same embedding dimension
        for (const auto& row : embedding_table) {
            if (row.size() != embedding_dim) {
                throw std::invalid_argument("All rows in the embedding table must have the same dimension.");
            }
        }
    }

    // Forward pass: Lookup embeddings for a list of indices
    std::vector<std::vector<float>> forward(const std::vector<size_t>& indices) {
        std::vector<std::vector<float>> embeddings;
        embeddings.reserve(indices.size());

        for (size_t idx : indices) {
            if (idx >= vocab_size) {
                throw std::out_of_range("Index out of range in embedding lookup.");
            }
            embeddings.push_back(embedding_matrix[idx]);
        }
        return embeddings;
    }

private:
    size_t vocab_size;
    size_t embedding_dim;
    std::vector<std::vector<float>> embedding_matrix;
};

#endif // EMBEDDING_H
