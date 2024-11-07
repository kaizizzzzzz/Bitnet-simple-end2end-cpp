#include <iostream>
#include <vector>
#include <stdexcept>
#include "../scr/embedding.h"  // Include the header with the Embedding class

int main() {
    try {
        // Define a small embedding table for testing
        std::vector<std::vector<float>> embedding_table = {
            {0.1f, 0.2f, 0.3f},
            {0.4f, 0.5f, 0.6f},
            {0.7f, 0.8f, 0.9f}
        };

        // Initialize the Embedding object with the table
        Embedding embedding(embedding_table);
        std::cout << "Embedding initialized successfully.\n";

        // Test the forward method with valid indices
        std::vector<size_t> indices = {0, 2};
        std::vector<std::vector<float>> result = embedding.forward(indices);

        std::cout << "Embedding lookup results:\n";
        for (const auto& vec : result) {
            for (float value : vec) {
                std::cout << value << " ";
            }
            std::cout << "\n";
        }

        // Test with an out-of-range index
        std::cout << "\nTesting with an out-of-range index...\n";
        try {
            std::vector<size_t> invalid_indices = {0, 3};  // 3 is out of range
            embedding.forward(invalid_indices);
        } catch (const std::out_of_range& e) {
            std::cout << "Caught expected out_of_range exception: " << e.what() << "\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
