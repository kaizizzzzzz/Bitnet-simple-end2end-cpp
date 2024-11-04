#ifndef RMS_NORM_H
#define RMS_NORM_H

#include <vector>
#include <cstdint>

// RMS Normalization function
std::vector<float> rms_norm(const std::vector<float> &hidden_states, const std::vector<float> &weight, float epsilon = 1e-6) {
    size_t hidden_size = hidden_states.size();
    float variance = 0.0f;

    for (float val : hidden_states) {
        variance += val * val;
    }

    variance = std::sqrt(variance / hidden_size + epsilon);
    std::vector<float> normalized_states(hidden_size);

    for (size_t i = 0; i < hidden_size; ++i) {
        normalized_states[i] = (hidden_states[i] / variance) * weight[i];
    }

    return normalized_states;
}

#endif // RMS_NORM_H