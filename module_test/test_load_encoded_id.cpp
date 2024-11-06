#include "../inference/inference.cpp"

int main() {
    std::string file_name = "encoded_prompt.bin";
    std::vector<float> encoded_id = get_encoded_id(file_name);
    // inference();
    return 0;
}