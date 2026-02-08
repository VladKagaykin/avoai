#include <iostream>
#include <vector>
#include <string>
#include <unordered_map>
#include <cmath>
#include <random>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <chrono>

// ==================== Класс нейросети ====================
class NeuralNet {
private:
    std::vector<std::vector<float>> weights1;
    std::vector<std::vector<float>> weights2;
    std::vector<float> bias1, bias2;
    
    size_t vocab_size;
    size_t hidden_size;
    float learning_rate;
    
public:
    NeuralNet(size_t vocab_size, size_t hidden_size = 128, float lr = 0.01f)
        : vocab_size(vocab_size), hidden_size(hidden_size), learning_rate(lr) {}
    
    inline float sigmoid(float x) {
        return 1.0f / (1.0f + expf(-x));
    }
    
    std::vector<float> softmax(const std::vector<float>& logits) {
        std::vector<float> exp_values(logits.size());
        float max_val = *std::max_element(logits.begin(), logits.end());
        float sum = 0.0f;
        
        for (size_t i = 0; i < logits.size(); ++i) {
            exp_values[i] = expf(logits[i] - max_val);
            sum += exp_values[i];
        }
        
        for (size_t i = 0; i < exp_values.size(); ++i) {
            exp_values[i] /= sum;
        }
        
        return exp_values;
    }
    
    void forward(const std::vector<int>& input,
                 std::vector<float>& hidden,
                 std::vector<float>& output) {
        
        hidden.assign(hidden_size, 0.0f);
        
        for (size_t i = 0; i < input.size(); ++i) {
            int idx = input[i];
            for (size_t j = 0; j < hidden_size; ++j) {
                hidden[j] += weights1[j][idx];
            }
        }
        
        for (size_t i = 0; i < hidden_size; ++i) {
            hidden[i] = sigmoid(hidden[i] + bias1[i]);
        }
        
        std::vector<float> logits(vocab_size, 0.0f);
        
        for (size_t i = 0; i < vocab_size; ++i) {
            float sum = 0.0f;
            for (size_t j = 0; j < hidden_size; ++j) {
                sum += weights2[i][j] * hidden[j];
            }
            logits[i] = sum + bias2[i];
        }
        
        output = softmax(logits);
    }
    
    int predict_next(const std::vector<int>& context) {
        std::vector<float> hidden, output;
        forward(context, hidden, output);
        
        return std::distance(output.begin(), 
                           std::max_element(output.begin(), output.end()));
    }
    
    void load(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        
        if (!file.is_open()) {
            throw std::runtime_error("Не могу открыть файл модели: " + filename);
        }
        
        file.read(reinterpret_cast<char*>(&vocab_size), sizeof(vocab_size));
        file.read(reinterpret_cast<char*>(&hidden_size), sizeof(hidden_size));
        file.read(reinterpret_cast<char*>(&learning_rate), sizeof(learning_rate));
        
        weights1.resize(hidden_size);
        for (auto& row : weights1) {
            row.resize(vocab_size);
            file.read(reinterpret_cast<char*>(row.data()), 
                     row.size() * sizeof(float));
        }
        
        weights2.resize(vocab_size);
        for (auto& row : weights2) {
            row.resize(hidden_size);
            file.read(reinterpret_cast<char*>(row.data()), 
                     row.size() * sizeof(float));
        }
        
        bias1.resize(hidden_size);
        file.read(reinterpret_cast<char*>(bias1.data()), 
                 bias1.size() * sizeof(float));
        
        bias2.resize(vocab_size);
        file.read(reinterpret_cast<char*>(bias2.data()), 
                 bias2.size() * sizeof(float));
        
        file.close();
    }
    
    size_t get_vocab_size() const { return vocab_size; }
};

// ==================== Класс чата ====================
class ChatModel {
private:
    NeuralNet model;
    std::vector<std::string> vocabulary;
    std::unordered_map<std::string, int> word_to_idx;
    
public:
    ChatModel(const std::string& model_path, const std::string& vocab_path)
        : model(1, 1) {
        load_vocabulary(vocab_path);
        model = NeuralNet(vocabulary.size(), 256);
        model.load(model_path);
    }
    
    void load_vocabulary(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error("Не могу открыть файл словаря: " + path);
        }
        
        std::string word;
        int idx = 0;
        
        while (std::getline(file, word)) {
            if (!word.empty()) {
                vocabulary.push_back(word);
                word_to_idx[word] = idx++;
            }
        }
        
        file.close();
    }
    
    std::vector<int> text_to_indices(const std::string& text) {
        std::vector<int> indices;
        std::stringstream ss(text);
        std::string word;
        
        while (ss >> word) {
            std::transform(word.begin(), word.end(), word.begin(), ::tolower);
            
            auto it = word_to_idx.find(word);
            if (it != word_to_idx.end()) {
                indices.push_back(it->second);
            }
        }
        
        return indices;
    }
    
    std::string generate_text(const std::string& prompt, int max_words = 30) {
        std::vector<int> context = text_to_indices(prompt);
        
        if (context.empty()) {
            return "Не могу понять ваш запрос. Попробуйте другие слова.";
        }
        
        const int CONTEXT_SIZE = 3;
        
        if (context.size() < CONTEXT_SIZE) {
            while (context.size() < CONTEXT_SIZE && !context.empty()) {
                context.insert(context.begin(), context[0]);
            }
        }
        
        if (context.size() < CONTEXT_SIZE) {
            return "Слишком мало слов для анализа.";
        }
        
        std::string result;
        std::vector<int> current_context(context.end() - CONTEXT_SIZE, context.end());
        
        for (int i = 0; i < max_words; ++i) {
            int next_word_idx = model.predict_next(current_context);
            
            if (next_word_idx < 0 || next_word_idx >= vocabulary.size()) {
                break;
            }
            
            std::string next_word = vocabulary[next_word_idx];
            
            if (i == 0) {
                result = next_word;
            } else {
                result += " " + next_word;
            }
            
            current_context.erase(current_context.begin());
            current_context.push_back(next_word_idx);
            
            if (next_word.find('.') != std::string::npos || 
                next_word.find('!') != std::string::npos ||
                next_word.find('?') != std::string::npos) {
                break;
            }
        }
        
        return result;
    }
    
    // Геттер для получения размера словаря
    size_t get_vocabulary_size() const {
        return vocabulary.size();
    }
};

// ==================== Основная функция ====================
int main() {
    try {
        std::cout << "==================== AvoAI Chat ====================" << std::endl;
        std::cout << "Загрузка модели и словаря..." << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        ChatModel chat("models/model.bin", "models/vocab.txt");
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Модель загружена за " << duration.count() << " мс" << std::endl;
        std::cout << "Размер словаря: " << chat.get_vocabulary_size() << " слов" << std::endl;
        std::cout << "\nНачинайте общение (для выхода введите 'exit' или 'quit'):" << std::endl;
        std::cout << "====================================================\n" << std::endl;
        
        std::string input;
        while (true) {
            std::cout << ">>> ";
            std::getline(std::cin, input);
            
            if (input == "exit" || input == "quit" || input == "выход") {
                break;
            }
            
            if (input.empty()) {
                continue;
            }
            
            try {
                std::string response = chat.generate_text(input, 25);
                std::cout << "AvoAI: " << response << std::endl << std::endl;
            } catch (const std::exception& e) {
                std::cout << "Ошибка генерации: " << e.what() << std::endl;
            }
        }
        
        std::cout << "\nДо новых встреч!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Критическая ошибка: " << e.what() << std::endl;
        std::cerr << "Убедитесь, что файлы models/model.bin и models/vocab.txt существуют" << std::endl;
        return 1;
    }
    
    return 0;
}