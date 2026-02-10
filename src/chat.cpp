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
#include <filesystem>

namespace fs = std::filesystem;

// ==================== ОРИГИНАЛЬНАЯ CPU МОДЕЛЬ ====================
class NeuralNetCPU {
private:
    std::vector<std::vector<float>> weights1;
    std::vector<std::vector<float>> weights2;
    std::vector<float> bias1, bias2;
    
    size_t vocab_size;
    size_t hidden_size;
    float learning_rate;
    
public:
    NeuralNetCPU(size_t vocab_size = 0, size_t hidden_size = 128, float lr = 0.01f)
        : vocab_size(vocab_size), hidden_size(hidden_size), learning_rate(lr) {}
    
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
        
        // Суммируем эмбеддинги слов контекста
        for (size_t i = 0; i < input.size(); ++i) {
            int idx = input[i];
            if (idx < 0 || static_cast<size_t>(idx) >= vocab_size) continue;
            for (size_t j = 0; j < hidden_size; ++j) {
                hidden[j] += weights1[j][idx];
            }
        }
        
        // Усредняем и применяем ReLU (как в train_cuda)
        if (!input.empty()) {
            float scale = 1.0f / input.size();
            for (size_t i = 0; i < hidden_size; ++i) {
                hidden[i] = hidden[i] * scale + bias1[i];
                // ReLU активация
                if (hidden[i] < 0) hidden[i] = 0.0f;
            }
        }
        
        // Выходной слой
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
        if (context.empty() || vocab_size == 0) return -1;
        
        std::vector<float> hidden, output;
        forward(context, hidden, output);
        
        if (output.empty()) return -1;
        
        // Выбираем слово с максимальной вероятностью
        int best_idx = 0;
        float best_prob = output[0];
        for (size_t i = 1; i < output.size(); ++i) {
            if (output[i] > best_prob) {
                best_prob = output[i];
                best_idx = i;
            }
        }
        return best_idx;
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
        
        // Проверяем загруженные веса
        std::cout << "  Загружено параметров: " 
                  << weights1.size() * weights1[0].size() + 
                     weights2.size() * weights2[0].size() +
                     bias1.size() + bias2.size() << std::endl;
    }
    
    size_t get_vocab_size() const { return vocab_size; }
};

// ==================== КЛАСС ЧАТА ====================
class ChatModel {
private:
    NeuralNetCPU model;
    std::vector<std::string> vocabulary;
    std::unordered_map<std::string, int> word_to_idx;
    
public:
    ChatModel(const std::string& model_path, const std::string& vocab_path) {
        // Загружаем словарь
        load_vocabulary(vocab_path);
        
        // Загружаем модель
        model = NeuralNetCPU(vocabulary.size(), 256);
        model.load(model_path);
        
        std::cout << "  Размер словаря модели: " << model.get_vocab_size() << std::endl;
    }
    
    void load_vocabulary(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error("Не могу открыть файл словаря: " + path);
        }
        
        std::string line;
        int idx = 0;
        
        // Простой формат: по одному слову на строку
        while (std::getline(file, line)) {
            if (!line.empty()) {
                // Убираем пробелы
                line.erase(0, line.find_first_not_of(" \t\n\r"));
                line.erase(line.find_last_not_of(" \t\n\r") + 1);
                
                if (!line.empty()) {
                    vocabulary.push_back(line);
                    word_to_idx[line] = idx++;
                }
            }
        }
        
        file.close();
        
        if (vocabulary.empty()) {
            throw std::runtime_error("Словарь пуст!");
        }
        
        std::cout << "  Загружено слов в словаре: " << vocabulary.size() << std::endl;
        
        // Выводим примеры слов для проверки
        std::cout << "  Примеры слов: ";
        for (int i = 0; i < std::min(10, static_cast<int>(vocabulary.size())); ++i) {
            std::cout << vocabulary[i] << " ";
        }
        std::cout << std::endl;
    }
    
    std::vector<int> text_to_indices(const std::string& text) {
        std::vector<int> indices;
        std::stringstream ss(text);
        std::string word;
        
        while (ss >> word) {
            // Приводим к нижнему регистру
            std::transform(word.begin(), word.end(), word.begin(), ::tolower);
            
            auto it = word_to_idx.find(word);
            if (it != word_to_idx.end()) {
                indices.push_back(it->second);
            } else {
                // Ищем слово без пунктуации
                std::string clean_word;
                for (char c : word) {
                    if (!std::ispunct(c)) {
                        clean_word += c;
                    }
                }
                if (!clean_word.empty()) {
                    auto it2 = word_to_idx.find(clean_word);
                    if (it2 != word_to_idx.end()) {
                        indices.push_back(it2->second);
                    } else {
                        std::cout << "  [Слово не найдено: '" << word << "'] ";
                    }
                }
            }
        }
        
        return indices;
    }
    
    std::string generate_text(const std::string& prompt, int max_words = 30) {
        std::vector<int> context = text_to_indices(prompt);
        
        std::cout << "  Контекст (" << context.size() << " слов): ";
        for (int idx : context) {
            if (idx >= 0 && static_cast<size_t>(idx) < vocabulary.size()) {
                std::cout << vocabulary[idx] << "(" << idx << ") ";
            } else {
                std::cout << "?" << idx << "? ";
            }
        }
        std::cout << std::endl;
        
        if (context.empty()) {
            return "Не могу понять ваш запрос. Попробуйте другие слова.";
        }
        
        const int CONTEXT_SIZE = 3;
        
        // Дополняем контекст если нужно
        if (context.size() < CONTEXT_SIZE) {
            while (context.size() < CONTEXT_SIZE && !context.empty()) {
                context.insert(context.begin(), context[0]);
            }
        }
        
        if (context.size() < CONTEXT_SIZE) {
            return "Слишком мало слов для анализа.";
        }
        
        // Берём последние CONTEXT_SIZE слов
        std::vector<int> current_context(
            context.end() - std::min<int>(CONTEXT_SIZE, static_cast<int>(context.size())),
            context.end()
        );
        
        std::string result;
        
        for (int i = 0; i < max_words; ++i) {
            int next_word_idx = model.predict_next(current_context);
            
            if (next_word_idx < 0 || static_cast<size_t>(next_word_idx) >= vocabulary.size()) {
                std::cout << "  Ошибка: idx=" << next_word_idx 
                         << " при размере словаря=" << vocabulary.size() << std::endl;
                if (i == 0) {
                    result = "[ошибка генерации]";
                }
                break;
            }
            
            std::string next_word = vocabulary[next_word_idx];
            
            if (i == 0) {
                result = next_word;
            } else {
                result += " " + next_word;
            }
            
            // Обновляем контекст
            current_context.erase(current_context.begin());
            current_context.push_back(next_word_idx);
            
            // Завершаем если встретили знак конца предложения
            if (next_word.find('.') != std::string::npos || 
                next_word.find('!') != std::string::npos ||
                next_word.find('?') != std::string::npos) {
                break;
            }
            
            // Ограничиваем длину
            if (result.size() > 200) {
                break;
            }
        }
        
        return result;
    }
    
    size_t get_vocabulary_size() const {
        return vocabulary.size();
    }
};

// ==================== ОСНОВНАЯ ФУНКЦИЯ ====================
int main() {
    try {
        std::cout << "==================== AvoAI Chat ====================" << std::endl;
        std::cout << "Универсальная версия (работает с .bin моделями)" << std::endl;
        
        // Проверяем наличие файлов модели
        if (!fs::exists("models/model.bin") || !fs::exists("models/vocab.txt")) {
            std::cerr << "Ошибка: файлы модели не найдены!" << std::endl;
            std::cerr << "Сначала обучите модель:" << std::endl;
            std::cerr << "  ./train_cuda  (для CUDA)" << std::endl;
            std::cerr << "  ИЛИ" << std::endl;
            std::cerr << "  ./train       (для CPU)" << std::endl;
            return 1;
        }
        
        std::cout << "Загрузка модели..." << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Загружаем модель
        ChatModel chat("models/model.bin", "models/vocab.txt");
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Модель загружена за " << duration.count() << " мс" << std::endl;
        std::cout << "Размер словаря: " << chat.get_vocabulary_size() << " слов" << std::endl;
        std::cout << "\nНачинайте общение (для выхода введите 'exit'):" << std::endl;
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
                auto gen_start = std::chrono::high_resolution_clock::now();
                std::string response = chat.generate_text(input, 20);
                auto gen_end = std::chrono::high_resolution_clock::now();
                auto gen_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                    gen_end - gen_start
                );
                
                std::cout << "AvoAI: " << response << std::endl;
                std::cout << "[Сгенерировано за " << gen_time.count() << " мс]" 
                          << std::endl << std::endl;
            } catch (const std::exception& e) {
                std::cout << "Ошибка генерации: " << e.what() << std::endl;
            }
        }
        
        std::cout << "\nДо новых встреч!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Ошибка: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}