#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <codecvt>
#include <locale>
#include <unordered_map>

namespace fs = std::filesystem;

// ==================== UNICODE СЛОВАРЬ ====================
class UnicodeVocabulary {
private:
    std::unordered_map<char32_t, int> char_to_idx;
    std::vector<char32_t> idx_to_char;
    
public:
    // Конвертация UTF-8 в UTF-32
    static std::u32string utf8_to_utf32(const std::string& utf8) {
        std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
        try {
            return converter.from_bytes(utf8);
        } catch (...) {
            return U"";
        }
    }
    
    // Конвертация UTF-32 в UTF-8
    static std::string utf32_to_utf8(const std::u32string& utf32) {
        std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
        return converter.to_bytes(utf32);
    }
    
    void load(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Не могу открыть файл словаря: " + filename);
        }
        
        size_t vocab_size;
        file.read(reinterpret_cast<char*>(&vocab_size), sizeof(vocab_size));
        
        idx_to_char.resize(vocab_size);
        char_to_idx.clear();
        
        for (size_t i = 0; i < vocab_size; ++i) {
            uint32_t code_point;
            file.read(reinterpret_cast<char*>(&code_point), sizeof(code_point));
            idx_to_char[i] = static_cast<char32_t>(code_point);
            char_to_idx[idx_to_char[i]] = i;
        }
        
        file.close();
    }
    
    int get_index(char32_t c) const {
        auto it = char_to_idx.find(c);
        return it != char_to_idx.end() ? it->second : -1;
    }
    
    char32_t get_char(int idx) const {
        if (idx >= 0 && static_cast<size_t>(idx) < idx_to_char.size()) {
            return idx_to_char[idx];
        }
        return U'?';
    }
    
    size_t size() const { return idx_to_char.size(); }
    
    std::vector<int> text_to_indices(const std::string& text) {
        std::vector<int> indices;
        std::u32string unicode_text = utf8_to_utf32(text);
        
        for (char32_t c : unicode_text) {
            int idx = get_index(c);
            if (idx != -1) {
                indices.push_back(idx);
            }
        }
        
        return indices;
    }
    
    std::string indices_to_text(const std::vector<int>& indices) {
        std::u32string unicode_str;
        for (int idx : indices) {
            if (idx >= 0 && static_cast<size_t>(idx) < idx_to_char.size()) {
                unicode_str += idx_to_char[idx];
            }
        }
        return utf32_to_utf8(unicode_str);
    }
};

// ==================== ПРОСТАЯ МОДЕЛЬ ====================
class SimpleCharModel {
private:
    std::vector<float> weights1;  // [vocab_size, hidden_size]
    std::vector<float> weights2;  // [hidden_size, vocab_size]
    std::vector<float> bias1, bias2;
    
    size_t vocab_size;
    size_t hidden_size;
    int context_size;
    
public:
    SimpleCharModel() : vocab_size(0), hidden_size(0), context_size(50) {}
    
    void load(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Не могу открыть файл модели: " + filename);
        }
        
        file.read(reinterpret_cast<char*>(&vocab_size), sizeof(vocab_size));
        file.read(reinterpret_cast<char*>(&hidden_size), sizeof(hidden_size));
        
        weights1.resize(vocab_size * hidden_size);
        weights2.resize(hidden_size * vocab_size);
        bias1.resize(hidden_size);
        bias2.resize(vocab_size);
        
        file.read(reinterpret_cast<char*>(weights1.data()), 
                 weights1.size() * sizeof(float));
        file.read(reinterpret_cast<char*>(weights2.data()), 
                 weights2.size() * sizeof(float));
        file.read(reinterpret_cast<char*>(bias1.data()), 
                 bias1.size() * sizeof(float));
        file.read(reinterpret_cast<char*>(bias2.data()), 
                 bias2.size() * sizeof(float));
        
        file.close();
        
        std::cout << "  Загружена модель: " << vocab_size << " символов, "
                  << hidden_size << " нейронов" << std::endl;
    }
    
    // ReLU активация
    inline float relu(float x) const {
        return x > 0.0f ? x : 0.0f;
    }
    
    // Softmax
    void softmax(std::vector<float>& logits) const {
        float max_val = *std::max_element(logits.begin(), logits.end());
        float sum = 0.0f;
        
        for (float& val : logits) {
            val = expf(val - max_val);
            sum += val;
        }
        
        float inv_sum = 1.0f / sum;
        for (float& val : logits) {
            val *= inv_sum;
        }
    }
    
    // Прямой проход
    std::vector<float> forward(const std::vector<int>& context) {
        std::vector<float> hidden(hidden_size, 0.0f);
        
        // Усреднение one-hot векторов контекста
        float scale = 1.0f / context.size();
        
        for (int idx : context) {
            if (idx < 0 || static_cast<size_t>(idx) >= vocab_size) continue;
            
            const float* w_ptr = &weights1[idx * hidden_size];
            for (size_t j = 0; j < hidden_size; ++j) {
                hidden[j] += w_ptr[j] * scale;
            }
        }
        
        // ReLU
        for (size_t j = 0; j < hidden_size; ++j) {
            hidden[j] = relu(hidden[j] + bias1[j]);
        }
        
        // Выходной слой
        std::vector<float> output(vocab_size, 0.0f);
        for (size_t i = 0; i < vocab_size; ++i) {
            float sum = 0.0f;
            for (size_t j = 0; j < hidden_size; ++j) {
                sum += weights2[j * vocab_size + i] * hidden[j];
            }
            output[i] = sum + bias2[i];
        }
        
        softmax(output);
        return output;
    }
    
    int predict_next(const std::vector<int>& context) {
        if (context.empty()) return -1;
        
        auto output = forward(context);
        
        // Выбираем наиболее вероятный символ
        int best_idx = 0;
        for (size_t i = 1; i < output.size(); ++i) {
            if (output[i] > output[best_idx]) {
                best_idx = static_cast<int>(i);
            }
        }
        
        return best_idx;
    }
    
    size_t get_vocab_size() const { return vocab_size; }
    int get_context_size() const { return context_size; }
};

// ==================== ОСНОВНАЯ ФУНКЦИЯ ====================
int main() {
    try {
        std::cout << "==================== AvoAI Unicode Chat ====================" << std::endl;
        std::cout << "Поддерживает все Unicode символы" << std::endl;
        std::cout << "============================================================\n" << std::endl;
        
        // Проверяем наличие файлов
        if (!fs::exists("models/model_unicode.bin") || !fs::exists("models/vocab_unicode.bin")) {
            std::cerr << "Ошибка: файлы модели не найдены!" << std::endl;
            std::cerr << "Сначала обучите модель:" << std::endl;
            std::cerr << "  ./train_unicode" << std::endl;
            return 1;
        }
        
        // Загружаем словарь
        std::cout << "Загрузка словаря..." << std::endl;
        UnicodeVocabulary vocab;
        vocab.load("models/vocab_unicode.bin");
        std::cout << "  Загружено символов: " << vocab.size() << std::endl;
        
        // Загружаем модель
        std::cout << "Загрузка модели..." << std::endl;
        SimpleCharModel model;
        model.load("models/model_unicode.bin");
        
        // Определяем размер контекста из модели (должен быть 50 по умолчанию)
        int target_context = 50;  // Используем фиксированный размер для чата
        
        std::cout << "\nГотово к общению!" << std::endl;
        std::cout << "Размер контекста: " << target_context << " символов" << std::endl;
        std::cout << "Вводите текст на любом языке (для выхода: exit)\n" << std::endl;
        
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
                auto start = std::chrono::high_resolution_clock::now();
                
                // Конвертируем в индексы
                std::vector<int> context = vocab.text_to_indices(input);
                
                if (context.empty()) {
                    std::cout << "AvoAI: Не могу понять ваш запрос." << std::endl;
                    continue;
                }
                
                // Дополняем контекст если нужно
                if (context.size() < static_cast<size_t>(target_context)) {
                    while (context.size() < static_cast<size_t>(target_context) && !context.empty()) {
                        context.insert(context.begin(), context[0]);
                    }
                }
                
                if (context.size() < static_cast<size_t>(target_context)) {
                    std::cout << "AvoAI: Слишком короткий запрос." << std::endl;
                    continue;
                }
                
                // Берем последние target_context символов
                size_t start_pos = context.size() - target_context;
                std::vector<int> current_context(
                    context.begin() + static_cast<long>(start_pos),
                    context.end()
                );
                
                // Генерируем текст
                std::vector<int> generated;
                
                for (int i = 0; i < 100; ++i) {
                    int next_idx = model.predict_next(current_context);
                    
                    if (next_idx < 0 || static_cast<size_t>(next_idx) >= vocab.size()) {
                        break;
                    }
                    
                    generated.push_back(next_idx);
                    
                    // Обновляем контекст
                    current_context.erase(current_context.begin());
                    current_context.push_back(next_idx);
                    
                    // Проверяем на конец предложения
                    char32_t next_char = vocab.get_char(next_idx);
                    if (next_char == U'.' || next_char == U'!' || next_char == U'?' || 
                        next_char == U'\n' || next_char == U'。' || next_char == U'！') {
                        break;
                    }
                }
                
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
                
                std::string response = vocab.indices_to_text(generated);
                std::cout << "AvoAI: " << response << std::endl;
                std::cout << "[Сгенерировано за " << duration.count() << " мс]" << std::endl;
                
                // Показываем информацию о символах
                if (!response.empty()) {
                    std::u32string unicode_res = UnicodeVocabulary::utf8_to_utf32(response);
                    if (!unicode_res.empty()) {
                        std::cout << "[Символы: ";
                        for (size_t i = 0; i < std::min(static_cast<size_t>(3), unicode_res.size()); ++i) {
                            std::cout << "U+" << std::hex << static_cast<uint32_t>(unicode_res[i]) << std::dec << " ";
                        }
                        if (unicode_res.size() > 3) std::cout << "...";
                        std::cout << "]" << std::endl;
                    }
                }
                
                std::cout << std::endl;
                
            } catch (const std::exception& e) {
                std::cout << "Ошибка: " << e.what() << std::endl;
            }
        }
        
        std::cout << "\nДо новых встреч!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Критическая ошибка: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}