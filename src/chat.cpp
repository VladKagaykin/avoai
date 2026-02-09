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
#include <memory>

// ==================== ВЕРСИЯ С LIBTORCH (для CUDA моделей) ====================
#ifdef USE_LIBTORCH
#include <torch/torch.h>
#include <torch/script.h>

// Структура модели, идентичная той, что использовалась в train_cuda.cpp
struct NeuralNetImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};

    NeuralNetImpl(int64_t vocab_size, int64_t hidden_size)
        : fc1(register_module("fc1", torch::nn::Linear(vocab_size, hidden_size))),
          fc2(register_module("fc2", torch::nn::Linear(hidden_size, vocab_size))) {}

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1(x));
        x = fc2(x);
        return x;
    }
};
TORCH_MODULE(NeuralNet);

class PyTorchModel {
private:
    NeuralNet module = nullptr;
    torch::Device device = torch::Device(torch::kCPU); // Всегда CPU для чата
    int64_t vocab_size = 0;
    int64_t hidden_size = 0;
    
public:
    PyTorchModel(const std::string& model_path) {
        std::cout << "  Используем CPU для инференса (рекомендовано для чата)" << std::endl;
        
        try {
            // 1. Читаем метаданные
            std::string meta_path = "models/meta_cuda.txt";
            std::ifstream meta_file(meta_path);
            if (meta_file.is_open()) {
                std::string line;
                while (std::getline(meta_file, line)) {
                    std::istringstream iss(line);
                    std::string key;
                    int64_t value;
                    if (iss >> key >> value) {
                        if (key == "vocab_size") vocab_size = value;
                        if (key == "hidden_size") hidden_size = value;
                    }
                }
                meta_file.close();
            }
            
            if (vocab_size == 0 || hidden_size == 0) {
                throw std::runtime_error("Не удалось прочитать размеры модели из " + meta_path);
            }
            
            std::cout << "  Размеры модели: vocab=" << vocab_size 
                      << ", hidden=" << hidden_size << std::endl;
            
            // 2. Создаём модель с правильными размерами
            module = NeuralNet(vocab_size, hidden_size);
            
            // 3. Загружаем веса
            torch::load(module, model_path);
            
            module->eval();
            std::cout << "  Модель PyTorch загружена успешно" << std::endl;
            
        } catch (const c10::Error& e) {
            throw std::runtime_error("Ошибка загрузки модели PyTorch: " + 
                                    std::string(e.what()));
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Ошибка: ") + e.what());
        }
    } // ← ЗАКРЫВАЮЩАЯ СКОБКА КОНСТРУКТОРА
    
    int predict_next(const std::vector<int>& context) {
        if (context.empty() || !module) return -1;
        
        try {
            // Создаем тензор на CPU
            auto options = torch::TensorOptions().dtype(torch::kInt64);
            torch::Tensor cpu_tensor = torch::from_blob(
                const_cast<int*>(context.data()), 
                {1, static_cast<int64_t>(context.size())}, 
                options
            ).clone();
            
            // Преобразуем в one-hot
            auto one_hot = torch::nn::functional::one_hot(cpu_tensor, vocab_size)
                           .to(torch::kFloat32);
            
            // Усредняем по контексту
            auto input = one_hot.mean(/*dim=*/1);
            
            // Прямой проход
            torch::NoGradGuard no_grad;
            auto output = module->forward(input);
            
            return torch::argmax(output, 1).item<int>();
            
        } catch (const c10::Error& e) {
            std::cerr << "  Ошибка при инференсе: " << e.what() << std::endl;
            return -1;
        }
    }
};
#endif

// ==================== ОРИГИНАЛЬНЫЙ КЛАСС НЕЙРОСЕТИ (без CUDA) ====================
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
            if (idx < 0 || idx >= (int)vocab_size) continue;
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
        if (context.empty() || vocab_size == 0) return -1;
        
        std::vector<float> hidden, output;
        forward(context, hidden, output);
        
        if (output.empty()) return -1;
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
    size_t get_hidden_size() const { return hidden_size; }
};

// ==================== ОБНОВЛЁННЫЙ КЛАСС ЧАТА ====================
class ChatModel {
private:
    NeuralNetCPU original_model;
    #ifdef USE_LIBTORCH
    std::unique_ptr<PyTorchModel> pytorch_model;
    #endif
    
    std::vector<std::string> vocabulary;
    std::unordered_map<std::string, int> word_to_idx;
    
    bool use_pytorch_model = false;
    std::string model_type = "original";
    
public:
    ChatModel(const std::string& model_path, const std::string& vocab_path) {
        // Загружаем словарь
        load_vocabulary(vocab_path);
        
        // Определяем тип модели по расширению файла
        std::filesystem::path model_file(model_path);
        std::string extension = model_file.extension().string();
        
        if (extension == ".pt" || extension == ".pth") {
            #ifdef USE_LIBTORCH
            std::cout << "  Обнаружена модель PyTorch (.pt)" << std::endl;
            pytorch_model = std::make_unique<PyTorchModel>(model_path);
            use_pytorch_model = true;
            model_type = "pytorch";
            #else
            throw std::runtime_error(
                "Обнаружена модель PyTorch, но компиляция без поддержки LibTorch.\n"
                "Перекомпилируйте с флагом: g++ -DUSE_LIBTORCH ..."
            );
            #endif
        } else if (extension == ".bin") {
            std::cout << "  Обнаружена оригинальная модель (.bin)" << std::endl;
            original_model = NeuralNetCPU(vocabulary.size(), 256);
            original_model.load(model_path);
            use_pytorch_model = false;
            model_type = "original";
        } else {
            throw std::runtime_error(
                "Неизвестный формат модели. Поддерживаются:\n"
                "  - .bin (оригинальная бинарная модель)\n"
                "  - .pt/.pth (модель PyTorch)"
            );
        }
    }
    
    void load_vocabulary(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error("Не могу открыть файл словаря: " + path);
        }
        
        std::string line;
        int idx = 0;
        
        while (std::getline(file, line)) {
            if (!line.empty()) {
                // Поддерживаем оба формата: просто слово или "слово\tиндекс"
                std::string word;
                size_t tab_pos = line.find('\t');
                if (tab_pos != std::string::npos) {
                    word = line.substr(0, tab_pos);
                } else {
                    word = line;
                }
                
                vocabulary.push_back(word);
                word_to_idx[word] = idx++;
            }
        }
        
        file.close();
        
        if (vocabulary.empty()) {
            throw std::runtime_error("Словарь пуст!");
        }
        
        std::cout << "  Загружено слов в словаре: " << vocabulary.size() << std::endl;
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
            context.end() - std::min<int>(CONTEXT_SIZE, context.size()),
            context.end()
        );
        
        std::string result;
        
        for (int i = 0; i < max_words; ++i) {
            int next_word_idx = -1;
            
            if (use_pytorch_model) {
                #ifdef USE_LIBTORCH
                // БЫЛО: next_word_idx = pytorch_model->predict_next(current_context, vocab_size);
                // СТАЛО: передаем только контекст
                next_word_idx = pytorch_model->predict_next(current_context);
                #endif
            } else {
                next_word_idx = original_model.predict_next(current_context);
            }
            
            if (next_word_idx < 0 || next_word_idx >= (int)vocabulary.size()) {
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
        }
        
        return result;
    }
    
    size_t get_vocabulary_size() const {
        return vocabulary.size();
    }
    
    std::string get_model_type() const {
        return model_type;
    }
};

// ==================== ФУНКЦИЯ ДЛЯ ПОИСКА ФАЙЛОВ МОДЕЛИ ====================
std::pair<std::string, std::string> find_model_files() {
    std::vector<std::pair<std::string, std::string>> candidates = {
        {"models/model_cuda.pt", "models/vocab_cuda.txt"},
        {"models/model.bin", "models/vocab.txt"},
        {"model_cuda.pt", "vocab_cuda.txt"},
        {"model.bin", "vocab.txt"},
        {"../models/model_cuda.pt", "../models/vocab_cuda.txt"},
        {"../models/model.bin", "../models/vocab.txt"}
    };
    
    for (const auto& [model_path, vocab_path] : candidates) {
        std::ifstream model_file(model_path);
        std::ifstream vocab_file(vocab_path);
        
        if (model_file.good() && vocab_file.good()) {
            model_file.close();
            vocab_file.close();
            std::cout << "Найдена модель: " << model_path << std::endl;
            std::cout << "Найден словарь: " << vocab_path << std::endl;
            return {model_path, vocab_path};
        }
    }
    
    throw std::runtime_error(
        "Не найдены файлы модели и словаря. Разместите файлы в одной из папок:\n"
        "  - models/model_cuda.pt + models/vocab_cuda.txt\n"
        "  - models/model.bin + models/vocab.txt"
    );
}

// ==================== ОСНОВНАЯ ФУНКЦИЯ ====================
int main() {
    try {
        std::cout << "==================== AvoAI Chat ====================" << std::endl;
        std::cout << "Поиск и загрузка модели..." << std::endl;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Автоматически ищем файлы модели
        auto [model_path, vocab_path] = find_model_files();
        
        // Создаем модель чата
        ChatModel chat(model_path, vocab_path);
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        std::cout << "Модель загружена за " << duration.count() << " мс" << std::endl;
        std::cout << "Тип модели: " << chat.get_model_type() << std::endl;
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
                auto gen_start = std::chrono::high_resolution_clock::now();
                std::string response = chat.generate_text(input, 25);
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
        std::cerr << "Критическая ошибка: " << e.what() << std::endl;
        std::cerr << "Убедитесь, что файлы модели и словаря существуют" << std::endl;
        return 1;
    }
    
    return 0;
}