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
#include <iomanip>
#include <filesystem>
#include <mutex>
#include <omp.h>

namespace fs = std::filesystem;

// ==================== Класс словаря ====================
class Vocabulary {
private:
    std::unordered_map<std::string, int> word_to_idx;
    std::vector<std::string> idx_to_word;
    std::mutex mtx;
    int next_idx = 0;
    
public:
    int add_word(const std::string& word) {
        std::lock_guard<std::mutex> lock(mtx);
        auto it = word_to_idx.find(word);
        if (it != word_to_idx.end()) return it->second;
        
        int idx = next_idx++;
        word_to_idx[word] = idx;
        if (idx >= idx_to_word.size()) {
            idx_to_word.resize(idx + 1);
        }
        idx_to_word[idx] = word;
        return idx;
    }
    
    int get_index(const std::string& word) const {
        auto it = word_to_idx.find(word);
        return it != word_to_idx.end() ? it->second : -1;
    }
    
    const std::string& get_word(int idx) const {
        static const std::string empty = "";
        if (idx >= 0 && idx < idx_to_word.size()) {
            return idx_to_word[idx];
        }
        return empty;
    }
    
    size_t size() const { return word_to_idx.size(); }
    
    const std::vector<std::string>& get_words() const { return idx_to_word; }
};

// ==================== Класс нейросети (упрощенный) ====================
class NeuralNet {
private:
    std::vector<std::vector<float>> weights1;
    std::vector<std::vector<float>> weights2;
    std::vector<float> bias1, bias2;
    
    size_t vocab_size;
    size_t hidden_size;
    float learning_rate;
    
    std::mt19937 rng;
    
public:
    NeuralNet(size_t vocab_size, size_t hidden_size = 128, float lr = 0.01f)
        : vocab_size(vocab_size), hidden_size(hidden_size), learning_rate(lr) {
        
        rng.seed(std::random_device{}());
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
        
        std::cout << "Инициализация нейросети..." << std::endl;
        std::cout << "Размер словаря: " << vocab_size << std::endl;
        std::cout << "Скрытый слой: " << hidden_size << " нейронов" << std::endl;
        
        // Инициализация весов
        weights1.resize(hidden_size, std::vector<float>(vocab_size));
        weights2.resize(vocab_size, std::vector<float>(hidden_size));
        
        // Инициализируем веса в одном потоке для простоты
        for (size_t i = 0; i < hidden_size; ++i) {
            for (size_t j = 0; j < vocab_size; ++j) {
                weights1[i][j] = dist(rng) / sqrtf(vocab_size);
            }
        }
        
        for (size_t i = 0; i < vocab_size; ++i) {
            for (size_t j = 0; j < hidden_size; ++j) {
                weights2[i][j] = dist(rng) / sqrtf(hidden_size);
            }
        }
        
        bias1.resize(hidden_size, 0.0f);
        bias2.resize(vocab_size, 0.0f);
        
        std::cout << "Инициализация завершена!" << std::endl;
    }
    
    inline float sigmoid(float x) {
        return 1.0f / (1.0f + expf(-x));
    }
    
    inline float sigmoid_derivative(float x) {
        return x * (1.0f - x);
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
    
    float train_step(const std::vector<int>& input,
                     int target,
                     float lr) {
        
        std::vector<float> hidden, output;
        forward(input, hidden, output);
        
        // Вычисление потерь
        float loss = -logf(output[target] + 1e-8f);
        
        // Градиенты выходного слоя
        std::vector<float> delta2(vocab_size);
        delta2[target] = -1.0f;
        
        for (size_t i = 0; i < vocab_size; ++i) {
            delta2[i] += output[i];
            
            for (size_t j = 0; j < hidden_size; ++j) {
                weights2[i][j] -= lr * delta2[i] * hidden[j];
            }
            bias2[i] -= lr * delta2[i];
        }
        
        // Градиенты скрытого слоя
        std::vector<float> delta1(hidden_size, 0.0f);
        
        for (size_t j = 0; j < hidden_size; ++j) {
            float sum = 0.0f;
            for (size_t i = 0; i < vocab_size; ++i) {
                sum += weights2[i][j] * delta2[i];
            }
            delta1[j] = sum * sigmoid_derivative(hidden[j]);
            
            for (size_t k = 0; k < input.size(); ++k) {
                int idx = input[k];
                weights1[j][idx] -= lr * delta1[j];
            }
            bias1[j] -= lr * delta1[j];
        }
        
        return loss;
    }
    
    int predict_next(const std::vector<int>& context) {
        std::vector<float> hidden, output;
        forward(context, hidden, output);
        
        return std::distance(output.begin(), 
                           std::max_element(output.begin(), output.end()));
    }
    
    void save(const std::string& filename) {
        std::cout << "Сохранение модели в " << filename << "..." << std::endl;
        
        std::ofstream file(filename, std::ios::binary);
        
        file.write(reinterpret_cast<const char*>(&vocab_size), sizeof(vocab_size));
        file.write(reinterpret_cast<const char*>(&hidden_size), sizeof(hidden_size));
        file.write(reinterpret_cast<const char*>(&learning_rate), sizeof(learning_rate));
        
        for (const auto& row : weights1) {
            file.write(reinterpret_cast<const char*>(row.data()), 
                      row.size() * sizeof(float));
        }
        
        for (const auto& row : weights2) {
            file.write(reinterpret_cast<const char*>(row.data()), 
                      row.size() * sizeof(float));
        }
        
        file.write(reinterpret_cast<const char*>(bias1.data()), 
                  bias1.size() * sizeof(float));
        file.write(reinterpret_cast<const char*>(bias2.data()), 
                  bias2.size() * sizeof(float));
        
        file.close();
        std::cout << "Модель сохранена!" << std::endl;
    }
    
    void load(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        
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
    
    std::mt19937& get_rng() { return rng; }
};

// ==================== Вспомогательные функции ====================
void print_progress_bar(float progress, int bar_width = 50, const std::string& label = "") {
    int pos = bar_width * progress;
    std::cout << "\r" << label << " [";
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << std::setw(3) << int(progress * 100.0) << "%";
    std::cout.flush();
}

std::vector<std::string> load_all_texts(const std::string& data_dir) {
    std::vector<std::string> all_words;
    
    if (!fs::exists(data_dir)) {
        std::cerr << "Ошибка: директория " << data_dir << " не существует!" << std::endl;
        return all_words;
    }
    
    std::vector<std::string> files;
    for (const auto& entry : fs::directory_iterator(data_dir)) {
        if (entry.path().extension() == ".txt") {
            files.push_back(entry.path().string());
            std::cout << "  Найден: " << entry.path().filename() << std::endl;
        }
    }
    
    if (files.empty()) {
        std::cerr << "Ошибка: не найдено .txt файлов в директории " << data_dir << std::endl;
        return all_words;
    }
    
    std::cout << "Обработка " << files.size() << " файлов..." << std::endl;
    
    // Читаем файлы последовательно для стабильности
    for (size_t i = 0; i < files.size(); ++i) {
        std::ifstream file(files[i]);
        if (file.is_open()) {
            std::stringstream buffer;
            buffer << file.rdbuf();
            std::string text = buffer.str();
            file.close();
            
            std::string word;
            for (char c : text) {
                if (std::isspace(c) || std::ispunct(c)) {
                    if (!word.empty()) {
                        all_words.push_back(word);
                        word.clear();
                    }
                } else {
                    word += std::tolower(c);
                }
            }
            if (!word.empty()) {
                all_words.push_back(word);
            }
            
            print_progress_bar(static_cast<float>(i + 1) / files.size(), 30, "Загрузка файлов ");
        }
    }
    
    std::cout << "\r" << std::string(60, ' ') << "\r";
    return all_words;
}

// ==================== Основная функция (упрощенная) ====================
int main() {
    try {
        std::cout << "=========================================" << std::endl;
        std::cout << "        Обучение AvoAI Нейросети        " << std::endl;
        std::cout << "=========================================" << std::endl;
        
        auto start_total = std::chrono::high_resolution_clock::now();
        
        // Этап 1: Загрузка данных
        std::cout << "\n1. Загрузка данных из data/..." << std::endl;
        auto words = load_all_texts("data/");
        
        if (words.empty()) {
            std::cerr << "Ошибка: не загружены слова!" << std::endl;
            std::cerr << "Проверьте наличие .txt файлов в директории data/" << std::endl;
            return 1;
        }
        
        std::cout << "✓ Загружено слов: " << words.size() << std::endl;
        
        // Этап 2: Создание словаря
        std::cout << "\n2. Создание словаря..." << std::endl;
        Vocabulary vocab;
        
        // Создаем словарь в одном потоке для стабильности
        int word_counter = 0;
        for (const auto& word : words) {
            vocab.add_word(word);
            word_counter++;
            
            if (word_counter % 50000 == 0) {
                print_progress_bar(static_cast<float>(word_counter) / words.size(), 
                                 30, "Обработка слов    ");
            }
        }
        
        std::cout << "\r" << std::string(60, ' ') << "\r";
        std::cout << "✓ Словарь создан! Уникальных слов: " << vocab.size() << std::endl;
        
        // Этап 3: Подготовка обучающих данных
        std::cout << "\n3. Подготовка обучающих данных..." << std::endl;
        std::vector<std::pair<std::vector<int>, int>> training_data;
        
        const int CONTEXT_SIZE = 3;
        size_t total_pairs = words.size() - CONTEXT_SIZE;
        
        for (size_t i = 0; i < total_pairs; ++i) {
            std::vector<int> context;
            bool valid = true;
            
            for (int j = 0; j < CONTEXT_SIZE; ++j) {
                int idx = vocab.get_index(words[i + j]);
                if (idx == -1) {
                    valid = false;
                    break;
                }
                context.push_back(idx);
            }
            
            if (valid) {
                int target_idx = vocab.get_index(words[i + CONTEXT_SIZE]);
                if (target_idx != -1) {
                    training_data.emplace_back(context, target_idx);
                }
            }
            
            if (i % 50000 == 0) {
                print_progress_bar(static_cast<float>(i) / total_pairs, 
                                 30, "Подготовка данных ");
            }
        }
        
        std::cout << "\r" << std::string(60, ' ') << "\r";
        std::cout << "✓ Создано обучающих примеров: " << training_data.size() << std::endl;
        
        if (training_data.empty()) {
            std::cerr << "Ошибка: не созданы обучающие данные!" << std::endl;
            return 1;
        }
        
        // Этап 4: Обучение модели
        std::cout << "\n4. Обучение нейросети..." << std::endl;
        std::cout << "   Размер словаря: " << vocab.size() << std::endl;
        std::cout << "   Примеров для обучения: " << training_data.size() << std::endl;
        
        // Уменьшаем размер скрытого слоя для ускорения обучения
        size_t hidden_size = std::min(static_cast<size_t>(256), vocab.size() / 2);
        if (hidden_size < 64) hidden_size = 64;
        
        NeuralNet model(vocab.size(), hidden_size, 0.01f);
        
        const int EPOCHS = 10;  // Уменьшаем для теста
        const int BATCH_SIZE = 32;
        
        auto start_train = std::chrono::high_resolution_clock::now();
        
        std::cout << "\nНачало обучения (эпох: " << EPOCHS << ", батч: " << BATCH_SIZE << ")..." << std::endl;
        
        // Основной цикл обучения
        for (int epoch = 0; epoch < EPOCHS; ++epoch) {
            auto epoch_start = std::chrono::high_resolution_clock::now();
            
            std::cout << "Эпоха " << (epoch + 1) << "/" << EPOCHS << ": ";
            
            // Перемешивание данных
            std::shuffle(training_data.begin(), training_data.end(), model.get_rng());
            
            float epoch_loss = 0.0f;
            int processed = 0;
            
            // Адаптивная скорость обучения
            float current_lr = 0.01f * (1.0f - static_cast<float>(epoch) / EPOCHS);
            
            // Простой цикл обучения
            for (size_t i = 0; i < training_data.size(); ++i) {
                const auto& [context, target] = training_data[i];
                float loss = model.train_step(context, target, current_lr);
                epoch_loss += loss;
                processed++;
                
                if (i % 1000 == 0) {
                    print_progress_bar(static_cast<float>(i) / training_data.size(), 
                                     30, "Обучение         ");
                }
            }
            
            auto epoch_end = std::chrono::high_resolution_clock::now();
            auto epoch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end - epoch_start);
            
            std::cout << "\r" << std::string(60, ' ') << "\r";
            float avg_loss = epoch_loss / processed;
            std::cout << "Эпоха " << std::setw(2) << (epoch + 1) << " завершена: ";
            std::cout << "потери=" << std::fixed << std::setprecision(4) << avg_loss;
            std::cout << ", время=" << epoch_duration.count() << "мс" << std::endl;
        }
        
        auto end_train = std::chrono::high_resolution_clock::now();
        auto train_duration = std::chrono::duration_cast<std::chrono::seconds>(end_train - start_train);
        
        // Этап 5: Сохранение модели
        std::cout << "\n5. Сохранение модели..." << std::endl;
        
        if (!fs::exists("models")) {
            fs::create_directory("models");
        }
        
        model.save("models/model.bin");
        
        // Сохранение словаря
        std::ofstream vocab_file("models/vocab.txt");
        for (size_t i = 0; i < vocab.size(); ++i) {
            vocab_file << vocab.get_word(i) << "\n";
        }
        vocab_file.close();
        std::cout << "✓ Словарь сохранен (" << vocab.size() << " слов)" << std::endl;
        
        // Этап 6: Тестирование
        std::cout << "\n6. Тестирование модели..." << std::endl;
        
        if (training_data.size() > 5) {
            std::cout << "\nПримеры предсказаний:" << std::endl;
            std::cout << "-------------------" << std::endl;
            
            for (int i = 0; i < 3 && i < training_data.size(); ++i) {
                const auto& [context, target] = training_data[i];
                
                std::cout << "Контекст: ";
                for (int idx : context) {
                    std::cout << vocab.get_word(idx) << " ";
                }
                
                int prediction = model.predict_next(context);
                std::cout << "\nПредсказано: '" << vocab.get_word(prediction) << "'";
                std::cout << " | Ожидалось: '" << vocab.get_word(target) << "'";
                
                if (prediction == target) {
                    std::cout << " ✓" << std::endl;
                } else {
                    std::cout << " ✗" << std::endl;
                }
                std::cout << std::endl;
            }
        }
        
        // Итоги
        auto end_total = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(end_total - start_total);
        
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!" << std::endl;
        std::cout << "Общее время: " << total_duration.count() << " секунд" << std::endl;
        std::cout << "Время обучения: " << train_duration.count() << " секунд" << std::endl;
        std::cout << "Обучающих примеров: " << training_data.size() << std::endl;
        std::cout << "Размер словаря: " << vocab.size() << " слов" << std::endl;
        std::cout << "\nФайлы сохранены:" << std::endl;
        std::cout << "  • Модель: models/model.bin" << std::endl;
        std::cout << "  • Словарь: models/vocab.txt" << std::endl;
        std::cout << "\nДля общения с моделью запустите:" << std::endl;
        std::cout << "  ./bin/chat" << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "\n✗ ОШИБКА: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}