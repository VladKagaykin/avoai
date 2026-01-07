#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <random>
#include <filesystem>
#include <unordered_map>
#include <chrono>
#include <iomanip>
#include <omp.h>

namespace fs = std::filesystem;

// ==================== Конфигурация ====================
struct Config {
    std::string data_dir = "data";
    std::string model_dir = "models";
    std::string text_file = "models/processed_text.txt";
    std::string model_file = "models/model.bin";
    int context_size = 20;
    int hidden_size = 128;
    float learning_rate = 0.01f;
    int min_epochs = 50;
    int max_epochs = 500;
    int batch_size = 32;
    int num_threads = 8;
};

// ==================== Чтение файлов ====================
class TextFileReader {
public:
    static std::string readAllTextFiles(const std::string& directory_path) {
        std::string all_text;
        
        try {
            if (!fs::exists(directory_path)) {
                std::cerr << "Директория не существует: " << directory_path << std::endl;
                return all_text;
            }
            
            std::cout << "Чтение текстовых файлов из: " << directory_path << std::endl;
            int file_count = 0;
            size_t total_bytes = 0;
            
            for (const auto& entry : fs::directory_iterator(directory_path)) {
                if (entry.is_regular_file()) {
                    std::string filename = entry.path().string();
                    std::string extension = entry.path().extension().string();
                    
                    if (extension == ".txt" || extension == ".TXT" || extension.empty()) {
                        std::ifstream file(filename, std::ios::binary);
                        if (file.is_open()) {
                            std::stringstream buffer;
                            buffer << file.rdbuf();
                            std::string content = buffer.str();
                            
                            if (!content.empty()) {
                                all_text += content + "\n";
                                total_bytes += content.size();
                                file_count++;
                                std::cout << "  " << entry.path().filename() 
                                          << " (" << content.size() << " байт)" << std::endl;
                            }
                            file.close();
                        }
                    }
                }
            }
            
            std::cout << "Всего прочитано файлов: " << file_count << std::endl;
            std::cout << "Общий объем данных: " << total_bytes << " байт" << std::endl;
            std::cout << "UTF-8 байтов (0-255): " << total_bytes << std::endl;
            
        } catch (const fs::filesystem_error& e) {
            std::cerr << "Ошибка чтения файлов: " << e.what() << std::endl;
        }
        
        return all_text;
    }
};

// ==================== Обработка текста (байты) ====================
class ByteProcessor {
public:
    // Оставляем текст как есть (бинарные данные)
    static std::string cleanText(const std::string& text) {
        return text;
    }
    
    // Создаем обучающие пары (контекст байтов -> следующий байт)
    static std::vector<std::pair<std::string, unsigned char>> createTrainingPairs(
        const std::string& text, int context_size = 20) {
        
        std::vector<std::pair<std::string, unsigned char>> pairs;
        pairs.reserve(text.size() - context_size);
        
        for (size_t i = 0; i < text.size() - context_size; ++i) {
            std::string context = text.substr(i, context_size);
            unsigned char next_byte = static_cast<unsigned char>(text[i + context_size]);
            pairs.push_back({context, next_byte});
        }
        
        return pairs;
    }
    
    // Векторизация контекста (one-hot для каждого байта)
    static std::vector<float> vectorizeContext(
        const std::string& context, 
        int context_size) {
        
        std::vector<float> vec(256 * context_size, 0.0f);
        
        // Исправление: явно приводим context_size к size_t
        size_t ctx_size = static_cast<size_t>(context_size);
        size_t limit = std::min(context.size(), ctx_size);
        
        for (size_t i = 0; i < limit; ++i) {
            unsigned char c = static_cast<unsigned char>(context[i]);
            int idx = static_cast<int>(i) * 256 + static_cast<int>(c);
            if (idx < static_cast<int>(vec.size())) {
                vec[idx] = 1.0f;
            }
        }
        
        return vec;
    }
    
    // Векторизация целевого байта
    static std::vector<float> vectorizeTarget(unsigned char target_byte) {
        std::vector<float> vec(256, 0.0f);
        vec[target_byte] = 1.0f;
        return vec;
    }
};

// ==================== Нейронная сеть с многопоточностью ====================
class ParallelNeuralNetwork {
public:
    int input_size;
    int hidden_size;
    int output_size;
    
private:
    std::vector<std::vector<float>> w1; // вход->скрытый
    std::vector<std::vector<float>> w2; // скрытый->выход
    std::vector<float> b1, b2;
    
public:
    ParallelNeuralNetwork(int in_size, int hid_size, int out_size) 
        : input_size(in_size), hidden_size(hid_size), output_size(out_size) {
        
        // Инициализация весов
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, 0.1f);
        
        w1.resize(input_size);
        #pragma omp parallel for
        for (int i = 0; i < input_size; ++i) {
            w1[i].resize(hidden_size);
            for (int j = 0; j < hidden_size; ++j) {
                w1[i][j] = dist(gen);
            }
        }
        
        w2.resize(hidden_size);
        #pragma omp parallel for
        for (int i = 0; i < hidden_size; ++i) {
            w2[i].resize(output_size);
            for (int j = 0; j < output_size; ++j) {
                w2[i][j] = dist(gen);
            }
        }
        
        b1.resize(hidden_size, 0.1f);
        b2.resize(output_size, 0.1f);
        
        std::cout << "Создана многопоточная нейронная сеть: " << input_size 
                  << " -> " << hidden_size << " -> " << output_size << std::endl;
    }
    
    float sigmoid(float x) const {
        return 1.0f / (1.0f + std::exp(-x));
    }
    
    float sigmoidDerivative(float x) const {
        return x * (1.0f - x);
    }
    
    std::vector<float> softmax(const std::vector<float>& x) const {
        std::vector<float> result(x.size());
        float max_val = *std::max_element(x.begin(), x.end());
        float sum = 0.0f;
        
        #pragma omp parallel for reduction(+:sum)
        for (size_t i = 0; i < x.size(); ++i) {
            result[i] = std::exp(x[i] - max_val);
            sum += result[i];
        }
        
        #pragma omp parallel for
        for (size_t i = 0; i < result.size(); ++i) {
            result[i] /= sum;
        }
        
        return result;
    }
    
    // Прямое распространение с многопоточностью
    std::vector<float> forward(const std::vector<float>& input) const {
        std::vector<float> hidden(hidden_size, 0.0f);
        std::vector<float> output(output_size, 0.0f);
        
        // Параллельно вычисляем скрытый слой
        #pragma omp parallel for
        for (int i = 0; i < hidden_size; ++i) {
            float sum = b1[i];
            for (int j = 0; j < input_size; ++j) {
                sum += input[j] * w1[j][i];
            }
            hidden[i] = sigmoid(sum);
        }
        
        // Параллельно вычисляем выходной слой
        #pragma omp parallel for
        for (int i = 0; i < output_size; ++i) {
            float sum = b2[i];
            for (int j = 0; j < hidden_size; ++j) {
                sum += hidden[j] * w2[j][i];
            }
            output[i] = sum;
        }
        
        return softmax(output);
    }
    
    // Обучение с мини-батчами и многопоточностью
    void trainBatch(const std::vector<std::vector<float>>& inputs,
                    const std::vector<std::vector<float>>& targets,
                    float learning_rate) {
        
        int batch_size = inputs.size();
        if (batch_size == 0) return;
        
        // Векторы для накопления градиентов
        std::vector<std::vector<float>> w1_grad(input_size, std::vector<float>(hidden_size, 0.0f));
        std::vector<std::vector<float>> w2_grad(hidden_size, std::vector<float>(output_size, 0.0f));
        std::vector<float> b1_grad(hidden_size, 0.0f);
        std::vector<float> b2_grad(output_size, 0.0f);
        
        // Параллельная обработка мини-батча
        #pragma omp parallel
        {
            // Локальные градиенты для каждого потока
            std::vector<std::vector<float>> local_w1_grad(input_size, std::vector<float>(hidden_size, 0.0f));
            std::vector<std::vector<float>> local_w2_grad(hidden_size, std::vector<float>(output_size, 0.0f));
            std::vector<float> local_b1_grad(hidden_size, 0.0f);
            std::vector<float> local_b2_grad(output_size, 0.0f);
            
            #pragma omp for
            for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
                const auto& input = inputs[batch_idx];
                const auto& target = targets[batch_idx];
                
                // Прямое распространение
                std::vector<float> hidden(hidden_size, 0.0f);
                for (int i = 0; i < hidden_size; ++i) {
                    float sum = b1[i];
                    for (int j = 0; j < input_size; ++j) {
                        sum += input[j] * w1[j][i];
                    }
                    hidden[i] = sigmoid(sum);
                }
                
                std::vector<float> output(output_size, 0.0f);
                for (int i = 0; i < output_size; ++i) {
                    float sum = b2[i];
                    for (int j = 0; j < hidden_size; ++j) {
                        sum += hidden[j] * w2[j][i];
                    }
                    output[i] = sum;
                }
                
                auto softmax_output = softmax(output);
                
                // Ошибка выходного слоя
                std::vector<float> output_error(output_size);
                for (int i = 0; i < output_size; ++i) {
                    output_error[i] = target[i] - softmax_output[i];
                }
                
                // Градиенты для выходного слоя
                for (int i = 0; i < hidden_size; ++i) {
                    for (int j = 0; j < output_size; ++j) {
                        local_w2_grad[i][j] += output_error[j] * hidden[i];
                    }
                }
                
                for (int i = 0; i < output_size; ++i) {
                    local_b2_grad[i] += output_error[i];
                }
                
                // Ошибка скрытого слоя
                std::vector<float> hidden_error(hidden_size, 0.0f);
                for (int i = 0; i < hidden_size; ++i) {
                    float error = 0.0f;
                    for (int j = 0; j < output_size; ++j) {
                        error += output_error[j] * w2[i][j];
                    }
                    hidden_error[i] = error * sigmoidDerivative(hidden[i]);
                }
                
                // Градиенты для скрытого слоя
                for (int i = 0; i < input_size; ++i) {
                    for (int j = 0; j < hidden_size; ++j) {
                        local_w1_grad[i][j] += hidden_error[j] * input[i];
                    }
                }
                
                for (int i = 0; i < hidden_size; ++i) {
                    local_b1_grad[i] += hidden_error[i];
                }
            }
            
            // Собираем градиенты от всех потоков
            #pragma omp critical
            {
                for (int i = 0; i < input_size; ++i) {
                    for (int j = 0; j < hidden_size; ++j) {
                        w1_grad[i][j] += local_w1_grad[i][j];
                    }
                }
                
                for (int i = 0; i < hidden_size; ++i) {
                    for (int j = 0; j < output_size; ++j) {
                        w2_grad[i][j] += local_w2_grad[i][j];
                    }
                }
                
                for (int i = 0; i < hidden_size; ++i) {
                    b1_grad[i] += local_b1_grad[i];
                }
                
                for (int i = 0; i < output_size; ++i) {
                    b2_grad[i] += local_b2_grad[i];
                }
            }
        }
        
        // Обновление весов (усредняем градиенты по батчу)
        float scale = learning_rate / batch_size;
        
        #pragma omp parallel for
        for (int i = 0; i < input_size; ++i) {
            for (int j = 0; j < hidden_size; ++j) {
                w1[i][j] += scale * w1_grad[i][j];
            }
        }
        
        #pragma omp parallel for
        for (int i = 0; i < hidden_size; ++i) {
            for (int j = 0; j < output_size; ++j) {
                w2[i][j] += scale * w2_grad[i][j];
            }
        }
        
        #pragma omp parallel for
        for (int i = 0; i < hidden_size; ++i) {
            b1[i] += scale * b1_grad[i];
        }
        
        #pragma omp parallel for
        for (int i = 0; i < output_size; ++i) {
            b2[i] += scale * b2_grad[i];
        }
    }
    
    void save(const std::string& filename) const {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Не могу открыть файл: " << filename << std::endl;
            return;
        }
        
        file.write(reinterpret_cast<const char*>(&input_size), sizeof(int));
        file.write(reinterpret_cast<const char*>(&hidden_size), sizeof(int));
        file.write(reinterpret_cast<const char*>(&output_size), sizeof(int));
        
        for (const auto& row : w1) {
            file.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
        }
        
        for (const auto& row : w2) {
            file.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
        }
        
        file.write(reinterpret_cast<const char*>(b1.data()), b1.size() * sizeof(float));
        file.write(reinterpret_cast<const char*>(b2.data()), b2.size() * sizeof(float));
        
        std::cout << "Модель сохранена: " << filename << std::endl;
    }
};

// ==================== Основная функция ====================
int main() {
    std::cout << "=== ПРОГРАММА ОБУЧЕНИЯ НЕЙРОННОЙ СЕТИ НА БАЙТАХ UTF-8 ===" << std::endl;
    std::cout << std::endl;
    
    Config config;
    
    // Установка количества потоков
    omp_set_num_threads(config.num_threads);
    std::cout << "Используется потоков: " << config.num_threads << std::endl;
    
    // Создаем директории
    fs::create_directories(config.data_dir);
    fs::create_directories(config.model_dir);
    
    // 1. Загрузка данных
    std::cout << "1. ЗАГРУЗКА ДАННЫХ (БАЙТЫ UTF-8)" << std::endl;
    std::cout << "---------------------------------" << std::endl;
    
    std::string all_text = TextFileReader::readAllTextFiles(config.data_dir);
    if (all_text.empty()) {
        std::cout << "Добавьте текстовые файлы в папку data/ и перезапустите программу." << std::endl;
        return 1;
    }
    
    // 2. Очистка текста (оставляем как есть - бинарные данные)
    std::cout << std::endl;
    std::cout << "2. ОБРАБОТКА ТЕКСТА (СОХРАНЕНИЕ ВСЕХ БАЙТОВ)" << std::endl;
    std::cout << "--------------------------------------------" << std::endl;
    
    std::string processed_text = ByteProcessor::cleanText(all_text);
    std::cout << "Текст обработан: " << processed_text.size() << " байт" << std::endl;
    
    // 3. Подготовка данных для обучения
    std::cout << std::endl;
    std::cout << "3. ПОДГОТОВКА ДАННЫХ ДЛЯ ОБУЧЕНИЯ (БАЙТЫ)" << std::endl;
    std::cout << "------------------------------------------" << std::endl;
    
    auto start_pairs = std::chrono::high_resolution_clock::now();
    auto training_pairs = ByteProcessor::createTrainingPairs(processed_text, config.context_size);
    auto end_pairs = std::chrono::high_resolution_clock::now();
    auto pairs_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_pairs - start_pairs);
    
    std::cout << "Создано обучающих пар: " << training_pairs.size() << std::endl;
    std::cout << "Размер словаря: 256 байтов (0-255)" << std::endl;
    std::cout << "Время создания пар: " << pairs_duration.count() << " мс" << std::endl;
    
    if (training_pairs.empty()) {
        std::cout << "Недостаточно данных для обучения!" << std::endl;
        return 1;
    }
    
    // 4. Создание нейронной сети
    std::cout << std::endl;
    std::cout << "4. СОЗДАНИЕ НЕЙРОННОЙ СЕТИ" << std::endl;
    std::cout << "--------------------------" << std::endl;
    
    int input_size = config.context_size * 256;  // 256 возможных байтов на позицию
    int output_size = 256;  // 256 возможных байтов на выходе
    
    ParallelNeuralNetwork nn(input_size, config.hidden_size, output_size);
    
    // 5. Расчет количества эпох
    int epochs = config.min_epochs + (processed_text.size() / 50000);
    if (epochs > config.max_epochs) epochs = config.max_epochs;
    
    std::cout << "Будет выполнено эпох: " << epochs << std::endl;
    std::cout << "Размер входного слоя: " << input_size << " (контекст: " 
              << config.context_size << " × 256)" << std::endl;
    std::cout << "Размер выходного слоя: " << output_size << " (256 байтов)" << std::endl;
    std::cout << "Размер мини-батча: " << config.batch_size << std::endl;
    
    // 6. Подготовка мини-батчей
    std::cout << std::endl;
    std::cout << "5. ПОДГОТОВКА МИНИ-БАТЧЕЙ" << std::endl;
    std::cout << "-------------------------" << std::endl;
    
    // Перемешиваем данные
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(training_pairs.begin(), training_pairs.end(), g);
    
    // Ограничиваем количество примеров для ускорения
    size_t total_samples = std::min<size_t>(training_pairs.size(), 50000);
    size_t num_batches = (total_samples + config.batch_size - 1) / config.batch_size;
    
    std::cout << "Всего примеров: " << total_samples << std::endl;
    std::cout << "Количество батчей: " << num_batches << std::endl;
    
    // 7. Обучение с мини-батчами
    std::cout << std::endl;
    std::cout << "6. ОБУЧЕНИЕ С МИНИ-БАТЧАМИ" << std::endl;
    std::cout << "---------------------------" << std::endl;
    
    auto start_training = std::chrono::high_resolution_clock::now();
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;
        size_t samples_processed = 0;
        
        // Обработка батчей
        for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
            size_t start = batch_idx * config.batch_size;
            size_t end = std::min(start + config.batch_size, total_samples);
            size_t current_batch_size = end - start;
            
            // Подготовка мини-батча
            std::vector<std::vector<float>> batch_inputs;
            std::vector<std::vector<float>> batch_targets;
            batch_inputs.reserve(current_batch_size);
            batch_targets.reserve(current_batch_size);
            
            // Параллельная векторизация батча
            #pragma omp parallel for
            for (size_t i = start; i < end; ++i) {
                auto input_vec = ByteProcessor::vectorizeContext(
                    training_pairs[i].first, config.context_size);
                auto target_vec = ByteProcessor::vectorizeTarget(
                    training_pairs[i].second);
                
                #pragma omp critical
                {
                    batch_inputs.push_back(std::move(input_vec));
                    batch_targets.push_back(std::move(target_vec));
                }
            }
            
            // Обучение на мини-батче
            nn.trainBatch(batch_inputs, batch_targets, config.learning_rate);
            
            // Вычисление потерь для этого батча
            #pragma omp parallel for reduction(+:total_loss)
            for (size_t i = 0; i < batch_inputs.size(); ++i) {
                auto output = nn.forward(batch_inputs[i]);
                float loss = 0.0f;
                for (size_t j = 0; j < output.size(); ++j) {
                    float diff = batch_targets[i][j] - output[j];
                    loss += diff * diff;
                }
                total_loss += loss;
            }
            
            samples_processed += current_batch_size;
        }
        
        float avg_loss = total_loss / samples_processed;
        
        // Прогресс-бар
        float progress = (epoch + 1) * 100.0f / epochs;
        int bar_width = 50;
        int pos = bar_width * progress / 100.0;
        
        std::cout << "  Эпоха " << std::setw(3) << (epoch + 1) << "/" << epochs << " [";
        for (int j = 0; j < bar_width; ++j) {
            if (j < pos) std::cout << "=";
            else if (j == pos) std::cout << ">";
            else std::cout << " ";
        }
        std::cout << "] " << std::fixed << std::setprecision(1) << progress << "% "
                  << "Потери: " << std::setprecision(4) << avg_loss << std::endl;
    }
    
    auto end_training = std::chrono::high_resolution_clock::now();
    auto training_duration = std::chrono::duration_cast<std::chrono::seconds>(end_training - start_training);
    
    std::cout << std::endl;
    std::cout << "  Обучение заняло: " << training_duration.count() << " секунд" << std::endl;
    
    // 8. Сохранение результатов
    std::cout << std::endl;
    std::cout << "7. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ" << std::endl;
    std::cout << "-------------------------" << std::endl;
    
    // Сохраняем обработанный текст
    std::ofstream text_file(config.text_file, std::ios::binary);
    text_file << processed_text;
    text_file.close();
    std::cout << "Текст сохранен: " << config.text_file << " (" 
              << processed_text.size() << " байт)" << std::endl;
    
    // Сохраняем модель
    nn.save(config.model_file);
    
    std::cout << std::endl;
    std::cout << "=== ОБУЧЕНИЕ ЗАВЕРШЕНО ===" << std::endl;
    std::cout << "Модель обучена на всех 256 байтах UTF-8." << std::endl;
    std::cout << "Для генерации текста запустите: ./chat" << std::endl;
    
    return 0;
}