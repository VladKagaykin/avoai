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
#include <thread>
#include <atomic>
#include <cpuid.h>
#include <unistd.h>
#include <omp.h>

namespace fs = std::filesystem;

// ==================== Детектор железа (ТОЛЬКО ДЛЯ ПАРАЛЛЕЛИЗМА) ====================
class HardwareDetector {
private:
    int logical_cores;
    size_t total_ram;
    
public:
    HardwareDetector() {
        logical_cores = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
        
        detect_ram();
        
        // Используем ВСЕ логические ядра
        omp_set_num_threads(logical_cores);
        omp_set_dynamic(0);
        omp_set_schedule(omp_sched_dynamic, 32);
    }
    
    void detect_ram() {
        long pages = sysconf(_SC_PHYS_PAGES);
        long page_size = sysconf(_SC_PAGE_SIZE);
        if (pages > 0 && page_size > 0) {
            total_ram = pages * page_size;
        } else {
            total_ram = 8ULL * 1024 * 1024 * 1024;
        }
    }
    
    void print_info() const {
        std::cout << "=== Анализ системы ===" << std::endl;
        std::cout << "Логических ядер: " << logical_cores << std::endl;
        std::cout << "ОЗУ: " << total_ram / 1024 / 1024 / 1024 << " GB" << std::endl;
        std::cout << "=====================" << std::endl;
    }
    
    int get_logical_cores() const { return logical_cores; }
    size_t get_total_ram() const { return total_ram; }
    
    int recommend_threads() const {
        return logical_cores; // Используем ВСЕ логические ядра
    }
};

// ==================== Адаптивный словарь ====================
class AdaptiveVocabulary {
private:
    std::unordered_map<std::string, int> word_to_idx;
    std::vector<std::string> idx_to_word;
    std::mutex mtx;
    int next_idx = 0;
    
public:
    AdaptiveVocabulary(size_t vocab_size_limit = 50000) {
        idx_to_word.reserve(vocab_size_limit);
        word_to_idx.reserve(vocab_size_limit);
    }
    
    int add_word(const std::string& word) {
        std::lock_guard<std::mutex> lock(mtx);
        auto it = word_to_idx.find(word);
        if (it != word_to_idx.end()) return it->second;
        
        int idx = next_idx++;
        word_to_idx[word] = idx;
        idx_to_word.push_back(word);
        return idx;
    }
    
    int get_index(const std::string& word) const {
        auto it = word_to_idx.find(word);
        return it != word_to_idx.end() ? it->second : -1;
    }
    
    const std::string& get_word(int idx) const {
        static const std::string empty = "";
        if (idx >= 0 && idx < static_cast<int>(idx_to_word.size())) {
            return idx_to_word[idx];
        }
        return empty;
    }
    
    size_t size() const { return word_to_idx.size(); }
    
    const std::vector<std::string>& get_words() const { return idx_to_word; }
};

// ==================== Адаптивная нейросеть (НАСТРОЙКИ НА ОСНОВЕ ДАННЫХ) ====================
class AdaptiveNeuralNet {
private:
    std::vector<float> weights1;  // vocab_size * hidden_size
    std::vector<float> weights2;  // hidden_size * vocab_size
    std::vector<float> bias1, bias2;
    
    size_t vocab_size;
    size_t hidden_size;
    float learning_rate;
    
    std::mt19937 rng;
    
public:
    AdaptiveNeuralNet(size_t vocab_size, size_t training_samples, float lr = 0.01f)
        : vocab_size(vocab_size), learning_rate(lr) {
        
        rng.seed(std::chrono::steady_clock::now().time_since_epoch().count());
        
        // РАЗМЕР СКРЫТОГО СЛОЯ ЗАВИСИТ ОТ ОБЪЕМА ДАННЫХ
        if (training_samples < 50000) {
            hidden_size = 128;  // Мало данных - маленькая модель
        } else if (training_samples < 200000) {
            hidden_size = 256;  // Средний объем данных
        } else if (training_samples < 500000) {
            hidden_size = 384;  // Много данных
        } else {
            hidden_size = 512;  // Очень много данных
        }
        
        // Ограничиваем размер модели в зависимости от словаря
        size_t max_hidden_by_vocab = static_cast<size_t>(sqrt(vocab_size * 10));
        hidden_size = std::min(hidden_size, max_hidden_by_vocab);
        hidden_size = std::max(hidden_size, static_cast<size_t>(64));  // Минимум 64
        hidden_size = std::min(hidden_size, static_cast<size_t>(1024)); // Максимум 1024
        
        std::cout << "Конфигурация нейросети:" << std::endl;
        std::cout << "  • Размер словаря: " << vocab_size << std::endl;
        std::cout << "  • Скрытый слой: " << hidden_size << " нейронов (на основе " 
                  << training_samples << " примеров)" << std::endl;
        
        // Проверка на переполнение памяти (не более 2GB)
        size_t required_memory = (vocab_size * hidden_size + hidden_size * vocab_size + 
                                 hidden_size + vocab_size) * sizeof(float);
        if (required_memory > 2ULL * 1024 * 1024 * 1024) {
            std::cerr << "Предупреждение: модель большая, уменьшаем скрытый слой..." << std::endl;
            hidden_size = vocab_size / 100;  // Эвристика: 1% от словаря
            hidden_size = std::max(hidden_size, static_cast<size_t>(64));
            hidden_size = std::min(hidden_size, static_cast<size_t>(512));
        }
        
        weights1.resize(vocab_size * hidden_size, 0.0f);
        weights2.resize(hidden_size * vocab_size, 0.0f);
        bias1.resize(hidden_size, 0.0f);
        bias2.resize(vocab_size, 0.0f);
        
        initialize_weights();
    }
    
    void initialize_weights() {
        std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
        float stddev = sqrtf(2.0f / (vocab_size + hidden_size));
        
        // Инициализация Xavier/Glorot
        #pragma omp parallel for
        for (size_t i = 0; i < vocab_size * hidden_size; ++i) {
            weights1[i] = dist(rng) * stddev;
        }
        
        #pragma omp parallel for
        for (size_t i = 0; i < hidden_size * vocab_size; ++i) {
            weights2[i] = dist(rng) * stddev;
        }
    }
    
    inline float fast_sigmoid(float x) const {
        x = 0.5f * x;
        x = x / (1.0f + fabsf(x));
        return x + 0.5f;
    }
    
    inline float sigmoid_derivative(float x) const {
        return x * (1.0f - x);
    }
    
    void forward_optimized(const std::vector<int>& input,
                          std::vector<float>& hidden,
                          std::vector<float>& output) {
        
        hidden.assign(hidden_size, 0.0f);
        
        // Безопасное суммирование эмбеддингов
        for (int idx : input) {
            if (idx < 0 || idx >= static_cast<int>(vocab_size)) continue;
            
            const float* w_ptr = &weights1[idx * hidden_size];
            for (size_t j = 0; j < hidden_size; ++j) {
                hidden[j] += w_ptr[j];
            }
        }
        
        // Сигмоида с bias
        for (size_t j = 0; j < hidden_size; ++j) {
            hidden[j] = fast_sigmoid(hidden[j] + bias1[j]);
        }
        
        // Выходной слой
        output.assign(vocab_size, 0.0f);
        #pragma omp parallel for
        for (size_t i = 0; i < vocab_size; ++i) {
            float sum = 0.0f;
            for (size_t j = 0; j < hidden_size; ++j) {
                sum += weights2[j * vocab_size + i] * hidden[j];
            }
            output[i] = sum + bias2[i];
        }
        
        softmax(output);
    }
    
    void softmax(std::vector<float>& logits) {
        float max_val = *std::max_element(logits.begin(), logits.end());
        float sum = 0.0f;
        
        for (size_t i = 0; i < logits.size(); ++i) {
            logits[i] = expf(logits[i] - max_val);
            sum += logits[i];
        }
        
        float inv_sum = 1.0f / sum;
        for (size_t i = 0; i < logits.size(); ++i) {
            logits[i] *= inv_sum;
        }
    }
    
    // БЫСТРОЕ обучение с проверками границ
    float train_batch_fast(const std::vector<std::pair<std::vector<int>, int>>& batch,
                          float lr) {
        
        size_t batch_size = batch.size();
        const size_t hs = hidden_size;
        const size_t vs = vocab_size;
        
        if (batch_size == 0) return 0.0f;
        
        // Глобальные градиенты
        std::vector<float> grad_w1(vs * hs, 0.0f);
        std::vector<float> grad_w2(hs * vs, 0.0f);
        std::vector<float> grad_b1(hs, 0.0f);
        std::vector<float> grad_b2(vs, 0.0f);
        
        float total_loss = 0.0f;
        
        #pragma omp parallel reduction(+:total_loss)
        {
            // Локальные буферы
            std::vector<float> local_hidden(hs, 0.0f);
            std::vector<float> local_output(vs, 0.0f);
            std::vector<float> delta2(vs, 0.0f);
            std::vector<float> delta1(hs, 0.0f);
            std::vector<float> local_grad_w1(vs * hs, 0.0f);
            std::vector<float> local_grad_w2(hs * vs, 0.0f);
            std::vector<float> local_grad_b1(hs, 0.0f);
            std::vector<float> local_grad_b2(vs, 0.0f);
            
            #pragma omp for schedule(static)
            for (size_t b = 0; b < batch_size; ++b) {
                const auto& [input, target] = batch[b];
                
                // ПРОВЕРКА: target должен быть в пределах словаря
                if (target < 0 || target >= static_cast<int>(vs)) continue;
                
                // Прямой проход
                std::fill(local_hidden.begin(), local_hidden.end(), 0.0f);
                
                for (int idx : input) {
                    if (idx < 0 || idx >= static_cast<int>(vs)) continue;
                    const float* w_ptr = &weights1[idx * hs];
                    for (size_t j = 0; j < hs; ++j) {
                        local_hidden[j] += w_ptr[j];
                    }
                }
                
                for (size_t j = 0; j < hs; ++j) {
                    local_hidden[j] = fast_sigmoid(local_hidden[j] + bias1[j]);
                }
                
                // Выходной слой
                for (size_t i = 0; i < vs; ++i) {
                    float sum = 0.0f;
                    for (size_t j = 0; j < hs; ++j) {
                        sum += weights2[j * vs + i] * local_hidden[j];
                    }
                    local_output[i] = sum + bias2[i];
                }
                
                // Softmax и loss
                std::vector<float> temp_output = local_output;
                softmax(temp_output);
                
                total_loss += -logf(temp_output[target] + 1e-8f);
                
                // Градиенты выходного слоя
                std::copy(temp_output.begin(), temp_output.end(), delta2.begin());
                delta2[target] -= 1.0f;
                
                // Накопление градиентов weights2 и bias2
                for (size_t j = 0; j < hs; ++j) {
                    float grad = local_hidden[j];
                    float* grad_ptr = &local_grad_w2[j * vs];
                    for (size_t i = 0; i < vs; ++i) {
                        grad_ptr[i] += delta2[i] * grad;
                    }
                }
                
                for (size_t i = 0; i < vs; ++i) {
                    local_grad_b2[i] += delta2[i];
                }
                
                // Градиенты скрытого слоя
                for (size_t j = 0; j < hs; ++j) {
                    float sum = 0.0f;
                    for (size_t i = 0; i < vs; ++i) {
                        sum += weights2[j * vs + i] * delta2[i];
                    }
                    delta1[j] = sum * sigmoid_derivative(local_hidden[j]);
                    local_grad_b1[j] += delta1[j];
                }
                
                // Градиенты weights1
                for (int idx : input) {
                    if (idx < 0 || idx >= static_cast<int>(vs)) continue;
                    float* grad_ptr = &local_grad_w1[idx * hs];
                    for (size_t j = 0; j < hs; ++j) {
                        grad_ptr[j] += delta1[j];
                    }
                }
            }
            
            // Сбор градиентов
            #pragma omp critical
            {
                for (size_t i = 0; i < grad_w1.size(); ++i) grad_w1[i] += local_grad_w1[i];
                for (size_t i = 0; i < grad_w2.size(); ++i) grad_w2[i] += local_grad_w2[i];
                for (size_t i = 0; i < grad_b1.size(); ++i) grad_b1[i] += local_grad_b1[i];
                for (size_t i = 0; i < grad_b2.size(); ++i) grad_b2[i] += local_grad_b2[i];
            }
        }
        
        // Обновление весов с momentum
        float scale = lr / batch_size;
        
        #pragma omp parallel for
        for (size_t i = 0; i < vs * hs; ++i) {
            weights1[i] -= scale * grad_w1[i];
        }
        
        #pragma omp parallel for
        for (size_t i = 0; i < hs; ++i) {
            bias1[i] -= scale * grad_b1[i];
        }
        
        #pragma omp parallel for
        for (size_t i = 0; i < hs * vs; ++i) {
            weights2[i] -= scale * grad_w2[i];
        }
        
        #pragma omp parallel for
        for (size_t i = 0; i < vs; ++i) {
            bias2[i] -= scale * grad_b2[i];
        }
        
        return total_loss / batch_size;
    }
    
    int predict_next(const std::vector<int>& context) {
        std::vector<float> hidden(hidden_size);
        std::vector<float> output(vocab_size);
        
        forward_optimized(context, hidden, output);
        
        return std::distance(output.begin(), 
                           std::max_element(output.begin(), output.end()));
    }
    
    void save(const std::string& filename) {
        std::ofstream file(filename, std::ios::binary);
        
        file.write(reinterpret_cast<const char*>(&vocab_size), sizeof(vocab_size));
        file.write(reinterpret_cast<const char*>(&hidden_size), sizeof(hidden_size));
        
        file.write(reinterpret_cast<const char*>(weights1.data()), 
                  weights1.size() * sizeof(float));
        file.write(reinterpret_cast<const char*>(weights2.data()), 
                  weights2.size() * sizeof(float));
        file.write(reinterpret_cast<const char*>(bias1.data()), 
                  bias1.size() * sizeof(float));
        file.write(reinterpret_cast<const char*>(bias2.data()), 
                  bias2.size() * sizeof(float));
        
        file.close();
    }
    
    bool load(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) return false;
        
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
        return true;
    }
    
    std::mt19937& get_rng() { return rng; }
    
    size_t get_hidden_size() const { return hidden_size; }
    size_t get_vocab_size() const { return vocab_size; }
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

std::vector<std::string> load_texts_fast(const std::string& data_dir) {
    std::vector<std::string> all_words;
    
    if (!fs::exists(data_dir)) {
        std::cerr << "Ошибка: директория " << data_dir << " не существует!" << std::endl;
        return all_words;
    }
    
    std::vector<std::string> files;
    for (const auto& entry : fs::directory_iterator(data_dir)) {
        if (entry.path().extension() == ".txt") {
            files.push_back(entry.path().string());
        }
    }
    
    if (files.empty()) {
        std::cerr << "Ошибка: не найдено .txt файлов!" << std::endl;
        return all_words;
    }
    
    std::cout << "Найдено файлов: " << files.size() << std::endl;
    
    // Читаем все файлы последовательно для лучшей производительности диска
    for (const auto& filepath : files) {
        std::ifstream file(filepath);
        if (file.is_open()) {
            std::stringstream buffer;
            buffer << file.rdbuf();
            std::string text = buffer.str();
            file.close();
            
            // Быстрый парсинг слов
            std::string word;
            word.reserve(64);
            for (char c : text) {
                if (std::isspace(c) || std::ispunct(c)) {
                    if (!word.empty()) {
                        std::transform(word.begin(), word.end(), word.begin(), ::tolower);
                        all_words.push_back(word);
                        word.clear();
                    }
                } else {
                    word += c;
                }
            }
            if (!word.empty()) {
                std::transform(word.begin(), word.end(), word.begin(), ::tolower);
                all_words.push_back(word);
            }
        }
    }
    
    return all_words;
}

// ==================== Основная функция ====================
int main() {
    try {
        std::cout << "=========================================" << std::endl;
        std::cout << "   AvoAI - Адаптивное обучение на данных" << std::endl;
        std::cout << "=========================================" << std::endl;
        
        HardwareDetector hw;
        hw.print_info();
        
        auto start_total = std::chrono::high_resolution_clock::now();
        
        std::cout << "\n1. Загрузка данных..." << std::endl;
        auto words = load_texts_fast("data/");
        
        if (words.empty()) {
            std::cerr << "Ошибка: не загружены слова!" << std::endl;
            return 1;
        }
        
        std::cout << "✓ Загружено слов: " << words.size() << std::endl;
        
        std::cout << "\n2. Создание словаря..." << std::endl;
        AdaptiveVocabulary vocab(50000);  // Ограничиваем размер словаря
        
        // Добавляем только уникальные слова
        std::unordered_map<std::string, int> word_freq;
        for (const auto& word : words) {
            word_freq[word]++;
        }
        
        // Добавляем самые частые слова (первые 50к)
        std::vector<std::pair<std::string, int>> sorted_words(word_freq.begin(), word_freq.end());
        std::sort(sorted_words.begin(), sorted_words.end(),
                 [](const auto& a, const auto& b) { return a.second > b.second; });
        
        size_t vocab_limit = std::min(sorted_words.size(), static_cast<size_t>(50000));
        for (size_t i = 0; i < vocab_limit; ++i) {
            vocab.add_word(sorted_words[i].first);
        }
        
        std::cout << "✓ Словарь создан: " << vocab.size() << " самых частых слов" << std::endl;
        
        std::cout << "\n3. Подготовка данных..." << std::endl;
        
        int context_size = 3;
        std::vector<std::pair<std::vector<int>, int>> training_data;
        
        // Создаем обучающие примеры с проверкой индексов
        size_t valid_pairs = 0;
        for (size_t i = 0; i + context_size < words.size(); ++i) {
            std::vector<int> context;
            bool valid = true;
            
            for (int j = 0; j < context_size; ++j) {
                int idx = vocab.get_index(words[i + j]);
                if (idx == -1) {
                    valid = false;
                    break;
                }
                context.push_back(idx);
            }
            
            if (valid) {
                int target_idx = vocab.get_index(words[i + context_size]);
                if (target_idx != -1) {
                    training_data.emplace_back(context, target_idx);
                    valid_pairs++;
                }
            }
            
            // Ограничиваем количество примеров для разумного времени обучения
            if (valid_pairs >= 500000) {
                std::cout << "  Достигнут лимит в 500к примеров, останавливаем сбор данных..." << std::endl;
                break;
            }
        }
        
        std::cout << "✓ Создано примеров: " << training_data.size() << std::endl;
        
        if (training_data.empty()) {
            std::cerr << "Ошибка: нет данных для обучения!" << std::endl;
            return 1;
        }
        
        std::cout << "\n4. Создание адаптивной нейросети..." << std::endl;
        
        // ПАРАМЕТРЫ ОБУЧЕНИЯ ЗАВИСЯТ ОТ ОБЪЕМА ДАННЫХ
        int epochs;
        int batch_size;
        float learning_rate;
        
        if (training_data.size() < 10000) {
            epochs = 50;
            batch_size = 32;
            learning_rate = 0.02f;
        } else if (training_data.size() < 50000) {
            epochs = 30;
            batch_size = 64;
            learning_rate = 0.015f;
        } else if (training_data.size() < 200000) {
            epochs = 20;
            batch_size = 128;
            learning_rate = 0.01f;
        } else {
            epochs = 15;
            batch_size = 256;
            learning_rate = 0.005f;
        }
        
        std::cout << "Параметры обучения (на основе " << training_data.size() << " примеров):" << std::endl;
        std::cout << "  • Эпох: " << epochs << std::endl;
        std::cout << "  • Размер батча: " << batch_size << std::endl;
        std::cout << "  • Скорость обучения: " << learning_rate << std::endl;
        std::cout << "  • Потоков: " << hw.recommend_threads() << std::endl;
        
        AdaptiveNeuralNet model(vocab.size(), training_data.size(), learning_rate);
        
        auto start_train = std::chrono::high_resolution_clock::now();
        
        std::cout << "\nНачало обучения..." << std::endl;
        
        for (int epoch = 0; epoch < epochs; ++epoch) {
            auto epoch_start = std::chrono::high_resolution_clock::now();
            
            std::cout << "Эпоха " << (epoch + 1) << "/" << epochs << ": ";
            
            // Перемешиваем данные
            std::shuffle(training_data.begin(), training_data.end(), model.get_rng());
            
            float epoch_loss = 0.0f;
            float current_lr = learning_rate * powf(0.95f, epoch);
            
            // Обучение с прогресс-баром
            for (size_t batch_start = 0; batch_start < training_data.size(); batch_start += batch_size) {
                size_t batch_end = std::min(batch_start + batch_size, training_data.size());
                std::vector<std::pair<std::vector<int>, int>> batch(
                    training_data.begin() + batch_start,
                    training_data.begin() + batch_end
                );
                
                float batch_loss = model.train_batch_fast(batch, current_lr);
                epoch_loss += batch_loss * batch.size();
                
                if (batch_start % (batch_size * 50) == 0) {
                    float progress = static_cast<float>(batch_start) / training_data.size();
                    print_progress_bar(progress, 30, "Обучение ");
                }
            }
            
            auto epoch_end = std::chrono::high_resolution_clock::now();
            auto epoch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end - epoch_start);
            
            std::cout << "\r" << std::string(60, ' ') << "\r";
            float avg_loss = epoch_loss / training_data.size();
            std::cout << "Эпоха " << std::setw(2) << (epoch + 1) << " | ";
            std::cout << "Потери: " << std::fixed << std::setprecision(4) << avg_loss << " | ";
            std::cout << "Время: " << epoch_duration.count() << "мс" << std::endl;
        }
        
        auto end_train = std::chrono::high_resolution_clock::now();
        auto train_duration = std::chrono::duration_cast<std::chrono::seconds>(end_train - start_train);
        
        std::cout << "\n5. Сохранение модели..." << std::endl;
        
        if (!fs::exists("models")) {
            fs::create_directory("models");
        }
        
        model.save("models/model.bin");
        
        std::ofstream vocab_file("models/vocab.txt");
        for (size_t i = 0; i < vocab.size(); ++i) {
            vocab_file << vocab.get_word(i) << "\n";
        }
        vocab_file.close();
        
        std::cout << "✓ Модель сохранена" << std::endl;
        
        auto end_total = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(end_total - start_total);
        
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "ОБУЧЕНИЕ ЗАВЕРШЕНО!" << std::endl;
        std::cout << "Время обучения: " << train_duration.count() << " сек" << std::endl;
        std::cout << "Общее время: " << total_duration.count() << " сек" << std::endl;
        std::cout << "Скорость: ~" << std::fixed << std::setprecision(1) 
                  << (training_data.size() * epochs / train_duration.count() / 1000.0)
                  << " тыс. примеров/сек" << std::endl;
        std::cout << "Размер модели: " << model.get_hidden_size() << " скрытых нейронов" << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "\n✗ ОШИБКА: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}