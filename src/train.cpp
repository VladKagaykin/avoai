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

// ==================== Детектор железа ====================
class HardwareDetector {
private:
    int logical_cores;
    int physical_cores;
    size_t l3_cache_size;
    size_t total_ram;
    bool has_avx;
    bool has_avx2;
    bool has_fma;
    
public:
    HardwareDetector() {
        logical_cores = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
        physical_cores = std::max(1, logical_cores / 2);
        
        detect_cpu();
        detect_cache();
        detect_ram();
        
        // Оптимальные настройки для OMP
        omp_set_num_threads(physical_cores);
        omp_set_dynamic(0);
        omp_set_schedule(omp_sched_static, 0);
    }
    
    void detect_cpu() {
        unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
        
        __cpuid(1, eax, ebx, ecx, edx);
        has_avx = (ecx & (1 << 28)) != 0;
        has_fma = (ecx & (1 << 12)) != 0;
        
        __cpuid_count(7, 0, eax, ebx, ecx, edx);
        has_avx2 = (ebx & (1 << 5)) != 0;
    }
    
    void detect_cache() {
        if (logical_cores <= 4) {
            l3_cache_size = 4 * 1024 * 1024;
        } else if (logical_cores <= 8) {
            l3_cache_size = 8 * 1024 * 1024;
        } else {
            l3_cache_size = 16 * 1024 * 1024;
        }
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
        std::cout << "Физических ядер: " << physical_cores << std::endl;
        std::cout << "Кэш L3: " << l3_cache_size / 1024 / 1024 << " MB" << std::endl;
        std::cout << "ОЗУ: " << total_ram / 1024 / 1024 / 1024 << " GB" << std::endl;
        std::cout << "AVX: " << (has_avx ? "✓" : "✗") << std::endl;
        std::cout << "AVX2: " << (has_avx2 ? "✓" : "✗") << std::endl;
        std::cout << "FMA: " << (has_fma ? "✓" : "✗") << std::endl;
        std::cout << "=====================" << std::endl;
    }
    
    int get_logical_cores() const { return logical_cores; }
    int get_physical_cores() const { return physical_cores; }
    size_t get_l3_cache_size() const { return l3_cache_size; }
    size_t get_total_ram() const { return total_ram; }
    bool get_has_avx() const { return has_avx; }
    bool get_has_avx2() const { return has_avx2; }
    bool get_has_fma() const { return has_fma; }
    
    size_t recommend_hidden_size(size_t vocab_size) const {
        size_t cache_bytes = l3_cache_size;
        size_t optimal = static_cast<size_t>(
            sqrt(cache_bytes / (4.0 * vocab_size * sizeof(float)))
        );
        
        if (optimal < 128) optimal = 128;
        if (optimal > 512) optimal = 512;
        
        // Выравнивание для SIMD
        optimal = ((optimal + 31) / 32) * 32;
        return optimal;
    }
    
    int recommend_batch_size(size_t hidden_size, size_t vocab_size) const {
        size_t model_size_per_example = hidden_size * vocab_size * sizeof(float);
        size_t optimal_batch = l3_cache_size / model_size_per_example;
        
        if (optimal_batch < 16) optimal_batch = 16;
        if (optimal_batch > 128) optimal_batch = 128;
        
        int batch = 16;
        while (batch * 2 <= optimal_batch) batch *= 2;
        return batch;
    }
    
    int recommend_threads() const {
        return physical_cores;
    }
    
    std::string get_optimization_flags() const {
        if (has_avx2 && has_fma) return "AVX2+FMA";
        if (has_avx) return "AVX";
        return "SSE";
    }
    
    bool use_fast_math() const {
        return has_fma;
    }
};

// ==================== Адаптивный словарь ====================
class AdaptiveVocabulary {
private:
    std::unordered_map<std::string, int> word_to_idx;
    std::vector<std::string> idx_to_word;
    std::mutex mtx;
    int next_idx = 0;
    size_t max_size;
    
public:
    AdaptiveVocabulary(size_t available_memory_mb) {
        max_size = std::min<size_t>((available_memory_mb * 1024 * 1024) / 100, 50000);
        if (max_size < 10000) max_size = 10000;
        
        idx_to_word.reserve(max_size);
        word_to_idx.reserve(max_size);
    }
    
    int add_word(const std::string& word) {
        if (size() >= max_size) {
            return -1;
        }
        
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
    
    size_t get_max_size() const { return max_size; }
};

// ==================== УПРОЩЕННАЯ Адаптивная нейросеть ====================
class AdaptiveNeuralNet {
private:
    HardwareDetector hw;
    std::vector<float> weights1;  // vocab_size * hidden_size
    std::vector<float> weights2;  // hidden_size * vocab_size
    std::vector<float> bias1, bias2;
    
    size_t vocab_size;
    size_t hidden_size;
    float learning_rate;
    
    std::mt19937 rng;
    
public:
    AdaptiveNeuralNet(size_t vocab_size, const HardwareDetector& hw_detector, float lr = 0.01f)
        : hw(hw_detector), vocab_size(vocab_size), learning_rate(lr) {
        
        rng.seed(std::chrono::steady_clock::now().time_since_epoch().count());
        hidden_size = hw.recommend_hidden_size(vocab_size);
        
        std::cout << "Конфигурация нейросети:" << std::endl;
        std::cout << "  • Скрытый слой: " << hidden_size << " нейронов" << std::endl;
        std::cout << "  • Размер словаря: " << vocab_size << std::endl;
        std::cout << "  • Оптимизация: " << hw.get_optimization_flags() << std::endl;
        
        // Правильные размеры матриц
        weights1.resize(vocab_size * hidden_size, 0.0f);
        weights2.resize(hidden_size * vocab_size, 0.0f);
        bias1.resize(hidden_size, 0.0f);
        bias2.resize(vocab_size, 0.0f);
        
        initialize_weights();
    }
    
    void initialize_weights() {
        std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
        float stddev = sqrtf(2.0f / (vocab_size + hidden_size));
        
        // Инициализация weights1 (vocab_size x hidden_size)
        for (size_t i = 0; i < vocab_size * hidden_size; ++i) {
            weights1[i] = dist(rng) * stddev;
        }
        
        // Инициализация weights2 (hidden_size x vocab_size)
        for (size_t i = 0; i < hidden_size * vocab_size; ++i) {
            weights2[i] = dist(rng) * stddev;
        }
    }
    
    // Быстрая сигмоида
    inline float fast_sigmoid(float x) const {
        x = 0.5f * x;
        x = x / (1.0f + fabsf(x));
        return x + 0.5f;
    }
    
    inline float sigmoid_derivative(float x) const {
        return x * (1.0f - x);
    }
    
    // Упрощенный прямой проход
    void forward_optimized(const std::vector<int>& input,
                          std::vector<float>& hidden,
                          std::vector<float>& output) {
        
        hidden.assign(hidden_size, 0.0f);
        
        // Суммируем эмбеддинги слов (правильный доступ к памяти)
        for (int idx : input) {
            if (idx < 0 || idx >= static_cast<int>(vocab_size)) continue;
            
            const float* w_ptr = &weights1[idx * hidden_size];
            for (size_t j = 0; j < hidden_size; ++j) {
                hidden[j] += w_ptr[j];
            }
        }
        
        // Применяем сигмоиду и bias
        for (size_t j = 0; j < hidden_size; ++j) {
            hidden[j] = fast_sigmoid(hidden[j] + bias1[j]);
        }
        
        // Выходной слой
        output.assign(vocab_size, 0.0f);
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
    
    // Упрощенное обучение с проверкой границ
    float train_batch_simple(const std::vector<std::pair<std::vector<int>, int>>& batch,
                            float lr) {
        
        size_t batch_size = batch.size();
        const size_t hs = hidden_size;
        const size_t vs = vocab_size;
        
        // Градиенты
        std::vector<float> grad_w1(vs * hs, 0.0f);
        std::vector<float> grad_w2(hs * vs, 0.0f);
        std::vector<float> grad_b1(hs, 0.0f);
        std::vector<float> grad_b2(vs, 0.0f);
        
        float total_loss = 0.0f;
        const int context_size = 3; // Фиксированный размер контекста
        
        #pragma omp parallel reduction(+:total_loss)
        {
            // Локальные буферы для каждого потока
            std::vector<float> local_hidden(hs, 0.0f);
            std::vector<float> local_output(vs, 0.0f);
            std::vector<float> local_grad_w1(vs * hs, 0.0f);
            std::vector<float> local_grad_w2(hs * vs, 0.0f);
            std::vector<float> local_grad_b1(hs, 0.0f);
            std::vector<float> local_grad_b2(vs, 0.0f);
            
            #pragma omp for schedule(static)
            for (size_t b = 0; b < batch_size; ++b) {
                const auto& [input, target] = batch[b];
                
                // Проверка корректности индексов
                if (input.size() != context_size) continue;
                bool valid = true;
                for (int idx : input) {
                    if (idx < 0 || idx >= static_cast<int>(vs)) {
                        valid = false;
                        break;
                    }
                }
                if (!valid || target < 0 || target >= static_cast<int>(vs)) {
                    continue;
                }
                
                // Прямой проход
                std::fill(local_hidden.begin(), local_hidden.end(), 0.0f);
                
                for (int idx : input) {
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
                
                // Обратное распространение
                std::vector<float> delta2(vs, 0.0f);
                for (size_t i = 0; i < vs; ++i) {
                    delta2[i] = temp_output[i];
                }
                delta2[target] -= 1.0f;
                
                // Градиенты выходного слоя
                for (size_t j = 0; j < hs; ++j) {
                    float grad = local_hidden[j];
                    for (size_t i = 0; i < vs; ++i) {
                        local_grad_w2[j * vs + i] += delta2[i] * grad;
                    }
                }
                
                for (size_t i = 0; i < vs; ++i) {
                    local_grad_b2[i] += delta2[i];
                }
                
                // Градиенты скрытого слоя
                std::vector<float> delta1(hs, 0.0f);
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
                    for (size_t j = 0; j < hs; ++j) {
                        local_grad_w1[idx * hs + j] += delta1[j];
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
        
        // Обновление весов
        float scale = lr / batch_size;
        
        // Обновление weights1 и bias1
        for (size_t i = 0; i < vs * hs; ++i) {
            weights1[i] -= scale * grad_w1[i];
        }
        
        for (size_t j = 0; j < hs; ++j) {
            bias1[j] -= scale * grad_b1[j];
        }
        
        // Обновление weights2 и bias2
        for (size_t i = 0; i < hs * vs; ++i) {
            weights2[i] -= scale * grad_w2[i];
        }
        
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
    const HardwareDetector& get_hardware_detector() const { return hw; }
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

std::vector<std::string> load_texts_adaptive(const std::string& data_dir, int threads) {
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
    
    omp_set_num_threads(threads);
    std::vector<std::vector<std::string>> thread_words(threads);
    
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < files.size(); ++i) {
        std::ifstream file(files[i]);
        if (file.is_open()) {
            std::stringstream buffer;
            buffer << file.rdbuf();
            std::string text = buffer.str();
            file.close();
            
            // Быстрый парсинг слов
            std::string word;
            for (char c : text) {
                if (std::isspace(c) || std::ispunct(c)) {
                    if (!word.empty()) {
                        std::transform(word.begin(), word.end(), word.begin(), ::tolower);
                        thread_words[omp_get_thread_num()].push_back(word);
                        word.clear();
                    }
                } else {
                    word += c;
                }
            }
            if (!word.empty()) {
                std::transform(word.begin(), word.end(), word.begin(), ::tolower);
                thread_words[omp_get_thread_num()].push_back(word);
            }
        }
    }
    
    // Сбор всех слов
    size_t total_words = 0;
    for (const auto& words : thread_words) {
        total_words += words.size();
    }
    all_words.reserve(total_words);
    
    for (const auto& words : thread_words) {
        all_words.insert(all_words.end(), words.begin(), words.end());
    }
    
    return all_words;
}

// ==================== Основная функция ====================
int main() {
    try {
        std::cout << "=========================================" << std::endl;
        std::cout << "   AvoAI - Оптимизированное обучение    " << std::endl;
        std::cout << "=========================================" << std::endl;
        
        HardwareDetector hw;
        hw.print_info();
        
        auto start_total = std::chrono::high_resolution_clock::now();
        
        std::cout << "\n1. Загрузка данных..." << std::endl;
        auto words = load_texts_adaptive("data/", hw.recommend_threads());
        
        if (words.empty()) {
            std::cerr << "Ошибка: не загружены слова!" << std::endl;
            return 1;
        }
        
        std::cout << "✓ Загружено слов: " << words.size() << std::endl;
        
        std::cout << "\n2. Создание словаря..." << std::endl;
        size_t available_memory_mb = hw.get_total_ram() / 1024 / 1024 / 4;
        AdaptiveVocabulary vocab(available_memory_mb);
        
        // Добавляем слова в словарь
        for (const auto& word : words) {
            vocab.add_word(word);
        }
        
        std::cout << "✓ Словарь создан: " << vocab.size() << " слов" << std::endl;
        
        std::cout << "\n3. Подготовка данных..." << std::endl;
        
        int context_size = 3;
        std::vector<std::pair<std::vector<int>, int>> training_data;
        
        // Создаем обучающие примеры
        for (size_t i = 0; i < words.size() - context_size; ++i) {
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
                }
            }
        }
        
        std::cout << "✓ Создано примеров: " << training_data.size() << std::endl;
        
        if (training_data.empty()) {
            std::cerr << "Ошибка: нет данных для обучения!" << std::endl;
            return 1;
        }
        
        std::cout << "\n4. Создание адаптивной нейросети..." << std::endl;
        AdaptiveNeuralNet model(vocab.size(), hw, 0.01f);
        
        int epochs = 5; // Уменьшил количество эпох для тестирования
        int batch_size = 32;
        float learning_rate = 0.01f;
        
        std::cout << "Параметры обучения:" << std::endl;
        std::cout << "  • Эпох: " << epochs << std::endl;
        std::cout << "  • Размер батча: " << batch_size << std::endl;
        std::cout << "  • Скорость обучения: " << learning_rate << std::endl;
        std::cout << "  • Потоков: " << hw.recommend_threads() << std::endl;
        
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
                
                float batch_loss = model.train_batch_simple(batch, current_lr);
                epoch_loss += batch_loss * batch.size();
                
                if (batch_start % (batch_size * 100) == 0) {
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
        std::cout << std::string(50, '=') << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "\n✗ ОШИБКА: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}