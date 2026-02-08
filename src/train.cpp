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
#include <unistd.h>  // Добавлено для sysconf
#include <omp.h>

namespace fs = std::filesystem;

// ==================== Детектор железа ====================
class HardwareDetector {
private:
    int logical_cores;
    int physical_cores;
    size_t l1_cache_size;
    size_t l2_cache_size;
    size_t l3_cache_size;
    size_t total_ram;
    bool has_avx;
    bool has_avx2;
    bool has_fma;
    
public:
    HardwareDetector() {
        detect_cpu();
        detect_cache();
        detect_ram();
    }
    
    void detect_cpu() {
        // Количество ядер и потоков
        logical_cores = std::thread::hardware_concurrency();
        physical_cores = logical_cores / 2; // Оценка для Hyper-Threading
        
        // Определение возможностей CPU
        unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;
        
        // CPUID для AVX/AVX2/FMA
        __cpuid(1, eax, ebx, ecx, edx);
        has_avx = (ecx & (1 << 28)) != 0;
        has_fma = (ecx & (1 << 12)) != 0;
        
        __cpuid_count(7, 0, eax, ebx, ecx, edx);
        has_avx2 = (ebx & (1 << 5)) != 0;
    }
    
    void detect_cache() {
        // Примерные размеры кэша для типичных процессоров
        if (logical_cores <= 4) {
            l1_cache_size = 32 * 1024;     // 32 KB
            l2_cache_size = 256 * 1024;    // 256 KB
            l3_cache_size = 4 * 1024 * 1024; // 4 MB
        } else if (logical_cores <= 8) {
            l1_cache_size = 32 * 1024;     // 32 KB
            l2_cache_size = 256 * 1024;    // 256 KB
            l3_cache_size = 8 * 1024 * 1024; // 8 MB
        } else {
            l1_cache_size = 32 * 1024;     // 32 KB
            l2_cache_size = 512 * 1024;    // 512 KB
            l3_cache_size = 16 * 1024 * 1024; // 16 MB
        }
    }
    
    void detect_ram() {
        long pages = sysconf(_SC_PHYS_PAGES);
        long page_size = sysconf(_SC_PAGE_SIZE);
        if (pages > 0 && page_size > 0) {
            total_ram = pages * page_size;
        } else {
            total_ram = 8ULL * 1024 * 1024 * 1024; // 8GB по умолчанию
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
    
    // Геттеры для приватных полей
    int get_logical_cores() const { return logical_cores; }
    int get_physical_cores() const { return physical_cores; }
    size_t get_l3_cache_size() const { return l3_cache_size; }
    size_t get_total_ram() const { return total_ram; }
    bool get_has_avx() const { return has_avx; }
    bool get_has_avx2() const { return has_avx2; }
    bool get_has_fma() const { return has_fma; }
    
    // Рекомендации для нейросети
    size_t recommend_hidden_size(size_t vocab_size) const {
        // Основано на размере кэша и ОЗУ
        size_t target_memory_mb = total_ram / 1024 / 1024 / 4; // 25% ОЗУ
        
        // Формула: hidden_size ~ sqrt(доступная_память / vocab_size)
        size_t recommended = static_cast<size_t>(
            sqrt(target_memory_mb * 1024.0 * 1024.0 / (vocab_size * sizeof(float)))
        );
        
        // Ограничения
        if (recommended < 128) recommended = 128;
        if (recommended > 2048) recommended = 2048;
        
        // Выравнивание под векторные инструкции
        recommended = ((recommended + 31) / 32) * 32;
        
        return recommended;
    }
    
    int recommend_batch_size(size_t hidden_size, size_t vocab_size) const {
        // Оптимальный батч для заполнения кэша
        size_t model_size_per_example = (hidden_size * vocab_size + vocab_size * hidden_size) * sizeof(float);
        size_t optimal_batch = l3_cache_size / model_size_per_example;
        
        if (optimal_batch < 8) optimal_batch = 8;
        if (optimal_batch > 256) optimal_batch = 256;
        
        // Округление до степени двойки
        int batch = 32;
        while (batch * 2 <= optimal_batch) batch *= 2;
        
        return batch;
    }
    
    int recommend_threads() const {
        // Для CPU-bound задач используем все логические ядра
        // Для memory-bound может быть меньше
        return logical_cores;
    }
    
    std::string get_optimization_flags() const {
        if (has_avx2 && has_fma) return "AVX2+FMA";
        if (has_avx) return "AVX";
        return "SSE";
    }
    
    bool use_fast_math() const {
        // Использовать быструю математику если есть FMA
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
        // Оцениваем максимальный размер словаря по доступной памяти
        // ~100 байт на слово в среднем
        max_size = (available_memory_mb * 1024 * 1024) / 100;
        if (max_size > 1000000) max_size = 1000000; // Максимум 1M слов
        if (max_size < 10000) max_size = 10000;     // Минимум 10K слов
        
        idx_to_word.reserve(max_size);
    }
    
    int add_word(const std::string& word) {
        if (size() >= max_size) {
            // Если словарь переполнен, возвращаем наиболее частый индекс
            return 0; // Простой fallback
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
        if (idx >= 0 && idx < idx_to_word.size()) {
            return idx_to_word[idx];
        }
        return empty;
    }
    
    size_t size() const { return word_to_idx.size(); }
    
    const std::vector<std::string>& get_words() const { return idx_to_word; }
    
    size_t get_max_size() const { return max_size; }
};

// ==================== Адаптивная нейросеть ====================
class AdaptiveNeuralNet {
private:
    HardwareDetector hw;
    std::vector<float> weights1;  // hidden_size * vocab_size
    std::vector<float> weights2;  // vocab_size * hidden_size
    std::vector<float> bias1, bias2;
    
    size_t vocab_size;
    size_t hidden_size;
    size_t aligned_hidden;
    size_t aligned_vocab;
    float learning_rate;
    
    std::mt19937 rng;
    
    // Методы оптимизации
    enum OptimMethod { 
        METHOD_SIMPLE, 
        METHOD_OMP, 
        METHOD_VECTOR,
        METHOD_AVX 
    };
    OptimMethod current_method;
    
public:
    AdaptiveNeuralNet(size_t vocab_size, const HardwareDetector& hw_detector, float lr = 0.01f)
        : hw(hw_detector), vocab_size(vocab_size), learning_rate(lr) {
        
        rng.seed(std::random_device{}());
        
        // Автоматический подбор размера скрытого слоя
        hidden_size = hw.recommend_hidden_size(vocab_size);
        
        // Выравнивание для векторных инструкций
        aligned_hidden = ((hidden_size + 31) / 32) * 32;
        aligned_vocab = ((vocab_size + 31) / 32) * 32;
        
        // Выбор метода оптимизации
        if (hw.get_optimization_flags() == "AVX2+FMA") {
            current_method = METHOD_AVX;
        } else if (hw.get_optimization_flags() == "AVX") {
            current_method = METHOD_VECTOR;
        } else {
            current_method = METHOD_OMP;
        }
        
        std::cout << "Конфигурация нейросети:" << std::endl;
        std::cout << "  • Скрытый слой: " << hidden_size << " нейронов" << std::endl;
        std::cout << "  • Метод оптимизации: " << get_method_name() << std::endl;
        std::cout << "  • Потоков: " << hw.recommend_threads() << std::endl;
        
        // Инициализация памяти
        weights1.resize(aligned_hidden * aligned_vocab, 0.0f);
        weights2.resize(aligned_vocab * aligned_hidden, 0.0f);
        bias1.resize(aligned_hidden, 0.0f);
        bias2.resize(aligned_vocab, 0.0f);
        
        // Инициализация весов
        initialize_weights();
    }
    
    void initialize_weights() {
        std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
        
        // Инициализация He для ReLU-like активаций
        float stddev = sqrtf(2.0f / vocab_size);
        
        #pragma omp parallel for
        for (size_t i = 0; i < hidden_size; ++i) {
            for (size_t j = 0; j < vocab_size; ++j) {
                weights1[i * aligned_vocab + j] = dist(rng) * stddev;
            }
        }
        
        stddev = sqrtf(2.0f / hidden_size);
        #pragma omp parallel for
        for (size_t i = 0; i < vocab_size; ++i) {
            for (size_t j = 0; j < hidden_size; ++j) {
                weights2[i * aligned_hidden + j] = dist(rng) * stddev;
            }
        }
    }
    
    std::string get_method_name() const {
        switch (current_method) {
            case METHOD_AVX: return "AVX2+FMA";
            case METHOD_VECTOR: return "AVX/SSE";
            case METHOD_OMP: return "OpenMP";
            default: return "Простой";
        }
    }
    
    // Адаптивная функция активации
    inline float adaptive_activation(float x) const {
        if (hw.use_fast_math()) {
            // Быстрая аппроксимация для FMA
            x = 0.5f * x;
            x = x / (1.0f + fabsf(x));
            return x + 0.5f;
        } else {
            // Точная сигмоида
            return 1.0f / (1.0f + expf(-x));
        }
    }
    
    inline float activation_derivative(float x) const {
        return x * (1.0f - x);
    }
    
    // Оптимизированный forward в зависимости от железа
    void forward_adaptive(const std::vector<int>& input,
                          std::vector<float>& hidden,
                          std::vector<float>& output) {
        
        switch (current_method) {
            case METHOD_AVX:
                forward_avx(input, hidden, output);
                break;
            case METHOD_VECTOR:
                forward_vector(input, hidden, output);
                break;
            case METHOD_OMP:
                forward_omp(input, hidden, output);
                break;
            default:
                forward_simple(input, hidden, output);
                break;
        }
    }
    
    void forward_simple(const std::vector<int>& input,
                        std::vector<float>& hidden,
                        std::vector<float>& output) {
        
        hidden.assign(aligned_hidden, 0.0f);
        
        for (int idx : input) {
            float* w_ptr = &weights1[idx];
            for (size_t j = 0; j < hidden_size; ++j) {
                hidden[j] += w_ptr[j * aligned_vocab];
            }
        }
        
        for (size_t j = 0; j < hidden_size; ++j) {
            hidden[j] = adaptive_activation(hidden[j] + bias1[j]);
        }
        
        output.resize(vocab_size);
        for (size_t i = 0; i < vocab_size; ++i) {
            float sum = 0.0f;
            float* w2_ptr = &weights2[i * aligned_hidden];
            for (size_t j = 0; j < hidden_size; ++j) {
                sum += w2_ptr[j] * hidden[j];
            }
            output[i] = sum + bias2[i];
        }
        
        softmax(output);
    }
    
    void forward_omp(const std::vector<int>& input,
                     std::vector<float>& hidden,
                     std::vector<float>& output) {
        
        hidden.assign(aligned_hidden, 0.0f);
        
        #pragma omp parallel for
        for (size_t i = 0; i < input.size(); ++i) {
            int idx = input[i];
            float* w_ptr = &weights1[idx];
            #pragma omp simd
            for (size_t j = 0; j < hidden_size; ++j) {
                #pragma omp atomic
                hidden[j] += w_ptr[j * aligned_vocab];
            }
        }
        
        #pragma omp parallel for
        for (size_t j = 0; j < hidden_size; ++j) {
            hidden[j] = adaptive_activation(hidden[j] + bias1[j]);
        }
        
        output.resize(vocab_size);
        #pragma omp parallel for
        for (size_t i = 0; i < vocab_size; ++i) {
            float sum = 0.0f;
            float* w2_ptr = &weights2[i * aligned_hidden];
            #pragma omp simd reduction(+:sum)
            for (size_t j = 0; j < hidden_size; ++j) {
                sum += w2_ptr[j] * hidden[j];
            }
            output[i] = sum + bias2[i];
        }
        
        softmax(output);
    }
    
    void forward_vector(const std::vector<int>& input,
                        std::vector<float>& hidden,
                        std::vector<float>& output) {
        
        // Векторизованная версия
        forward_omp(input, hidden, output); // Пока используем OMP версию
    }
    
    void forward_avx(const std::vector<int>& input,
                     std::vector<float>& hidden,
                     std::vector<float>& output) {
        
        // AVX версия
        forward_omp(input, hidden, output); // Пока используем OMP версию
    }
    
    void softmax(std::vector<float>& logits) {
        float max_val = *std::max_element(logits.begin(), logits.end());
        float sum = 0.0f;
        
        #pragma omp parallel for reduction(+:sum)
        for (size_t i = 0; i < logits.size(); ++i) {
            logits[i] = expf(logits[i] - max_val);
            sum += logits[i];
        }
        
        float inv_sum = 1.0f / sum;
        #pragma omp simd
        for (size_t i = 0; i < logits.size(); ++i) {
            logits[i] *= inv_sum;
        }
    }
    
    float train_batch_adaptive(const std::vector<std::pair<std::vector<int>, int>>& batch,
                               float lr) {
        
        // Автоматический подбор размера блока для OMP
        int chunk_size = std::max(1, static_cast<int>(batch.size() / (hw.recommend_threads() * 4)));
        
        std::vector<float> grad_w1(aligned_hidden * aligned_vocab, 0.0f);
        std::vector<float> grad_w2(aligned_vocab * aligned_hidden, 0.0f);
        std::vector<float> grad_b1(aligned_hidden, 0.0f);
        std::vector<float> grad_b2(aligned_vocab, 0.0f);
        
        float total_loss = 0.0f;
        size_t batch_size = batch.size();
        
        #pragma omp parallel reduction(+:total_loss)
        {
            std::vector<float> local_hidden(aligned_hidden, 0.0f);
            std::vector<float> local_output(vocab_size, 0.0f);
            
            #pragma omp for schedule(dynamic, chunk_size)
            for (size_t b = 0; b < batch_size; ++b) {
                const auto& [input, target] = batch[b];
                
                // Forward
                std::fill(local_hidden.begin(), local_hidden.end(), 0.0f);
                
                for (int idx : input) {
                    float* w_ptr = &weights1[idx];
                    for (size_t j = 0; j < hidden_size; ++j) {
                        local_hidden[j] += w_ptr[j * aligned_vocab];
                    }
                }
                
                for (size_t j = 0; j < hidden_size; ++j) {
                    local_hidden[j] = adaptive_activation(local_hidden[j] + bias1[j]);
                }
                
                for (size_t i = 0; i < vocab_size; ++i) {
                    float sum = 0.0f;
                    float* w2_ptr = &weights2[i * aligned_hidden];
                    for (size_t j = 0; j < hidden_size; ++j) {
                        sum += w2_ptr[j] * local_hidden[j];
                    }
                    local_output[i] = sum + bias2[i];
                }
                
                softmax(local_output);
                
                // Loss
                total_loss += -logf(local_output[target] + 1e-8f);
                
                // Backward (локально в потоке)
                std::vector<float> local_grad_w1(aligned_hidden * aligned_vocab, 0.0f);
                std::vector<float> local_grad_w2(aligned_vocab * aligned_hidden, 0.0f);
                std::vector<float> local_grad_b1(aligned_hidden, 0.0f);
                std::vector<float> local_grad_b2(aligned_vocab, 0.0f);
                
                std::vector<float> delta2(vocab_size, 0.0f);
                delta2[target] = -1.0f;
                
                for (size_t i = 0; i < vocab_size; ++i) {
                    delta2[i] += local_output[i];
                    local_grad_b2[i] += delta2[i];
                    
                    float* grad_ptr = &local_grad_w2[i * aligned_hidden];
                    float delta = delta2[i];
                    for (size_t j = 0; j < hidden_size; ++j) {
                        grad_ptr[j] += delta * local_hidden[j];
                    }
                }
                
                std::vector<float> delta1(hidden_size, 0.0f);
                for (size_t j = 0; j < hidden_size; ++j) {
                    float sum = 0.0f;
                    for (size_t i = 0; i < vocab_size; ++i) {
                        sum += weights2[i * aligned_hidden + j] * delta2[i];
                    }
                    delta1[j] = sum * activation_derivative(local_hidden[j]);
                    local_grad_b1[j] += delta1[j];
                    
                    float* grad_ptr = &local_grad_w1[j * aligned_vocab];
                    float delta = delta1[j];
                    for (int idx : input) {
                        grad_ptr[idx] += delta;
                    }
                }
                
                // Аккумулирование градиентов
                #pragma omp critical
                {
                    for (size_t i = 0; i < grad_w1.size(); ++i) grad_w1[i] += local_grad_w1[i];
                    for (size_t i = 0; i < grad_w2.size(); ++i) grad_w2[i] += local_grad_w2[i];
                    for (size_t i = 0; i < grad_b1.size(); ++i) grad_b1[i] += local_grad_b1[i];
                    for (size_t i = 0; i < grad_b2.size(); ++i) grad_b2[i] += local_grad_b2[i];
                }
            }
        }
        
        // Обновление весов
        float scale = lr / batch_size;
        
        #pragma omp parallel for
        for (size_t j = 0; j < hidden_size; ++j) {
            bias1[j] -= scale * grad_b1[j];
            for (size_t i = 0; i < vocab_size; ++i) {
                weights1[j * aligned_vocab + i] -= scale * grad_w1[j * aligned_vocab + i];
            }
        }
        
        #pragma omp parallel for
        for (size_t i = 0; i < vocab_size; ++i) {
            bias2[i] -= scale * grad_b2[i];
            for (size_t j = 0; j < hidden_size; ++j) {
                weights2[i * aligned_hidden + j] -= scale * grad_w2[i * aligned_hidden + j];
            }
        }
        
        return total_loss / batch_size;
    }
    
    int predict_next(const std::vector<int>& context) {
        std::vector<float> hidden(aligned_hidden, 0.0f);
        std::vector<float> output(vocab_size, 0.0f);
        
        forward_adaptive(context, hidden, output);
        
        return std::distance(output.begin(), 
                           std::max_element(output.begin(), output.end()));
    }
    
    void save(const std::string& filename) {
        std::ofstream file(filename, std::ios::binary);
        
        file.write(reinterpret_cast<const char*>(&vocab_size), sizeof(vocab_size));
        file.write(reinterpret_cast<const char*>(&hidden_size), sizeof(hidden_size));
        file.write(reinterpret_cast<const char*>(&learning_rate), sizeof(learning_rate));
        
        size_t w1_size = weights1.size();
        size_t w2_size = weights2.size();
        file.write(reinterpret_cast<const char*>(&w1_size), sizeof(w1_size));
        file.write(reinterpret_cast<const char*>(&w2_size), sizeof(w2_size));
        
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
    
    void load(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        
        file.read(reinterpret_cast<char*>(&vocab_size), sizeof(vocab_size));
        file.read(reinterpret_cast<char*>(&hidden_size), sizeof(hidden_size));
        file.read(reinterpret_cast<char*>(&learning_rate), sizeof(learning_rate));
        
        size_t w1_size, w2_size;
        file.read(reinterpret_cast<char*>(&w1_size), sizeof(w1_size));
        file.read(reinterpret_cast<char*>(&w2_size), sizeof(w2_size));
        
        weights1.resize(w1_size);
        weights2.resize(w2_size);
        
        file.read(reinterpret_cast<char*>(weights1.data()), 
                 weights1.size() * sizeof(float));
        file.read(reinterpret_cast<char*>(weights2.data()), 
                 weights2.size() * sizeof(float));
        
        aligned_hidden = ((hidden_size + 31) / 32) * 32;
        aligned_vocab = ((vocab_size + 31) / 32) * 32;
        
        bias1.resize(aligned_hidden);
        bias2.resize(aligned_vocab);
        
        file.read(reinterpret_cast<char*>(bias1.data()), 
                 bias1.size() * sizeof(float));
        file.read(reinterpret_cast<char*>(bias2.data()), 
                 bias2.size() * sizeof(float));
        
        file.close();
    }
    
    std::mt19937& get_rng() { return rng; }
    
    size_t get_hidden_size() const { return hidden_size; }
    size_t get_vocab_size() const { return vocab_size; }
    std::string get_optimization_method() const { return get_method_name(); }
    const HardwareDetector& get_hardware_detector() const { return hw; }
};

// ==================== Адаптивный тренировщик ====================
class AdaptiveTrainer {
private:
    HardwareDetector hw;
    const AdaptiveNeuralNet* model;
    
public:
    AdaptiveTrainer(const AdaptiveNeuralNet* net) : model(net) {
        if (net) {
            hw = net->get_hardware_detector();
        }
    }
    
    int recommend_epochs(size_t dataset_size) const {
        // Больше данных - меньше эпох нужно
        if (dataset_size < 10000) return 50;
        if (dataset_size < 100000) return 30;
        if (dataset_size < 1000000) return 20;
        return 15;
    }
    
    int recommend_batch_size() const {
        if (!model) return 32;
        return hw.recommend_batch_size(model->get_hidden_size(), model->get_vocab_size());
    }
    
    float recommend_learning_rate() const {
        // Адаптивная скорость обучения
        if (!model) return 0.01f;
        
        size_t total_params = model->get_hidden_size() * model->get_vocab_size() * 2;
        if (total_params > 10000000) return 0.005f;
        if (total_params > 1000000) return 0.01f;
        return 0.02f;
    }
    
    int recommend_context_size() const {
        // Больше кэша - больше контекст
        size_t l3_mb = hw.get_l3_cache_size() / 1024 / 1024;
        if (l3_mb >= 16) return 4;
        if (l3_mb >= 8) return 3;
        return 2;
    }
    
    void print_training_plan(size_t dataset_size) const {
        std::cout << "\n=== План обучения ===" << std::endl;
        std::cout << "Эпох: " << recommend_epochs(dataset_size) << std::endl;
        std::cout << "Размер батча: " << recommend_batch_size() << std::endl;
        std::cout << "Скорость обучения: " << recommend_learning_rate() << std::endl;
        std::cout << "Размер контекста: " << recommend_context_size() << std::endl;
        std::cout << "Потоков: " << hw.recommend_threads() << std::endl;
        std::cout << "====================" << std::endl;
    }
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
    
    // Адаптивная параллельная загрузка
    std::vector<std::vector<std::string>> thread_words(threads);
    
    #pragma omp parallel for num_threads(threads) schedule(dynamic)
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
                        thread_words[omp_get_thread_num()].push_back(word);
                        word.clear();
                    }
                } else {
                    word += std::tolower(c);
                }
            }
            if (!word.empty()) {
                thread_words[omp_get_thread_num()].push_back(word);
            }
        }
    }
    
    // Объединение
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
        std::cout << "   AvoAI - Адаптивное обучение         " << std::endl;
        std::cout << "=========================================" << std::endl;
        
        // Детекция железа
        HardwareDetector hw;
        hw.print_info();
        
        // Настройка OpenMP под железо
        omp_set_num_threads(hw.recommend_threads());
        omp_set_schedule(omp_sched_dynamic, 16);
        
        auto start_total = std::chrono::high_resolution_clock::now();
        
        // 1. Загрузка данных
        std::cout << "\n1. Загрузка данных..." << std::endl;
        auto words = load_texts_adaptive("data/", hw.recommend_threads());
        
        if (words.empty()) {
            std::cerr << "Ошибка: не загружены слова!" << std::endl;
            return 1;
        }
        
        std::cout << "✓ Загружено слов: " << words.size() << std::endl;
        
        // 2. Создание адаптивного словаря
        std::cout << "\n2. Создание словаря..." << std::endl;
        size_t available_memory_mb = hw.get_total_ram() / 1024 / 1024 / 4; // 25% ОЗУ
        AdaptiveVocabulary vocab(available_memory_mb);
        
        #pragma omp parallel for
        for (size_t i = 0; i < words.size(); ++i) {
            vocab.add_word(words[i]);
        }
        
        std::cout << "✓ Словарь создан: " << vocab.size() << " / " << vocab.get_max_size() << " слов" << std::endl;
        
        // 3. Подготовка данных
        std::cout << "\n3. Подготовка данных..." << std::endl;
        
        // Создаем временный детектор для рекомендаций до создания модели
        HardwareDetector temp_hw;
        int context_size = 3; // По умолчанию
        size_t l3_mb = temp_hw.get_l3_cache_size() / 1024 / 1024;
        if (l3_mb >= 16) context_size = 4;
        else if (l3_mb >= 8) context_size = 3;
        else context_size = 2;
        
        std::vector<std::pair<std::vector<int>, int>> training_data;
        size_t total_pairs = words.size() - context_size;
        
        #pragma omp parallel
        {
            std::vector<std::pair<std::vector<int>, int>> local_data;
            
            #pragma omp for nowait schedule(dynamic, 1000)
            for (size_t i = 0; i < total_pairs; ++i) {
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
                        local_data.emplace_back(context, target_idx);
                    }
                }
            }
            
            #pragma omp critical
            training_data.insert(training_data.end(), local_data.begin(), local_data.end());
        }
        
        std::cout << "✓ Создано примеров: " << training_data.size() << std::endl;
        
        if (training_data.empty()) {
            std::cerr << "Ошибка: нет данных для обучения!" << std::endl;
            return 1;
        }
        
        // 4. Создание и обучение адаптивной модели
        std::cout << "\n4. Создание адаптивной нейросети..." << std::endl;
        AdaptiveNeuralNet model(vocab.size(), hw);
        
        // Адаптивный планировщик обучения
        AdaptiveTrainer trainer(&model);
        trainer.print_training_plan(training_data.size());
        
        int epochs = trainer.recommend_epochs(training_data.size());
        int batch_size = trainer.recommend_batch_size();
        float learning_rate = trainer.recommend_learning_rate();
        
        auto start_train = std::chrono::high_resolution_clock::now();
        
        std::cout << "\nНачало обучения..." << std::endl;
        
        for (int epoch = 0; epoch < epochs; ++epoch) {
            auto epoch_start = std::chrono::high_resolution_clock::now();
            
            std::cout << "Эпоха " << (epoch + 1) << "/" << epochs << ": ";
            
            // Перемешивание
            std::shuffle(training_data.begin(), training_data.end(), model.get_rng());
            
            float epoch_loss = 0.0f;
            float current_lr = learning_rate * powf(0.95f, epoch);
            
            // Обучение
            #pragma omp parallel for reduction(+:epoch_loss) schedule(dynamic)
            for (size_t batch_start = 0; batch_start < training_data.size(); batch_start += batch_size) {
                size_t batch_end = std::min(batch_start + batch_size, training_data.size());
                std::vector<std::pair<std::vector<int>, int>> batch(
                    training_data.begin() + batch_start,
                    training_data.begin() + batch_end
                );
                
                float batch_loss = model.train_batch_adaptive(batch, current_lr);
                epoch_loss += batch_loss * batch.size();
                
                if (omp_get_thread_num() == 0) {
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
        
        // 5. Сохранение
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
        
        // 6. Тестирование
        std::cout << "\n6. Тестирование..." << std::endl;
        
        if (!training_data.empty()) {
            std::cout << "\nПример генерации:" << std::endl;
            auto& [context, target] = training_data[0];
            
            int prediction = model.predict_next(context);
            std::cout << "Контекст: ";
            for (int idx : context) {
                std::cout << vocab.get_word(idx) << " ";
            }
            std::cout << "\nПредсказано: " << vocab.get_word(prediction);
            std::cout << " (правильно: " << vocab.get_word(target) << ")" << std::endl;
        }
        
        auto end_total = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(end_total - start_total);
        
        std::cout << "\n" << std::string(50, '=') << std::endl;
        std::cout << "ОБУЧЕНИЕ ЗАВЕРШЕНО!" << std::endl;
        std::cout << "Время обучения: " << train_duration.count() << " сек" << std::endl;
        std::cout << "Общее время: " << total_duration.count() << " сек" << std::endl;
        std::cout << "Использовано ОЗУ: ~" 
                  << (model.get_hidden_size() * model.get_vocab_size() * 4 * 2) / 1024 / 1024 
                  << " MB" << std::endl;
        std::cout << "\nДля общения: ./bin/chat" << std::endl;
        std::cout << std::string(50, '=') << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "\n✗ ОШИБКА: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}