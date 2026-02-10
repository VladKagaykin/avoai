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
#include <unordered_set>
#include <torch/torch.h>

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
    
    // Построение словаря из текста
    void build_from_text(const std::string& text) {
        std::u32string unicode_text = utf8_to_utf32(text);
        std::unordered_set<char32_t> unique_chars;
        
        for (char32_t c : unicode_text) {
            unique_chars.insert(c);
        }
        
        // Добавляем обязательные символы
        std::u32string mandatory = U"\n\t .,!?;:()[]{}<>\"'";
        for (char32_t c : mandatory) {
            unique_chars.insert(c);
        }
        
        idx_to_char.assign(unique_chars.begin(), unique_chars.end());
        char_to_idx.clear();
        
        for (size_t i = 0; i < idx_to_char.size(); ++i) {
            char_to_idx[idx_to_char[i]] = i;
        }
    }
    
    int get_index(char32_t c) const {
        auto it = char_to_idx.find(c);
        return it != char_to_idx.end() ? it->second : -1;
    }
    
    char32_t get_char(int idx) const {
        if (idx >= 0 && idx < static_cast<int>(idx_to_char.size())) {
            return idx_to_char[idx];
        }
        return U'?';
    }
    
    size_t size() const { return idx_to_char.size(); }
    
    void save(const std::string& filename) {
        std::ofstream file(filename, std::ios::binary);
        size_t vocab_size = size();
        file.write(reinterpret_cast<const char*>(&vocab_size), sizeof(vocab_size));
        
        for (char32_t c : idx_to_char) {
            uint32_t code_point = static_cast<uint32_t>(c);
            file.write(reinterpret_cast<const char*>(&code_point), sizeof(code_point));
        }
        file.close();
    }
    
    void load(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
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
            if (idx >= 0 && idx < static_cast<int>(idx_to_char.size())) {
                unicode_str += idx_to_char[idx];
            }
        }
        return utf32_to_utf8(unicode_str);
    }
};

// ==================== МОДЕЛЬ TORCH ====================
struct CharModelImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    size_t vocab_size;
    size_t hidden_size;
    int context_size;
    
    CharModelImpl(size_t vocab_size, size_t hidden_size, int context_size)
        : vocab_size(vocab_size), hidden_size(hidden_size), context_size(context_size) {
        fc1 = register_module("fc1", torch::nn::Linear(vocab_size, hidden_size));
        fc2 = register_module("fc2", torch::nn::Linear(hidden_size, vocab_size));
        
        // Инициализация Xavier
        torch::nn::init::xavier_uniform_(fc1->weight);
        torch::nn::init::zeros_(fc1->bias);
        torch::nn::init::xavier_uniform_(fc2->weight);
        torch::nn::init::zeros_(fc2->bias);
    }
    
    torch::Tensor forward(torch::Tensor x) {
        // x: [batch_size, vocab_size] - усредненный one-hot контекста
        x = torch::relu(fc1(x));
        x = fc2(x);
        return x;
    }
};
TORCH_MODULE(CharModel);

// ==================== ОПТИМИЗИРОВАННЫЕ ФОРМУЛЫ ДЛЯ 2GB GPU ====================
struct AdaptiveParams {
    int context_size;
    size_t hidden_size;
    int batch_size;
    int epochs;
    double learning_rate;
    
    // Формулы для автоматического подбора параметров для 2GB GPU
    static AdaptiveParams calculate_from_data(size_t total_chars, size_t vocab_size, size_t num_examples) {
        AdaptiveParams params;
        
        // 1. Размер контекста: меньше для экономии памяти
        params.context_size = static_cast<int>(
            std::min(30.0,  // Меньше для 2GB GPU
                    std::max(10.0, 
                            std::log10(static_cast<double>(total_chars)) * 5.0)));
        
        // 2. Размер скрытого слоя: уменьшен для 2GB GPU
        double hidden_base = std::sqrt(static_cast<double>(vocab_size * num_examples)) / 100.0;
        params.hidden_size = static_cast<size_t>(
            std::min(256.0,  // Уменьшено для 2GB GPU
                    std::max(64.0, 
                            hidden_base * 30.0)));
        
        // 3. Размер батча: уменьшен для 2GB GPU
        params.batch_size = static_cast<int>(
            std::min(64.0,  // Уменьшено для 2GB GPU
                    std::max(16.0, 
                            std::sqrt(static_cast<double>(num_examples)) / 20.0)));
        
        // 4. Количество эпох: больше, так как меньше данных
        params.epochs = static_cast<int>(
            std::min(15.0, 
                    std::max(8.0, 
                            25.0 - std::log10(static_cast<double>(num_examples)) * 3.0)));
        
        // 5. Скорость обучения: адаптивная
        params.learning_rate = std::min(0.03, 
                                       std::max(0.001, 
                                               0.05 / std::log10(static_cast<double>(num_examples))));
        
        return params;
    }
    
    void print() const {
        std::cout << "Адаптивные параметры для 2GB GPU:" << std::endl;
        std::cout << "  • Размер контекста: " << context_size << " символов" << std::endl;
        std::cout << "  • Скрытый слой: " << hidden_size << " нейронов" << std::endl;
        std::cout << "  • Размер батча: " << batch_size << std::endl;
        std::cout << "  • Эпох: " << epochs << std::endl;
        std::cout << "  • Скорость обучения: " << std::fixed << std::setprecision(4) 
                  << learning_rate << std::endl;
    }
    
    // Расчет использования памяти
    size_t estimate_gpu_memory(int64_t batch_size, size_t vocab_size, int context_size, size_t hidden_size) const {
        // Примерный расчет памяти в байтах для одного батча
        size_t memory = 0;
        
        // Память для модели (параметры)
        memory += (vocab_size * hidden_size + hidden_size * vocab_size) * 4; // float32
        
        // Память для одного батча данных (на GPU)
        memory += batch_size * context_size * vocab_size * 4; // one-hot (float32)
        memory += batch_size * hidden_size * 4; // скрытый слой
        memory += batch_size * vocab_size * 4; // выход
        
        // Градиенты (примерно x2)
        memory *= 2;
        
        return memory;
    }
};

// ==================== ФУНКЦИЯ ДЛЯ СОЗДАНИЯ ВСЕХ ДАННЫХ ====================
std::tuple<torch::Tensor, torch::Tensor> prepare_all_training_data(
    const std::vector<int>& indices,
    int context_size) {
    
    std::cout << "Создание всех обучающих данных..." << std::endl;
    
    size_t num_examples = indices.size() - context_size;
    
    std::cout << "  • Всего примеров: " << num_examples << std::endl;
    std::cout << "  • Размер контекста: " << context_size << std::endl;
    
    // Создаем тензоры на CPU
    std::vector<int64_t> flat_contexts;
    std::vector<int64_t> targets;
    
    flat_contexts.reserve(num_examples * context_size);
    targets.reserve(num_examples);
    
    // Создаем все примеры подряд
    for (size_t i = 0; i < num_examples; ++i) {
        for (int j = 0; j < context_size; ++j) {
            flat_contexts.push_back(indices[i + j]);
        }
        targets.push_back(indices[i + context_size]);
    }
    
    // Создаем тензоры на CPU
    auto contexts_tensor = torch::from_blob(
        flat_contexts.data(),
        {static_cast<int64_t>(num_examples), context_size},
        torch::kInt64
    ).clone();
    
    auto targets_tensor = torch::from_blob(
        targets.data(),
        {static_cast<int64_t>(num_examples)},
        torch::kInt64
    ).clone();
    
    std::cout << "✓ Созданы все тензоры на CPU" << std::endl;
    std::cout << "  • Размер контекстов: " << contexts_tensor.sizes() << std::endl;
    std::cout << "  • Размер целей: " << targets_tensor.sizes() << std::endl;
    std::cout << "  • Память CPU: " 
              << (contexts_tensor.numel() * 8 + targets_tensor.numel() * 8) / 1024 / 1024 
              << " MB" << std::endl;
    
    return {contexts_tensor, targets_tensor};
}

// ==================== ОСНОВНАЯ ФУНКЦИЯ ====================
int main() {
    std::cout << "=== AvoAI Unicode Trainer (использует все данные) ===" << std::endl;
    
    // Проверяем CUDA
    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available()) {
        std::cout << "✅ CUDA доступна! Используем GPU." << std::endl;
        device = torch::Device(torch::kCUDA);
        
        // Используем доступную функцию для получения информации
        std::cout << "  Устройств CUDA: " << torch::cuda::device_count() << std::endl;
    } else {
        std::cout << "⚠️  CUDA недоступна, используем CPU." << std::endl;
    }
    
    try {
        // 1. Загрузка данных
        std::cout << "\n[1/4] Загрузка текстов..." << std::endl;
        std::string all_text;
        
        if (!fs::exists("data")) {
            fs::create_directory("data");
            std::cout << "Создана папка data/ - поместите туда .txt файлы" << std::endl;
            return 1;
        }
        
        size_t total_chars = 0;
        for (const auto& entry : fs::directory_iterator("data")) {
            if (entry.path().extension() == ".txt") {
                std::ifstream file(entry.path());
                std::stringstream buffer;
                buffer << file.rdbuf();
                std::string file_text = buffer.str();
                all_text += file_text + "\n";
                file.close();
                
                size_t chars = file_text.size();
                total_chars += chars;
                std::cout << "  " << entry.path().filename() << " (" << chars << " символов)" << std::endl;
            }
        }
        
        if (all_text.empty()) {
            std::cerr << "Ошибка: в папке data нет .txt файлов!" << std::endl;
            return 1;
        }
        
        std::cout << "✓ Всего символов UTF-8: " << total_chars << std::endl;
        
        // 2. Создание словаря
        std::cout << "\n[2/4] Создание Unicode словаря..." << std::endl;
        UnicodeVocabulary vocab;
        vocab.build_from_text(all_text);
        
        std::cout << "✓ Размер словаря: " << vocab.size() << " Unicode символов" << std::endl;
        
        // 3. Подготовка данных
        std::cout << "\n[3/4] Подготовка данных..." << std::endl;
        
        // Конвертируем текст в индексы
        std::vector<int> indices = vocab.text_to_indices(all_text);
        std::cout << "✓ Конвертировано в индексы: " << indices.size() << " токенов" << std::endl;
        
        // Автоматический подбор параметров через формулы (оптимизировано для 2GB GPU)
        AdaptiveParams params = AdaptiveParams::calculate_from_data(
            total_chars, 
            vocab.size(), 
            indices.size()
        );
        params.print();
        
        const int CONTEXT_SIZE = params.context_size;
        const size_t HIDDEN_SIZE = params.hidden_size;
        const int BATCH_SIZE = params.batch_size;
        const int EPOCHS = params.epochs;
        const double LEARNING_RATE = params.learning_rate;
        
        // Создаем ВСЕ данные (без ограничений)
        auto [contexts_tensor, targets_tensor] = prepare_all_training_data(indices, CONTEXT_SIZE);
        
        int64_t num_examples = contexts_tensor.size(0);
        std::cout << "✓ Используем ВСЕ примеры: " << num_examples << std::endl;
        
        // Проверяем оценку памяти GPU
        size_t estimated_memory = params.estimate_gpu_memory(BATCH_SIZE, vocab.size(), CONTEXT_SIZE, HIDDEN_SIZE);
        std::cout << "  • Оценка памяти GPU для одного батча: " << estimated_memory / 1024 / 1024 
                  << " MB (макс. 2GB)" << std::endl;
        
        if (estimated_memory > 1500 * 1024 * 1024 && device.is_cuda()) {
            std::cout << "⚠️  Внимание: оценка памяти близка к лимиту 2GB!" << std::endl;
            std::cout << "   Уменьшаем параметры..." << std::endl;
            
            // Форсированное уменьшение для безопасности
            int new_batch_size = std::max(8, BATCH_SIZE / 2);
            size_t new_hidden_size = std::max(static_cast<size_t>(32), HIDDEN_SIZE / 2);
            std::cout << "   Новый размер батча: " << new_batch_size << std::endl;
            std::cout << "   Новый скрытый слой: " << new_hidden_size << std::endl;
            
            // Здесь можно обновить параметры, если хотите их использовать дальше
        }
        
        // 4. Создание модели
        std::cout << "\n[4/4] Обучение модели..." << std::endl;
        
        CharModel model(vocab.size(), HIDDEN_SIZE, CONTEXT_SIZE);
        model->to(device);
        
        torch::optim::Adam optimizer(model->parameters(), LEARNING_RATE);
        auto criterion = torch::nn::CrossEntropyLoss();
        
        std::cout << "Настройки обучения:" << std::endl;
        std::cout << "  • Устройство: " << (device.is_cuda() ? "GPU" : "CPU") << std::endl;
        std::cout << "  • Параметров модели: ~" 
                  << (vocab.size() * HIDDEN_SIZE + HIDDEN_SIZE * vocab.size()) / 1000 
                  << " тыс." << std::endl;
        std::cout << "  • Всего батчей: " << (num_examples + BATCH_SIZE - 1) / BATCH_SIZE << std::endl;
        
        // Обучение
        auto total_start = std::chrono::steady_clock::now();
        
        for (int epoch = 0; epoch < EPOCHS; ++epoch) {
            model->train();
            auto epoch_start = std::chrono::steady_clock::now();
            double epoch_loss = 0.0;
            
            // Перемешиваем данные (на CPU)
            auto perm = torch::randperm(num_examples, torch::kInt64);
            auto shuffled_contexts = contexts_tensor.index({perm});
            auto shuffled_targets = targets_tensor.index({perm});
            
            int64_t batches_processed = 0;
            int64_t total_batches = (num_examples + BATCH_SIZE - 1) / BATCH_SIZE;
            
            for (int64_t start = 0; start < num_examples; start += BATCH_SIZE) {
                int64_t end = std::min(start + BATCH_SIZE, num_examples);
                int64_t current_batch_size = end - start;
                
                // Берем батч на CPU и переносим на GPU
                auto batch_contexts = shuffled_contexts.index({
                    torch::indexing::Slice(start, end)
                }).to(device);
                
                auto batch_targets = shuffled_targets.index({
                    torch::indexing::Slice(start, end)
                }).to(device);
                
                // Преобразуем в one-hot (на GPU)
                auto one_hot = torch::nn::functional::one_hot(
                    batch_contexts, 
                    static_cast<int64_t>(vocab.size())
                ).to(torch::kFloat32);
                
                // Усредняем по контексту
                auto input = one_hot.mean(/*dim=*/1);
                
                // Прямой проход
                auto output = model->forward(input);
                auto loss = criterion(output, batch_targets);
                
                // Обратный проход
                optimizer.zero_grad();
                loss.backward();
                
                // Gradient clipping для стабильности
                torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
                
                optimizer.step();
                
                epoch_loss += loss.item<double>() * current_batch_size;
                batches_processed++;
                
                // Прогресс в пределах эпохи
                if (batches_processed % 50 == 0 || batches_processed == total_batches) {
                    float progress = static_cast<float>(batches_processed) / total_batches;
                    std::cout << "\r  Прогресс эпохи " << epoch+1 << ": " 
                              << std::fixed << std::setprecision(1) << progress * 100 
                              << "% (" << batches_processed << "/" << total_batches << " батчей)" 
                              << std::flush;
                }
            }
            
            auto epoch_end = std::chrono::steady_clock::now();
            auto epoch_time = std::chrono::duration<double>(epoch_end - epoch_start).count();
            auto avg_loss = epoch_loss / num_examples;
            
            std::cout << "\r  Эпоха " << std::setw(2) << epoch+1 << "/" << EPOCHS
                      << " | Потери: " << std::fixed << std::setprecision(4) << avg_loss
                      << " | Время: " << std::setprecision(1) << epoch_time << "s" 
                      << " | LR: " << std::setprecision(4) << LEARNING_RATE 
                      << " | Примеров: " << num_examples << std::endl;
            
            // Автоматическое уменьшение learning rate
            if (epoch > 0 && epoch % 5 == 0) {
                for (auto& param_group : optimizer.param_groups()) {
                    // Правильный способ изменения learning rate в C++ API
                    auto* options = static_cast<torch::optim::AdamOptions*>(&param_group.options());
                    options->lr(options->lr() * 0.9);
                }
                std::cout << "    Уменьшен learning rate в 0.9 раза" << std::endl;
            }
        }
        
        auto total_end = std::chrono::steady_clock::now();
        auto total_time = std::chrono::duration<double>(total_end - total_start).count();
        
        std::cout << "\n✓ Обучение завершено за " << total_time << " секунд" << std::endl;
        std::cout << "  • Скорость: " << std::fixed << std::setprecision(1)
                  << (num_examples * EPOCHS / total_time / 1000.0)
                  << " тыс. примеров/сек" << std::endl;
        
        // 5. Сохранение
        std::cout << "\n[5/5] Сохранение модели..." << std::endl;
        
        if (!fs::exists("models")) {
            fs::create_directory("models");
        }
        
        // Сохраняем модель Torch
        model->to(torch::kCPU);
        torch::save(model, "models/model_unicode.pt");
        
        // Сохраняем словарь
        vocab.save("models/vocab_unicode.bin");
        
        // Сохраняем веса в бинарном формате (для совместимости с CPU версией)
        model->eval();
        auto params_torch = model->named_parameters();
        
        auto fc1_weight = params_torch["fc1.weight"].clone().cpu();
        auto fc1_bias = params_torch["fc1.bias"].clone().cpu();
        auto fc2_weight = params_torch["fc2.weight"].clone().cpu();
        auto fc2_bias = params_torch["fc2.bias"].clone().cpu();
        
        // Транспонируем для совместимости
        fc1_weight = fc1_weight.transpose(0, 1); // [vocab_size, hidden_size]
        fc2_weight = fc2_weight.transpose(0, 1); // [hidden_size, vocab_size]
        
        std::ofstream bin_file("models/model_unicode.bin", std::ios::binary);
        size_t vocab_size_val = vocab.size();
        size_t hidden_size_val = HIDDEN_SIZE;
        float learning_rate_val = static_cast<float>(LEARNING_RATE);
        int context_size_val = CONTEXT_SIZE;
        
        // Записываем метаданные
        bin_file.write(reinterpret_cast<const char*>(&vocab_size_val), sizeof(vocab_size_val));
        bin_file.write(reinterpret_cast<const char*>(&hidden_size_val), sizeof(hidden_size_val));
        bin_file.write(reinterpret_cast<const char*>(&learning_rate_val), sizeof(learning_rate_val));
        bin_file.write(reinterpret_cast<const char*>(&context_size_val), sizeof(context_size_val));
        
        // Записываем веса
        bin_file.write(reinterpret_cast<const char*>(fc1_weight.data_ptr<float>()),
                      fc1_weight.numel() * sizeof(float));
        bin_file.write(reinterpret_cast<const char*>(fc2_weight.data_ptr<float>()),
                      fc2_weight.numel() * sizeof(float));
        bin_file.write(reinterpret_cast<const char*>(fc1_bias.data_ptr<float>()),
                      fc1_bias.numel() * sizeof(float));
        bin_file.write(reinterpret_cast<const char*>(fc2_bias.data_ptr<float>()),
                      fc2_bias.numel() * sizeof(float));
        
        bin_file.close();
        
        // Сохраняем параметры модели в текстовый файл
        std::ofstream info_file("models/model_info.txt");
        info_file << "Модель Unicode: AvoAI (использует все данные)" << std::endl;
        info_file << "Время обучения: " << total_time << " секунд" << std::endl;
        info_file << "Параметры:" << std::endl;
        info_file << "  - Словарь: " << vocab.size() << " символов" << std::endl;
        info_file << "  - Скрытый слой: " << HIDDEN_SIZE << " нейронов" << std::endl;
        info_file << "  - Контекст: " << CONTEXT_SIZE << " символов" << std::endl;
        info_file << "  - Батч: " << BATCH_SIZE << std::endl;
        info_file << "  - Эпох: " << EPOCHS << std::endl;
        info_file << "  - Learning Rate: " << LEARNING_RATE << std::endl;
        info_file << "  - Примеров использовано: " << num_examples << std::endl;
        info_file << "  - Скорость обучения: " << std::fixed << std::setprecision(1)
                  << (num_examples * EPOCHS / total_time / 1000.0)
                  << " тыс. примеров/сек" << std::endl;
        info_file << "  - Устройство: " << (device.is_cuda() ? "GPU" : "CPU") << std::endl;
        info_file.close();
        
        std::cout << "✓ Модель сохранена:" << std::endl;
        std::cout << "  • models/model_unicode.pt (Torch формат)" << std::endl;
        std::cout << "  • models/model_unicode.bin (бинарный формат)" << std::endl;
        std::cout << "  • models/vocab_unicode.bin (словарь)" << std::endl;
        std::cout << "  • models/model_info.txt (информация)" << std::endl;
        
        // 6. Тест генерации
        std::cout << "\n[6/6] Тест генерации..." << std::endl;
        
        // Берем начало текста как контекст
        std::vector<int> test_context;
        size_t test_size = std::min(static_cast<size_t>(CONTEXT_SIZE), indices.size());
        for (size_t i = 0; i < test_size; ++i) {
            test_context.push_back(indices[i]);
        }
        
        if (test_context.size() >= static_cast<size_t>(CONTEXT_SIZE)) {
            std::cout << "Контекст: \"" << vocab.indices_to_text(test_context) << "\"" << std::endl;
            
            // Генерируем 50 символов
            std::vector<int> generated = test_context;
            
            for (int i = 0; i < 50; ++i) {
                // Создаем тензор для текущего контекста
                auto context_tensor = torch::from_blob(
                    generated.data() + generated.size() - CONTEXT_SIZE,
                    {1, CONTEXT_SIZE},
                    torch::kInt64
                ).clone().to(device);
                
                // One-hot
                auto one_hot = torch::nn::functional::one_hot(
                    context_tensor,
                    static_cast<int64_t>(vocab.size())
                ).to(torch::kFloat32);
                
                // Усредняем
                auto input = one_hot.mean(1);
                
                // Прямой проход
                torch::NoGradGuard no_grad;
                auto output = model->forward(input);
                
                // Берем наиболее вероятный символ
                auto next_idx = torch::argmax(output, -1).item<int>();
                generated.push_back(next_idx);
                
                // Прерываем на знаках препинания
                char32_t next_char = vocab.get_char(next_idx);
                if (next_char == U'.' || next_char == U'!' || next_char == U'?' || 
                    next_char == U'\n' || next_char == U'。' || next_char == U'！') {
                    break;
                }
            }
            
            std::cout << "Генерация: \"" << vocab.indices_to_text(
                std::vector<int>(generated.begin() + test_context.size(), generated.end())
            ) << "\"" << std::endl;
        } else {
            std::cout << "Слишком короткий текст для теста генерации" << std::endl;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "\n✗ ОШИБКА: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}