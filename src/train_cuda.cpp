#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <unordered_map>
#include <chrono>
#include <algorithm>
#include <iomanip>

namespace fs = std::filesystem;

// 1. ОПРЕДЕЛЕНИЕ МОДЕЛИ (совместимой с NeuralNetCPU)
struct NeuralNetImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};

    NeuralNetImpl(int64_t vocab_size, int64_t hidden_size)
        : fc1(register_module("fc1", torch::nn::Linear(vocab_size, hidden_size))),
          fc2(register_module("fc2", torch::nn::Linear(hidden_size, vocab_size))) {
        // Инициализация весов как у Xavier/Glorot
        torch::nn::init::xavier_uniform_(fc1->weight);
        torch::nn::init::zeros_(fc1->bias);
        torch::nn::init::xavier_uniform_(fc2->weight);
        torch::nn::init::zeros_(fc2->bias);
    }

    torch::Tensor forward(torch::Tensor x) {
        // x: [batch_size, vocab_size] (one-hot усредненного контекста)
        x = torch::relu(fc1(x));
        x = fc2(x);
        return x; // Логиты, softmax в функции потерь
    }
};
TORCH_MODULE(NeuralNet);

// 2. ФУНКЦИИ ДЛЯ РАБОТЫ С ДАННЫМИ
std::vector<std::string> load_words(const std::string& data_dir) {
    std::vector<std::string> words;
    for (const auto& entry : fs::directory_iterator(data_dir)) {
        if (entry.path().extension() == ".txt") {
            std::ifstream file(entry.path());
            std::stringstream buffer;
            buffer << file.rdbuf();
            std::string text = buffer.str();
            file.close();

            std::string word;
            for (char c : text) {
                if (std::isspace(c) || std::ispunct(c)) {
                    if (!word.empty()) {
                        std::transform(word.begin(), word.end(), word.begin(), ::tolower);
                        words.push_back(word);
                        word.clear();
                    }
                } else {
                    word += c;
                }
            }
            if (!word.empty()) {
                std::transform(word.begin(), word.end(), word.begin(), ::tolower);
                words.push_back(word);
            }
        }
    }
    std::cout << "Загружено слов: " << words.size() << std::endl;
    return words;
}

std::tuple<torch::Tensor, torch::Tensor, std::vector<std::string>>
prepare_training_data(const std::vector<std::string>& words, int context_size, int64_t* vocab_size_out) {
    std::unordered_map<std::string, int64_t> word_to_idx;
    std::vector<std::string> idx_to_word;
    int64_t next_id = 0;

    for (const auto& w : words) {
        if (word_to_idx.find(w) == word_to_idx.end()) {
            word_to_idx[w] = next_id++;
            idx_to_word.push_back(w);
        }
    }
    *vocab_size_out = word_to_idx.size();
    std::cout << "Размер словаря: " << *vocab_size_out << std::endl;

    // Подготовка примеров (context -> target)
    std::vector<int64_t> contexts, targets;
    size_t n = words.size();

    for (size_t i = 0; i + context_size < n; ++i) {
        bool valid = true;
        std::vector<int64_t> ctx_indices;
        for (int j = 0; j < context_size; ++j) {
            auto it = word_to_idx.find(words[i + j]);
            if (it == word_to_idx.end()) { valid = false; break; }
            ctx_indices.push_back(it->second);
        }
        auto target_it = word_to_idx.find(words[i + context_size]);
        if (!valid || target_it == word_to_idx.end()) continue;

        contexts.insert(contexts.end(), ctx_indices.begin(), ctx_indices.end());
        targets.push_back(target_it->second);
    }

    size_t num_examples = targets.size();
    std::cout << "Создано примеров: " << num_examples << std::endl;

    // Конвертируем в тензоры
    auto ctx_tensor = torch::from_blob(contexts.data(), {static_cast<int64_t>(num_examples), context_size},
                                       torch::dtype(torch::kInt64)).clone();
    auto tgt_tensor = torch::from_blob(targets.data(), {static_cast<int64_t>(num_examples)},
                                       torch::dtype(torch::kInt64)).clone();

    return {ctx_tensor, tgt_tensor, idx_to_word};
}

// 3. ФУНКЦИЯ ДЛЯ СОХРАНЕНИЯ В .BIN ФОРМАТЕ
void save_model_bin(const std::string& filename, 
                    const torch::Tensor& weights1,
                    const torch::Tensor& weights2,
                    const torch::Tensor& bias1,
                    const torch::Tensor& bias2,
                    size_t vocab_size,
                    size_t hidden_size,
                    float learning_rate = 0.01f) {
    
    std::ofstream file(filename, std::ios::binary);
    
    if (!file.is_open()) {
        throw std::runtime_error("Не могу открыть файл для записи: " + filename);
    }
    
    // Записываем размеры
    file.write(reinterpret_cast<const char*>(&vocab_size), sizeof(vocab_size));
    file.write(reinterpret_cast<const char*>(&hidden_size), sizeof(hidden_size));
    file.write(reinterpret_cast<const char*>(&learning_rate), sizeof(learning_rate));
    
    // Получаем данные из тензоров
    auto w1_cpu = weights1.cpu().contiguous();
    auto w2_cpu = weights2.cpu().contiguous();
    auto b1_cpu = bias1.cpu().contiguous();
    auto b2_cpu = bias2.cpu().contiguous();
    
    // Проверяем размеры
    if (w1_cpu.numel() != static_cast<int64_t>(hidden_size * vocab_size) ||
        w2_cpu.numel() != static_cast<int64_t>(vocab_size * hidden_size) ||
        b1_cpu.numel() != static_cast<int64_t>(hidden_size) ||
        b2_cpu.numel() != static_cast<int64_t>(vocab_size)) {
        throw std::runtime_error("Несовпадение размеров при сохранении модели");
    }
    
    // Записываем weights1 (hidden_size x vocab_size)
    file.write(reinterpret_cast<const char*>(w1_cpu.data_ptr<float>()),
               w1_cpu.numel() * sizeof(float));
    
    // Записываем weights2 (vocab_size x hidden_size)
    file.write(reinterpret_cast<const char*>(w2_cpu.data_ptr<float>()),
               w2_cpu.numel() * sizeof(float));
    
    // Записываем bias1
    file.write(reinterpret_cast<const char*>(b1_cpu.data_ptr<float>()),
               b1_cpu.numel() * sizeof(float));
    
    // Записываем bias2
    file.write(reinterpret_cast<const char*>(b2_cpu.data_ptr<float>()),
               b2_cpu.numel() * sizeof(float));
    
    file.close();
    std::cout << "Модель сохранена в бинарном формате: " << filename << std::endl;
}

// 4. ФУНКЦИЯ ДЛЯ СОХРАНЕНИЯ СЛОВАРЯ
void save_vocabulary(const std::vector<std::string>& vocabulary, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Не могу открыть файл словаря: " + filename);
    }
    
    for (const auto& word : vocabulary) {
        file << word << "\n";
    }
    
    file.close();
    std::cout << "Словарь сохранен: " << filename << " (" << vocabulary.size() << " слов)" << std::endl;
}

// 5. ГЛАВНАЯ ФУНКЦИЯ
int main() {
    std::cout << "\n=== AvoAI CUDA Trainer (бинарный формат) ===\n" << std::endl;

    // A. ВЫБОР УСТРОЙСТВА (GPU/CPU)
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        std::cout << "✅ CUDA доступна! Используем GPU." << std::endl;
        device = torch::Device(torch::kCUDA);
    } else {
        std::cout << "⚠️  CUDA недоступна, используем CPU." << std::endl;
    }

    // B. ПАРАМЕТРЫ
    const int CONTEXT_SIZE = 3;
    const int64_t HIDDEN_SIZE = 256;
    const int64_t BATCH_SIZE = 128;
    const int EPOCHS = 15;
    const double LEARNING_RATE = 0.01;

    // C. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
    std::cout << "\n[1/4] Загрузка данных..." << std::endl;
    auto words = load_words("data");
    if (words.empty()) {
        std::cerr << "Ошибка: не загружено слов!" << std::endl;
        return 1;
    }

    int64_t vocab_size;
    torch::Tensor all_contexts, all_targets;
    std::vector<std::string> vocabulary;
    std::tie(all_contexts, all_targets, vocabulary) = prepare_training_data(words, CONTEXT_SIZE, &vocab_size);

    // Перенос данных на GPU (если доступно)
    all_contexts = all_contexts.to(device);
    all_targets = all_targets.to(device);

    // D. СОЗДАНИЕ МОДЕЛИ И ОПТИМИЗАТОРА
    std::cout << "\n[2/4] Инициализация модели..." << std::endl;
    NeuralNet model(vocab_size, HIDDEN_SIZE);
    model->to(device);
    torch::optim::Adam optimizer(model->parameters(), LEARNING_RATE);
    auto criterion = torch::nn::CrossEntropyLoss();

    int64_t num_samples = all_targets.size(0);
    std::cout << "    Всего примеров: " << num_samples << std::endl;
    std::cout << "    Размер батча: " << BATCH_SIZE << std::endl;
    std::cout << "    Параметров модели: ~" << (vocab_size * HIDDEN_SIZE + HIDDEN_SIZE * vocab_size) / 1e6 << " млн." << std::endl;

    // E. ЦИКЛ ОБУЧЕНИЯ
    std::cout << "\n[3/4] Начало обучения..." << std::endl;
    auto total_start = std::chrono::steady_clock::now();

    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        model->train();
        auto epoch_start = std::chrono::steady_clock::now();
        double epoch_loss = 0.0;
        int64_t batches_processed = 0;

        // Перемешиваем данные каждый раз
        auto perm = torch::randperm(num_samples, torch::TensorOptions().dtype(torch::kInt64).device(device));
        auto shuffled_contexts = all_contexts.index({perm});
        auto shuffled_targets = all_targets.index({perm});

        for (int64_t start = 0; start < num_samples; start += BATCH_SIZE) {
            int64_t end = std::min(start + BATCH_SIZE, num_samples);
            auto batch_contexts = shuffled_contexts.index({
                torch::indexing::Slice(start, end)
            });
            auto batch_targets = shuffled_targets.index({
                torch::indexing::Slice(start, end)
            });

            // Преобразуем контекст в one-hot и усредняем
            auto one_hot = torch::nn::functional::one_hot(batch_contexts, vocab_size)
                               .to(torch::kFloat32); // [batch, context_size, vocab_size]
            auto input = one_hot.mean(/*dim=*/1); // Усреднение по контексту -> [batch, vocab_size]

            // Прямой проход
            auto output = model->forward(input);
            auto loss = criterion(output, batch_targets);

            // Обратный проход
            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            epoch_loss += loss.item<double>();
            batches_processed++;
        }

        auto epoch_end = std::chrono::steady_clock::now();
        auto epoch_time = std::chrono::duration<double>(epoch_end - epoch_start).count();
        auto avg_loss = epoch_loss / batches_processed;

        std::cout << "  Эпоха [" << epoch+1 << "/" << EPOCHS << "] "
                  << "Потери: " << std::fixed << std::setprecision(4) << avg_loss
                  << " | Время: " << std::setprecision(1) << epoch_time << "с"
                  << " | Устройство: " << (device.is_cuda() ? "GPU" : "CPU")
                  << std::endl;
    }

    auto total_end = std::chrono::steady_clock::now();
    auto total_time = std::chrono::duration<double>(total_end - total_start).count();
    std::cout << "\n    Общее время обучения: " << std::setprecision(1) << total_time << " секунд." << std::endl;

    // F. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ В БИНАРНОМ ФОРМАТЕ
    std::cout << "\n[4/4] Сохранение модели и словаря..." << std::endl;
    if (!fs::exists("models")) {
        fs::create_directory("models");
    }

    // Извлекаем веса из модели
    model->eval();
    
    // Получаем параметры модели
    auto params = model->named_parameters();
    
    // weights1: fc1.weight (hidden_size x vocab_size)
    auto fc1_weight = params["fc1.weight"].clone().cpu();
    // bias1: fc1.bias (hidden_size)
    auto fc1_bias = params["fc1.bias"].clone().cpu();
    // weights2: fc2.weight (vocab_size x hidden_size)
    auto fc2_weight = params["fc2.weight"].clone().cpu();
    // bias2: fc2.bias (vocab_size)
    auto fc2_bias = params["fc2.bias"].clone().cpu();
    
    // Сохраняем модель в .bin формате
    save_model_bin("models/model.bin", 
                   fc1_weight,   // weights1
                   fc2_weight,   // weights2 (уже правильный размер)
                   fc1_bias,     // bias1
                   fc2_bias,     // bias2
                   static_cast<size_t>(vocab_size), 
                   static_cast<size_t>(HIDDEN_SIZE),
                   0.01f);
    
    // Сохраняем словарь
    save_vocabulary(vocabulary, "models/vocab.txt");

    std::cout << "\n✅ Обучение завершено успешно!" << std::endl;
    std::cout << "   Модель:     models/model.bin" << std::endl;
    std::cout << "   Словарь:    models/vocab.txt" << std::endl;
    std::cout << "   Формат:     бинарный (.bin) - совместим с обычной версией" << std::endl;
    return 0;
}