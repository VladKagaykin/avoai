#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <random>
#include <iomanip>

// ==================== Нейронная сеть ====================
class SimpleNeuralNetwork {
private:
    std::vector<std::vector<float>> w1, w2;
    std::vector<float> b1, b2;
    
public:
    int input_size, hidden_size, output_size;

    SimpleNeuralNetwork() : input_size(0), hidden_size(0), output_size(0) {}
    
    void load(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Не могу открыть файл модели: " << filename << std::endl;
            return;
        }
        
        file.read(reinterpret_cast<char*>(&input_size), sizeof(int));
        file.read(reinterpret_cast<char*>(&hidden_size), sizeof(int));
        file.read(reinterpret_cast<char*>(&output_size), sizeof(int));
        
        std::cout << "Модель: " << input_size << " -> " << hidden_size << " -> " << output_size << std::endl;
        
        w1.resize(input_size);
        for (auto& row : w1) row.resize(hidden_size);
        w2.resize(hidden_size);
        for (auto& row : w2) row.resize(output_size);
        b1.resize(hidden_size);
        b2.resize(output_size);
        
        for (auto& row : w1) {
            file.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(float));
        }
        
        for (auto& row : w2) {
            file.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(float));
        }
        
        file.read(reinterpret_cast<char*>(b1.data()), b1.size() * sizeof(float));
        file.read(reinterpret_cast<char*>(b2.data()), b2.size() * sizeof(float));
        
        std::cout << "Модель загружена успешно" << std::endl;
    }
    
    float sigmoid(float x) const {
        return 1.0f / (1.0f + std::exp(-x));
    }
    
    std::vector<float> softmax(const std::vector<float>& x) const {
        std::vector<float> result(x.size());
        float max_val = *std::max_element(x.begin(), x.end());
        float sum = 0.0f;
        
        for (size_t i = 0; i < x.size(); ++i) {
            result[i] = std::exp(x[i] - max_val);
            sum += result[i];
        }
        
        for (auto& val : result) {
            val /= sum;
        }
        
        return result;
    }
    
    std::vector<float> predict(const std::vector<float>& input) const {
        // Проверка размера входных данных
        if (input.size() != static_cast<size_t>(input_size)) {
            std::cerr << "Ошибка: неверный размер входных данных! Ожидается " 
                      << input_size << ", получено " << input.size() << std::endl;
            return std::vector<float>(output_size, 0.0f);
        }
        
        // Скрытый слой
        std::vector<float> hidden(hidden_size, 0.0f);
        for (int i = 0; i < hidden_size; ++i) {
            float sum = b1[i];
            for (int j = 0; j < input_size; ++j) {
                sum += input[j] * w1[j][i];
            }
            hidden[i] = sigmoid(sum);
        }
        
        // Выходной слой
        std::vector<float> output(output_size, 0.0f);
        for (int i = 0; i < output_size; ++i) {
            float sum = b2[i];
            for (int j = 0; j < hidden_size; ++j) {
                sum += hidden[j] * w2[j][i];
            }
            output[i] = sum;
        }
        
        return softmax(output);
    }
};

// ==================== Вспомогательные функции для работы с байтами ====================
std::string loadText(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        return "";
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

std::vector<float> vectorizeContext(
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
    
    // Заполняем оставшиеся позиции нулевыми байтами
    for (size_t i = limit; i < ctx_size; ++i) {
        int idx = static_cast<int>(i) * 256 + 0;  // NULL байт
        if (idx < static_cast<int>(vec.size())) {
            vec[idx] = 1.0f;
        }
    }
    
    return vec;
}

unsigned char sampleFromDistribution(const std::vector<float>& probs) {
    // Проверяем размер вектора вероятностей
    if (probs.size() != 256) {
        std::cerr << "Ошибка: размер вектора вероятностей должен быть 256, получено " 
                  << probs.size() << std::endl;
        return ' ';
    }
    
    // Выбираем случайный байт на основе вероятностей
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> dist(probs.begin(), probs.end());
    
    int sampled_byte = dist(gen);
    
    // Ограничиваем диапазон 0-255
    if (sampled_byte < 0) sampled_byte = 0;
    if (sampled_byte > 255) sampled_byte = 255;
    
    return static_cast<unsigned char>(sampled_byte);
}

// Функция для вывода байта (без фильтрации)
void printByte(unsigned char c) {
    std::cout << c;  // Просто выводим любой байт
}

// ==================== Основная функция ====================
int main() {
    std::cout << "=== ГЕНЕРАЦИЯ ТЕКСТА НЕЙРОННОЙ СЕТЬЮ (БАЙТЫ UTF-8) ===" << std::endl;
    std::cout << std::endl;
    
    // Загрузка модели
    SimpleNeuralNetwork nn;
    std::cout << "Загрузка модели..." << std::endl;
    nn.load("models/model.bin");
    
    if (nn.input_size == 0) {
        std::cout << "Модель не загружена. Сначала обучите модель: ./train" << std::endl;
        return 1;
    }
    
    // Вычисляем размер контекста
    int context_size = nn.input_size / 256;  // 256 байтов на позицию
    std::cout << "Размер словаря: 256 байтов (0-255)" << std::endl;
    std::cout << "Размер контекста: " << context_size << " байтов" << std::endl;
    
    if (nn.output_size != 256) {
        std::cout << "Ошибка: выходной слой должен быть размером 256, получено " 
                  << nn.output_size << std::endl;
        return 1;
    }
    
    // Загрузка текста для начального контекста
    std::string initial_text = loadText("models/processed_text.txt");
    if (initial_text.empty()) {
        std::cout << "Текст для контекста не найден." << std::endl;
        return 1;
    }
    
    std::cout << std::endl;
    std::cout << "Готово! Задавайте вопросы или вводите текст для продолжения." << std::endl;
    std::cout << "Модель работает со всеми 256 байтами UTF-8." << std::endl;
    std::cout << "Команды:" << std::endl;
    std::cout << "  /exit - выход" << std::endl;
    std::cout << "  /len <число> - изменить длину ответа (по умолчанию: 100)" << std::endl;
    std::cout << "  /clear - очистить контекст" << std::endl;
    std::cout << "  /raw - показать сырые байты" << std::endl;
    std::cout << std::endl;
    
    std::string context;
    int generate_length = 100;
    bool show_raw_bytes = false;
    
    // Инициализируем контекст
    if (initial_text.size() > static_cast<size_t>(context_size)) {
        context = initial_text.substr(initial_text.size() - context_size);
    } else {
        context = initial_text;
        // Дополняем нулевыми байтами до нужного размера
        context.append(context_size - context.size(), '\0');
    }
    
    // Функция генерации ответа
    auto generateAnswer = [&](const std::string& prompt) {
        std::string current_context;
        
        if (prompt.size() >= static_cast<size_t>(context_size)) {
            current_context = prompt.substr(prompt.size() - context_size);
        } else {
            current_context = prompt;
            // Дополняем нулевыми байтами до нужного размера
            current_context = std::string(context_size - current_context.size(), '\0') + current_context;
        }
        
        std::cout << "\nОтвет (" << generate_length << " байтов): ";
        std::string generated = current_context;
        
        for (int i = 0; i < generate_length; ++i) {
            // Векторизация текущего контекста
            auto input_vec = vectorizeContext(
                generated.substr(generated.size() - context_size), 
                context_size);
            
            // Получение предсказания
            auto probs = nn.predict(input_vec);
            
            // Выбор следующего байта
            unsigned char next_byte = sampleFromDistribution(probs);
            
            // Добавление байта к сгенерированному тексту
            generated += next_byte;
            
            // Выводим только новую часть (после первоначального контекста)
            if (i >= 0) {
                if (show_raw_bytes) {
                    // Показываем сырой байт в hex формате
                    std::cout << std::hex << std::setw(2) << std::setfill('0') 
                              << static_cast<int>(next_byte) << " ";
                } else {
                    printByte(next_byte);
                }
                std::cout.flush();
            }
            
            // Простая логика остановки
            if (next_byte == '.' && i > generate_length / 3) {
                if (rand() % 4 == 0) break;
            }
            if ((next_byte == '\n' || next_byte == '\r') && i > generate_length / 2) {
                if (rand() % 3 == 0) break;
            }
        }
        
        if (show_raw_bytes) {
            std::cout << std::dec;  // Возвращаем десятичный формат
        }
        std::cout << "\n" << std::endl;
        
        // Обновляем контекст для следующей генерации
        if (generated.size() > static_cast<size_t>(context_size)) {
            context = generated.substr(generated.size() - context_size);
        } else {
            context = generated;
        }
    };
    
    while (true) {
        std::cout << ">>> ";
        std::string input;
        std::getline(std::cin, input);
        
        if (input == "/exit" || input == "выход" || input == "exit") {
            break;
        }
        else if (input == "/clear" || input == "очистить") {
            context = std::string(context_size, '\0');
            std::cout << "Контекст очищен (заполнен NULL байтами)." << std::endl;
        }
        else if (input.substr(0, 4) == "/len") {
            try {
                int new_length = std::stoi(input.substr(5));
                if (new_length > 0 && new_length <= 5000) {
                    generate_length = new_length;
                    std::cout << "Длина ответа установлена: " << generate_length << " байтов" << std::endl;
                } else {
                    std::cout << "Длина должна быть от 1 до 5000 байтов" << std::endl;
                }
            } catch (...) {
                std::cout << "Использование: /len <число>" << std::endl;
            }
        }
        else if (input == "/raw") {
            show_raw_bytes = !show_raw_bytes;
            std::cout << "Режим сырых байтов: " << (show_raw_bytes ? "ВКЛ" : "ВЫКЛ") << std::endl;
        }
        else if (!input.empty()) {
            // Генерируем ответ на введенный текст
            generateAnswer(input);
        }
        else {
            // Пустой ввод - продолжаем с текущим контекстом
            if (context != std::string(context_size, '\0')) {
                std::cout << "Продолжаю с предыдущего контекста (" << context_size << " байтов)..." << std::endl;
                generateAnswer(context);
            } else {
                std::cout << "Введите вопрос или текст для продолжения." << std::endl;
            }
        }
    }
    
    std::cout << "До свидания!" << std::endl;
    return 0;
}