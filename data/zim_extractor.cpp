#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <filesystem>
#include <vector>
#include <algorithm>

namespace fs = std::filesystem;

int main() {
    std::cout << "=== Извлечение всего текста из ZIM файла ===" << std::endl;
    
    // Создаем временную директорию для извлечения
    std::string tempDir = "temp_zim_extract";
    fs::create_directories(tempDir);
    
    // Используем правильную команду zimdump
    std::string command = "zimdump dump --dir=" + tempDir + " wikipedia_ru_all_maxi_2025-06.zim";
    std::cout << "Выполняю: " << command << std::endl;
    
    int result = system(command.c_str());
    
    if (result != 0) {
        std::cerr << "Ошибка при выполнении zimdump! Код: " << result << std::endl;
        
        // Проверяем файл ошибок
        std::string errorFile = tempDir + "/dump_errors.log";
        if (fs::exists(errorFile)) {
            std::cout << "Содержимое dump_errors.log:" << std::endl;
            system(("cat " + errorFile + " | head -20").c_str());
        }
        
        return 1;
    }
    
    // Теперь обрабатываем извлеченные файлы
    std::cout << "\nОбработка извлеченных файлов..." << std::endl;
    
    std::ofstream out("data/all_text.txt", std::ios::binary);
    if (!out.is_open()) {
        std::cerr << "Не могу открыть выходной файл!" << std::endl;
        return 1;
    }
    
    int fileCount = 0;
    long long totalChars = 0;
    
    // Рекурсивно проходим по всем файлам
    for (const auto& entry : fs::recursive_directory_iterator(tempDir)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().string();
            
            // Пропускаем файл ошибок
            if (filename.find("dump_errors.log") != std::string::npos) {
                continue;
            }
            
            std::ifstream file(filename, std::ios::binary);
            if (file) {
                // Читаем весь файл
                std::string content((std::istreambuf_iterator<char>(file)),
                                   std::istreambuf_iterator<char>());
                
                file.close();
                
                // Простая очистка: удаляем HTML-теги
                std::string cleanContent;
                bool inTag = false;
                
                for (char c : content) {
                    if (c == '<') {
                        inTag = true;
                    } else if (c == '>') {
                        inTag = false;
                    } else if (!inTag && c != 0) { // Пропускаем нулевые байты
                        // Сохраняем все символы, включая спецсимволы
                        cleanContent += c;
                    }
                }
                
                // Записываем очищенный контент
                if (!cleanContent.empty()) {
                    out << cleanContent << "\n";
                    totalChars += cleanContent.size();
                    fileCount++;
                    
                    if (fileCount % 100 == 0) {
                        std::cout << "\rОбработано файлов: " << fileCount 
                                  << ", символов: " << totalChars / 1024 << " KB" << std::flush;
                    }
                }
            }
        }
    }
    
    out.close();
    
    // Удаляем временную директорию
    system(("rm -rf " + tempDir).c_str());
    
    std::cout << "\n\n=== Завершено ===" << std::endl;
    std::cout << "Обработано файлов: " << fileCount << std::endl;
    std::cout << "Извлечено символов: " << totalChars << std::endl;
    std::cout << "Размер текста: " << totalChars / 1024 / 1024 << " MB" << std::endl;
    std::cout << "Файл сохранен: data/all_text.txt" << std::endl;
    
    // Показываем первые 500 символов для проверки
    std::cout << "\nПервые 500 символов извлеченного текста:" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    std::ifstream check("data/all_text.txt");
    if (check) {
        std::string first500;
        check.read(&first500[0], 500);
        first500.resize(check.gcount());
        
        std::cout << first500 << std::endl;
        check.close();
    }
    
    return 0;
}