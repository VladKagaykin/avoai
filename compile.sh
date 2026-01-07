#!/bin/bash

echo "=== КОМПИЛЯЦИЯ НЕЙРОННОЙ СЕТИ ДЛЯ РАБОТЫ СО ВСЕМИ БАЙТАМИ UTF-8 ==="
echo ""

# Проверка компилятора
if ! command -v g++ &> /dev/null; then
    echo "Ошибка: g++ не найден. Установите компилятор:"
    echo "sudo apt-get install g++"
    exit 1
fi

# Проверка поддержки OpenMP
echo "Проверка поддержки OpenMP..."
g++ -fopenmp --version 2>/dev/null | head -1

# Создаем директории
echo "Создание структуры директорий..."
mkdir -p data models
mkdir -p src

# Компиляция train.cpp с оптимизациями и поддержкой OpenMP
echo ""
echo "1. КОМПИЛЯЦИЯ TRAIN.CPP (ОБУЧЕНИЕ)"
echo "----------------------------------"

echo "Компиляция с оптимизацией O3 и OpenMP..."
g++ -std=c++17 -O3 -march=native -ffast-math -funroll-loops -fopenmp \
    -Wall -Wextra -Wpedantic \
    src/train.cpp -o train -pthread -lm

if [ $? -eq 0 ]; then
    echo "✓ train успешно скомпилирован с оптимизациями"
    echo "  Флаги: -O3 -march=native -ffast-math -funroll-loops -fopenmp"
else
    echo "Пробую компиляцию без -march=native..."
    g++ -std=c++17 -O3 -ffast-math -funroll-loops -fopenmp \
        -Wall -Wextra \
        src/train.cpp -o train -pthread -lm
    
    if [ $? -eq 0 ]; then
        echo "✓ train успешно скомпилирован"
    else
        echo "✗ КРИТИЧЕСКАЯ ОШИБКА КОМПИЛЯЦИИ TRAIN.CPP"
        echo "Пробую минимальную компиляцию..."
        g++ -std=c++17 -O2 -fopenmp src/train.cpp -o train -pthread
        
        if [ $? -eq 0 ]; then
            echo "✓ train скомпилирован с минимальными оптимизациями"
        else
            echo "✗ Не удалось скомпилировать train.cpp"
            echo "Проверьте наличие исходного файла: src/train.cpp"
            exit 1
        fi
    fi
fi

# Компиляция chat.cpp
echo ""
echo "2. КОМПИЛЯЦИЯ CHAT.CPP (ГЕНЕРАЦИЯ)"
echo "----------------------------------"

echo "Компиляция с оптимизацией O2..."
g++ -std=c++17 -O2 -Wall -Wextra \
    src/chat.cpp -o chat -pthread -lm

if [ $? -eq 0 ]; then
    echo "✓ chat успешно скомпилирован"
else
    echo "✗ ОШИБКА КОМПИЛЯЦИИ CHAT.CPP"
    echo "Пробую минимальную компиляцию..."
    g++ -std=c++17 -O1 src/chat.cpp -o chat -pthread
    
    if [ $? -eq 0 ]; then
        echo "✓ chat скомпилирован с минимальными оптимизациями"
    else
        echo "✗ Не удалось скомпилировать chat.cpp"
        echo "Проверьте наличие исходного файла: src/chat.cpp"
        exit 1
    fi
fi

# Проверка размера исполняемых файлов
echo ""
echo "3. ПРОВЕРКА РАЗМЕРА ИСПОЛНЯЕМЫХ ФАЙЛОВ"
echo "--------------------------------------"

if [ -f "train" ]; then
    train_size=$(stat -c%s "train")
    echo "  train: $train_size байт"
else
    echo "  train: не найден"
fi

if [ -f "chat" ]; then
    chat_size=$(stat -c%s "chat")
    echo "  chat: $chat_size байт"
else
    echo "  chat: не найден"
fi

# Установка прав на выполнение
chmod +x train chat 2>/dev/null

echo ""
echo "=== КОМПИЛЯЦИЯ ЗАВЕРШЕНА ==="
echo ""
echo "ИНФОРМАЦИЯ О ПРОЕКТЕ:"
echo "• Модель работает со ВСЕМИ 256 байтами UTF-8"
echo "• Входной слой: контекст × 256 нейронов"
echo "• Выходной слой: 256 нейронов (вероятности для каждого байта)"
echo "• Обучение: предсказание следующего байта в последовательности"
echo ""
echo "ИНСТРУКЦИЯ ПО ИСПОЛЬЗОВАНИЮ:"
echo "1. Добавьте текстовые файлы в папку data/ (поддерживаются любые кодировки UTF-8)"
echo "2. Запустите обучение: ./train"
echo "3. Для генерации текста: ./chat"
echo ""
echo "КОМАНДЫ В CHAT:"
echo "  /exit           - выход из программы"
echo "  /len <число>    - изменить длину генерируемого текста"
echo "  /clear          - очистить контекст"
echo "  /raw            - переключить режим отображения сырых байтов"
echo ""
echo "ПРИМЕЧАНИЯ:"
echo "• Модель может генерировать любые UTF-8 символы (кириллица, эмодзи, иероглифы)"
echo "• Для лучших результатов используйте больше данных в папке data/"
echo "• В режиме /raw байты отображаются в hex-формате"
echo ""
echo "УДАЧНОГО ИСПОЛЬЗОВАНИЯ!"