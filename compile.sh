#!/bin/bash

echo "=== КОМПИЛЯЦИЯ МНОГОСЛОЙНОЙ НЕЙРОННОЙ СЕТИ С ПОДДЕРЖКОЙ GPU ==="
echo ""

# Проверка компилятора
if ! command -v g++ &> /dev/null; then
    echo "Ошибка: g++ не найден. Установите компилятор:"
    echo "  Ubuntu/Debian: sudo apt-get install g++"
    echo "  Fedora: sudo dnf install gcc-c++"
    echo "  Arch: sudo pacman -S gcc"
    exit 1
fi

# Проверка наличия исходных файлов
echo "Проверка исходных файлов..."
if [ ! -f "src/train.cpp" ]; then
    echo "✗ Ошибка: файл src/train.cpp не найден"
    echo "Убедитесь, что исходные файлы находятся в папке src/"
    exit 1
fi

if [ ! -f "src/chat.cpp" ]; then
    echo "⚠ Внимание: файл src/chat.cpp не найден"
    echo "Будет скомпилирован только train"
fi

# Проверка OpenCL
echo "Проверка наличия OpenCL..."
if [ -f "/usr/include/CL/cl.h" ] || [ -f "/usr/local/include/CL/cl.h" ] || [ -f "/opt/local/include/CL/cl.h" ]; then
    echo "✓ OpenCL заголовочные файлы найдены"
    OPENCL_AVAILABLE=1
else
    echo "⚠ OpenCL заголовочные файлы не найдены"
    echo "  Для GPU ускорения установите OpenCL:"
    echo "  Ubuntu/Debian: sudo apt-get install ocl-icd-opencl-dev"
    echo "  Fedora: sudo dnf install ocl-icd-devel"
    echo "  Будет использоваться только CPU"
    OPENCL_AVAILABLE=0
fi

# Проверка поддержки OpenMP
echo "Проверка поддержки OpenMP..."
g++ -fopenmp --version 2>/dev/null | head -1

# Создаем директории
echo ""
echo "Создание структуры директорий..."
mkdir -p data models

# Компиляция train.cpp с оптимизациями
echo ""
echo "1. КОМПИЛЯЦИЯ TRAIN.CPP (ОБУЧЕНИЕ С GPU/CPU)"
echo "--------------------------------------------"

COMPILE_SUCCESS=0
COMPILE_OPTIONS="-std=c++17 -O3 -march=native -ffast-math -funroll-loops -fopenmp -Wall -Wextra -Wno-deprecated-declarations -Wno-reorder"

# Добавляем определение версии OpenCL для устранения предупреждения
if [ $OPENCL_AVAILABLE -eq 1 ]; then
    echo "Попытка компиляции с поддержкой GPU (OpenCL)..."
    g++ $COMPILE_OPTIONS -DCL_TARGET_OPENCL_VERSION=300 src/train.cpp -o train -pthread -lm -lOpenCL
    
    if [ $? -eq 0 ]; then
        COMPILE_SUCCESS=1
        echo "✓ train успешно скомпилирован с поддержкой GPU"
        echo "  Флаги: $COMPILE_OPTIONS -DCL_TARGET_OPENCL_VERSION=300 -lOpenCL"
    else
        echo "⚠ Не удалось скомпилировать с OpenCL, пробую без GPU..."
    fi
fi

if [ $COMPILE_SUCCESS -eq 0 ]; then
    echo "Компиляция без GPU (только CPU)..."
    g++ $COMPILE_OPTIONS src/train.cpp -o train -pthread -lm
    
    if [ $? -eq 0 ]; then
        COMPILE_SUCCESS=1
        echo "✓ train успешно скомпилирован (только CPU)"
        echo "  Флаги: $COMPILE_OPTIONS"
    else
        echo "⚠ Не удалось скомпилировать с текущими оптимизациями, пробую упрощенные флаги..."
        g++ -std=c++17 -O2 -fopenmp -Wall -Wextra -Wno-deprecated-declarations -Wno-reorder src/train.cpp -o train -pthread -lm
        
        if [ $? -eq 0 ]; then
            COMPILE_SUCCESS=1
            echo "✓ train скомпилирован с упрощенными оптимизациями"
        else
            echo "✗ КРИТИЧЕСКАЯ ОШИБКА КОМПИЛЯЦИИ"
            echo "Пробую минимальную компиляцию без оптимизаций..."
            g++ -std=c++17 -O1 src/train.cpp -o train -pthread
            
            if [ $? -eq 0 ]; then
                COMPILE_SUCCESS=1
                echo "✓ train скомпилирован с минимальными оптимизациями"
            else
                echo "✗ Не удалось скомпилировать train.cpp"
                echo "Проверьте наличие файла src/train.cpp"
                exit 1
            fi
        fi
    fi
fi

# Компиляция chat.cpp
echo ""
echo "2. КОМПИЛЯЦИЯ CHAT.CPP (ГЕНЕРАЦИЯ)"
echo "----------------------------------"

if [ -f "src/chat.cpp" ]; then
    echo "Компиляция chat.cpp..."
    g++ -std=c++17 -O2 -Wall -Wextra src/chat.cpp -o chat -pthread -lm
    
    if [ $? -eq 0 ]; then
        echo "✓ chat успешно скомпилирован"
    else
        echo "⚠ Не удалось скомпилировать chat.cpp, будет использован только train"
    fi
else
    echo "⚠ chat.cpp не найден в src/chat.cpp"
    echo "  Вы можете добавить его позже для генерации текста"
fi

# Проверка размера исполняемых файлов
echo ""
echo "3. ПРОВЕРКА ИСПОЛНЯЕМЫХ ФАЙЛОВ"
echo "-------------------------------"

if [ -f "train" ]; then
    train_size=$(stat -c%s "train" 2>/dev/null || stat -f%z "train" 2>/dev/null)
    echo "  train: $train_size байт"
else
    echo "✗ train не найден после компиляции"
    exit 1
fi

if [ -f "chat" ]; then
    chat_size=$(stat -c%s "chat" 2>/dev/null || stat -f%z "chat" 2>/dev/null)
    echo "  chat: $chat_size байт"
fi

# Установка прав на выполнение
chmod +x train 2>/dev/null
if [ -f "chat" ]; then
    chmod +x chat 2>/dev/null
fi

echo ""
echo "=== КОМПИЛЯЦИЯ ЗАВЕРШЕНА ==="
echo ""
echo "🎯 ИНФОРМАЦИЯ О МОДЕЛИ:"
echo "• Многослойная нейронная сеть с автоматической настройкой архитектуры"
echo "• GPU ускорение: $(if [ $OPENCL_AVAILABLE -eq 1 ] && [ $COMPILE_SUCCESS -eq 1 ] && ldd train 2>/dev/null | grep -q OpenCL; then echo 'ДА'; else echo 'НЕТ'; fi)"
echo "• Количество слоев зависит от размера данных (1 слой на 0.25 МБ)"
echo "• Работает со ВСЕМИ 256 байтами UTF-8"
echo "• Параллельные вычисления с OpenMP"
echo ""
echo "📁 СТРУКТУРА ПРОЕКТА:"
echo "  src/      - исходные файлы (train.cpp, chat.cpp)"
echo "  data/     - текстовые файлы для обучения"
echo "  models/   - сохраненные модели и обработанный текст"
echo "  train     - программа обучения"
echo "  chat      - программа генерации текста (если скомпилирована)"
echo ""
echo "🚀 ИНСТРУКЦИЯ ПО ИСПОЛЬЗОВАНИЮ:"
echo "1. Добавьте текстовые файлы в папку data/"
echo "2. Запустите обучение: ./train"
echo "3. Сеть автоматически настроит архитектуру под размер данных"
if [ -f "chat" ]; then
    echo "4. Для генерации текста запустите: ./chat"
else
    echo "4. Для генерации текста скомпилируйте chat.cpp:"
    echo "   g++ -std=c++17 -O2 src/chat.cpp -o chat"
fi
echo ""
echo "✅ ГОТОВО К РАБОТЕ!"