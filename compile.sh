#!/bin/bash

# Создание директорий если их нет
mkdir -p bin models

echo "Компиляция train.cpp..."
g++ -std=c++17 -O2 -march=native -ffast-math -pthread \
    src/train.cpp -o bin/train

echo "Компиляция chat.cpp..."
g++ -std=c++17 -O2 -march=native -ffast-math -pthread \
    src/chat.cpp -o bin/chat

echo ""
echo "Компиляция завершена!"
echo "Исполняемые файлы в bin/"
echo ""
echo "Для обучения запустите:"
echo "  ./bin/train"
echo ""
echo "Для общения с моделью запустите:"
echo "  ./bin/chat"