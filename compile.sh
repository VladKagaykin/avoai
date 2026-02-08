#!/bin/bash

mkdir -p bin models

# Автоматическое определение CPU
CPU_FEATURES=""
if grep -q avx2 /proc/cpuinfo; then
    CPU_FEATURES="-march=native -mavx2 -mfma"
    echo "Обнаружен AVX2, включаю оптимизации"
elif grep -q avx /proc/cpuinfo; then
    CPU_FEATURES="-march=native -mavx"
    echo "Обнаружен AVX, включаю оптимизации"
else
    CPU_FEATURES="-msse4.2"
    echo "Использую SSE4.2 (базовые оптимизации)"
fi

# Определение количества потоков
THREADS=$(nproc)
echo "Потоков CPU: $THREADS"

# Адаптивная компиляция
echo "Компиляция train.cpp (адаптивная)..."
g++ -std=c++17 -O3 $CPU_FEATURES \
    -ffast-math -flto -funroll-loops \
    -pthread -fopenmp -DNDEBUG \
    src/train.cpp -o bin/train

echo "Компиляция chat.cpp..."
g++ -std=c++17 -O2 -march=native -pthread \
    src/chat.cpp -o bin/chat

echo ""
echo "✅ Адаптивная компиляция завершена!"
echo "   Оптимизации: $CPU_FEATURES"
echo "   Потоков: $THREADS"
echo ""
echo "Для обучения: ./bin/train"
echo "Для общения:  ./bin/chat"

# Создание конфига для runtime
cat > models/hardware.cfg << EOF
# Конфигурация AvoAI
threads=$THREADS
features=$CPU_FEATURES
timestamp=$(date +%Y%m%d_%H%M%S)
EOF