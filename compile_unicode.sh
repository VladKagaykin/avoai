#!/bin/bash

# Unicode модель с CUDA
echo "Компиляция train_unicode (CUDA)..."
g++ -std=c++17 -O2 -Wall src/train_unicode.cpp -o train_unicode \
    -I/usr/include/torch \
    -I/usr/include/torch/csrc/api/include \
    -L/usr/lib/x86_64-linux-gnu \
    -Wl,--no-as-needed \
    -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda \
    -Wl,--as-needed \
    -Wl,-rpath,/usr/lib/x86_64-linux-gnu \
    -D_GLIBCXX_USE_CXX11_ABI=1

# Unicode чат (без зависимостей)
echo "Компиляция chat_unicode..."
g++ -std=c++17 -O2 -Wall src/chat_unicode.cpp -o chat_unicode

echo "Готово!"