g++ -O3 -march=native -fopenmp src/train.cpp -o train -std=c++17

g++ -std=c++17 -O2 -Wall src/chat.cpp -o chat -I.

g++ -std=c++17 -O2 -Wall src/train_cuda.cpp -o train_cuda \
    -I/usr/include/torch \
    -I/usr/include/torch/csrc/api/include \
    -L/usr/lib/x86_64-linux-gnu \
    -Wl,--no-as-needed \
    -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda \
    -Wl,--as-needed \
    -Wl,-rpath,/usr/lib/x86_64-linux-gnu \
    -D_GLIBCXX_USE_CXX11_ABI=1

g++ -std=c++17 -O2 -Wall src/chat.cpp -o chat_cuda \
    -DUSE_LIBTORCH \
    -I/usr/include/torch \
    -I/usr/include/torch/csrc/api/include \
    -L/usr/lib/x86_64-linux-gnu \
    -Wl,--no-as-needed \
    -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda \
    -Wl,--as-needed \
    -Wl,-rpath,/usr/lib/x86_64-linux-gnu \
    -D_GLIBCXX_USE_CXX11_ABI=1