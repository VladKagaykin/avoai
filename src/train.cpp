#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <random>
#include <filesystem>
#include <unordered_map>
#include <chrono>
#include <iomanip>
#include <omp.h>
#include <thread>
#include <atomic>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <sys/sysctl.h>
#else
#include <CL/cl.h>
#include <sys/sysinfo.h>
#endif

namespace fs = std::filesystem;

// ==================== Конфигурация ====================
struct Config {
    std::string data_dir = "data";
    std::string model_dir = "models";
    std::string text_file = "models/processed_text.txt";
    std::string model_file = "models/model.bin";
    int context_size = 20;
    int base_hidden_size = 128;
    float learning_rate = 0.01f;
    int min_epochs = 50;
    int max_epochs = 500;
    int batch_size = 32;
    int num_threads = 8;
    bool use_gpu = true;
    int max_hidden_layers = 20;
    double mb_per_layer = 0.25;
    
    // Автоматически настроенные параметры
    int actual_batch_size = 32;
    int actual_hidden_layers = 3;
    int actual_hidden_size = 128;
    size_t available_gpu_memory = 0;
    size_t available_system_memory = 0;
    bool gpu_available = false;
};

// ==================== Системные утилиты ====================
class SystemInfo {
public:
    static size_t getAvailableSystemMemory() {
#ifdef __linux__
        struct sysinfo info;
        if (sysinfo(&info) == 0) {
            return info.freeram * info.mem_unit;
        }
#elif __APPLE__
        int mib[2];
        size_t len;
        uint64_t mem;
        
        mib[0] = CTL_HW;
        mib[1] = HW_MEMSIZE;
        len = sizeof(mem);
        sysctl(mib, 2, &mem, &len, NULL, 0);
        return mem;
#endif
        return 4ULL * 1024 * 1024 * 1024;
    }
    
    static size_t getTotalSystemMemory() {
#ifdef __linux__
        struct sysinfo info;
        if (sysinfo(&info) == 0) {
            return info.totalram * info.mem_unit;
        }
#elif __APPLE__
        int mib[2];
        size_t len;
        uint64_t mem;
        
        mib[0] = CTL_HW;
        mib[1] = HW_MEMSIZE;
        len = sizeof(mem);
        sysctl(mib, 2, &mem, &len, NULL, 0);
        return mem;
#endif
        return 8ULL * 1024 * 1024 * 1024;
    }
    
    static int getOptimalThreadCount() {
        int threads = std::thread::hardware_concurrency();
        return threads > 0 ? threads : 4;
    }
};

// ==================== GPU утилиты с определением памяти ====================
class GPUContext {
private:
    cl_context context;
    cl_command_queue command_queue;
    cl_device_id device_id;
    cl_platform_id platform_id;
    bool initialized;
    size_t gpu_memory;
    
public:
    GPUContext() : initialized(false), gpu_memory(0) {
        cl_int ret;
        
        ret = clGetPlatformIDs(1, &platform_id, NULL);
        if (ret != CL_SUCCESS) {
            std::cerr << "Ошибка получения платформы OpenCL: " << ret << std::endl;
            return;
        }
        
        ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
        if (ret != CL_SUCCESS) {
            std::cerr << "GPU не найдено, пытаемся использовать CPU..." << std::endl;
            ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
            if (ret != CL_SUCCESS) {
                std::cerr << "Ошибка получения устройства OpenCL: " << ret << std::endl;
                return;
            }
        }
        
        context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
        if (ret != CL_SUCCESS) {
            std::cerr << "Ошибка создания контекста OpenCL: " << ret << std::endl;
            return;
        }
        
        command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
        if (ret != CL_SUCCESS) {
            std::cerr << "Ошибка создания очереди команд OpenCL: " << ret << std::endl;
            clReleaseContext(context);
            return;
        }
        
        cl_ulong global_mem_size;
        ret = clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, 
                             sizeof(cl_ulong), &global_mem_size, NULL);
        if (ret == CL_SUCCESS) {
            gpu_memory = static_cast<size_t>(global_mem_size);
            std::cout << "  Доступно памяти GPU: " 
                      << (gpu_memory / (1024*1024)) << " MB" << std::endl;
        } else {
            gpu_memory = 512 * 1024 * 1024;
        }
        
        initialized = true;
        std::cout << "✓ GPU контекст успешно инициализирован" << std::endl;
    }
    
    ~GPUContext() {
        if (initialized) {
            clReleaseCommandQueue(command_queue);
            clReleaseContext(context);
        }
    }
    
    bool isInitialized() const { return initialized; }
    size_t getGPUMemory() const { return gpu_memory; }
    cl_context getContext() const { return context; }
    cl_command_queue getCommandQueue() const { return command_queue; }
    cl_device_id getDeviceId() const { return device_id; }
    
    void matrixMultiply(const std::vector<float>& A, const std::vector<float>& B, 
                       std::vector<float>& C, int m, int n, int k) {
        if (!initialized) {
            #pragma omp parallel for
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    float sum = 0.0f;
                    for (int l = 0; l < k; ++l) {
                        sum += A[i * k + l] * B[l * n + j];
                    }
                    C[i * n + j] = sum;
                }
            }
            return;
        }
        
        cl_int ret;
        
        cl_mem a_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         m * k * sizeof(float), (void*)A.data(), &ret);
        cl_mem b_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         k * n * sizeof(float), (void*)B.data(), &ret);
        cl_mem c_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                         m * n * sizeof(float), NULL, &ret);
        
        if (ret != CL_SUCCESS) {
            clReleaseMemObject(a_buffer);
            clReleaseMemObject(b_buffer);
            clReleaseMemObject(c_buffer);
            
            #pragma omp parallel for
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    float sum = 0.0f;
                    for (int l = 0; l < k; ++l) {
                        sum += A[i * k + l] * B[l * n + j];
                    }
                    C[i * n + j] = sum;
                }
            }
            return;
        }
        
        const char* source = 
            "__kernel void matrix_mult(__global const float* A, __global const float* B, "
            "                         __global float* C, int M, int N, int K) {"
            "    int i = get_global_id(0);"
            "    int j = get_global_id(1);"
            "    if (i < M && j < N) {"
            "        float sum = 0.0f;"
            "        for (int l = 0; l < K; ++l) {"
            "            sum += A[i * K + l] * B[l * N + j];"
            "        }"
            "        C[i * N + j] = sum;"
            "    }"
            "}";
        
        cl_program program = clCreateProgramWithSource(context, 1, &source, NULL, &ret);
        ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
        
        cl_kernel kernel = clCreateKernel(program, "matrix_mult", &ret);
        
        ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &a_buffer);
        ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &b_buffer);
        ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), &c_buffer);
        ret = clSetKernelArg(kernel, 3, sizeof(int), &m);
        ret = clSetKernelArg(kernel, 4, sizeof(int), &n);
        ret = clSetKernelArg(kernel, 5, sizeof(int), &k);
        
        size_t global_work_size[2] = {static_cast<size_t>(m), static_cast<size_t>(n)};
        ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
        
        ret = clEnqueueReadBuffer(command_queue, c_buffer, CL_TRUE, 0,
                                 m * n * sizeof(float), C.data(), 0, NULL, NULL);
        
        clReleaseMemObject(a_buffer);
        clReleaseMemObject(b_buffer);
        clReleaseMemObject(c_buffer);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
    }
};

// ==================== Автоматический подбор параметров ====================
class AutoConfigurator {
public:
    static void configureForHardware(Config& config, size_t data_size) {
        std::cout << "\n⚙️  АВТОМАТИЧЕСКАЯ НАСТРОЙКА ПОД ВОЗМОЖНОСТИ ЖЕЛЕЗА\n";
        std::cout << "=============================================\n";
        
        size_t system_memory = SystemInfo::getAvailableSystemMemory();
        size_t total_system_memory = SystemInfo::getTotalSystemMemory();
        config.available_system_memory = system_memory;
        
        std::cout << "  Доступно оперативной памяти: " 
                  << (system_memory / (1024*1024)) << " MB из "
                  << (total_system_memory / (1024*1024)) << " MB\n";
        
        config.num_threads = SystemInfo::getOptimalThreadCount();
        omp_set_num_threads(config.num_threads);
        std::cout << "  Оптимальное количество потоков: " << config.num_threads << std::endl;
        
        GPUContext test_gpu;
        if (test_gpu.isInitialized()) {
            config.gpu_available = true;
            config.available_gpu_memory = test_gpu.getGPUMemory();
            config.use_gpu = true;
        } else {
            config.gpu_available = false;
            config.available_gpu_memory = 0;
            config.use_gpu = false;
            std::cout << "  GPU недоступно, используется CPU\n";
        }
        
        int input_size = config.context_size * 256;
        size_t memory_per_sample = input_size * sizeof(float) * 3;
        size_t available_memory = config.use_gpu ? 
            config.available_gpu_memory * 0.7 :
            system_memory * 0.5;
        
        int max_batch_by_memory = static_cast<int>(available_memory / memory_per_sample);
        config.actual_batch_size = std::min(256, std::max(4, max_batch_by_memory));
        
        // ИСПРАВЛЕНО: Увеличен лимит батча для GPU с 2 ГБ
        if (config.use_gpu && config.available_gpu_memory < 2ULL * 1024 * 1024 * 1024) {
            config.actual_batch_size = std::min(config.actual_batch_size, 128); // Было 16
        }
        
        std::cout << "  Размер мини-батча: " << config.actual_batch_size 
                  << " (максимально возможный: " << max_batch_by_memory << ")\n";
        
        double text_size_mb = static_cast<double>(data_size) / (1024.0 * 1024.0);
        int layers_by_data = 1 + static_cast<int>(text_size_mb / config.mb_per_layer);
        size_t memory_per_layer = config.base_hidden_size * config.base_hidden_size * sizeof(float);
        int max_layers_by_memory = static_cast<int>(available_memory / memory_per_layer);
        
        config.actual_hidden_layers = std::min(
            std::min(layers_by_data, max_layers_by_memory),
            config.max_hidden_layers
        );
        
        if (config.actual_hidden_layers < 1) config.actual_hidden_layers = 1;
        
        if (config.use_gpu && config.available_gpu_memory < 2ULL * 1024 * 1024 * 1024) {
            config.actual_hidden_size = 128; // Было 64
            if (config.available_gpu_memory < 1ULL * 1024 * 1024 * 1024) {
                config.actual_hidden_size = 64; // Было 32
            }
        } else if (!config.use_gpu && system_memory < 4ULL * 1024 * 1024 * 1024) {
            config.actual_hidden_size = 128; // Было 64
        } else {
            config.actual_hidden_size = config.base_hidden_size;
        }
        
        std::cout << "  Количество скрытых слоев: " << config.actual_hidden_layers << std::endl;
        std::cout << "  Размер скрытого слоя: " << config.actual_hidden_size << std::endl;
        
        if (max_batch_by_memory < 4) {
            std::cout << "  ⚠ ВНИМАНИЕ: Очень мало памяти! Рекомендуется добавить оперативной памяти.\n";
        }
        
        std::cout << "✓ Конфигурация завершена\n";
    }
};

// ==================== Чтение файлов ====================
class TextFileReader {
public:
    static std::string readAllTextFiles(const std::string& directory_path) {
        std::string all_text;
        
        try {
            if (!fs::exists(directory_path)) {
                std::cerr << "Директория не существует: " << directory_path << std::endl;
                return all_text;
            }
            
            std::cout << "Чтение текстовых файлов из: " << directory_path << std::endl;
            int file_count = 0;
            size_t total_bytes = 0;
            
            for (const auto& entry : fs::directory_iterator(directory_path)) {
                if (entry.is_regular_file()) {
                    std::string filename = entry.path().string();
                    std::string extension = entry.path().extension().string();
                    
                    if (extension == ".txt" || extension == ".TXT" || extension.empty()) {
                        std::ifstream file(filename, std::ios::binary);
                        if (file.is_open()) {
                            std::stringstream buffer;
                            buffer << file.rdbuf();
                            std::string content = buffer.str();
                            
                            if (!content.empty()) {
                                all_text += content + "\n";
                                total_bytes += content.size();
                                file_count++;
                                std::cout << "  " << entry.path().filename() 
                                          << " (" << content.size() << " байт)" << std::endl;
                            }
                            file.close();
                        }
                    }
                }
            }
            
            std::cout << "Всего прочитано файлов: " << file_count << std::endl;
            std::cout << "Общий объем данных: " << total_bytes << " байт" << std::endl;
            
        } catch (const fs::filesystem_error& e) {
            std::cerr << "Ошибка чтения файлов: " << e.what() << std::endl;
        }
        
        return all_text;
    }
};

// ==================== Обработка текста (байты) ====================
class ByteProcessor {
public:
    static std::string cleanText(const std::string& text) {
        return text;
    }
    
    static std::vector<std::pair<std::string, unsigned char>> createTrainingPairs(
        const std::string& text, int context_size = 20) {
        
        std::vector<std::pair<std::string, unsigned char>> pairs;
        if (text.size() <= static_cast<size_t>(context_size)) {
            return pairs;
        }
        
        pairs.reserve(text.size() - context_size);
        
        #pragma omp parallel for
        for (size_t i = 0; i < text.size() - context_size; ++i) {
            std::string context = text.substr(i, context_size);
            unsigned char next_byte = static_cast<unsigned char>(text[i + context_size]);
            #pragma omp critical
            {
                pairs.push_back({context, next_byte});
            }
        }
        
        return pairs;
    }
    
    static std::vector<float> vectorizeContext(
        const std::string& context, 
        int context_size) {
        
        std::vector<float> vec(256 * context_size, 0.0f);
        size_t limit = std::min(context.size(), static_cast<size_t>(context_size));
        
        #pragma omp parallel for
        for (size_t i = 0; i < limit; ++i) {
            unsigned char c = static_cast<unsigned char>(context[i]);
            int idx = static_cast<int>(i) * 256 + static_cast<int>(c);
            if (idx < static_cast<int>(vec.size())) {
                vec[idx] = 1.0f;
            }
        }
        
        return vec;
    }
    
    static std::vector<float> vectorizeTarget(unsigned char target_byte) {
        std::vector<float> vec(256, 0.0f);
        vec[target_byte] = 1.0f;
        return vec;
    }
};

// ==================== Многослойная нейронная сеть с GPU ====================
class MultiLayerNeuralNetwork {
private:
    struct Layer {
        std::vector<std::vector<float>> weights;
        std::vector<float> biases;
        std::vector<float> activations;
        std::vector<float> gradients;
        int input_size;
        int output_size;
        
        Layer(int in_size, int out_size) : 
            input_size(in_size), output_size(out_size) {
            
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / in_size));
            
            weights.resize(input_size);
            #pragma omp parallel for
            for (int i = 0; i < input_size; ++i) {
                weights[i].resize(output_size);
                for (int j = 0; j < output_size; ++j) {
                    weights[i][j] = dist(gen);
                }
            }
            
            biases.resize(output_size, 0.1f);
            activations.resize(output_size, 0.0f);
            gradients.resize(output_size, 0.0f);
        }
    };
    
    std::vector<Layer> layers;
    GPUContext* gpu_context;
    bool use_gpu;
    int input_size;
    int output_size;
    std::atomic<int> current_batch;
    std::atomic<int> total_batches;
    
public:
    MultiLayerNeuralNetwork(int in_size, int hidden_size, int out_size, 
                          int num_hidden_layers, bool use_gpu_acceleration,
                          [[maybe_unused]] const Config& config)
        : input_size(in_size), output_size(out_size), use_gpu(use_gpu_acceleration),
          current_batch(0), total_batches(0) {
        
        if (use_gpu) {
            gpu_context = new GPUContext();
            if (!gpu_context->isInitialized()) {
                std::cout << "  GPU недоступно, используется CPU\n";
                use_gpu = false;
                delete gpu_context;
                gpu_context = nullptr;
            }
        } else {
            gpu_context = nullptr;
        }
        
        std::vector<int> layer_sizes;
        layer_sizes.push_back(hidden_size);
        
        for (int i = 1; i < num_hidden_layers; ++i) {
            int layer_size = static_cast<int>(hidden_size * std::exp(-0.15 * i));
            if (layer_size < 16) layer_size = 16;
            layer_sizes.push_back(layer_size);
        }
        
        layers.emplace_back(in_size, layer_sizes[0]);
        
        for (size_t i = 1; i < layer_sizes.size(); ++i) {
            layers.emplace_back(layer_sizes[i-1], layer_sizes[i]);
        }
        
        layers.emplace_back(layer_sizes.back(), out_size);
        
        std::cout << "\n  Создана многослойная нейронная сеть:\n";
        std::cout << "    Входной слой: " << in_size << " нейронов\n";
        for (size_t i = 0; i < layers.size(); ++i) {
            std::cout << "    Слой " << i+1 << ": " << layers[i].input_size 
                      << " -> " << layers[i].output_size << " нейронов\n";
        }
        std::cout << "    GPU ускорение: " << (use_gpu ? "ДА" : "НЕТ") << std::endl;
    }
    
    ~MultiLayerNeuralNetwork() {
        if (gpu_context) delete gpu_context;
    }
    
    float relu(float x) const {
        return x > 0 ? x : 0.0f;
    }
    
    float reluDerivative(float x) const {
        return x > 0 ? 1.0f : 0.0f;
    }
    
    std::vector<float> softmax(const std::vector<float>& x) const {
        std::vector<float> result(x.size());
        float max_val = *std::max_element(x.begin(), x.end());
        float sum = 0.0f;
        
        #pragma omp parallel for reduction(+:sum)
        for (size_t i = 0; i < x.size(); ++i) {
            result[i] = std::exp(x[i] - max_val);
            sum += result[i];
        }
        
        #pragma omp parallel for
        for (size_t i = 0; i < result.size(); ++i) {
            result[i] /= sum;
        }
        
        return result;
    }
    
    // ИСПРАВЛЕНО: Оптимизированный forward с единым параллельным регионом
    std::vector<float> forward(const std::vector<float>& input) {
        std::vector<float> current_activations = input;
        
        #pragma omp parallel
        {
            for (auto& layer : layers) {
                std::vector<float> new_activations(layer.output_size, 0.0f);
                
                #pragma omp for
                for (int i = 0; i < layer.output_size; ++i) {
                    float sum = layer.biases[i];
                    for (int j = 0; j < layer.input_size; ++j) {
                        sum += current_activations[j] * layer.weights[j][i];
                    }
                    new_activations[i] = relu(sum);
                }
                
                #pragma omp single
                {
                    layer.activations = new_activations;
                    current_activations = new_activations;
                }
            }
        }
        
        return softmax(current_activations);
    }
    
    // ИСПРАВЛЕНО: Оптимизированный backward с единым параллельным регионом
    void backward(const std::vector<float>& input, 
                  const std::vector<float>& target, 
                  float learning_rate) {
        
        forward(input);
        
        auto& output_layer = layers.back();
        std::vector<float> output_error(output_layer.output_size);
        std::vector<float> output_activations = layers.back().activations;
        
        #pragma omp parallel for
        for (int i = 0; i < output_layer.output_size; ++i) {
            output_error[i] = output_activations[i] - target[i];
        }
        
        std::vector<float> next_gradients = output_error;
        
        #pragma omp parallel
        {
            for (int layer_idx = layers.size() - 1; layer_idx >= 0; --layer_idx) {
                auto& layer = layers[layer_idx];
                std::vector<float> current_gradients(layer.output_size, 0.0f);
                
                #pragma omp for
                for (int i = 0; i < layer.output_size; ++i) {
                    current_gradients[i] = next_gradients[i] * reluDerivative(layer.activations[i]);
                }
                
                #pragma omp single
                {
                    layer.gradients = current_gradients;
                }
                
                std::vector<float> prev_activations;
                if (layer_idx > 0) {
                    prev_activations = layers[layer_idx - 1].activations;
                } else {
                    prev_activations = input;
                }
                
                #pragma omp for
                for (int i = 0; i < layer.input_size; ++i) {
                    for (int j = 0; j < layer.output_size; ++j) {
                        layer.weights[i][j] -= learning_rate * current_gradients[j] * prev_activations[i];
                    }
                }
                
                #pragma omp for
                for (int i = 0; i < layer.output_size; ++i) {
                    layer.biases[i] -= learning_rate * current_gradients[i];
                }
                
                if (layer_idx > 0) {
                    std::vector<float> new_gradients(layer.input_size, 0.0f);
                    
                    #pragma omp for
                    for (int i = 0; i < layer.input_size; ++i) {
                        float sum = 0.0f;
                        for (int j = 0; j < layer.output_size; ++j) {
                            sum += layer.weights[i][j] * current_gradients[j];
                        }
                        new_gradients[i] = sum;
                    }
                    
                    #pragma omp single
                    {
                        next_gradients = new_gradients;
                    }
                }
            }
        }
    }
    
    // Обучение с мини-батчами (оптимизированная версия из изначального кода)
    void trainBatch(const std::vector<std::vector<float>>& inputs,
                    const std::vector<std::vector<float>>& targets,
                    float learning_rate, int batch_num, int total_batches) {
        
        int batch_size = inputs.size();
        if (batch_size == 0) return;
        
        #pragma omp parallel
        {
            // Локальные градиенты для каждого потока
            std::vector<std::vector<std::vector<float>>> local_w_grads(layers.size());
            std::vector<std::vector<float>> local_b_grads(layers.size());
            
            // Инициализируем градиенты для каждого слоя
            for (size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
                auto& layer = layers[layer_idx];
                local_w_grads[layer_idx].resize(layer.input_size);
                for (int i = 0; i < layer.input_size; ++i) {
                    local_w_grads[layer_idx][i].resize(layer.output_size, 0.0f);
                }
                local_b_grads[layer_idx].resize(layer.output_size, 0.0f);
            }
            
            #pragma omp for
            for (int i = 0; i < batch_size; ++i) {
                backward_with_grads(inputs[i], targets[i], learning_rate / batch_size, 
                                local_w_grads, local_b_grads);
            }
            
            // Агрегация градиентов
            for (size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
                auto& layer = layers[layer_idx];
                
                #pragma omp for
                for (int i = 0; i < layer.input_size; ++i) {
                    for (int j = 0; j < layer.output_size; ++j) {
                        #pragma omp atomic
                        layer.weights[i][j] -= local_w_grads[layer_idx][i][j];
                    }
                }
                
                #pragma omp for
                for (int i = 0; i < layer.output_size; ++i) {
                    #pragma omp atomic
                    layer.biases[i] -= local_b_grads[layer_idx][i];
                }
            }
        }
        
        current_batch = batch_num;
        this->total_batches = total_batches;
    }
    
    void backward_with_grads(const std::vector<float>& input, 
                           const std::vector<float>& target, 
                           float learning_rate,
                           std::vector<std::vector<std::vector<float>>>& w_grads,
                           std::vector<std::vector<float>>& b_grads) {
        
        forward(input);
        
        auto& output_layer = layers.back();
        std::vector<float> output_error(output_layer.output_size);
        std::vector<float> output_activations = layers.back().activations;
        
        for (int i = 0; i < output_layer.output_size; ++i) {
            output_error[i] = output_activations[i] - target[i];
        }
        
        std::vector<float> next_gradients = output_error;
        
        for (int layer_idx = layers.size() - 1; layer_idx >= 0; --layer_idx) {
            auto& layer = layers[layer_idx];
            std::vector<float> current_gradients(layer.output_size, 0.0f);
            
            for (int i = 0; i < layer.output_size; ++i) {
                current_gradients[i] = next_gradients[i] * reluDerivative(layer.activations[i]);
            }
            
            std::vector<float> prev_activations;
            if (layer_idx > 0) {
                prev_activations = layers[layer_idx - 1].activations;
            } else {
                prev_activations = input;
            }
            
            for (int i = 0; i < layer.input_size; ++i) {
                for (int j = 0; j < layer.output_size; ++j) {
                    w_grads[layer_idx][i][j] += learning_rate * current_gradients[j] * prev_activations[i];
                }
            }
            
            for (int i = 0; i < layer.output_size; ++i) {
                b_grads[layer_idx][i] += learning_rate * current_gradients[i];
            }
            
            if (layer_idx > 0) {
                next_gradients.assign(layer.input_size, 0.0f);
                for (int i = 0; i < layer.input_size; ++i) {
                    float sum = 0.0f;
                    for (int j = 0; j < layer.output_size; ++j) {
                        sum += layer.weights[i][j] * current_gradients[j];
                    }
                    next_gradients[i] = sum;
                }
            }
        }
    }
    
    void printProgress() {
        if (total_batches > 0) {
            float progress = (current_batch * 100.0f) / total_batches;
            int bar_width = 30;
            int pos = bar_width * progress / 100.0;
            
            std::cout << "\r  Прогресс: [";
            for (int i = 0; i < bar_width; ++i) {
                if (i < pos) std::cout << "=";
                else if (i == pos) std::cout << ">";
                else std::cout << " ";
            }
            std::cout << "] " << std::fixed << std::setprecision(1) << progress << "%";
            std::cout.flush();
        }
    }
    
    void save(const std::string& filename) const {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Не могу открыть файл: " << filename << std::endl;
            return;
        }
        
        int num_layers = layers.size();
        file.write(reinterpret_cast<const char*>(&num_layers), sizeof(int));
        file.write(reinterpret_cast<const char*>(&input_size), sizeof(int));
        file.write(reinterpret_cast<const char*>(&output_size), sizeof(int));
        
        for (const auto& layer : layers) {
            int layer_input = layer.input_size;
            int layer_output = layer.output_size;
            file.write(reinterpret_cast<const char*>(&layer_input), sizeof(int));
            file.write(reinterpret_cast<const char*>(&layer_output), sizeof(int));
            
            for (int i = 0; i < layer.input_size; ++i) {
                file.write(reinterpret_cast<const char*>(layer.weights[i].data()), 
                          layer.output_size * sizeof(float));
            }
            
            file.write(reinterpret_cast<const char*>(layer.biases.data()), 
                      layer.output_size * sizeof(float));
        }
        
        std::cout << "\n✓ Модель сохранена: " << filename << std::endl;
    }
};

// ==================== Основная функция ====================
int main() {
    std::cout << "=== ПРОГРАММА ОБУЧЕНИЯ МНОГОСЛОЙНОЙ НЕЙРОННОЙ СЕТИ ===" << std::endl;
    std::cout << "       (автоматическая настройка под железо)\n" << std::endl;
    
    Config config;
    
    fs::create_directories(config.data_dir);
    fs::create_directories(config.model_dir);
    
    // 1. Загрузка данных
    std::cout << "1. ЗАГРУЗКА ДАННЫХ" << std::endl;
    std::cout << "------------------" << std::endl;
    
    std::string all_text = TextFileReader::readAllTextFiles(config.data_dir);
    if (all_text.empty()) {
        std::cout << "Добавьте текстовые файлы в папку data/ и перезапустите программу." << std::endl;
        return 1;
    }
    
    // 2. Очистка текста
    std::cout << "\n2. ОБРАБОТКА ТЕКСТА" << std::endl;
    std::cout << "-------------------" << std::endl;
    
    std::string processed_text = ByteProcessor::cleanText(all_text);
    std::cout << "Текст обработан: " << processed_text.size() << " байт" << std::endl;
    
    // 3. Автоматическая настройка под железо
    AutoConfigurator::configureForHardware(config, processed_text.size());
    
    // ИСПРАВЛЕНО: Проверка для отключения GPU только для очень маленьких сетей
    std::cout << "\n" << std::endl;
    if (config.use_gpu && config.actual_hidden_size <= 32) {
        std::cout << "⚠  Сеть имеет маленький скрытый слой (" << config.actual_hidden_size << " нейронов)." << std::endl;
        std::cout << "   Для снижения накладных расходов переключаюсь на вычисления CPU..." << std::endl;
        config.use_gpu = false;
    }
    
    // 4. Подготовка данных для обучения
    std::cout << "\n3. ПОДГОТОВКА ДАННЫХ ДЛЯ ОБУЧЕНИЯ" << std::endl;
    std::cout << "-----------------------------------" << std::endl;
    
    auto start_pairs = std::chrono::high_resolution_clock::now();
    auto training_pairs = ByteProcessor::createTrainingPairs(processed_text, config.context_size);
    auto end_pairs = std::chrono::high_resolution_clock::now();
    auto pairs_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_pairs - start_pairs);
    
    std::cout << "Создано обучающих пар: " << training_pairs.size() << std::endl;
    std::cout << "Время создания пар: " << pairs_duration.count() << " мс" << std::endl;
    
    if (training_pairs.empty()) {
        std::cout << "Недостаточно данных для обучения!" << std::endl;
        return 1;
    }
    
    // 5. Создание нейронной сети
    std::cout << "\n4. СОЗДАНИЕ НЕЙРОННОЙ СЕТИ" << std::endl;
    std::cout << "---------------------------" << std::endl;
    
    int input_size = config.context_size * 256;
    int output_size = 256;
    
    MultiLayerNeuralNetwork nn(input_size, config.actual_hidden_size, output_size, 
                              config.actual_hidden_layers, config.use_gpu, config);
    
    // 6. Расчет количества эпох
    int epochs = config.min_epochs + (processed_text.size() / 50000);
    if (epochs > config.max_epochs) epochs = config.max_epochs;
    
    std::cout << "\n5. ПАРАМЕТРЫ ОБУЧЕНИЯ" << std::endl;
    std::cout << "--------------------" << std::endl;
    std::cout << "  Будет выполнено эпох: " << epochs << std::endl;
    std::cout << "  Размер мини-батча: " << config.actual_batch_size << std::endl;
    std::cout << "  Скорость обучения: " << config.learning_rate << std::endl;
    
    // 7. Подготовка мини-батчей (оптимизированная версия из изначального кода)
    std::cout << "\n6. ПОДГОТОВКА МИНИ-БАТЧЕЙ" << std::endl;
    std::cout << "-------------------------" << std::endl;
    
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(training_pairs.begin(), training_pairs.end(), g);
    
    size_t total_samples = std::min<size_t>(training_pairs.size(), 50000);
    size_t num_batches = (total_samples + config.actual_batch_size - 1) / config.actual_batch_size;
    
    std::cout << "  Всего примеров: " << total_samples << std::endl;
    std::cout << "  Количество батчей: " << num_batches << std::endl;
    
    // 8. Обучение с мини-батчами
    std::cout << "\n7. ОБУЧЕНИЕ С МИНИ-БАТЧАМИ" << std::endl;
    std::cout << "---------------------------" << std::endl;
    
    auto start_training = std::chrono::high_resolution_clock::now();
    
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;
        size_t samples_processed = 0;
        
        std::cout << "\n  Эпоха " << (epoch + 1) << "/" << epochs << ":" << std::endl;
        
        for (size_t batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
            size_t start = batch_idx * config.actual_batch_size;
            size_t end = std::min(start + config.actual_batch_size, total_samples);
            size_t current_batch_size = end - start;
            
            std::vector<std::vector<float>> batch_inputs(current_batch_size);
            std::vector<std::vector<float>> batch_targets(current_batch_size);
            
            #pragma omp parallel for
            for (size_t i = 0; i < current_batch_size; ++i) {
                size_t idx = start + i;
                batch_inputs[i] = ByteProcessor::vectorizeContext(
                    training_pairs[idx].first, config.context_size);
                batch_targets[i] = ByteProcessor::vectorizeTarget(
                    training_pairs[idx].second);
            }
            
            nn.trainBatch(batch_inputs, batch_targets, config.learning_rate, 
                         batch_idx + 1, num_batches);
            
            float batch_loss = 0.0f;
            #pragma omp parallel for reduction(+:batch_loss)
            for (size_t i = 0; i < batch_inputs.size(); ++i) {
                auto output = nn.forward(batch_inputs[i]);
                float sample_loss = 0.0f;
                for (size_t j = 0; j < output.size(); ++j) {
                    float diff = batch_targets[i][j] - output[j];
                    sample_loss += diff * diff;
                }
                batch_loss += sample_loss;
            }
            
            total_loss += batch_loss;
            samples_processed += current_batch_size;
            
            if ((batch_idx + 1) % 10 == 0 || (batch_idx + 1) == num_batches) {
                nn.printProgress();
            }
        }
        
        float avg_loss = total_loss / samples_processed;
        std::cout << "\n  Потери: " << std::setprecision(4) << avg_loss << std::endl;
    }
    
    auto end_training = std::chrono::high_resolution_clock::now();
    auto training_duration = std::chrono::duration_cast<std::chrono::seconds>(end_training - start_training);
    
    std::cout << "\n  Обучение заняло: " << training_duration.count() << " секунд" << std::endl;
    
    // 9. Сохранение результатов
    std::cout << "\n8. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ" << std::endl;
    std::cout << "-------------------------" << std::endl;
    
    std::ofstream text_file(config.text_file, std::ios::binary);
    text_file << processed_text;
    text_file.close();
    std::cout << "  Текст сохранен: " << config.text_file << " (" 
              << processed_text.size() << " байт)" << std::endl;
    
    nn.save(config.model_file);
    
    std::cout << "\n=== ОБУЧЕНИЕ ЗАВЕРШЕНО ===" << std::endl;
    std::cout << "  Использовано GPU: " << (config.gpu_available ? "ДА" : "НЕТ") << std::endl;
    std::cout << "  Скрытых слоев: " << config.actual_hidden_layers << std::endl;
    std::cout << "  Размер скрытого слоя: " << config.actual_hidden_size << std::endl;
    std::cout << "  Размер батча: " << config.actual_batch_size << std::endl;
    std::cout << "  Для генерации текста запустите: ./chat" << std::endl;
    
    return 0;
}