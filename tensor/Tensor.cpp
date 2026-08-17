#include "Tensor.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <cmath>

Tensor* tensor::create(Arena* arena, u32 rows, u32 cols) {
        Tensor* tensor = arena->alloc<Tensor>();
        if(tensor == nullptr) return nullptr;

        tensor->rows = rows;
        tensor->cols = cols;
        tensor->data = arena->alloc_array<float>(rows * cols);
        if(tensor->data == nullptr) return nullptr;

        return tensor;
}

Tensor* tensor::load(Arena* arena, u32 rows, u32 cols, const std::string filename) {            // helper function for the mnist dataset. could be other uses
        Tensor* tensor = tensor::create(arena, rows, cols);
        if(!tensor) return nullptr;

        std::ifstream file(filename, std::ios::binary);
        if(!file) return nullptr;

        file.seekg(0, std::ios::end);

        std::streampos end = file.tellg();
        if(end == std::streampos(-1)) return nullptr;

        std::size_t size = static_cast<std::size_t>(end);

        file.seekg(0, std::ios::beg);

        size = std::min(size, sizeof(float) * rows * cols);

        file.read(reinterpret_cast<char*>(tensor->data), size);

        return tensor;
}

void tensor::copy(Tensor* dst, const Tensor* src) {
        if(dst->rows != src->rows || dst->cols != src->cols) return;

        std::size_t count = src->rows * src->cols;
        for(std::size_t i = 0; i < count; i++) {
                dst->data[i] = src->data[i];
        }
}

void tensor::clear(Tensor* tnsr) {
        if(tnsr == nullptr || tnsr->data == nullptr) return;

        std::size_t count = tnsr->rows * tnsr->cols;
        for(std::size_t i = 0; i < count; i++) {
                tnsr->data[i] = 0;
        }
}

void tensor::fill(Tensor* tnsr, float data) {
        if(tnsr == nullptr || tnsr->data == nullptr || tnsr->rows == 0 || tnsr->cols == 0) return;

        std::size_t count = tnsr->rows * tnsr->cols;
        for(std::size_t i = 0; i < count; i++) {
                tnsr->data[i] = data;
        }
}

void tensor::scale(Tensor* tnsr, float scale) {
        if(tnsr == nullptr || tnsr->data == nullptr || tnsr->rows == 0 || tnsr->cols == 0) return;

        std::size_t count = tnsr->rows * tnsr->cols;
        for(std::size_t i = 0; i < count; i++) {
                tnsr->data[i] *= scale;
        }
}

void tensor::print(const Tensor* tnsr) {
        std::size_t count = tnsr->rows * tnsr->cols;
        for(std::size_t i = 0; i < count; i++) {
                std::cout << tnsr->data[i] << " ";
                if((i + 1) % tnsr->cols == 0) std::cout << "\n";
        }
}

bool tensor::add(Tensor* out, const Tensor* a, const Tensor* b) {
        if(a == nullptr || a->data == nullptr || a->rows == 0 || a->cols == 0 ||
                b == nullptr || b->data == nullptr || b->rows == 0 || b->cols == 0 ||
                out == nullptr || out->data == nullptr || out->rows == 0 || out->cols == 0 ||
                a->rows != b->rows || a->cols != b->cols ||
                out->rows != a->rows || out->cols != a->cols) return false;

        std::size_t count = a->rows * a->cols;
        for(std::size_t i = 0; i < count; i++) {
                out->data[i] = a->data[i] + b->data[i];
        }

        return true;
}

bool tensor::sub(Tensor* out, const Tensor* a, const Tensor* b) {
        if(a == nullptr || a->data == nullptr || a->rows == 0 || a->cols == 0 ||
                b == nullptr || b->data == nullptr || b->rows == 0 || b->cols == 0 ||
                out == nullptr || out->data == nullptr || out->rows == 0 || out->cols == 0 ||
                a->rows != b->rows || a->cols != b->cols ||
                out->rows != a->rows || out->cols != a->cols) return false;

        std::size_t count = a->rows * a->cols;
        for(std::size_t i = 0; i < count; i++) {
                out->data[i] = a->data[i] - b->data[i];
        }

        return true;
}

bool tensor::dot(Tensor* out, const Tensor* a, const Tensor* b) {
        if(a == nullptr || a->data == nullptr || a->rows == 0 || a->cols == 0 ||
                b == nullptr || b->data == nullptr || b->rows == 0 || b->cols == 0 ||
                out == nullptr || out->data == nullptr || out->rows == 0 || out->cols == 0 ||
                a->cols != b->rows || out->rows != a->rows || out->cols != b->cols) return false;

        std::size_t a_i = a->rows;
        std::size_t a_j = a->cols;
        std::size_t b_j = b->cols;

        for(std::size_t i = 0; i < a_i; i++) {
                for(std::size_t j = 0; j < b_j; j++) {
                        for(std::size_t k = 0; k < a_j; k++) {
                                out->data[i * out->cols + j] = a->data[i * a->cols + k] * b->data[k * b->cols + j];
                        }
                }
        }

        return true;
}

bool tensor::relu(Tensor* out, const Tensor* in) {
        if(in == nullptr || in->data == nullptr || in->rows == 0 || in->cols == 0 ||
                out == nullptr || out->data == nullptr || out->rows == 0 || out->cols == 0 ||
                out->rows != in->rows || out->cols != in->cols) return false;

        std::size_t count = in->rows * in->cols;
        for(std::size_t i = 0; i < count; i++) {
                out->data[i] = (in->data[i] > 0) ? in->data[i] : 0;
        }

        return true;
}

bool tensor::softmax(Tensor* out, const Tensor* in) {
        if(in == nullptr || in->data == nullptr || in->rows == 0 || in->cols == 0 ||
                out == nullptr || out->data == nullptr || out->rows == 0 || out->cols == 0 ||
                out->rows != in->rows || out->cols != in->cols) return false;

        std::size_t count = in->rows * in->cols;
        float sum {};
        for(std::size_t i = 0; i < count; i++) {
                out->data[i] = expf(in->data[i]);
                sum += out->data[i];
        }
        tensor::scale(out, 1.0f / sum);

        return true;
}

bool tensor::cross_entropy(Tensor* out, const Tensor* a, Tensor* b) {
        if(a == nullptr || a->data == nullptr || a->rows == 0 || a->cols == 0 ||
                b == nullptr || b->data == nullptr || b->rows == 0 || b->cols == 0 ||
                a->rows != b->rows || a->cols != b->cols ||
                out == nullptr || out->data == nullptr || out->rows == 0 || out->cols == 0 ||
                out->rows != a->rows || out->cols != a->cols) return false;

        std::size_t count = a->rows * a->cols;
        for(std::size_t i = 0; i < count; i++) {
                out->data[i] = (a->data[i] == 0.0f) ? 0.0f : a->data[i] * -logf(b->data[i]);
        }

        return true;
}
