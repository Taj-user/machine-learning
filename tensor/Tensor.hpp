#include "../arena/Arena.cpp"
#include "../types/types.hpp"

struct Tensor {
        u32 rows {};
        u32 cols {};
        float* data{};
};

namespace tensor {
        Tensor* create(Arena* arena, u32 rows, u32 cols);
        Tensor* load(Arena* arena, u32 rows, u32 cols, const std::string filename);
        void copy(Tensor* dst, const Tensor* src);
        void clear(Tensor* tnsr);
        void fill(Tensor* tnsr, float data);
        void scale(Tensor* tnsr, float scale);
        void print(const Tensor* tnsr);
        bool add(Tensor* out, const Tensor* a, const Tensor* b);
        bool sub(Tensor* out, const Tensor* a, const Tensor* b);
        bool dot(Tensor* out, const Tensor* a, const Tensor* b);
        bool relu(Tensor* out, const Tensor* in);
        bool softmax(Tensor* out, const Tensor* in);
        bool cross_entropy(Tensor* out, const Tensor* a, Tensor* b);
        bool relu_grad(Tensor* out, const Tensor* in);
        bool softmax_grad(Tensor* out, const Tensor* in);
        bool cross_entropy_grad(Tensor* out, const Tensor* a, const Tensor* b);
}
