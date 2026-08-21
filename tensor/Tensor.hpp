#include "../arena/Arena.cpp"
#include "../types/types.hpp"

#include <string>

#define MODEL_VAR_MAX_INPUTS 2
#define MV_NUM_INPUTS(op) ((op) < MV_OP_UNARY_START ? 0 : ((op) < MV_OP_BINARY_START ? 1 : 2))

struct Tensor {
        u32 rows {};
        u32 cols {};
        float* data{};
};

enum class ModelVarFlags {
        MV_FLAG_NONE            = 0,
        MV_FLAG_REQUIRES_GRAD   = (1 << 0),
        MV_FLAG_PARAMETER       = (1 << 1),
        MV_FLAG_INPUT           = (1 << 2),
        MV_FLAG_OUTPUT          = (1 << 3),
        MV_FLAG_DESIRED_OUTPUT  = (1 << 4),
        MV_FLAG_COST            = (1 << 5)
};

enum class ModelVarOperator {
        MV_OP_NONE,
        MV_OP_CREATE,
        MV_OP_UNARY_START,
        MV_OP_BINARY_START,
        MV_OP_RELU,
        MV_OP_SOFTMAX,
        MV_OP_ADD,
        MV_OP_SUB,
        MV_OP_DOT,
        MV_OP_CROSS_ENTROPY,
};

struct ModelVar {
        u32 index;
        u32 flags;

        Tensor* val;
        Tensor* grad;

        ModelVarOperator op;
        ModelVar* inputs[MODEL_VAR_MAX_INPUTS];
};

struct ModelProgram {
        ModelVar** vars;
        u32 size;
};

struct ModelContext {
        u32 num_vars;

        ModelVar* input;
        ModelVar* output;
        ModelVar* desired_output;
        ModelVar* cost;

        ModelProgram forward_prog;
        ModelProgram cost_prog;
};

struct ModelTrainDesc {
        u32 epochs;
        u32 batch_size;
        float learning_rate;

        Tensor* train_images;
        Tensor* train_labels;
        Tensor* test_images;
        Tensor* test_labels;
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
        bool cross_entropy(Tensor* out, const Tensor* a, const Tensor* b);
        bool relu_grad(Tensor* out, const Tensor* in);
        bool softmax_grad(Tensor* out, const Tensor* in);
        bool cross_entropy_grad(Tensor* out, const Tensor* a, const Tensor* b);
}

namespace model {
        namespace var {
                ModelVar* create(Arena* arena, ModelContext* model, u32 rows, u32 cols, u32 flags);
                ModelVar* relu(Arena* arena, ModelContext* model, ModelVar* input, u32 flags);
                ModelVar* softmax(Arena* arena, ModelContext* model, ModelVar* input, u32 flags);
                ModelVar* add(Arena* arena, ModelContext* model, ModelVar* a, ModelVar* b, u32 flags);
                ModelVar* sub(Arena* arena, ModelContext* model, ModelVar* a, ModelVar* b, u32 flags);
                ModelVar* dot(Arena* arena, ModelContext* model, ModelVar* a, ModelVar* b, u32 flags);
                ModelVar* cross_entropy(Arena* arena, ModelContext* model, ModelVar* a, ModelVar* b, u32 flags);
        }

        namespace prog {
                ModelProgram* create(Arena* arena, ModelContext* model, ModelVar* out);
                void compute(ModelProgram* prog);
                void compute_grads(ModelProgram* prog);
        }

        namespace cntxt {
                ModelContext* create(Arena* arena);
                void compile(Arena* arena, ModelContext* model);
                void feed_forward(ModelContext* model);
                void train(ModelContext* model, const ModelTrainDesc* train_desc);
        }
}
