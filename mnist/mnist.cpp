#include "../tensor/Tensor.hpp"

#include <iostream>

void draw_mnist_digit(float* data) {
        for(u32 y = 0; y < 28; y += 2) {
                for(u32 x = 0; x < 28; x++) {
                        float top = data[x + y * 28];
                        float bottom = data[x + (y + 1) * 28];

                        u32 top_col = 232 + static_cast<u32>(top * 23);
                        u32 bottom_col = 232 + static_cast<u32>(bottom * 23);

                        std::cout
                                << "\x1b[38;5;" << top_col << "m"
                                << "\x1b[48;5;" << bottom_col << "m"
                                << "\u2580";
                }
                std::cout << "\x1b[0m\n";
        }
}

int main(void) {
        Arena arena {Arena::GiB(1)};

        Tensor* train_images = tensor::load(&arena, 60'000, 784, "mnist/train_images.bin");
        [[maybe_unused]] Tensor* test_images = tensor::load(&arena, 10'000, 784, "mnist/test_images.bin");

        Tensor* train_labels = tensor::create(&arena, 60'000, 10);
        Tensor* test_labels = tensor::create(&arena, 10'000, 10);

        {
                Tensor* train_labels_file = tensor::load(&arena, 60'000, 1, "mnist/train_labels.bin");
                Tensor* test_labels_file = tensor::load(&arena, 10'000, 1, "mnist/test_labels.bin");

                for(u32 i = 0; i < 60'000; i++) {
                        u32 num = train_labels_file->data[i];
                        train_labels->data[i * 10 + num] = 1.0f;
                }

                for(u32 i = 0; i < 10'000; i++) {
                        u32 num = test_labels_file->data[i];
                        test_labels->data[i * 10 + num] = 1.0f;
                }
        }

        draw_mnist_digit(train_images->data);
        for(u32 i = 0; i < 10; i++) {
                std::cout << train_labels->data[i];
        }

        return 0;
}
