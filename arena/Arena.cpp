#include <cstddef>
#include <cstdint>
#include <new>
#include <utility>

class Arena {
        public:
                explicit Arena(std::size_t capacity)
                        : m_capacity (capacity)
                        , m_offset (0) {
                        m_buffer = static_cast<std::byte*>(::operator new(m_capacity));
                }

                ~Arena() {
                        ::operator delete(m_buffer);
                }

                template<typename T, typename... Args>
                T* alloc(Args&&... args) {
                        void* object { alloc(sizeof(T), alignof(T)) };
                        if(object == nullptr) return nullptr;

                        return new (object) T { std::forward<Args>(args)... };
                }

                template<typename T>
                T* alloc_array(std::size_t size) {
                        void* object { alloc(sizeof(T) * size, alignof(T)) };
                        if(object == nullptr) return nullptr;

                        return static_cast<T*>(object);
                }

                void reset() {
                        m_offset = 0;
                }

                std::size_t used() const {
                        return m_offset;
                }

                std::size_t remaining() {
                        return m_capacity - m_offset;
                }

                static std::size_t KiB(std::size_t n) {
                        return static_cast<std::size_t>(n << 10);
                }

                static std::size_t MiB(std::size_t n) {
                        return static_cast<std::size_t>(n << 20);
                }

                static std::size_t GiB(std::size_t n) {
                        return static_cast<std::size_t>(n << 30);
                }

        private:
                void* alloc(std::size_t size, std::size_t alignment) {
                        uintptr_t current { reinterpret_cast<uintptr_t>(m_buffer + m_offset) };
                        uintptr_t aligned { (current + alignment - 1) & ~(alignment - 1) };
                        std::size_t padding { aligned - current };

                        if(m_offset + size + padding > m_capacity) return nullptr;

                        m_offset += padding;
                        void* result = { m_buffer + m_offset };
                        m_offset += size;

                        return result;
                }

                std::byte* m_buffer {};
                std::size_t m_capacity {};
                std::size_t m_offset {};
};
