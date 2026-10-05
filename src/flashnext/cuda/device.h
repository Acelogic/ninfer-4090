// Minimal CUDA helpers for the Flash-Next engine: error checks and a device allocation (owning, or a view).
#pragma once
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

#include <cuda_runtime.h>

namespace ninfer::flashnext::cuda {

inline void check(cudaError_t err, const char * what) {
    if (err != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t bytes) : bytes_(bytes) {
        if (bytes) check(cudaMalloc(&ptr_, bytes), "cudaMalloc");
    }
    // A non-owning view of bytes at p (part of a larger allocation); never freed by this object.
    static DeviceBuffer view(void * p, std::size_t bytes) {
        DeviceBuffer b;
        b.ptr_ = p;
        b.bytes_ = bytes;
        b.owned_ = false;
        return b;
    }
    ~DeviceBuffer() { if (ptr_ && owned_) cudaFree(ptr_); }
    DeviceBuffer(DeviceBuffer && o) noexcept
        : ptr_(std::exchange(o.ptr_, nullptr)), bytes_(std::exchange(o.bytes_, 0)), owned_(std::exchange(o.owned_, true)) {}
    DeviceBuffer & operator=(DeviceBuffer && o) noexcept {
        if (this != &o) {
            if (ptr_ && owned_) cudaFree(ptr_);
            ptr_ = std::exchange(o.ptr_, nullptr);
            bytes_ = std::exchange(o.bytes_, 0);
            owned_ = std::exchange(o.owned_, true);
        }
        return *this;
    }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer & operator=(const DeviceBuffer &) = delete;

    void * get() const { return ptr_; }
    template <class T> T * as() const { return static_cast<T *>(ptr_); }
    std::size_t bytes() const { return bytes_; }

private:
    void * ptr_ = nullptr;
    std::size_t bytes_ = 0;
    bool owned_ = true;
};

}  // namespace ninfer::flashnext::cuda
