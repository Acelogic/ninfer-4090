// VRAM shared with other programs. On Windows the OS gives every process a budget of the GPU's memory (DXGI) and,
// when a process holds more than its budget because another one needs memory (the desktop when a display wakes up,
// a game), pages that process's allocations out to system RAM; GPU work on them then crawls over PCIe. VramBudget
// reads the budget; ElasticArena is device memory whose top can be given back and taken again while its addresses
// stay the same (CUDA graphs keep pointers into it).
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninfer::flashnext {

class VramBudget {
public:
    // The budget of the adapter that runs CUDA device `device` (matched by LUID). Unavailable outside Windows or
    // without DXGI 1.4. NINFER_FN_VRAM_BUDGET_MIB caps the budget it reports (tests).
    explicit VramBudget(int device);
    ~VramBudget();
    VramBudget(const VramBudget &) = delete;
    VramBudget & operator=(const VramBudget &) = delete;

    bool available() const { return adapter_ != nullptr; }
    struct Reading {
        std::int64_t budget = 0;  // bytes of the GPU's memory the OS lets this process use without paging it
        std::int64_t usage = 0;   // bytes this process uses (every CUDA allocation counts)
        bool valid = false;       // false: the query failed (nothing to act on)
    };
    Reading read() const;  // a few microseconds

private:
    void * adapter_ = nullptr;  // IDXGIAdapter3
    std::int64_t cap_ = 0;
};

class ElasticArena {
public:
    ElasticArena() = default;
    // Reserves `bytes` (rounded up to whole chunks) of device addresses and maps memory to all of them. Without the
    // driver's virtual memory management it is one plain allocation, and shrink_to / grow_to change nothing.
    explicit ElasticArena(std::size_t bytes);
    ~ElasticArena();
    ElasticArena(ElasticArena && o) noexcept { *this = static_cast<ElasticArena &&>(o); }
    ElasticArena & operator=(ElasticArena && o) noexcept;
    ElasticArena(const ElasticArena &) = delete;
    ElasticArena & operator=(const ElasticArena &) = delete;

    std::uint8_t * base() const { return reinterpret_cast<std::uint8_t *>(base_); }
    std::size_t bytes() const { return bytes_; }    // reserved
    std::size_t mapped() const { return mapped_; }  // [0, mapped()) has memory
    std::size_t chunk() const { return chunk_; }
    bool elastic() const { return chunk_ != 0; }
    // Gives back the memory at and above `bytes` (rounded up to a chunk). Nothing on the device may use it any more.
    void shrink_to(std::size_t bytes);
    // Maps memory up to `bytes` (rounded up to a chunk, at most bytes()); stops early when the device has none left.
    // Returns mapped().
    std::size_t grow_to(std::size_t bytes);

private:
    void release();
    std::uintptr_t base_ = 0;
    std::size_t bytes_ = 0, mapped_ = 0, chunk_ = 0;
    int device_ = 0;
    std::vector<unsigned long long> handles_;  // CUmemGenericAllocationHandle per mapped chunk
    void * plain_ = nullptr;                   // the fallback allocation
};

}  // namespace ninfer::flashnext
