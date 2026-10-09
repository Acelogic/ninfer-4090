#include "flashnext/vram.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dxgi1_4.h>
#endif

namespace ninfer::flashnext {

namespace {

void cu_check(CUresult r, const char * what) {
    if (r == CUDA_SUCCESS) return;
    const char * s = nullptr;
    cuGetErrorString(r, &s);
    throw std::runtime_error(std::string("engine: ") + what + ": " + (s ? s : "CUDA driver error"));
}

void rt_check(cudaError_t e, const char * what) {
    if (e == cudaSuccess) return;
    throw std::runtime_error(std::string("engine: ") + what + ": " + cudaGetErrorString(e));
}

CUmemAllocationProp device_memory(int device) {
    CUmemAllocationProp p{};
    p.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    p.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    p.location.id = device;
    return p;
}

// Small enough to give back about what the budget lacks, large enough that a 14 GiB cache maps in ~450 calls
// (measured on an RTX 4090 under WDDM: 10 GiB in 17 ms, 4 GiB unmapped and released in 29 ms).
constexpr std::size_t kChunk = std::size_t(32) << 20;

}  // namespace

// ---------------------------------------------------------------------------------------------------------------------

VramBudget::VramBudget(int device) {
    if (const char * c = std::getenv("NINFER_FN_VRAM_BUDGET_MIB")) cap_ = std::int64_t(std::atoll(c)) << 20;
#ifdef _WIN32
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        cudaGetLastError();
        return;
    }
    LUID luid;
    std::memcpy(&luid, prop.luid, sizeof luid);
    IDXGIFactory4 * factory = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory4), reinterpret_cast<void **>(&factory)))) return;
    IDXGIAdapter1 * a = nullptr;
    IDXGIAdapter3 * found = nullptr;
    for (UINT i = 0; !found && factory->EnumAdapters1(i, &a) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 d;
        if (SUCCEEDED(a->GetDesc1(&d)) && d.AdapterLuid.LowPart == luid.LowPart && d.AdapterLuid.HighPart == luid.HighPart)
            a->QueryInterface(__uuidof(IDXGIAdapter3), reinterpret_cast<void **>(&found));
        a->Release();
    }
    factory->Release();
    adapter_ = found;
#else
    (void)device;
#endif
}

VramBudget::~VramBudget() {
#ifdef _WIN32
    if (adapter_) static_cast<IDXGIAdapter3 *>(adapter_)->Release();
#endif
}

VramBudget::Reading VramBudget::read() const {
    Reading r;
#ifdef _WIN32
    DXGI_QUERY_VIDEO_MEMORY_INFO info{};
    if (adapter_ && SUCCEEDED(static_cast<IDXGIAdapter3 *>(adapter_)->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info))) {
        r.budget = std::int64_t(info.Budget);
        r.usage = std::int64_t(info.CurrentUsage);
        r.valid = r.budget > 0;
    }
#endif
    if (r.valid && cap_ > 0) r.budget = std::min(r.budget, cap_);
    return r;
}

// ---------------------------------------------------------------------------------------------------------------------

ElasticArena::ElasticArena(std::size_t bytes) {
    if (!bytes) return;
    rt_check(cudaGetDevice(&device_), "expert cache");
    rt_check(cudaFree(nullptr), "expert cache");  // the device's primary context is current for the driver calls
    CUdevice dev = 0;
    int vmm = 0;
    std::size_t gran = 0;
    const CUmemAllocationProp prop = device_memory(device_);
    const bool want = std::getenv("NINFER_FN_NO_VMM") == nullptr;
    if (want && cuDeviceGet(&dev, device_) == CUDA_SUCCESS &&
        cuDeviceGetAttribute(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, dev) == CUDA_SUCCESS && vmm &&
        cuMemGetAllocationGranularity(&gran, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED) == CUDA_SUCCESS && gran) {
        const std::size_t chunk = (kChunk + gran - 1) / gran * gran;
        const std::size_t reserved = (bytes + chunk - 1) / chunk * chunk;
        CUdeviceptr va = 0;
        if (cuMemAddressReserve(&va, reserved, 0, 0, 0) == CUDA_SUCCESS) {
            base_ = std::uintptr_t(va);
            bytes_ = reserved;
            chunk_ = chunk;
            if (grow_to(bytes_) == bytes_) return;
            release();
            throw std::runtime_error("engine: not enough VRAM for the expert cache");
        }
    }
    rt_check(cudaMalloc(&plain_, bytes), "expert cache");
    base_ = reinterpret_cast<std::uintptr_t>(plain_);
    bytes_ = mapped_ = bytes;
}

ElasticArena::~ElasticArena() { release(); }

ElasticArena & ElasticArena::operator=(ElasticArena && o) noexcept {
    if (this == &o) return *this;
    release();
    base_ = o.base_;
    bytes_ = o.bytes_;
    mapped_ = o.mapped_;
    chunk_ = o.chunk_;
    device_ = o.device_;
    handles_ = std::move(o.handles_);
    plain_ = o.plain_;
    o.base_ = 0;
    o.bytes_ = o.mapped_ = o.chunk_ = 0;
    o.handles_.clear();
    o.plain_ = nullptr;
    return *this;
}

void ElasticArena::release() {
    if (chunk_ && base_) {
        cudaSetDevice(device_);
        if (mapped_) cuMemUnmap(CUdeviceptr(base_), mapped_);
        for (unsigned long long h : handles_) cuMemRelease(CUmemGenericAllocationHandle(h));
        cuMemAddressFree(CUdeviceptr(base_), bytes_);
    } else if (plain_) {
        cudaFree(plain_);
    }
    handles_.clear();
    base_ = 0;
    bytes_ = mapped_ = chunk_ = 0;
    plain_ = nullptr;
}

std::size_t ElasticArena::grow_to(std::size_t bytes) {
    if (!chunk_) return mapped_;
    bytes = std::min(bytes_, (bytes + chunk_ - 1) / chunk_ * chunk_);
    if (bytes <= mapped_) return mapped_;
    rt_check(cudaSetDevice(device_), "expert cache");
    const CUmemAllocationProp prop = device_memory(device_);
    const std::size_t from = mapped_;
    while (mapped_ < bytes) {
        CUmemGenericAllocationHandle h = 0;
        if (cuMemCreate(&h, chunk_, &prop, 0) != CUDA_SUCCESS) break;  // out of memory: keep what was mapped
        if (cuMemMap(CUdeviceptr(base_ + mapped_), chunk_, 0, h, 0) != CUDA_SUCCESS) {
            cuMemRelease(h);
            break;
        }
        handles_.push_back(h);
        mapped_ += chunk_;
    }
    if (mapped_ > from) {
        CUmemAccessDesc access{};
        access.location = prop.location;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        cu_check(cuMemSetAccess(CUdeviceptr(base_ + from), mapped_ - from, &access, 1), "expert cache access");
    }
    return mapped_;
}

void ElasticArena::shrink_to(std::size_t bytes) {
    if (!chunk_) return;
    bytes = (bytes + chunk_ - 1) / chunk_ * chunk_;
    if (bytes >= mapped_) return;
    rt_check(cudaSetDevice(device_), "expert cache");
    cu_check(cuMemUnmap(CUdeviceptr(base_ + bytes), mapped_ - bytes), "expert cache unmap");
    while (mapped_ > bytes) {
        cuMemRelease(CUmemGenericAllocationHandle(handles_.back()));
        handles_.pop_back();
        mapped_ -= chunk_;
    }
}

}  // namespace ninfer::flashnext
