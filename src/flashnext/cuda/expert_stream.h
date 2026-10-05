// Streaming routed experts from host memory to the GPU for prompt processing.
//
// CpuExperts keeps every expert in RAM in its own lossless repacks (Q4L or Q4X gate/up, IQ4L or Q8_0
// down; quants.h), one contiguous blob [gate | up | down] per expert. A prompt chunk copies the blobs
// of the experts that are not in the VRAM cache into a ring of device memory (pinned host memory, one
// DMA per run of consecutive experts) and converts them on the GPU, bit for bit, into the cache's slot
// layout (experts.h), where the same kernels as for cached experts compute them. The conversion is the
// GPU version of CpuExperts::export_expert followed by pack_expert_rows.
#pragma once
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "flashnext/cuda/experts.h"

namespace ninfer::flashnext::cuda {

// The host format of one layer's experts (CpuExperts::HostLayer).
struct HostExpertFormat {
    bool gate_q4x = false;       // gate/up Q4X (IQ4_XS layers), else Q4L (IQ3_S layers)
    bool down_q8 = false;        // down Q8_0, else IQ4L (IQ4_NL layers)
    std::size_t gate_bytes = 0;  // bytes of the gate (and of the up) matrix
    std::size_t down_bytes = 0;
    std::size_t blob_bytes() const { return 2 * gate_bytes + down_bytes; }
};

// The slot layout the host format converts to (the GGUF's types: IQ3_S/IQ4_XS gate/up, IQ4_NL/Q8_0 down).
ExpertLayout stream_layout(const HostExpertFormat & f);

constexpr int kMaxConvert = 64;  // experts per convert_experts call

struct ConvertBatch {
    const std::uint8_t * src[kMaxConvert];  // device addresses of host-format blobs
    std::uint8_t * dst[kMaxConvert];        // device addresses of slots (layout.slot_bytes each)
    int n = 0;
};

// Converts b.n experts from their host format into slot images. Asynchronous on `stream`; src must stay
// valid until the kernel has run. Every byte of every slot is written (identical to pack_expert).
void convert_experts(const HostExpertFormat & f, const ExpertLayout & layout, const ConvertBatch & b, cudaStream_t stream);

// Uploads the conversion tables; called by convert_experts on first use (call it before graph capture).
void expert_stream_init();

}  // namespace ninfer::flashnext::cuda
