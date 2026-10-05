// The vision encoder of Qwen3.8-Flash-Next: Qwen3-VL's ViT (27 layers of 1152, 16 heads, 2-D rope, learned position
// embedding resized to the image) and its 2x2 patch merger, from the mmproj GGUF that llama.cpp uses
// (general.architecture = clip, clip.projector_type = qwen3vl_merger). It turns an image's patches into the rows the
// language model reads at its <|image_pad|> positions (ForwardInputs::embeddings).
//
// Numerics: FP32 activations, the F16 weights exact (converted to FP32 for cuBLAS SGEMM, no TF32), FP32 attention.
// The merger's GELU is exact (erf), as in the model's definition (Qwen3VLVisionPatchMerger uses nn.GELU()); the ViT's
// MLP uses the tanh approximation (gelu_pytorch_tanh), as llama.cpp does for both.
//
// VRAM: none between images. The weights (0.87 GB, F16 as in the file) live in pinned host memory; encode() runs in a
// workspace the caller lends it (Engine::lend_vram, from the expert cache) and streams one layer at a time into it
// over PCIe, overlapped with the previous layer's compute.
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace ninfer::flashnext {

struct VisionConfig {
    int image_size = 0;   // 768: the position embedding's grid is image_size / patch per side
    int patch = 0;        // 16
    int hidden = 0;       // 1152
    int ff = 0;           // 4304
    int layers = 0;       // 27
    int heads = 0;        // 16
    int merge = 0;        // 2: tokens are 2x2 blocks of patches
    int out_dim = 0;      // 2560 (the language model's hidden size)
    float eps = 0;        // 1e-6, every LayerNorm
    std::string mean_std; // "0.5,0.5,0.5/0.5,0.5,0.5": normalization the patches must have
};

struct VisionTiming {
    double total_ms = 0;  // wall time of encode()
    double gpu_ms = 0;    // from the first upload to the last kernel
};

class VisionEncoder {
public:
    // Reads the mmproj GGUF (the file must be a qwen3vl_merger clip model without deepstack layers) into pinned host
    // memory. Allocates no VRAM but a cuBLAS handle.
    explicit VisionEncoder(const std::string & mmproj_path);
    ~VisionEncoder();
    VisionEncoder(const VisionEncoder &) = delete;
    VisionEncoder & operator=(const VisionEncoder &) = delete;

    const VisionConfig & config() const;
    // Values per patch: 3 channels x 2 frames x patch x patch (an image repeats its frame), normalized pixels.
    int patch_values() const;
    // Device memory encode() needs for an image of `patches` patches.
    std::size_t workspace_bytes(int patches) const;

    // patches: [P][patch_values()] FP32, P = grid_h * grid_w (both even), in the frontend's merge order (2x2 blocks of
    // the patch grid row by row; within a block (0,0), (0,1), (1,0), (1,1)), each patch [channel][frame][y][x].
    // workspace: workspace_bytes(P) bytes of device memory. out: [P / 4][out_dim] host floats, one row per token, in
    // the same block order (row-major over the (grid_h / 2) x (grid_w / 2) token grid). Synchronous.
    VisionTiming encode(const float * patches, int grid_h, int grid_w, void * workspace, float * out);
    // The merger's GELU: exact (erf, the default, as the model defines it) or the tanh approximation (llama.cpp's).
    void set_exact_merger_gelu(bool exact);

    struct Impl;

private:
    std::unique_ptr<Impl> impl_;
};

}  // namespace ninfer::flashnext
