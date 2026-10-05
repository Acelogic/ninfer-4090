// Prompt processing helpers for the Flash-Next engine (engine.cpp): the per-stage profile, the layout of
// the prompt buffers that a chunk borrows from the expert cache, and the plan that streams the experts
// that are not cached through a ring of device memory.
//
// How a long prompt runs (see docs/maintainer/flash-next-engine.md, "Prompts"):
//  - The chunk size is chosen per prompt: the largest (256-token grid, up to 8192) whose buffers fit in
//    the VRAM the expert cache can lend. Those buffers are carved from the tail of the cache's arena at
//    the start of the prompt; the experts that lived there are put back (the best ones by the cache's
//    ranking) when the prompt ends.
//  - Big chunks compute every expert on the GPU: the experts that are not cached are copied from the
//    CPU's resident copy (pinned) into the ring on a copy stream, layer by layer in expert-id order (an
//    order known before routing, so the host never waits), converted on the GPU into the cache's slot
//    layout and computed with the same kernels as cached experts. Their results do not depend on where
//    the weights came from.
//  - Smaller chunks keep the hybrid path: cached experts on the GPU, the others on the CPU.
#pragma once
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace ninfer::flashnext {

// Wall time per stage of prompt chunks, from CUDA events recorded between stages on the main stream:
// the time between two marks goes to the stage named by the second one (so a stage also carries any
// time the GPU waited for the host before it). Off unless enabled; never used inside graph capture.
class StageProfile {
public:
    ~StageProfile();
    void enable(bool on) { on_ = on; }
    bool on() const { return on_; }
    void begin(cudaStream_t s);                 // start of a chunk
    void mark(const char * stage, cudaStream_t s);
    void end_chunk();                           // after the stream has finished the chunk
    void add_host(const char * stage, double ms) { host_[stage] += ms; }
    // Prints the totals since the last report (and resets them).
    void report(std::FILE * f, const char * title);

private:
    bool on_ = false;
    std::vector<cudaEvent_t> pool_;
    std::vector<const char *> names_;
    int used_ = 0;
    std::map<std::string, double> gpu_, host_;
    std::vector<std::string> order_;
    int chunks_ = 0;
};

// A run of bytes inside a contiguous device region, by offset.
struct Carve {
    std::size_t at = 0;
    std::size_t take(std::size_t bytes) {
        const std::size_t o = at;
        at += (bytes + 255) / 256 * 256;
        return o;
    }
};

// The streamed experts of one prompt chunk: groups of up to `group` experts of one layer, in the order
// the layers run; each group's blobs sit contiguously in the ring at `ring_off`.
struct StreamGroup {
    int layer = 0;
    int key0 = 0, key1 = 0;         // keys (processing ranks) of the layer: [key0, key1)
    std::vector<int> experts;       // expert ids, ascending
    std::size_t ring_off = 0, bytes = 0;
};

struct StreamLayer {
    std::vector<int> resident;      // cached experts (keys 0 .. n-1), ascending ids
    std::vector<int> cpu;           // experts left to the CPU (key -1)
    int first_group = 0, n_groups = 0;
};

// Which uncached experts of a layer a streamed chunk of T tokens leaves to the CPU: those predicted to get
// the fewest tokens, as long as the CPU (which starts on a layer once its routing is known, so after the
// layer's dense part) finishes them no later than the copy engine finishes the layer's other experts.
// rest: the uncached experts; share[e]: predicted fraction of the layer's pairs routed to e; us_per_copy:
// microseconds to copy one expert. Returns the chosen experts (ascending ids).
struct ShareModel {
    double us_per_expert = 50;  // CPU: reading an expert's weights
    double us_per_pair = 5;     // CPU: one token through one expert
    double us_dense = 7;        // GPU: a layer's dense part, per token
};
std::vector<int> pick_cpu_share(const std::vector<int> & rest, const std::vector<double> & share, int T, double us_per_copy, const ShareModel & m);

// Places the groups in a ring of ring_bytes: consecutive, wrapping to 0 when a group would cross the end.
// Returns false if one group is larger than the ring.
bool place_groups(std::vector<StreamGroup> & groups, std::size_t ring_bytes);

}  // namespace ninfer::flashnext
