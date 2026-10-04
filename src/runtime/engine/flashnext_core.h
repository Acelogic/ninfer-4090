#pragma once

// Generation on the Flash-Next engine (Qwen3.8-Flash-Next GGUF models, architecture qwen4exp).
//
// The engine runs one sequence: a FIFO admits one request at a time under the Engine's queue
// contract (Overloaded, QueueTimeout, cancellation, streaming publication). The Qwen3.5 frontend
// renders, tokenizes and parses output exactly as for a .ninfer artifact; sampling runs on the host
// with ops::sample() semantics. Prefix reuse keeps the live sequence and host snapshots of the
// recurrent state at prompt ends and at the frontend's identity, anchor and shared-prefix
// frontiers; attention keys and values stay in the engine by position, so a snapshot is usable
// while the engine still holds its tokens.

#include "ninfer/types.h"
#include "core/device.h"
#include "models/qwen3_5/frontend/frontend.h"
#include "runtime/contract/request.h"

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

namespace ninfer::runtime {

class FlashNextCore {
public:
    using Clock = std::chrono::steady_clock;
    struct Request;
    class State;

    class Submission {
    public:
        Submission() noexcept = default;
        ~Submission();
        Submission(Submission&& other) noexcept;
        Submission& operator=(Submission&& other) noexcept;
        Submission(const Submission&)            = delete;
        Submission& operator=(const Submission&) = delete;

        GenerationResult wait(OutputSink* sink, const CancellationView& cancellation);

    private:
        Submission(State& owner, std::shared_ptr<Request> request) noexcept;
        void reset() noexcept;

        State* owner_ = nullptr;
        std::shared_ptr<Request> request_;

        friend class FlashNextCore;
    };

    // Opens the GGUF (and its sibling shards), builds the frontend and loads the engine on the
    // calling thread, which must have `device` bound.
    FlashNextCore(const EngineOptions& options, DeviceContext& device);
    ~FlashNextCore();
    FlashNextCore(const FlashNextCore&)            = delete;
    FlashNextCore& operator=(const FlashNextCore&) = delete;

    [[nodiscard]] const models::qwen3_5::Frontend& frontend() const noexcept;
    [[nodiscard]] std::uint32_t capacity() const noexcept;
    [[nodiscard]] const LoadSummary& load_summary() const noexcept;
    [[nodiscard]] ModelSamplingDefaults sampling_defaults() const;

    [[nodiscard]] Submission
    submit(models::qwen3_5::PreparedPrompt prompt, PromptSummary prompt_summary,
           double prepare_seconds, ResolvedRequestOptions options, OutputConsumerMode consumer_mode,
           GenerationObservationOptions observation, Clock::time_point pending_deadline = {});

    [[nodiscard]] MemorySummary memory_summary() const;
    [[nodiscard]] bool healthy() const noexcept;
    [[nodiscard]] bool is_available() const;
    [[nodiscard]] RuntimeStats runtime_stats() const;

    void reset_memory_peaks() noexcept {}

    // The one sequence: running request or retained live state, with its reusable snapshots.
    [[nodiscard]] std::vector<SlotState> slot_states() const;

private:
    std::unique_ptr<State> state_;
};

} // namespace ninfer::runtime
