#include "runtime/engine/flashnext_core.h"

#include "core/startup.h"
#include "flashnext/engine.h"
#include "flashnext/gguf.h"
#include "flashnext/shards.h"
#include "flashnext/vision.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "runtime/engine/flashnext_frontend.h"
#include "runtime/engine/generation_budget.h"
#include "runtime/engine/host_sampler.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdio>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <list>
#include <exception>
#include <limits>
#include <mutex>
#include <new>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <variant>

namespace ninfer::runtime {
namespace {

namespace fn = ninfer::flashnext;
using Clock  = FlashNextCore::Clock;

constexpr std::string_view kArchitecture = "qwen4exp";
// Prompt tokens per engine call: bounds the latency of cancellation and of progress publication
// during long prompts. A multiple of every useful engine prefill chunk, so splitting a long suffix
// keeps the engine's batched decomposition.
constexpr std::uint32_t kPrefillPieceTokens = 8192;

std::uint64_t elapsed_ns(Clock::time_point started, Clock::time_point finished) noexcept {
    if (finished <= started) { return 0; }
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(finished - started).count());
}

double seconds_between(Clock::time_point started, Clock::time_point finished) noexcept {
    return std::chrono::duration<double>(finished - started).count();
}

std::size_t free_device_bytes() {
    std::size_t free_bytes  = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    return free_bytes;
}

void validate_options(const EngineOptions& options) {
    const auto reject = [](const char* what) {
        throw std::invalid_argument(std::string("the Flash-Next backend does not support ") + what);
    };
    if (options.purpose != EnginePurpose::Generation) { reject("causal scoring"); }
    if (options.max_concurrency != 1) { reject("max_concurrency above 1 (it runs one sequence)"); }
    if (options.speculative.backend != SpeculativeBackend::None) { reject("speculative decoding"); }
    if (options.enable_vision && options.flashnext.vision_path.empty()) {
        reject("--vision without a vision encoder (images need --flashnext-vision <mmproj.gguf>)");
    }
    if (options.kv_cache != KvCacheStorage::BFloat16) { reject("KV cache formats"); }
    if (options.auto_save_evicted) { reject("slot auto-save"); }
    if (options.max_pending_requests == 0 || options.pending_timeout_ms == 0) {
        throw std::invalid_argument("Engine pending request capacity and timeout must be nonzero");
    }
}

// The engine verifies a step of at most four tokens: the fed token and up to three drafts.
constexpr std::uint32_t kMaximumDraftTokens = 3;
constexpr std::uint32_t kDefaultDraftTokens = 2;

std::uint32_t resolve_draft_tokens(const FlashNextOptions& options) {
    if (options.mtp_path.empty()) {
        if (options.draft_tokens != 0) {
            throw std::invalid_argument("Flash-Next draft tokens need an MTP head");
        }
        return 0;
    }
    const std::uint32_t k = options.draft_tokens == 0 ? kDefaultDraftTokens : options.draft_tokens;
    if (k > kMaximumDraftTokens) {
        throw std::invalid_argument("Flash-Next drafts at most 3 tokens per step");
    }
    return k;
}

// Per prompt position, what besides the token id its input is: 0 for text, else a key of the image (its content digest
// and grid) and of the position's index in it. Prefix reuse compares these with the token ids, since every image
// position carries the same <|image_pad|> id. An empty vector stands for all zeros (text only).
using MediaKeys = std::vector<std::uint64_t>;

std::uint64_t media_key_at(const MediaKeys& keys, std::size_t i) noexcept { return i < keys.size() ? keys[i] : 0; }

bool media_equal(const MediaKeys& a, const MediaKeys& b, std::size_t depth) noexcept {
    for (std::size_t i = 0; i < depth; ++i) {
        if (media_key_at(a, i) != media_key_at(b, i)) { return false; }
    }
    return true;
}

std::uint64_t image_key(const models::qwen3_5::VisionItem& item) noexcept {
    std::uint64_t h = 1469598103934665603ull;
    for (const std::uint8_t b : item.content_digest) { h = (h ^ b) * 1099511628211ull; }
    for (const std::int32_t v : {item.grid.temporal, item.grid.height, item.grid.width}) {
        h = (h ^ std::uint64_t(std::uint32_t(v))) * 1099511628211ull;
    }
    return h;
}

MediaKeys prompt_media_keys(const models::qwen3_5::PreparedPromptData& data) {
    MediaKeys keys;
    if (!data.has_media()) { return keys; }
    keys.assign(data.token_ids.size(), 0);
    for (const auto& item : data.vision_items) {
        const std::uint64_t key = image_key(item);
        for (const auto& span : item.token_spans) {
            for (std::size_t i = 0; i < span.count && span.begin + i < keys.size(); ++i) {
                keys[span.begin + i] = ((key + i) * 0x9E3779B97F4A7C15ull) | 1u;
            }
        }
    }
    return keys;
}

// The frontend stores an image's patches as bf16 of u / 127.5 - 1 (u the 8-bit pixel value, mean and std 0.5): each of
// the 256 values maps back to its exact pixel, which is normalized in FP32 as llama.cpp does, (u / 255 - 0.5) / 0.5.
class PatchPixels {
public:
    PatchPixels() {
        table_.fill(-1);
        for (int u = 0; u < 256; ++u) {
            std::uint32_t bits = std::bit_cast<std::uint32_t>(float(u) / 127.5f - 1.0f);
            bits += 0x7fffU + ((bits >> 16U) & 1U);  // the frontend's round-to-nearest-even bf16
            const auto b = static_cast<std::uint16_t>(bits >> 16U);
            if (table_[b] >= 0) { throw std::logic_error("Flash-Next: bf16 pixel values collide"); }
            table_[b] = std::int16_t(u);
        }
    }
    void convert(std::span<const std::uint16_t> in, float* out) const {
        for (std::size_t i = 0; i < in.size(); ++i) {
            const int u = table_[in[i]];
            if (u < 0) { throw std::invalid_argument("Flash-Next: an image patch value is not a normalized pixel"); }
            out[i] = (float(u) / 255.0f - 0.5f) / 0.5f;
        }
    }

private:
    std::array<std::int16_t, 65536> table_{};
};

// A host image of the engine's recurrent state at one token frontier.
struct Snapshot {
    fn::EngineSnapshot state;
    MediaKeys media;  // the media keys of its positions
    // Next-token logits over the public domain when taken at a prompt end: a later prompt that
    // ends at the same frontier samples from them without executing a token.
    std::vector<float> logits;
    PrefixReusePath path    = PrefixReusePath::PrivateEndpoint;
    std::uint64_t last_used = 0;

    [[nodiscard]] std::uint32_t depth() const noexcept {
        return static_cast<std::uint32_t>(state.tokens.size());
    }
};

// Strength of a frontier's reuse class when several requests name the same token prefix.
int path_rank(PrefixReusePath path) noexcept {
    switch (path) {
    case PrefixReusePath::PrivateTurnClosure:
    case PrefixReusePath::PrivateResponseReplay:
        return 3;
    case PrefixReusePath::SharedStablePrefix:
        return 2;
    case PrefixReusePath::PrivateLongAnchor:
        return 1;
    default:
        return 0;
    }
}

} // namespace

struct FlashNextCore::Request {
    using StreamEvent = std::variant<OutputDelta, GenerationTimingObservation>;

    Request(std::uint64_t request_id, models::qwen3_5::PreparedPrompt prepared,
            models::qwen3_5::OutputSession session, PromptSummary summary, double preparation,
            ResolvedRequestOptions resolved, OutputConsumerMode mode,
            GenerationObservationOptions observe, Clock::time_point deadline,
            Clock::time_point submitted_at)
        : id(request_id), prompt(std::move(prepared)), output(std::move(session)),
          prompt_summary(summary), prepare_seconds(preparation), options(std::move(resolved)),
          consumer_mode(mode), observation(observe), pending_deadline(deadline),
          submitted(submitted_at) {}

    const std::uint64_t id;
    models::qwen3_5::PreparedPrompt prompt;
    models::qwen3_5::OutputSession output;
    const PromptSummary prompt_summary;
    const double prepare_seconds;
    const ResolvedRequestOptions options;
    const OutputConsumerMode consumer_mode;
    const GenerationObservationOptions observation;
    const Clock::time_point pending_deadline;
    const Clock::time_point submitted;
    std::atomic<bool> cancelled{false};

    // Worker-owned execution record.
    std::vector<TokenId> generated;
    std::string content;
    std::string reasoning;
    std::optional<Clock::time_point> admitted_at;
    std::optional<Clock::time_point> first_token;
    std::optional<Clock::time_point> last_token;
    std::uint32_t reused_prompt_tokens   = 0;
    std::uint32_t computed_prompt_tokens = 0;
    PrefixReusePath prefix_reuse_path    = PrefixReusePath::Root;
    double prefill_seconds               = 0.0;
    double decode_seconds                = 0.0;
    double vision_seconds                = 0.0;
    // the media keys of the prompt's positions, and per vision item its rows ([tokens][2560], null when its positions
    // are reused and need none)
    MediaKeys media;
    std::vector<std::shared_ptr<const std::vector<float>>> image_rows;
    SpeculativeStats speculative;

    // Consumer channel.
    std::mutex mutex;
    std::condition_variable cv;
    std::optional<GenerationStart> stream_start;
    std::optional<PromptProgress> stream_progress;
    std::vector<StreamEvent> events;
    bool response_done     = false;
    bool consumer_released = false;
    bool capacity_released = false;
    GenerationResult result;
    std::exception_ptr error;
};

// A prompt with images as forward() feeds it: its data (positions, image spans) and the request (media keys, image rows).
struct PromptMedia {
    const models::qwen3_5::PreparedPromptData* data = nullptr;
    const FlashNextCore::Request* request           = nullptr;
};

class FlashNextCore::State {
public:
    State(const EngineOptions& options, DeviceContext& device_context)
        : device(device_context), capacity(options.max_context),
          max_outstanding(1U + static_cast<std::size_t>(options.max_pending_requests)),
          pending_timeout(std::chrono::milliseconds(options.pending_timeout_ms)),
          reuse_enabled(options.context_cache.enabled),
          snapshot_capacity(options.context_cache.enabled ? options.context_cache.host_state_slots
                                                          : 0U),
          park_budget(options.context_cache.enabled && options.context_cache.host_state_slots != 0
                          ? std::size_t(options.flashnext.park_mib) << 20
                          : 0),
          routing_stats(options.flashnext.routing_stats.string()),
          draft_tokens(resolve_draft_tokens(options.flashnext)) {
        validate_options(options);
        if (capacity == 0) { throw std::invalid_argument("Engine max_context must be nonzero"); }
        const auto load_started = Clock::now();

        StartupPhaseScope inspect(options.startup_observer, StartupPhase::ArtifactInspect);
        gguf =
            std::make_unique<fn::GgufModel>(fn::gguf_shard_paths(options.artifact_path.string()));
        const std::string architecture = gguf->get_string("general.architecture");
        if (architecture != kArchitecture) {
            throw std::invalid_argument("GGUF architecture '" + architecture +
                                        "' is not served by NInfer (Flash-Next is qwen4exp)");
        }
        inspect.complete();

        if (!options.flashnext.vision_path.empty()) {
            // before the engine sizes its expert cache: the encoder keeps its weights in pinned RAM and only a cuBLAS
            // handle on the device
            StartupPhaseScope vision_phase(options.startup_observer, StartupPhase::ProgramInitialize);
            vision = std::make_unique<fn::VisionEncoder>(options.flashnext.vision_path.string());
            vision_phase.complete();
        }
        StartupPhaseScope frontend_phase(options.startup_observer,
                                         StartupPhase::FrontendInitialize);
        files                 = flashnext_frontend_files(*gguf, vision != nullptr);
        const auto& embedding = gguf->tensor("token_embd.weight");
        if (embedding.shape.size() < 2 || embedding.shape[1] <= 0 ||
            embedding.shape[1] > std::numeric_limits<std::uint32_t>::max()) {
            throw std::invalid_argument("Flash-Next GGUF has an invalid token embedding");
        }
        const models::qwen3_5::FrontendResources resources =
            flashnext_frontend_resources(files, static_cast<std::uint32_t>(embedding.shape[1]));
        public_tokens = resources.public_token_count;
        // with images: up to 32,768 image tokens per prompt (a conversation resends its images every turn), each
        // image at most 4,096 (kFlashNextImageMaxPixels)
        frontend.emplace(models::qwen3_5::make_frontend(
            resources, {.chat_template_path       = options.chat_template_path,
                        .architecture             = models::Architecture::Qwen3_5Moe,
                        .vision_enabled           = vision != nullptr,
                        .max_context              = capacity,
                        .media_cache_bytes        = options.media_cache_bytes,
                        .media_live_bytes         = options.media_live_bytes,
                        .media_preprocess_threads = options.media_preprocess_threads,
                        .vision_max_tokens        = static_cast<std::uint32_t>(models::qwen3_5::kMaximumPromptVisionTokens)}));
        if (vision) {
            const std::vector<int> pad = resources.tokenizer->encode("<|image_pad|>");
            if (pad.size() != 1) { throw std::invalid_argument("Flash-Next tokenizer has no <|image_pad|> token"); }
            image_pad = pad[0];
        }
        frontend_phase.complete();

        StartupPhaseScope program(options.startup_observer, StartupPhase::ProgramInitialize);
        fn::EngineOptions engine_options;
        engine_options.max_ctx            = capacity;
        engine_options.cuda_graphs        = options.use_cuda_graph;
        engine_options.expert_cache_mib   = options.flashnext.expert_cache_mib;
        engine_options.routing_stats      = routing_stats;
        engine_options.host_expert_images = options.flashnext.host_expert_images;
        engine_options.mtp_path           = options.flashnext.mtp_path.string();
        if (options.flashnext.expert_threads != 0) {
            engine_options.cpu_threads = static_cast<int>(options.flashnext.expert_threads);
        }
        const std::size_t free_before = free_device_bytes();
        const auto engine_started     = Clock::now();
        engine = std::make_unique<fn::Engine>(*gguf, engine_options);
        // The VRAM the engine took: dense weights, attention caches and the expert cache.
        load.host_to_device_bytes = free_before - std::min(free_before, free_device_bytes());
        load.upload_seconds       = seconds_between(engine_started, Clock::now());
        if (static_cast<std::uint32_t>(engine->n_vocab()) < public_tokens) {
            throw std::invalid_argument("Flash-Next logits do not cover the tokenizer domain");
        }
        if (draft_tokens != 0 && !engine->has_mtp()) {
            throw std::invalid_argument("the Flash-Next engine did not load the MTP head");
        }
        if (vision && engine->image_token_id() >= 0 && engine->image_token_id() != image_pad) {
            throw std::invalid_argument("the model's PLE image token is not the tokenizer's <|image_pad|>");
        }
        if (vision && vision->config().out_dim != 2560) {
            throw std::invalid_argument("the vision encoder's projection does not match the model's width");
        }
        std::fprintf(stderr, "Flash-Next: expert cache %.2f GiB (%lld experts)%s\n", engine->stats().cache_gib,
                     static_cast<long long>(engine->stats().cached_experts),
                     vision ? "; vision encoder loaded (weights in pinned RAM)" : "");
        program.complete();

        load.architecture = std::string(kArchitecture);
        load.model_name   = gguf->has("general.name") ? gguf->get_string("general.name")
                                                      : options.artifact_path.stem().string();
        std::set<std::string> formats;
        for (const auto& [name, tensor] : gguf->tensors()) {
            formats.emplace(fn::type_name(tensor.type));
            load.artifact_bytes_read += tensor.bytes;
        }
        load.weight_formats.assign(formats.begin(), formats.end());
        load.load_seconds = seconds_between(load_started, Clock::now());
        worker            = std::thread([this] { worker_loop(); });
    }

    ~State() {
        {
            std::lock_guard lock(queue_mutex);
            stopping = true;
        }
        stop_requested.store(true, std::memory_order_release);
        queue_cv.notify_all();
        if (worker.joinable()) { worker.join(); }
        try {
            device.bind_to_current_thread();
            save_routing_stats();
        } catch (...) {}
        engine.reset();
    }

    State(const State&)            = delete;
    State& operator=(const State&) = delete;

    // ---------------------------------------------------------------------------------------
    // Submission and consumption (caller threads)

    Submission submit(models::qwen3_5::PreparedPrompt prompt, PromptSummary prompt_summary,
                      double prepare_seconds, ResolvedRequestOptions options,
                      OutputConsumerMode consumer_mode, GenerationObservationOptions observation,
                      Clock::time_point pending_deadline) {
        const Clock::time_point submitted = Clock::now();
        if (pending_deadline == Clock::time_point{}) {
            pending_deadline = submitted + pending_timeout;
        }
        if (submitted >= pending_deadline) {
            throw RequestError(RequestErrorKind::QueueTimeout,
                               "inference request expired before submission");
        }
        std::uint64_t request_id = 0;
        {
            std::lock_guard lock(queue_mutex);
            if (stopping || failed) {
                throw RequestError(RequestErrorKind::Unavailable,
                                   "inference engine is unavailable");
            }
            if (outstanding >= max_outstanding) {
                throw RequestError(RequestErrorKind::Overloaded, "inference request queue is full");
            }
            ++outstanding;
            request_id = next_request_id++;
        }

        std::shared_ptr<Request> request;
        try {
            const auto& view = models::qwen3_5::PreparedPromptAccess::view(prompt);
            if (view.has_media() && !vision) {
                throw std::invalid_argument("this Flash-Next server takes text only: images need the vision "
                                            "encoder (--flashnext-vision <mmproj.gguf>)");
            }
            for (const auto& item : view.vision_items) {
                if (item.modality != models::qwen3_5::PromptModality::Image) {
                    throw std::invalid_argument("the Flash-Next backend takes images, not video");
                }
            }
            auto output = frontend->make_output_session(prompt, options.stop, options.output,
                                                        options.execution.thinking);
            const std::uint32_t capacity_output =
                capacity - prompt_summary.prompt_tokens + static_cast<std::uint32_t>(1);
            try {
                output.validate_generation_capacity(
                    std::min(options.execution.requested_output_tokens, capacity_output));
            } catch (const std::invalid_argument& error) {
                throw RequestError(RequestErrorKind::ThinkingBudgetCapacityInsufficient,
                                   error.what());
            }
            request = std::make_shared<Request>(
                request_id, std::move(prompt), std::move(output), prompt_summary, prepare_seconds,
                std::move(options), consumer_mode, observation, pending_deadline, submitted);
        } catch (...) {
            release_reserved_capacity();
            throw;
        }
        {
            std::lock_guard lock(queue_mutex);
            if (stopping || failed) {
                --outstanding;
                throw RequestError(RequestErrorKind::Unavailable,
                                   "inference engine is unavailable");
            }
            pending.push_back(request);
        }
        queue_cv.notify_one();
        publish_queue_length();
        return Submission(*this, std::move(request));
    }

    GenerationResult wait_for_request(const std::shared_ptr<Request>& request, OutputSink* sink,
                                      const CancellationView& cancellation) {
        struct ConsumerGuard {
            State* owner;
            const std::shared_ptr<Request>& request;

            ~ConsumerGuard() { owner->release_consumer(*request); }
        } guard{this, request};

        std::exception_ptr caller_error;
        std::optional<GenerationStart> start;
        std::optional<PromptProgress> progress;
        std::vector<Request::StreamEvent> events;
        for (;;) {
            start.reset();
            progress.reset();
            events.clear();
            bool done = false;
            {
                std::unique_lock lock(request->mutex);
                request->cv.wait_for(lock, std::chrono::milliseconds(10), [&] {
                    return request->response_done || request->stream_start.has_value() ||
                           request->stream_progress.has_value() || !request->events.empty();
                });
                start = std::move(request->stream_start);
                request->stream_start.reset();
                progress = std::move(request->stream_progress);
                request->stream_progress.reset();
                events.swap(request->events);
                done = request->response_done;
            }
            if (caller_error == nullptr && sink != nullptr) {
                try {
                    if (start) { sink->start(std::move(*start)); }
                    if (progress) { sink->progress(std::move(*progress)); }
                    for (auto& event : events) {
                        if (auto* timing = std::get_if<GenerationTimingObservation>(&event)) {
                            sink->timing(std::move(*timing));
                        } else {
                            sink->publish(std::move(std::get<OutputDelta>(event)));
                        }
                    }
                } catch (...) {
                    caller_error = std::current_exception();
                    request->cancelled.store(true, std::memory_order_release);
                }
            }
            if (caller_error == nullptr) {
                try {
                    if (cancellation.requested()) {
                        request->cancelled.store(true, std::memory_order_release);
                    }
                } catch (...) {
                    caller_error = std::current_exception();
                    request->cancelled.store(true, std::memory_order_release);
                }
            }
            if (!done) { continue; }
            if (caller_error != nullptr) { std::rethrow_exception(caller_error); }
            std::lock_guard lock(request->mutex);
            if (request->error != nullptr) { std::rethrow_exception(request->error); }
            return std::move(request->result);
        }
    }

    void abandon(Request& request) noexcept {
        request.cancelled.store(true, std::memory_order_release);
        release_consumer(request);
    }

    // ---------------------------------------------------------------------------------------
    // Introspection (any thread)

    [[nodiscard]] bool available() const noexcept {
        std::lock_guard lock(queue_mutex);
        return !stopping && !failed;
    }

    [[nodiscard]] RuntimeStats runtime_stats() const {
        std::lock_guard lock(stats_mutex);
        return stats;
    }

    [[nodiscard]] MemorySummary memory_summary() const {
        MemorySummary summary;
        summary.device                    = device.device;
        summary.max_context               = capacity;
        summary.kv_capacity_mode          = KvCapacityMode::Explicit;
        summary.kv_capacity               = capacity;
        summary.kv_cache                  = KvCacheStorage::BFloat16;
        summary.host_state_capacity_slots = snapshot_capacity;
        std::lock_guard lock(stats_mutex);
        summary.host_state_occupied_slots = published_snapshots;
        summary.gdn_state_bytes           = published_state_bytes;
        return summary;
    }

    [[nodiscard]] std::vector<SlotState> slot_states() const {
        std::lock_guard lock(stats_mutex);
        return {published_slot};
    }

    const models::qwen3_5::Frontend& frontend_ref() const noexcept { return *frontend; }

    DeviceContext& device;
    const std::uint32_t capacity;
    const std::size_t max_outstanding;
    const std::chrono::milliseconds pending_timeout;
    const bool reuse_enabled;
    const std::uint32_t snapshot_capacity;
    const std::size_t park_budget;  // host RAM for parked conversations (bytes); 0: none are kept
    const std::string routing_stats;
    // Tokens the MTP head drafts per decode step; zero without speculative decoding.
    const std::uint32_t draft_tokens;

    std::unique_ptr<fn::GgufModel> gguf;
    std::unique_ptr<fn::VisionEncoder> vision;
    int image_pad = -1;
    FlashNextFrontendFiles files;
    std::optional<models::qwen3_5::Frontend> frontend;
    std::uint32_t public_tokens = 0;
    LoadSummary load;
    std::unique_ptr<fn::Engine> engine;

private:
    // ---------------------------------------------------------------------------------------
    // Request completion

    void release_reserved_capacity() noexcept {
        {
            std::lock_guard lock(queue_mutex);
            if (outstanding != 0) { --outstanding; }
        }
        queue_cv.notify_all();
    }

    void release_consumer(Request& request) noexcept {
        bool release = false;
        {
            std::lock_guard lock(request.mutex);
            request.consumer_released = true;
            if (request.response_done && !request.capacity_released) {
                request.capacity_released = true;
                release                   = true;
            }
        }
        if (release) { release_reserved_capacity(); }
    }

    // Publishes a terminal response; capacity returns once both the response and the consumer
    // are done with the request.
    void finish(Request& request, GenerationResult* result, std::exception_ptr error) noexcept {
        bool release = false;
        {
            std::lock_guard lock(request.mutex);
            if (request.response_done) { return; }
            if (result != nullptr) {
                request.result = std::move(*result);
            } else {
                request.error = std::move(error);
            }
            request.response_done = true;
            if (request.consumer_released && !request.capacity_released) {
                request.capacity_released = true;
                release                   = true;
            }
        }
        request.cv.notify_one();
        if (release) { release_reserved_capacity(); }
    }

    void complete_error(Request& request, std::exception_ptr error) noexcept {
        finish(request, nullptr, std::move(error));
    }

    void complete_success(Request& request, FinishReason reason) {
        const Clock::time_point now = Clock::now();
        GenerationResult result;
        result.prompt                  = request.prompt_summary;
        result.generated_token_ids     = std::move(request.generated);
        result.content                 = std::move(request.content);
        result.reasoning               = std::move(request.reasoning);
        result.tool_calls              = request.output.take_tool_calls();
        result.tool_call_parse         = request.output.tool_call_parse_diagnostics();
        result.reasoning_tokens        = request.output.reasoning_tokens();
        result.finish_reason           = reason;
        result.matched_stop_string     = request.output.matched_stop_string();
        result.reused_prompt_tokens    = request.reused_prompt_tokens;
        result.prefix_reuse_path       = request.prefix_reuse_path;
        result.thinking                = request.output.thinking_stats();
        result.speculative             = std::move(request.speculative);
        result.timings.prepare_seconds = request.prepare_seconds;
        result.timings.prefill_seconds = request.prefill_seconds;
        result.timings.decode_seconds  = request.decode_seconds;
        result.timings.vision_seconds  = request.vision_seconds;
        if (request.first_token) {
            result.timings.first_token_seconds =
                request.prepare_seconds + seconds_between(request.submitted, *request.first_token);
            if (request.observation.phase_timings && request.admitted_at && request.last_token) {
                result.timings.prompt_wall_seconds =
                    seconds_between(*request.admitted_at, *request.first_token);
                result.timings.generation_wall_seconds =
                    seconds_between(*request.first_token, *request.last_token);
            }
        }
        result.timings.total_seconds =
            request.prepare_seconds + seconds_between(request.submitted, now);
        result.engine_timing.queue_wait_seconds =
            seconds_between(request.submitted, request.admitted_at.value_or(now));
        request.prompt = {};
        finish(request, &result, nullptr);
    }

    void complete_cancelled(Request& request) {
        (void)request.output.preview_terminal(FinishReason::Cancelled);
        append_output(request, request.output.commit_preview());
        complete_success(request, FinishReason::Cancelled);
    }

    // ---------------------------------------------------------------------------------------
    // Publication to the consumer

    void publish_generation_start(Request& request) {
        if (request.consumer_mode != OutputConsumerMode::Streaming) { return; }
        {
            std::lock_guard lock(request.mutex);
            request.stream_start = GenerationStart{
                .prompt               = request.prompt_summary,
                .reused_prompt_tokens = request.reused_prompt_tokens,
            };
        }
        request.cv.notify_one();
    }

    void publish_prompt_progress(Request& request) {
        if (!request.observation.prompt_progress || !request.admitted_at) { return; }
        const PromptProgress progress{
            .total_prompt_tokens  = request.prompt_summary.prompt_tokens,
            .reused_prompt_tokens = request.reused_prompt_tokens,
            .processed_prompt_tokens =
                request.reused_prompt_tokens + request.computed_prompt_tokens,
            .elapsed_ns = elapsed_ns(*request.admitted_at, Clock::now()),
        };
        {
            std::lock_guard lock(request.mutex);
            request.stream_progress = progress;
        }
        request.cv.notify_one();
    }

    std::optional<GenerationTimingObservation> record_committed_output(Request& request) {
        const Clock::time_point now = Clock::now();
        if (!request.first_token) { request.first_token = now; }
        request.last_token = now;
        if (!request.observation.live_timings || !request.admitted_at) { return std::nullopt; }
        return GenerationTimingObservation{
            .generated_tokens      = static_cast<std::uint32_t>(request.generated.size()),
            .prompt_elapsed_ns     = elapsed_ns(*request.admitted_at, *request.first_token),
            .generation_elapsed_ns = elapsed_ns(*request.first_token, now),
        };
    }

    void append_output(Request& request, models::qwen3_5::PublishedOutput output,
                       std::optional<GenerationTimingObservation> timing = std::nullopt) {
        if (output.empty() && !timing) { return; }
        const bool streaming = request.consumer_mode == OutputConsumerMode::Streaming;
        {
            std::lock_guard lock(request.mutex);
            if (streaming && timing) { request.events.emplace_back(std::move(*timing)); }
            for (OutputDelta& delta : output) {
                std::string& full =
                    delta.channel == OutputChannel::Reasoning ? request.reasoning : request.content;
                full += delta.text;
                if (streaming) { request.events.emplace_back(std::move(delta)); }
            }
        }
        if (streaming) { request.cv.notify_one(); }
    }

    // ---------------------------------------------------------------------------------------
    // Statistics

    void publish_queue_length() {
        std::size_t waiting = 0;
        {
            std::lock_guard lock(queue_mutex);
            waiting = pending.size();
        }
        std::lock_guard lock(stats_mutex);
        stats.waiting_requests = static_cast<std::uint32_t>(waiting);
    }

    // Worker thread: the sequence as /slots and the monitor report it.
    void publish_sequence(const Request* running, bool prefilling) {
        SlotState slot;
        if (running != nullptr) {
            slot.processing    = true;
            slot.prompt_tokens = running->prompt_summary.prompt_tokens;
            slot.cached_tokens = running->reused_prompt_tokens;
        } else if (live_reusable && live_tokens != 0) {
            slot.retained      = true;
            slot.prompt_tokens = live_tokens;
            slot.cached_tokens = live_tokens;
        }
        std::size_t state_bytes = 0;
        for (const Snapshot& snapshot : snapshots) {
            slot.checkpoints.push_back(SlotCheckpoint{.frontier = snapshot.depth()});
            state_bytes = snapshot.state.state.size();
        }
        std::sort(slot.checkpoints.begin(), slot.checkpoints.end(),
                  [](const SlotCheckpoint& a, const SlotCheckpoint& b) {
                      return a.frontier < b.frontier;
                  });
        std::lock_guard lock(stats_mutex);
        published_slot      = std::move(slot);
        published_snapshots = static_cast<std::uint32_t>(snapshots.size());
        if (state_bytes != 0) { published_state_bytes = state_bytes; }
        stats.running_requests      = running != nullptr ? 1U : 0U;
        stats.prefilling_requests   = running != nullptr && prefilling ? 1U : 0U;
        stats.decode_ready_requests = running != nullptr && !prefilling ? 1U : 0U;
    }

    // ---------------------------------------------------------------------------------------
    // Worker

    void worker_loop() noexcept {
        try {
            device.bind_to_current_thread();
        } catch (...) {
            fail_all(std::current_exception());
            return;
        }
        for (;;) {
            std::shared_ptr<Request> request;
            {
                std::unique_lock lock(queue_mutex);
                queue_cv.wait(lock, [&] { return stopping || !pending.empty(); });
                if (stopping) { break; }
                request = std::move(pending.front());
                pending.pop_front();
            }
            publish_queue_length();
            bool usable = true;
            try {
                execute(*request);
            } catch (...) {
                complete_error(*request, std::current_exception());
                usable = recover();
            }
            try {
                publish_sequence(nullptr, false);
                // Servers are usually stopped by terminating the process: write the statistics
                // whenever the queue drains.
                bool idle = false;
                {
                    std::lock_guard lock(queue_mutex);
                    idle = pending.empty();
                }
                if (idle) { save_routing_stats(); }
            } catch (...) {}
            if (!usable) {
                fail_all(std::make_exception_ptr(RequestError(
                    RequestErrorKind::Unavailable, "inference engine failed and was stopped")));
                return;
            }
        }
        fail_all(std::make_exception_ptr(
            RequestError(RequestErrorKind::Unavailable, "inference engine is shutting down")));
    }

    void fail_all(std::exception_ptr error) noexcept {
        std::deque<std::shared_ptr<Request>> waiting;
        {
            std::lock_guard lock(queue_mutex);
            failed = true;
            waiting.swap(pending);
        }
        for (const auto& request : waiting) { complete_error(*request, error); }
        try {
            publish_queue_length();
        } catch (...) {}
    }

    // After a failed request: drop every retained state and return the engine to an empty
    // sequence. False when the engine cannot even do that.
    bool recover() noexcept {
        try {
            snapshots.clear();
            live_reusable = false;
            live_tokens   = 0;
            kv_tokens.clear();
            kv_media.clear();
            engine->reset();
            return true;
        } catch (...) { return false; }
    }

    void save_routing_stats() {
        if (routing_stats.empty() || engine == nullptr) { return; }
        engine->save_routing_stats(routing_stats);
    }

    // Completes waiting requests whose deadline passed or whose consumer left while another
    // request runs.
    void expire_waiting() {
        const Clock::time_point now = Clock::now();
        std::vector<std::shared_ptr<Request>> expired;
        {
            std::lock_guard lock(queue_mutex);
            for (auto it = pending.begin(); it != pending.end();) {
                if (now >= (*it)->pending_deadline ||
                    (*it)->cancelled.load(std::memory_order_acquire)) {
                    expired.push_back(std::move(*it));
                    it = pending.erase(it);
                } else {
                    ++it;
                }
            }
        }
        if (expired.empty()) { return; }
        for (const auto& request : expired) {
            if (request->cancelled.load(std::memory_order_acquire)) {
                complete_cancelled(*request);
            } else {
                complete_error(*request, std::make_exception_ptr(RequestError(
                                             RequestErrorKind::QueueTimeout,
                                             "inference request expired in the queue")));
            }
        }
        publish_queue_length();
    }

    void check_running() const {
        if (stop_requested.load(std::memory_order_acquire)) {
            throw RequestError(RequestErrorKind::Unavailable, "inference engine is shutting down");
        }
    }

    // ---------------------------------------------------------------------------------------
    // Sequence state

    // Runs `tokens` at the live frontier and returns the logits of the last one, or of every one
    // ([size][n_vocab]) when all_logits is set. With a prompt's media (`media`: the prompt's keys, positions and image
    // rows; the tokens are prompt positions from the live frontier on) they go with the tokens.
    std::vector<float> forward(std::span<const TokenId> tokens, bool all_logits = false,
                               const PromptMedia& media = {}) {
        const std::size_t begin = live_tokens;
        const std::size_t end   = begin + tokens.size();
        if (end > capacity) { throw std::logic_error("Flash-Next sequence exceeds max_context"); }
        // The engine records the tokens by position before it executes them.
        if (kv_tokens.size() < end) { kv_tokens.resize(end); }
        std::copy(tokens.begin(), tokens.end(), kv_tokens.begin() + live_tokens);
        if (!kv_media.empty() || media.data != nullptr) {
            if (kv_media.size() < kv_tokens.size()) { kv_media.resize(kv_tokens.size(), 0); }
            for (std::size_t i = begin; i < end; ++i) {
                kv_media[i] = media.data != nullptr ? media_key_at(media.request->media, i) : 0;
            }
        }
        fn::ForwardInputs inputs;
        std::vector<std::int32_t> positions;
        std::vector<const float*> rows;
        if (media.data != nullptr) {
            // the frontend's rope positions ([3][prompt] axis-major) and each image position's row
            const auto& data      = *media.data;
            const std::size_t n   = data.token_ids.size();
            const std::size_t len = tokens.size();
            positions.resize(3 * len);
            for (std::size_t axis = 0; axis < 3; ++axis) {
                std::copy_n(data.positions.begin() + std::ptrdiff_t(axis * n + begin), len,
                            positions.begin() + std::ptrdiff_t(axis * len));
            }
            rows.assign(len, nullptr);
            for (std::size_t k = 0; k < data.vision_items.size(); ++k) {
                for (const auto& span : data.vision_items[k].token_spans) {
                    const std::size_t lo = std::max(span.begin, begin), hi = std::min(span.begin + span.count, end);
                    if (lo >= hi) { continue; }
                    const auto& r = media.request->image_rows[k];
                    if (!r) { throw std::logic_error("Flash-Next image rows are missing"); }
                    for (std::size_t i = lo; i < hi; ++i) {
                        rows[i - begin] = r->data() + (i - span.begin) * kImageWidth;
                    }
                }
            }
            inputs.positions  = positions.data();
            inputs.embeddings = rows.data();
        }
        const auto started = Clock::now();
        std::vector<float> logits =
            media.data != nullptr
                ? engine->forward(std::vector<std::int32_t>(tokens.begin(), tokens.end()), all_logits, inputs)
                : engine->forward(std::vector<std::int32_t>(tokens.begin(), tokens.end()), all_logits);
        live_tokens = static_cast<std::uint32_t>(end);
        check_sequence();
        engine_ns += elapsed_ns(started, Clock::now());
        return logits;
    }

    void check_sequence() const {
        if (engine->n_past() != static_cast<std::int64_t>(live_tokens)) {
            throw std::logic_error("Flash-Next engine position differs from the served sequence");
        }
    }

    [[nodiscard]] bool holds(const std::vector<TokenId>& tokens, const MediaKeys& media) const noexcept {
        return tokens.size() <= kv_tokens.size() &&
               std::equal(tokens.begin(), tokens.end(), kv_tokens.begin()) &&
               media_equal(media, kv_media, tokens.size());
    }

    // Drops the snapshots whose tokens (and images) the engine no longer holds by position.
    void drop_stale_snapshots() {
        std::erase_if(snapshots, [&](const Snapshot& snapshot) {
            return !holds(snapshot.state.tokens, snapshot.media);
        });
    }

    void capture(std::uint32_t depth, PrefixReusePath path, const std::vector<float>* logits) {
        if (snapshot_capacity == 0 || depth != live_tokens) { return; }
        drop_stale_snapshots();
        for (Snapshot& existing : snapshots) {
            if (existing.depth() == depth) { // the engine holds both: same tokens
                existing.last_used = ++use_clock;
                if (path_rank(path) > path_rank(existing.path)) { existing.path = path; }
                if (logits != nullptr && existing.logits.empty()) {
                    existing.logits.assign(logits->begin(), logits->begin() + public_tokens);
                }
                return;
            }
        }
        if (snapshots.size() >= snapshot_capacity) {
            snapshots.erase(std::min_element(
                snapshots.begin(), snapshots.end(),
                [](const Snapshot& a, const Snapshot& b) { return a.last_used < b.last_used; }));
        }
        const auto started = Clock::now();
        Snapshot snapshot;
        snapshot.state     = engine->snapshot();
        if (std::any_of(kv_media.begin(), kv_media.begin() + std::min<std::size_t>(depth, kv_media.size()),
                        [](std::uint64_t k) { return k != 0; })) {
            snapshot.media.assign(kv_media.begin(), kv_media.begin() + depth);
        }
        snapshot.path      = path;
        snapshot.last_used = ++use_clock;
        if (logits != nullptr) {
            snapshot.logits.assign(logits->begin(), logits->begin() + public_tokens);
        }
        const double seconds = seconds_between(started, Clock::now());
        {
            std::lock_guard lock(stats_mutex);
            ++stats.state_d2h_count;
            stats.state_d2h_bytes += snapshot.state.state.size();
            stats.state_d2h_seconds += seconds;
        }
        snapshots.push_back(std::move(snapshot));
    }

    struct BasePlan {
        std::uint32_t depth = 0;
        std::optional<std::size_t> snapshot;
        PrefixReusePath path = PrefixReusePath::Root;
    };

    // The deepest held state whose tokens begin the prompt: the live sequence when the prompt
    // extends it, else a snapshot, else an empty sequence. A state as long as the prompt needs
    // its next-token logits.
    BasePlan choose_base(const std::vector<TokenId>& prompt, const MediaKeys& media, bool reuse) {
        BasePlan plan;
        if (!reuse) { return plan; }
        drop_stale_snapshots();
        const std::size_t n = prompt.size();
        if (live_reusable && live_tokens != 0 && live_tokens < n &&
            std::equal(kv_tokens.begin(), kv_tokens.begin() + live_tokens, prompt.begin()) &&
            media_equal(kv_media, media, live_tokens)) {
            plan = BasePlan{.depth = live_tokens, .path = PrefixReusePath::PrivateEndpoint};
        }
        for (std::size_t index = 0; index < snapshots.size(); ++index) {
            const Snapshot& snapshot  = snapshots[index];
            const std::uint32_t depth = snapshot.depth();
            if (depth <= plan.depth || depth > n || (depth == n && snapshot.logits.empty())) {
                continue;
            }
            if (!std::equal(snapshot.state.tokens.begin(), snapshot.state.tokens.end(),
                            prompt.begin()) ||
                !media_equal(snapshot.media, media, depth)) {
                continue;
            }
            plan = BasePlan{.depth = depth, .snapshot = index, .path = snapshot.path};
        }
        return plan;
    }

    // ---------------------------------------------------------------------------------------
    // Parked conversations
    //
    // The engine holds one sequence by position. A request that does not continue it (a client's side task such as
    // a compaction summary, or another session) overwrites it from the depth it shares, and the conversation's next
    // turn then reads everything again: minutes at a few hundred thousand tokens. Instead the held conversation is
    // parked first (its positions and deepest snapshots copied to host RAM, about a second) and copied back when a
    // later request continues it.

    static constexpr std::uint32_t kParkMinTokens   = 16384; // shorter conversations are read again quickly
    static constexpr std::uint32_t kUnparkMinGain   = 4096;  // resume this much deeper than what is held, or not at all
    static constexpr std::size_t kParkedSnapshots   = 3;     // the deepest snapshots kept with a parked conversation

    // The deepest snapshot of `pool` that begins the prompt (0: none).
    static std::uint32_t resumable_depth(const std::vector<Snapshot>& pool, const std::vector<TokenId>& prompt,
                                         const MediaKeys& media) {
        std::uint32_t best = 0;
        for (const Snapshot& snapshot : pool) {
            const std::uint32_t depth = snapshot.depth();
            if (depth <= best || depth > prompt.size() || (depth == prompt.size() && snapshot.logits.empty())) {
                continue;
            }
            if (std::equal(snapshot.state.tokens.begin(), snapshot.state.tokens.end(), prompt.begin()) &&
                media_equal(snapshot.media, media, depth)) {
                best = depth;
            }
        }
        return best;
    }

    // Parks the held conversation when the next request keeps only its first `keep` positions and discards most of
    // it (a request that keeps most of it continues the conversation: its tail is the last answer, replaced). A
    // sequence still live past its last snapshot is snapshotted first, so it resumes where it ended. `protect` (a
    // last_used value) names a parked conversation that must not be dropped to make room. True when parked.
    bool park_held(std::uint32_t keep, std::optional<std::uint64_t> protect = std::nullopt) {
        if (park_budget == 0) { return false; }
        drop_stale_snapshots();
        std::uint32_t end = live_reusable ? live_tokens : 0;
        for (const Snapshot& snapshot : snapshots) { end = std::max(end, snapshot.depth()); }
        if (end < kParkMinTokens || keep > end / 2 || end - keep < kParkMinTokens) { return false; }
        // the snapshots the request keeps stay in use: no capture below may evict one of them
        for (Snapshot& snapshot : snapshots) {
            if (snapshot.depth() <= keep) { snapshot.last_used = ++use_clock; }
        }
        if (live_reusable && live_tokens == end) { capture(end, PrefixReusePath::PrivateEndpoint, nullptr); }
        std::vector<std::size_t> deep;
        for (std::size_t i = 0; i < snapshots.size(); ++i) {
            if (snapshots[i].depth() > keep) { deep.push_back(i); }
        }
        std::sort(deep.begin(), deep.end(),
                  [&](std::size_t a, std::size_t b) { return snapshots[a].depth() > snapshots[b].depth(); });
        if (deep.empty() || snapshots[deep.front()].depth() != end) { return false; }
        if (deep.size() > kParkedSnapshots) { deep.resize(kParkedSnapshots); }

        // room first: the estimate is the positions plus the snapshots, as Parked::bytes counts them
        std::size_t bytes = engine->park_bytes(end);
        for (const std::size_t i : deep) {
            const Snapshot& snapshot = snapshots[i];
            bytes += snapshot.state.state.size() + snapshot.state.tokens.size() * sizeof(std::int32_t) +
                     snapshot.logits.size() * sizeof(float) + snapshot.media.size() * sizeof(std::uint64_t);
        }
        if (bytes > park_budget) {
            std::fprintf(stderr, "Flash-Next: a %u-token conversation (%.1f GiB) does not fit --flashnext-park-mib; not parked\n",
                         end, double(bytes) / double(1 << 30));
            return false;
        }
        while (parked_bytes + bytes > park_budget) {
            auto oldest = parked.end();
            for (auto it = parked.begin(); it != parked.end(); ++it) {
                if (protect && it->last_used == *protect) { continue; }
                if (oldest == parked.end() || it->last_used < oldest->last_used) { oldest = it; }
            }
            if (oldest == parked.end()) {
                std::fprintf(stderr, "Flash-Next: no room to park a %u-token conversation; not parked\n", end);
                return false;
            }
            std::fprintf(stderr, "Flash-Next: dropped a parked %lld-token conversation for room\n",
                         static_cast<long long>(oldest->positions.end));
            parked_bytes -= oldest->bytes();
            parked.erase(oldest);
        }

        const auto started = Clock::now();
        Parked entry;
        try {
            entry.positions = engine->park(end);
        } catch (const std::bad_alloc&) {  // parking only reads the engine: without the RAM the request runs unparked
            std::fprintf(stderr, "Flash-Next: no host RAM to park a %u-token conversation; not parked\n", end);
            return false;
        }
        if (!kv_media.empty()) {
            entry.media.assign(kv_media.begin(), kv_media.begin() + std::min<std::size_t>(end, kv_media.size()));
        }
        std::sort(deep.begin(), deep.end(), std::greater<>());  // erase from the back
        for (const std::size_t i : deep) {
            entry.snapshots.push_back(std::move(snapshots[i]));
            snapshots.erase(snapshots.begin() + std::ptrdiff_t(i));
        }
        entry.last_used = ++use_clock;
        const std::size_t actual = entry.bytes();
        parked_bytes += actual;
        parked.push_back(std::move(entry));
        std::fprintf(stderr, "Flash-Next: parked a %u-token conversation (%.1f GiB, %.2f s; %zu parked, %.1f GiB)\n", end,
                     double(actual) / double(1 << 30), seconds_between(started, Clock::now()), parked.size(),
                     double(parked_bytes) / double(1 << 30));
        return true;
    }

    // Before a request runs from `base`: when a parked conversation lets it resume at least kUnparkMinGain deeper,
    // park what is held and copy that conversation back; otherwise park the held conversation if the request is about
    // to overwrite it. Returns the plan to run from (choose_base again whenever the snapshots changed).
    BasePlan switch_conversation(const std::vector<TokenId>& prompt, const MediaKeys& media, const BasePlan& base) {
        if (park_budget == 0) { return base; }
        std::optional<std::size_t> best;
        std::uint32_t best_depth = 0;
        for (std::size_t i = 0; i < parked.size(); ++i) {
            const std::uint32_t depth = resumable_depth(parked[i].snapshots, prompt, media);
            if (depth > best_depth) {
                best       = i;
                best_depth = depth;
            }
        }
        if (!best || best_depth < base.depth + kUnparkMinGain) {
            const bool continues_live = live_reusable && base.depth == live_tokens;
            if (!continues_live && park_held(base.depth)) { return choose_base(prompt, media, true); }
            return base;
        }
        const std::uint64_t chosen = parked[*best].last_used = ++use_clock;
        park_held(0, chosen);
        const auto it = std::find_if(parked.begin(), parked.end(), [&](const Parked& p) { return p.last_used == chosen; });
        if (it == parked.end()) { return choose_base(prompt, media, true); }  // protected above; not expected
        const std::size_t bytes = it->bytes();
        const long long end     = static_cast<long long>(it->positions.end);
        const auto started      = Clock::now();
        engine->unpark(it->positions);
        kv_tokens.assign(it->positions.tokens.begin(), it->positions.tokens.end());
        kv_media      = std::move(it->media);
        live_tokens   = 0;
        live_reusable = false;
        check_sequence();
        snapshots.clear();  // they named positions the unparked conversation now holds
        for (Snapshot& snapshot : it->snapshots) {
            snapshot.last_used = ++use_clock;
            snapshots.push_back(std::move(snapshot));
        }
        parked_bytes -= bytes;
        parked.erase(it);
        std::fprintf(stderr, "Flash-Next: unparked a %lld-token conversation in %.2f s; resuming at %u tokens\n", end,
                     seconds_between(started, Clock::now()), best_depth);
        return choose_base(prompt, media, true);
    }

    // ---------------------------------------------------------------------------------------
    // Request execution

    void execute(Request& request) {
        const Clock::time_point admitted = Clock::now();
        request.admitted_at              = admitted;
        engine_ns                        = 0;
        if (request.cancelled.load(std::memory_order_acquire)) {
            complete_cancelled(request);
            return;
        }
        if (admitted >= request.pending_deadline) {
            complete_error(request, std::make_exception_ptr(
                                        RequestError(RequestErrorKind::QueueTimeout,
                                                     "inference request expired in the queue")));
            return;
        }
        const models::qwen3_5::PreparedPromptData& data =
            models::qwen3_5::PreparedPromptAccess::view(request.prompt);
        const std::vector<TokenId>& prompt = data.token_ids;
        const std::size_t n                = prompt.size();
        if (n == 0 || n > capacity) {
            throw RequestError(RequestErrorKind::ContextLengthExceeded,
                               "prepared prompt does not fit the Engine max_context");
        }
        const std::uint32_t requested = request.options.execution.requested_output_tokens;
        const std::uint32_t capacity_output =
            capacity - static_cast<std::uint32_t>(n) + static_cast<std::uint32_t>(1);
        GenerationBudget budget(std::min(requested, capacity_output),
                                requested <= capacity_output ? FinishReason::OutputLimit
                                                             : FinishReason::ContextCapacity);
        request.generated.reserve(budget.remaining());

        // Reuse needs the request's permission, an identity the frontend can reproduce, and the
        // Engine's context cache. Without it the request runs from an empty sequence and leaves
        // nothing reusable behind.
        const bool reuse =
            reuse_enabled && request.options.execution.allow_prefix_reuse && data.identity.reusable;
        request.media                = prompt_media_keys(data);
        BasePlan base = choose_base(prompt, request.media, reuse);
        if (reuse) { base = switch_conversation(prompt, request.media, base); }
        request.reused_prompt_tokens = base.depth;
        request.prefix_reuse_path    = base.path;
        publish_sequence(&request, true);
        publish_generation_start(request);

        const Clock::time_point prefill_started  = Clock::now();
        encode_images(request, data, base.depth);
        std::optional<std::vector<float>> logits = prefill(request, data, base, reuse);
        // The sequence the engine holds is now exact for its tokens, whatever happens next.
        live_reusable = reuse;
        if (!logits) {
            complete_cancelled(request);
            return;
        }
        const Clock::time_point prefilled = Clock::now();
        request.prefill_seconds           = seconds_between(prefill_started, prefilled);
        {
            std::lock_guard lock(stats_mutex);
            account_unit(elapsed_ns(prefill_started, prefilled), false);
            stats.computed_prefill_tokens += request.computed_prompt_tokens;
            stats.prefill_seconds_total += request.prefill_seconds;
            stats.reused_prompt_tokens += base.depth;
            stats.last_selected_frontier_tokens = base.depth;
            switch (base.path) {
            case PrefixReusePath::Root:
                ++stats.root_selections;
                break;
            case PrefixReusePath::PrivateEndpoint:
                ++stats.private_endpoint_selections;
                break;
            case PrefixReusePath::PrivateTurnClosure:
                ++stats.private_turn_closure_selections;
                break;
            case PrefixReusePath::PrivateResponseReplay:
                ++stats.private_response_replay_selections;
                break;
            case PrefixReusePath::PrivateLongAnchor:
                ++stats.private_long_anchor_selections;
                break;
            case PrefixReusePath::SharedStablePrefix:
                ++stats.shared_stable_prefix_selections;
                break;
            }
        }
        publish_sequence(&request, false);

        decode(request, budget, std::move(*logits), static_cast<std::uint32_t>(n));
    }

    // The rows of every image whose positions the prompt still computes (those after `depth`): from the cache of
    // encoded images (by content and grid), else encoded now in VRAM the expert cache lends until the prompt runs.
    void encode_images(Request& request, const models::qwen3_5::PreparedPromptData& data, std::uint32_t depth) {
        request.image_rows.assign(data.vision_items.size(), nullptr);
        if (!data.has_media()) { return; }
        const auto started = Clock::now();
        std::vector<std::size_t> todo;
        std::vector<std::pair<std::size_t, std::size_t>> same;  // (item, earlier item of the same image)
        std::size_t max_patches = 0;
        for (std::size_t k = 0; k < data.vision_items.size(); ++k) {
            const auto& item = data.vision_items[k];
            const bool needed = std::any_of(item.token_spans.begin(), item.token_spans.end(), [&](const auto& span) {
                return span.begin + span.count > depth;
            });
            if (!needed) { continue; }
            const std::uint64_t key = image_key(item);
            const auto hit = std::find_if(image_cache.begin(), image_cache.end(), [&](const auto& e) { return e.first == key; });
            if (hit != image_cache.end()) {
                image_cache.splice(image_cache.end(), image_cache, hit);  // most recently used last
                request.image_rows[k] = hit->second;
                continue;
            }
            const auto twin = std::find_if(todo.begin(), todo.end(), [&](std::size_t j) { return image_key(data.vision_items[j]) == key; });
            if (twin != todo.end()) {
                same.emplace_back(k, *twin);
                continue;
            }
            todo.push_back(k);
            max_patches = std::max(max_patches, std::size_t(item.grid.height) * std::size_t(item.grid.width));
        }
        if (!todo.empty()) {
            void* workspace = engine->lend_vram(vision->workspace_bytes(int(max_patches)));
            std::vector<float> patches;
            for (const std::size_t k : todo) {
                const auto& item    = data.vision_items[k];
                const auto& payload = data.media_payloads.at(k);
                const int gh = item.grid.height, gw = item.grid.width;
                const std::size_t count = std::size_t(gh) * std::size_t(gw);
                if (!payload || item.grid.temporal != 1 || payload->patch_elements != count * std::size_t(vision->patch_values())) {
                    throw std::logic_error("Flash-Next image patches do not match the image's grid");
                }
                patches.resize(payload->patch_elements);
                patch_pixels.convert(payload->span(), patches.data());
                auto rows = std::make_shared<std::vector<float>>(count / 4 * kImageWidth);
                vision->encode(patches.data(), gh, gw, workspace, rows->data());
                request.image_rows[k] = rows;
                image_cache.emplace_back(image_key(item), rows);
                image_cache_bytes += rows->size() * sizeof(float);
                ++images_encoded;
            }
            while (image_cache_bytes > kImageCacheBytes && image_cache.size() > 1) {
                image_cache_bytes -= image_cache.front().second->size() * sizeof(float);
                image_cache.pop_front();
            }
        }
        for (const auto& [k, j] : same) { request.image_rows[k] = request.image_rows[j]; }
        request.vision_seconds = seconds_between(started, Clock::now());
    }

    // Establishes the base state and executes the rest of the prompt, capturing snapshots at the
    // frontend's frontiers. Empty when the request was cancelled first.
    std::optional<std::vector<float>> prefill(Request& request,
                                              const models::qwen3_5::PreparedPromptData& data,
                                              const BasePlan& base, bool reuse) {
        const std::vector<TokenId>& prompt = data.token_ids;
        const auto n                       = static_cast<std::uint32_t>(prompt.size());
        // Mid-request the live sequence is not a reusable state.
        live_reusable = false;
        if (base.snapshot) {
            Snapshot& snapshot = snapshots[*base.snapshot];
            const auto started = Clock::now();
            engine->restore(snapshot.state);
            live_tokens = snapshot.depth();
            check_sequence();
            snapshot.last_used = ++use_clock;
            std::lock_guard lock(stats_mutex);
            ++stats.state_restores;
            ++stats.state_h2d_count;
            stats.state_h2d_bytes += snapshot.state.state.size();
            stats.state_h2d_seconds += seconds_between(started, Clock::now());
        } else if (base.depth == 0) {
            engine->reset();
            kv_tokens.clear();
            kv_media.clear();
            live_tokens = 0;
            snapshots.clear(); // the engine no longer holds any position
        }
        if (base.depth == n) { return snapshots[*base.snapshot].logits; }

        // Frontiers to capture, and the frontend's execution frontiers that a resumed prompt
        // also splits at.
        std::vector<std::pair<std::uint32_t, PrefixReusePath>> captures;
        if (reuse && snapshot_capacity != 0) {
            if (data.identity.rewrite_checkpoint) {
                captures.emplace_back(data.identity.rewrite_checkpoint->frontier,
                                      data.identity.rewrite_checkpoint->kind ==
                                              models::qwen3_5::RewriteCheckpointKind::TurnClosure
                                          ? PrefixReusePath::PrivateTurnClosure
                                          : PrefixReusePath::PrivateResponseReplay);
            }
            for (const auto& opportunity : data.context_cache.opportunities) {
                captures.emplace_back(opportunity.frontier,
                                      opportunity.kind == PromptCacheMarkerKind::SharedStablePrefix
                                          ? PrefixReusePath::SharedStablePrefix
                                          : PrefixReusePath::PrivateLongAnchor);
            }
            captures.emplace_back(n, PrefixReusePath::PrivateEndpoint);
        }
        std::vector<std::uint32_t> splits;
        for (const auto& [frontier, path] : captures) { splits.push_back(frontier); }
        for (const std::uint32_t frontier : data.identity.rewrite_execution_frontiers) {
            splits.push_back(frontier);
        }
        std::erase_if(splits, [&](std::uint32_t frontier) {
            return frontier <= base.depth || frontier >= n;
        });
        splits.push_back(n);
        std::sort(splits.begin(), splits.end());
        splits.erase(std::unique(splits.begin(), splits.end()), splits.end());

        std::vector<float> logits;
        std::uint32_t cursor = base.depth;
        std::size_t next     = 0;
        while (cursor < n) {
            check_running();
            if (request.cancelled.load(std::memory_order_acquire)) { return std::nullopt; }
            expire_waiting();
            const std::uint32_t end = std::min(splits[next], cursor + kPrefillPieceTokens);
            logits = forward(std::span<const TokenId>(prompt).subspan(cursor, end - cursor), false,
                             data.has_media() ? PromptMedia{&data, &request} : PromptMedia{});
            request.computed_prompt_tokens += end - cursor;
            cursor = end;
            if (cursor == splits[next]) { ++next; }
            publish_prompt_progress(request);
            PrefixReusePath path = PrefixReusePath::Root;
            for (const auto& [frontier, kind] : captures) {
                if (frontier == cursor && path_rank(kind) >= path_rank(path)) { path = kind; }
            }
            if (path != PrefixReusePath::Root) {
                capture(cursor, path, cursor == n ? &logits : nullptr);
            }
        }
        return logits;
    }

    struct Committed {
        FinishReason finish_reason = FinishReason::None;
        bool control               = false; // the thinking budget is spent: force the close next
    };

    // Commits one model token through the output policy.
    Committed commit_model_token(Request& request, GenerationBudget& budget, HostSampler& sampler,
                                 TokenId token) {
        const OutputDecision decision = request.output.preview_model(
            std::span<const TokenId>(&token, 1), budget.remaining(), budget.limit_reason());
        if (decision.accepted_tokens != 1) {
            throw std::logic_error("output policy returned an invalid licensed prefix");
        }
        request.generated.push_back(token);
        sampler.commit(token);
        budget.commit(1);
        append_output(request, request.output.commit_preview(), record_committed_output(request));
        return Committed{
            .finish_reason = decision.finish_reason,
            .control       = decision.continuation == ContinuationAction::ApplyTargetControl,
        };
    }

    // Commits the canonical thinking close without sampling and returns its tokens.
    std::vector<TokenId> commit_control(Request& request, GenerationBudget& budget,
                                        HostSampler& sampler) {
        const std::span<const TokenId> control = request.output.pending_control_tokens();
        std::vector<TokenId> forced(control.begin(), control.end());
        const OutputDecision applied = request.output.preview_control(forced, budget.remaining());
        if (applied.accepted_tokens != forced.size() || applied.finished()) {
            throw std::logic_error("target control preview returned an invalid decision");
        }
        for (const TokenId token : forced) {
            request.generated.push_back(token);
            sampler.commit(token);
        }
        budget.commit(static_cast<std::uint32_t>(forced.size()));
        append_output(request, request.output.commit_preview(), record_committed_output(request));
        return forced;
    }

    // Splits a unit's wall time into the engine's execution (device wait, which includes the CPU
    // experts) and this core's own host work (sampling, output policy, bookkeeping).
    void account_unit(std::uint64_t wall_ns, bool decoding) {
        const std::uint64_t device = std::min(engine_ns, wall_ns);
        const std::uint64_t host   = wall_ns - device;
        engine_ns                  = 0;
        RuntimeHostWorkStats& work = stats.host_work;
        work.device_wait_ns += device;
        work.engine_commit_output_ns += host;
        (decoding ? work.decode_device_wait_ns : work.prefill_device_wait_ns) += device;
        (decoding ? work.decode_host_ns : work.prefill_host_ns) += host;
        if (!decoding) { ++work.prefill_units; }
    }

    void record_decode_step(Request& request, Clock::time_point started, std::uint64_t tokens) {
        const Clock::time_point now = Clock::now();
        const double seconds        = seconds_between(started, now);
        request.decode_seconds += seconds;
        std::lock_guard lock(stats_mutex);
        account_unit(elapsed_ns(started, now), true);
        stats.committed_decode_tokens += tokens;
        stats.decode_seconds_total += seconds;
        ++stats.decode_rounds;
        ++stats.decode_row_rounds;
    }

    // Generates from the prompt's next-token logits. Every token is sampled with the RNG key of the
    // position of the token executed before it; with the MTP head, a step feeds the last committed
    // token with drafts and accepts each draft that equals the token sampled from the row before
    // it, so the output distribution is the same as without drafts.
    void decode(Request& request, GenerationBudget& budget, std::vector<float> logits,
                std::uint32_t prompt_tokens) {
        HostSampler sampler(request.options.execution.sampling, public_tokens);
        if (draft_tokens != 0) {
            request.speculative.backend       = SpeculativeBackend::Mtp;
            request.speculative.enabled       = true;
            request.speculative.draft_window  = draft_tokens;
            request.speculative.verify_window = draft_tokens + 1;
            request.speculative.accepted_per_position.assign(draft_tokens, 0);
        }
        const auto vocabulary          = static_cast<std::size_t>(engine->n_vocab());
        std::int32_t position          = static_cast<std::int32_t>(prompt_tokens);
        HostSampler::Purpose purpose   = HostSampler::Purpose::Prefill;
        Clock::time_point step_started = Clock::now();
        // A committed token the engine has not executed yet; otherwise `logits` is current.
        std::optional<TokenId> unexecuted;
        const auto execute_control = [&](TokenId token) {
            std::vector<TokenId> step{token};
            const std::vector<TokenId> forced = commit_control(request, budget, sampler);
            step.insert(step.end(), forced.begin(), forced.end());
            {
                std::lock_guard lock(stats_mutex);
                stats.committed_decode_tokens += forced.size();
            }
            step_started = Clock::now();
            logits       = forward(step);
            position     = static_cast<std::int32_t>(live_tokens) - 1;
            purpose      = HostSampler::Purpose::Decode;
        };
        for (;;) {
            check_running();
            if (request.cancelled.load(std::memory_order_acquire)) {
                complete_cancelled(request);
                return;
            }
            if (!unexecuted) {
                const TokenId token     = sampler.sample(logits, position, purpose);
                const Committed outcome = commit_model_token(request, budget, sampler, token);
                if (purpose == HostSampler::Purpose::Decode) {
                    record_decode_step(request, step_started, 1);
                }
                if (outcome.finish_reason != FinishReason::None) {
                    complete_success(request, outcome.finish_reason);
                    return;
                }
                if (outcome.control) {
                    execute_control(token);
                } else {
                    unexecuted = token;
                }
                continue;
            }

            expire_waiting();
            const TokenId token = *unexecuted;
            unexecuted.reset();
            step_started = Clock::now();
            // Never draft past the output or thinking budget: the step commits at most k+1 tokens.
            const std::uint32_t model_budget =
                request.output.model_token_budget_remaining(budget.remaining());
            std::uint32_t k =
                model_budget == 0 ? 0 : std::min(choose_drafts(), model_budget - 1);
            if (k == 0) {
                logits   = forward(std::span<const TokenId>(&token, 1));
                position = static_cast<std::int32_t>(live_tokens) - 1;
                purpose  = HostSampler::Purpose::Decode;
                if (draft_tokens != 0) observe_step(0, Clock::now() - step_started);
                continue;
            }

            const std::uint32_t base          = live_tokens;
            std::vector<std::int32_t> drafted = engine->draft(token, static_cast<int>(k));
            if (drafted.size() > k) { throw std::logic_error("MTP head drafted a wrong count"); }
            // The engine stops drafting at an invalid draft (its output went non-finite); the step then
            // verifies the drafts it has, or feeds the token alone.
            if (drafted.size() < k) {
                k = static_cast<std::uint32_t>(drafted.size());
                if (k == 0) {
                    logits   = forward(std::span<const TokenId>(&token, 1));
                    position = static_cast<std::int32_t>(live_tokens) - 1;
                    purpose  = HostSampler::Purpose::Decode;
                    observe_step(0, Clock::now() - step_started);
                    continue;
                }
            }
            std::vector<TokenId> step{token};
            step.insert(step.end(), drafted.begin(), drafted.end());
            const std::vector<float> rows = forward(step, true);
            if (rows.size() != step.size() * vocabulary) {
                throw std::logic_error("Flash-Next verification returned a wrong logit count");
            }
            std::uint32_t kept = 0;
            TokenId last       = token;
            Committed outcome;
            for (std::uint32_t row = 0; row <= k; ++row) {
                last = sampler.sample(
                    std::span<const float>(rows).subspan(row * vocabulary, vocabulary),
                    static_cast<std::int32_t>(base + row), HostSampler::Purpose::Decode);
                outcome = commit_model_token(request, budget, sampler, last);
                if (row < k) observe_draft(row, last == drafted[row]);
                if (row < k && last == drafted[row] &&
                    outcome.finish_reason == FinishReason::None && !outcome.control) {
                    ++request.speculative.accepted_per_position[row];
                    ++kept;
                    continue;
                }
                break;
            }
            observe_step(k, Clock::now() - step_started);
            // Keep the fed token and the accepted drafts; `last` is committed but not executed.
            if (kept < k) { engine->rollback(static_cast<int>(1 + kept)); }
            live_tokens = base + 1 + kept;
            check_sequence();
            ++request.speculative.rounds;
            request.speculative.drafted_tokens += k;
            request.speculative.accepted_tokens += kept;
            record_decode_step(request, step_started, kept + 1);
            if (outcome.finish_reason != FinishReason::None) {
                complete_success(request, outcome.finish_reason);
                return;
            }
            if (outcome.control) {
                execute_control(last);
            } else {
                unexecuted = last;
            }
        }
    }

    // Adaptive draft length. Drafts pay off only when they are accepted often enough to cover the
    // longer verification step (on story-like text they did not: 63% accepted, slower than plain
    // decoding; on chat 93%, 30% faster). Moving averages of each draft position's acceptance (given
    // the earlier drafts were accepted) and of each length's step time pick the length with the most
    // expected tokens per second; a longer length is tried now and then to keep its estimates fresh.
    static constexpr std::uint32_t kMaxDrafts = 3;
    static constexpr double kAlpha = 0.05;
    std::array<double, kMaxDrafts> accept_ema{0.75, 0.75, 0.75};
    std::array<double, kMaxDrafts + 1> step_ms_ema{0.0, 0.0, 0.0, 0.0};  // 0 = not measured yet
    std::uint64_t speculative_steps = 0;

    std::uint32_t choose_drafts() {
        const std::uint32_t max_k = std::min(draft_tokens, kMaxDrafts);
        if (max_k == 0) return 0;
        const double t0 = step_ms_ema[0] > 0.0 ? step_ms_ema[0] : 20.0;
        double expected = 1.0, reach = 1.0, best_rate = -1.0;
        std::uint32_t best = 0;
        for (std::uint32_t k = 0; k <= max_k; ++k) {
            if (k > 0) {
                reach *= accept_ema[k - 1];
                expected += reach;
            }
            const double t    = step_ms_ema[k] > 0.0 ? step_ms_ema[k] : t0 * (1.0 + 0.5 * k);
            const double rate = expected / t;
            if (rate > best_rate) {
                best_rate = rate;
                best      = k;
            }
        }
        if (++speculative_steps % 32 == 0 && best < max_k) ++best;
        return best;
    }

    void observe_draft(std::uint32_t position, bool accepted) {
        if (position < kMaxDrafts) accept_ema[position] += kAlpha * ((accepted ? 1.0 : 0.0) - accept_ema[position]);
    }

    void observe_step(std::uint32_t k, Clock::duration elapsed) {
        const double ms = std::chrono::duration<double, std::milli>(elapsed).count();
        double & e      = step_ms_ema[std::min(k, kMaxDrafts)];
        e               = e > 0.0 ? e + kAlpha * (ms - e) : ms;
    }

    // Worker-owned sequence state. kv_tokens mirrors the engine's positional token history (what
    // its attention caches hold); live_tokens is the engine's n_past.
    std::vector<TokenId> kv_tokens;
    MediaKeys kv_media;  // the media keys of those positions (empty while there were no images)
    // encoded images by content (image_key), least recently used first, within kImageCacheBytes
    static constexpr std::size_t kImageWidth      = 2560;
    static constexpr std::size_t kImageCacheBytes = std::size_t(512) << 20;
    std::list<std::pair<std::uint64_t, std::shared_ptr<const std::vector<float>>>> image_cache;
    std::size_t image_cache_bytes = 0;
    std::uint64_t images_encoded  = 0;
    PatchPixels patch_pixels;
    std::uint32_t live_tokens = 0;
    bool live_reusable        = false;
    // Wall time inside the engine since the last accounted unit.
    std::uint64_t engine_ns = 0;
    std::vector<Snapshot> snapshots;
    std::uint64_t use_clock = 0;
    // Conversations parked by requests that did not continue them (park_held): their positions and their deepest
    // snapshots, least recently used first, within park_budget.
    struct Parked {
        fn::EngineParked positions;
        MediaKeys media;  // the media keys of those positions (empty without images)
        std::vector<Snapshot> snapshots;
        std::uint64_t last_used = 0;
        [[nodiscard]] std::size_t bytes() const noexcept {
            std::size_t n = positions.bytes() + media.size() * sizeof(std::uint64_t);
            for (const Snapshot& snapshot : snapshots) {
                n += snapshot.state.state.size() + snapshot.state.tokens.size() * sizeof(std::int32_t) +
                     snapshot.logits.size() * sizeof(float) + snapshot.media.size() * sizeof(std::uint64_t);
            }
            return n;
        }
    };
    std::vector<Parked> parked;
    std::size_t parked_bytes = 0;

    mutable std::mutex queue_mutex;
    std::condition_variable queue_cv;
    std::deque<std::shared_ptr<Request>> pending;
    std::size_t outstanding       = 0;
    std::uint64_t next_request_id = 1;
    bool stopping                 = false;
    bool failed                   = false;
    std::atomic<bool> stop_requested{false};

    mutable std::mutex stats_mutex;
    RuntimeStats stats;
    SlotState published_slot;
    std::uint32_t published_snapshots = 0;
    std::size_t published_state_bytes = 0;

    std::thread worker;
};

// -------------------------------------------------------------------------------------------

FlashNextCore::Submission::Submission(State& owner, std::shared_ptr<Request> request) noexcept
    : owner_(&owner), request_(std::move(request)) {}

FlashNextCore::Submission::~Submission() { reset(); }

FlashNextCore::Submission::Submission(Submission&& other) noexcept
    : owner_(std::exchange(other.owner_, nullptr)), request_(std::move(other.request_)) {}

FlashNextCore::Submission& FlashNextCore::Submission::operator=(Submission&& other) noexcept {
    if (this != &other) {
        reset();
        owner_   = std::exchange(other.owner_, nullptr);
        request_ = std::move(other.request_);
    }
    return *this;
}

void FlashNextCore::Submission::reset() noexcept {
    if (owner_ != nullptr && request_ != nullptr) { owner_->abandon(*request_); }
    owner_ = nullptr;
    request_.reset();
}

GenerationResult FlashNextCore::Submission::wait(OutputSink* sink,
                                                 const CancellationView& cancellation) {
    if (owner_ == nullptr || request_ == nullptr) {
        throw std::logic_error("Flash-Next submission is empty");
    }
    const bool streaming = request_->consumer_mode == OutputConsumerMode::Streaming;
    if (streaming != (sink != nullptr)) {
        throw std::invalid_argument(
            "GenerationHandle wait sink does not match its submitted consumer mode");
    }
    State* owner                     = std::exchange(owner_, nullptr);
    std::shared_ptr<Request> request = std::move(request_);
    return owner->wait_for_request(request, sink, cancellation);
}

FlashNextCore::FlashNextCore(const EngineOptions& options, DeviceContext& device)
    : state_(std::make_unique<State>(options, device)) {}

FlashNextCore::~FlashNextCore() = default;

const models::qwen3_5::Frontend& FlashNextCore::frontend() const noexcept {
    return state_->frontend_ref();
}

std::uint32_t FlashNextCore::capacity() const noexcept { return state_->capacity; }

const LoadSummary& FlashNextCore::load_summary() const noexcept { return state_->load; }

ModelSamplingDefaults FlashNextCore::sampling_defaults() const {
    return state_->frontend_ref().sampling_defaults();
}

FlashNextCore::Submission
FlashNextCore::submit(models::qwen3_5::PreparedPrompt prompt, PromptSummary prompt_summary,
                      double prepare_seconds, ResolvedRequestOptions options,
                      OutputConsumerMode consumer_mode, GenerationObservationOptions observation,
                      Clock::time_point pending_deadline) {
    return state_->submit(std::move(prompt), prompt_summary, prepare_seconds, std::move(options),
                          consumer_mode, observation, pending_deadline);
}

MemorySummary FlashNextCore::memory_summary() const { return state_->memory_summary(); }

bool FlashNextCore::healthy() const noexcept { return state_->available(); }

bool FlashNextCore::is_available() const { return state_->available(); }

RuntimeStats FlashNextCore::runtime_stats() const { return state_->runtime_stats(); }

std::vector<SlotState> FlashNextCore::slot_states() const { return state_->slot_states(); }

} // namespace ninfer::runtime
