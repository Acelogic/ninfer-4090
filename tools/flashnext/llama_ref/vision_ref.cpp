// vision_ref: llama.cpp's (mtmd's) side of the Flash-Next vision checks (tools/flashnext/fn_vision.cpp).
//
// Adapted from Strata's tools/vision/strata_vision.cpp (github.com/Niko1221/Strata), which turns images into embeddings
// with llama.cpp's mtmd library the same way:
//   MIT License, Copyright (c) 2026 Niko1221 and the Strata contributors
//
//   vision_ref encode --mmproj mmproj.gguf --model text-model.gguf --image img.png --out img.sve [--threads N]
//       The vision encoder's rows for the image (the text model is opened vocab-only).
//   vision_ref case --mmproj mmproj.gguf --model text-model-00001-of-00003.gguf --image a.png [--image b.png ...]
//                   --prompt "<|im_start|>user\n<__media__>Describe it.<|im_end|>\n<|im_start|>assistant\n" (or --prompt-file f)
//                   --out-dir dir [--n-predict 16] [--threads N]
//       The whole model on the CPU: tokenizes the prompt (one <__media__> per image), writes each image's rows
//       (imageK.sve) and pixels (imageK.ppm, what the encoder saw), runs the prompt, and writes dir/case.txt (tokens,
//       M-RoPE positions, images), dir/logits.bin (the first next-token logits, float32 [n_vocab]) and the greedy
//       continuation. Flash attention off (FP32 attention); one ubatch for the whole prompt, so an image's PLE n-grams
//       read the tokens before it in order.
// .sve: int32 {'SVE1', n, nx, ny, n_embd} then float32 [n][n_embd].
#include "gguf.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iterator>
#include <fstream>
#include <map>
#include <string>
#include <thread>
#include <vector>

namespace {

void quiet_log(ggml_log_level level, const char * text, void *) {
    if (level >= GGML_LOG_LEVEL_WARN) std::fputs(text, stderr);
}

int projection_dim(const std::string & mmproj) {
    gguf_init_params gp{true, nullptr};
    gguf_context * gg = gguf_init_from_file(mmproj.c_str(), gp);
    const int64_t k = gg ? gguf_find_key(gg, "clip.vision.projection_dim") : -1;
    const int n = k >= 0 ? (int) gguf_get_val_u32(gg, k) : 0;
    if (gg) gguf_free(gg);
    return n;
}

bool write_sve(const std::string & path, const float * rows, int n, int nx, int ny, int dim) {
    FILE * f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    const int32_t hdr[5] = {0x31455653, n, nx, ny, dim};
    const bool ok = std::fwrite(hdr, sizeof hdr, 1, f) == 1 && std::fwrite(rows, sizeof(float) * (size_t) dim, (size_t) n, f) == (size_t) n;
    std::fclose(f);
    return ok;
}

void write_ppm(const std::string & path, const mtmd_bitmap * bm) {
    std::ofstream f(path, std::ios::binary);
    f << "P6\n" << mtmd_bitmap_get_nx(bm) << " " << mtmd_bitmap_get_ny(bm) << "\n255\n";
    f.write(reinterpret_cast<const char *>(mtmd_bitmap_get_data(bm)), (std::streamsize) mtmd_bitmap_get_n_bytes(bm));
}

int fail(const std::string & msg) {
    std::fprintf(stderr, "vision_ref: %s\n", msg.c_str());
    return 1;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 2) return fail("usage: vision_ref encode|case ... (see the source)");
    const std::string mode = argv[1];
    std::string mmproj, model, out, out_dir, prompt;
    std::vector<std::string> images;
    int threads = 0, n_predict = 16;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--mmproj") mmproj = next();
        else if (a == "--model") model = next();
        else if (a == "--image") images.push_back(next());
        else if (a == "--out") out = next();
        else if (a == "--out-dir") out_dir = next();
        else if (a == "--prompt") prompt = next();
        else if (a == "--prompt-file") {
            std::ifstream f(next(), std::ios::binary);
            prompt.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
        }
        else if (a == "--threads") threads = std::atoi(next().c_str());
        else if (a == "--n-predict") n_predict = std::atoi(next().c_str());
        else return fail("unknown argument " + a);
    }
    if (mmproj.empty() || model.empty() || images.empty()) return fail("--mmproj, --model and --image are required");
    if (threads <= 0) threads = (int) std::max(1u, std::thread::hardware_concurrency() / 2);
    // replace the literal "\n" of a command line by newlines
    for (size_t p; (p = prompt.find("\\n")) != std::string::npos;) prompt.replace(p, 2, "\n");

    llama_log_set(quiet_log, nullptr);
    mtmd_helper_log_set(quiet_log, nullptr);
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = mode == "encode";
    mp.n_gpu_layers = 0;
    const auto t0 = std::chrono::steady_clock::now();
    llama_model * text = llama_model_load_from_file(model.c_str(), mp);
    if (!text) return fail("cannot open the text model " + model);
    mtmd_context_params cp = mtmd_context_params_default();
    cp.use_gpu = false;
    cp.print_timings = false;
    cp.warmup = false;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.n_threads = threads;
    mtmd_context * ctx = mtmd_init_from_file(mmproj.c_str(), text, cp);
    if (!ctx || !mtmd_support_vision(ctx)) return fail("cannot load the vision encoder " + mmproj);
    const int n_embd = projection_dim(mmproj);
    std::fprintf(stderr, "vision_ref: loaded in %.1f s\n",
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

    std::vector<mtmd_helper_bitmap_wrapper> bms;
    for (const std::string & img : images) {
        bms.push_back(mtmd_helper_bitmap_init_from_file(ctx, img.c_str(), false, mtmd_helper_init_opt_default()));
        if (!bms.back().bitmap) return fail("cannot read the image " + img);
    }
    std::vector<const mtmd_bitmap *> bp;
    for (auto & b : bms) bp.push_back(b.bitmap);
    if (mode == "encode") prompt = mtmd_default_marker();
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    mtmd_input_text txt{prompt.c_str(), prompt.size(), false, true};
    if (mtmd_tokenize(ctx, chunks, &txt, bp.data(), bp.size()) != 0) return fail("mtmd_tokenize failed (one <__media__> per image?)");

    // every image chunk: encode, write rows (and pixels)
    struct Img { int first, nx, ny; std::string ppm, sve; };
    std::vector<Img> imgs;
    std::vector<int32_t> tokens, pos[3];
    llama_pos n_past = 0;
    const llama_token image_token = 248056;  // qwen4exp.ple.image_token_id, what the engine expects at image positions
    for (size_t c = 0, k = 0; c < mtmd_input_chunks_size(chunks); ++c) {
        const mtmd_input_chunk * ch = mtmd_input_chunks_get(chunks, c);
        if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t n = 0;
            const llama_token * t = mtmd_input_chunk_get_tokens_text(ch, &n);
            for (size_t i = 0; i < n; ++i, ++n_past) {
                tokens.push_back(t[i]);
                for (auto & p : pos) p.push_back(n_past);
            }
            continue;
        }
        const auto e0 = std::chrono::steady_clock::now();
        if (mtmd_encode_chunk(ctx, ch) != 0) return fail("the vision encoder failed");
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - e0).count();
        const mtmd_image_tokens * it = mtmd_input_chunk_get_tokens_image(ch);
        const int n = (int) mtmd_input_chunk_get_n_tokens(ch);
        const mtmd_decoder_pos last = mtmd_image_tokens_get_decoder_pos(it, 0, (size_t) n - 1);
        const int nx = (int) last.x + 1, ny = (int) last.y + 1;
        if (nx * ny != n) return fail("the image grid is not rectangular");
        Img im{(int) tokens.size(), nx, ny, "", ""};
        const std::string base = mode == "encode" ? out : (std::filesystem::path(out_dir) / ("image" + std::to_string(k))).string();
        im.sve = mode == "encode" ? out : base + ".sve";
        if (!write_sve(im.sve, mtmd_get_output_embd(ctx), n, nx, ny, n_embd)) return fail("cannot write " + im.sve);
        if (mode != "encode") {
            im.ppm = base + ".ppm";
            write_ppm(im.ppm, bp[k]);
        }
        std::printf("image %zu: %ux%u -> %d tokens (%dx%d), encoded in %.0f ms -> %s\n", k, mtmd_bitmap_get_nx(bp[k]), mtmd_bitmap_get_ny(bp[k]), n,
                    nx, ny, ms, im.sve.c_str());
        for (int i = 0; i < n; ++i) {
            const mtmd_decoder_pos p = mtmd_image_tokens_get_decoder_pos(it, n_past, (size_t) i);
            tokens.push_back(image_token);
            pos[0].push_back((int32_t) p.t);
            pos[1].push_back((int32_t) p.y);
            pos[2].push_back((int32_t) p.x);
        }
        n_past += mtmd_input_chunk_get_n_pos(ch);
        imgs.push_back(im);
        ++k;
    }
    if (mode == "encode") return 0;

    // the whole model: the prompt in one ubatch, then greedy decoding
    llama_context_params lp = llama_context_default_params();
    lp.n_ctx = 16384;
    lp.n_batch = lp.n_ubatch = 8192;
    lp.n_threads = lp.n_threads_batch = threads;
    lp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context * lctx = llama_init_from_model(text, lp);
    if (!lctx) return fail("cannot create the llama context");
    const auto p0 = std::chrono::steady_clock::now();
    llama_pos new_past = 0;
    if (mtmd_helper_eval_chunks(ctx, lctx, chunks, 0, 0, (int32_t) lp.n_batch, true, &new_past) != 0) return fail("the prompt failed");
    std::printf("prompt of %zu tokens in %.1f s; next position %d\n", tokens.size(),
                std::chrono::duration<double>(std::chrono::steady_clock::now() - p0).count(), (int) new_past);
    const llama_vocab * vocab = llama_model_get_vocab(text);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const float * logits = llama_get_logits_ith(lctx, -1);
    {
        std::ofstream f(std::filesystem::path(out_dir) / "logits.bin", std::ios::binary);
        f.write(reinterpret_cast<const char *>(logits), (std::streamsize) (sizeof(float) * (size_t) n_vocab));
    }
    std::vector<int32_t> greedy;
    llama_pos p = new_past;
    for (int i = 0; i < n_predict; ++i) {
        const float * lg = llama_get_logits_ith(lctx, -1);
        llama_token best = (llama_token) (std::max_element(lg, lg + n_vocab) - lg);
        greedy.push_back(best);
        if (i + 1 == n_predict) break;
        llama_batch b = llama_batch_get_one(&best, 1);
        (void) p;
        if (llama_decode(lctx, b) != 0) return fail("decode failed");
    }
    std::ofstream cf(std::filesystem::path(out_dir) / "case.txt");
    auto line = [&](const char * key, const std::vector<int32_t> & v) {
        cf << key;
        for (int32_t x : v) cf << ' ' << x;
        cf << '\n';
    };
    line("tokens", tokens);
    line("pos_t", pos[0]);
    line("pos_h", pos[1]);
    line("pos_w", pos[2]);
    for (const Img & im : imgs)
        cf << "image " << im.first << ' ' << im.nx << ' ' << im.ny << ' ' << std::filesystem::path(im.ppm).filename().string() << ' '
           << std::filesystem::path(im.sve).filename().string() << '\n';
    line("greedy", greedy);
    cf << "logits logits.bin\n";
    std::printf("greedy:");
    for (int32_t t : greedy) std::printf(" %d", t);
    std::printf("\n");
    llama_free(lctx);
    mtmd_input_chunks_free(chunks);
    for (auto & b : bms) mtmd_bitmap_free(b.bitmap);
    mtmd_free(ctx);
    llama_model_free(text);
    llama_backend_free();
    return 0;
}
