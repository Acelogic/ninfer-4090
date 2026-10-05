"""Builds an abliterated Qwen3.8-Flash-Next (e.g. Huihui's) Qwen3.8-Flash-Next in Unsloth's UD-IQ4_XS recipe by patching the original file.

Huihui's abliteration changes only the four tensor kinds that write into the residual stream (attn_output, ssm_out,
ffn_down_exps, ffn_down_shexp); every other tensor is byte-identical to the original model. So the output is a copy of
Unsloth's UD-IQ4_XS shards in which those tensors are replaced by Huihui's BF16 weights quantized to the same type the
file already has there (IQ4_NL, or Q8_0), with ggml's own quantizer (ggml_quantize_chunk, as llama-quantize) and
mradermacher's importance matrix of the abliterated model. Sizes and offsets are unchanged, so the bytes are written in
place. Usage:

  python patch_tensors.py <quantized shard 1> <out dir> <delta dir> <delta.json> <imatrix.gguf> <ggml-base.dll> <out prefix>

<delta.json> lists the changed tensors (find_changed.py writes it); <delta dir> holds each one's raw BF16 bytes as
<name>.bf16 (download_changed.py fetches them). ggml-base.dll comes from a llama.cpp build.
"""
import ctypes, json, os, re, shutil, struct, sys, time, concurrent.futures as cf
import numpy as np

UD1, OUT, DELTA, DELTA_JSON, IMATRIX, GGML, PREFIX = sys.argv[1:8]
Q8_0, IQ4_NL = 8, 20
BLOCK = {Q8_0: (32, 34), IQ4_NL: (32, 18)}  # weights per block, bytes per block
KV_IQ4NL = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], dtype=np.float32)


def gguf_tensors(path):
    """name -> (type, dims, absolute data offset)"""
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF"
        f.read(4)
        nt, nkv = struct.unpack("<QQ", f.read(16))
        def s():
            n = struct.unpack("<Q", f.read(8))[0]
            return f.read(n).decode("utf-8", "replace")
        align = 32
        def val(t):
            if t in (0, 1, 7): return f.read(1)[0]
            if t in (2, 3): return f.read(2)
            if t in (4, 5): return struct.unpack("<I", f.read(4))[0]
            if t == 6: return f.read(4)
            if t in (10, 11): return struct.unpack("<Q", f.read(8))[0]
            if t == 12: return f.read(8)
            if t == 8: return s()
            if t == 9:
                et, n = struct.unpack("<IQ", f.read(12))
                if et == 8:
                    for _ in range(n): s()
                else:
                    f.read({0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}[et] * n)
        for _ in range(nkv):
            k = s()
            v = val(struct.unpack("<I", f.read(4))[0])
            if k == "general.alignment": align = v
        ts = []
        for _ in range(nt):
            name = s()
            nd = struct.unpack("<I", f.read(4))[0]
            dims = list(struct.unpack(f"<{nd}Q", f.read(8 * nd)))
            ty, off = struct.unpack("<IQ", f.read(12))
            ts.append((name, ty, dims, off))
        data0 = (f.tell() + align - 1) // align * align
    return {n: (ty, dims, data0 + off) for n, ty, dims, off in ts}


def load_imatrix(path):
    """llama-quantize's rule for GGUF imatrices: per expert, sums / count (1 where the expert saw no tokens)."""
    ts = gguf_tensors(path)
    out = {}
    with open(path, "rb") as f:
        for name in ts:
            if not name.endswith(".in_sum2"): continue
            base = name[: -len(".in_sum2")]
            st, sd, so = ts[name]
            ct, cd, co = ts[base + ".counts"]
            assert st == 0 and ct == 0, (name, st, ct)
            f.seek(so); sums = np.fromfile(f, dtype=np.float32, count=int(np.prod(sd)))
            f.seek(co); counts = np.fromfile(f, dtype=np.float32, count=int(np.prod(cd)))
            ne0 = sums.size // counts.size
            sums = sums.reshape(counts.size, ne0)
            e = np.ones_like(sums)
            nz = counts > 0
            e[nz] = sums[nz] / counts[nz, None]
            out[base] = e
    return out


def bf16_to_f32(raw):
    return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def dequant(ty, raw, n):
    b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, BLOCK[ty][1])
    d = b[:, :2].copy().view(np.float16).astype(np.float32)
    if ty == Q8_0:
        return (b[:, 2:].view(np.int8).astype(np.float32) * d).reshape(-1)[:n]
    q = b[:, 2:]
    lo, hi = KV_IQ4NL[q & 15], KV_IQ4NL[q >> 4]
    return (np.concatenate([lo, hi], axis=1) * d).reshape(-1)[:n]


def main():
    lib = ctypes.CDLL(GGML)
    lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                        ctypes.c_int64, ctypes.c_void_p]
    lib.ggml_quantize_chunk.restype = ctypes.c_size_t
    lib.ggml_quantize_init.argtypes = [ctypes.c_int]
    for t in (Q8_0, IQ4_NL): lib.ggml_quantize_init(t)

    imatrix = load_imatrix(IMATRIX)
    print(f"imatrix: {len(imatrix)} entries", flush=True)

    m = re.match(r"(.*)-00001-of-(\d{5})\.gguf$", os.path.basename(UD1))
    n_shards = int(m.group(2))
    src_dir = os.path.dirname(UD1)
    os.makedirs(OUT, exist_ok=True)
    shards = []
    for i in range(1, n_shards + 1):
        src = os.path.join(src_dir, f"{m.group(1)}-{i:05d}-of-{n_shards:05d}.gguf")
        dst = os.path.join(OUT, f"{PREFIX}-{i:05d}-of-{n_shards:05d}.gguf")
        if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(src):
            t0 = time.time(); shutil.copyfile(src, dst)
            print(f"copied shard {i} ({os.path.getsize(dst)/1e9:.1f} GB) in {time.time()-t0:.0f} s", flush=True)
        shards.append((dst, gguf_tensors(dst)))

    delta = json.load(open(DELTA_JSON))
    report = []
    pool = cf.ThreadPoolExecutor(16)
    t_all = time.time()
    for t in delta:
        name = t["name"]
        path, info = next((p, ts) for p, ts in shards if name in ts)
        ty, dims, off = info[name]
        if ty not in BLOCK: raise SystemExit(f"{name}: unexpected type {ty} in the UD file")
        ne0 = dims[0]
        rows_per_slice = dims[1] if len(dims) > 2 else int(np.prod(dims[1:]))
        n_slices = dims[2] if len(dims) > 2 else 1
        bpr = ne0 // BLOCK[ty][0] * BLOCK[ty][1]
        src = bf16_to_f32(open(os.path.join(DELTA, name + ".bf16"), "rb").read())
        assert src.size == ne0 * rows_per_slice * n_slices, name
        im = imatrix.get(name)
        if im is not None: assert im.shape == (n_slices, ne0), (name, im.shape)
        out = np.empty(n_slices * rows_per_slice * bpr, dtype=np.uint8)
        def q(e):
            x = np.ascontiguousarray(src[e * rows_per_slice * ne0:(e + 1) * rows_per_slice * ne0])
            w = np.ascontiguousarray(im[e]) if im is not None else None
            dst = out[e * rows_per_slice * bpr:(e + 1) * rows_per_slice * bpr]
            got = lib.ggml_quantize_chunk(ty, x.ctypes.data, dst.ctypes.data, 0, rows_per_slice, ne0,
                                          w.ctypes.data if w is not None else None)
            assert got == rows_per_slice * bpr, (name, e, got)
        list(pool.map(q, range(n_slices)))
        with open(path, "rb") as f:
            f.seek(off); old = np.frombuffer(f.read(out.size), dtype=np.uint8)
        # quality: relative RMS error of the new quantization against the BF16 source (sampled rows)
        rows = np.linspace(0, n_slices * rows_per_slice - 1, 64).astype(np.int64)
        err = []
        for r in rows:
            xr = src[r * ne0:(r + 1) * ne0]
            dq = dequant(ty, out[r * bpr:(r + 1) * bpr].tobytes(), ne0)
            err.append(np.sqrt(np.mean((dq - xr) ** 2) / max(np.mean(xr ** 2), 1e-30)))
        same = float(np.mean(old == out))
        with open(path, "r+b") as f:
            f.seek(off); f.write(out.tobytes())
        report.append({"name": name, "type": ty, "rel_rms_err": float(np.mean(err)), "bytes_same_as_original": same,
                       "imatrix": im is not None})
        print(f"{name:34s} {('Q8_0' if ty == Q8_0 else 'IQ4_NL'):6s} rel err {np.mean(err):.4f}  bytes unchanged vs "
              f"original {same:.3f}  imatrix {'yes' if im is not None else 'no'}", flush=True)
    json.dump(report, open(os.path.join(OUT, "patch-report.json"), "w"), indent=1)
    print(f"PATCHED {len(report)} tensors in {time.time()-t_all:.0f} s", flush=True)


main()
