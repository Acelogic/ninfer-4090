"""Finds the tensors an abliterated BF16 GGUF changed relative to the original BF16 GGUF, by comparing byte samples of
every tensor over HTTP range requests (nothing is downloaded in full), and writes the list for download_changed.py.

  python find_changed.py <abliterated shard URL template> <original shard URL template> <shards> <delta.json>

URL templates contain {} for the shard number, e.g.
  https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF/resolve/main/BF16/Qwen3.8-Flash-Next-BF16-0000{}-of-00008.gguf
For Huihui's Qwen3.8-Flash-Next it finds attn_output, ssm_out, ffn_down_exps and ffn_down_shexp in every layer that has
them (82 GB of the 354 GB): abliteration edits only the projections that write into the residual stream.
"""
import concurrent.futures as cf, json, re, struct, sys, urllib.request


def fetch(url, start, n):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start + n - 1}"})
    for _ in range(4):
        try:
            return urllib.request.urlopen(req, timeout=240).read()
        except Exception as e:
            err = e
    raise err


class Reader:
    def __init__(self, url): self.url, self.buf, self.pos = url, b"", 0
    def take(self, n):
        while self.pos + n > len(self.buf): self.buf += fetch(self.url, len(self.buf), max(4 << 20, n))
        b = self.buf[self.pos:self.pos + n]; self.pos += n; return b
    def u32(self): return struct.unpack("<I", self.take(4))[0]
    def u64(self): return struct.unpack("<Q", self.take(8))[0]
    def s(self): return self.take(self.u64()).decode("utf-8", "replace")
    def val(self, t):
        if t in (0, 1, 7): return self.take(1)[0]
        if t in (2, 3): return self.take(2)
        if t in (4, 5): return self.u32()
        if t == 6: return self.take(4)
        if t in (10, 11): return self.u64()
        if t == 12: return self.take(8)
        if t == 8: return self.s()
        et, n = self.u32(), self.u64()
        if et == 8:
            for _ in range(n): self.s()
        else:
            self.take({0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}[et] * n)


def layout(url):
    r = Reader(url)
    assert r.take(4) == b"GGUF"
    r.u32(); nt, nkv = r.u64(), r.u64(); align = 32
    for _ in range(nkv):
        k = r.s(); v = r.val(r.u32())
        if k == "general.alignment": align = v
    ts = []
    for _ in range(nt):
        name = r.s(); nd = r.u32(); dims = [r.u64() for _ in range(nd)]; ty = r.u32(); off = r.u64()
        ts.append((name, ty, dims, off))
    data0 = (r.pos + align - 1) // align * align
    return {n: (ty, dims, data0 + off) for n, ty, dims, off in ts}


def main():
    abl, orig, shards, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    a, o = {}, {}
    for i in range(1, shards + 1):
        for n, v in layout(abl.format(i)).items(): a[n] = (i,) + v
        for n, v in layout(orig.format(i)).items(): o[n] = (i,) + v
    assert set(a) == set(o) and all(a[n][1:3] == o[n][1:3] for n in a), "the two files differ in tensors or shapes"
    def size(n):
        ty, dims = a[n][1], a[n][2]; k = 1
        for d in dims: k *= d
        return k * {0: 4, 1: 2, 30: 2}[ty]
    def differs(n):
        sz = size(n)
        for frac in (0.1, 0.5, 0.9):
            pos = int(sz * frac) // 64 * 64; ln = min(32768, sz - pos)
            if fetch(abl.format(a[n][0]), a[n][3] + pos, ln) != fetch(orig.format(o[n][0]), o[n][3] + pos, ln): return n, True
        return n, False
    with cf.ThreadPoolExecutor(24) as ex:
        changed = sorted(n for n, d in ex.map(differs, list(a)) if d)
    delta = [{"name": n, "url": abl.format(a[n][0]), "off": a[n][3], "size": size(n), "dims": a[n][2]} for n in changed]
    json.dump(delta, open(out, "w"), indent=1)
    kinds = sorted({re.sub(r"^blk\.\d+\.", "", n) for n in changed})
    print(f"{len(changed)} changed tensors ({sum(d['size'] for d in delta) / 1e9:.1f} GB): {', '.join(kinds)}")


main()
