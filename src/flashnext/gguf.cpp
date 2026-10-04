#include "flashnext/gguf.h"

#include <cstring>
#include <stdexcept>

#if defined(_WIN32)
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace ninfer::flashnext {

const char * type_name(GgufType type) {
    switch (type) {
    case GgufType::F32: return "f32";
    case GgufType::F16: return "f16";
    case GgufType::Q8_0: return "q8_0";
    case GgufType::Q6_K: return "q6_K";
    case GgufType::IQ4_NL: return "iq4_nl";
    case GgufType::IQ3_S: return "iq3_s";
    case GgufType::IQ4_XS: return "iq4_xs";
    case GgufType::BF16: return "bf16";
    }
    return "unknown";
}

std::int64_t block_elems(GgufType type) {
    switch (type) {
    case GgufType::F32:
    case GgufType::F16:
    case GgufType::BF16: return 1;
    case GgufType::Q8_0:
    case GgufType::IQ4_NL: return 32;
    case GgufType::Q6_K:
    case GgufType::IQ3_S:
    case GgufType::IQ4_XS: return 256;
    }
    throw std::runtime_error("unsupported GGUF tensor type " + std::to_string(std::uint32_t(type)));
}

std::size_t block_bytes(GgufType type) {
    switch (type) {
    case GgufType::F32: return 4;
    case GgufType::F16:
    case GgufType::BF16: return 2;
    case GgufType::Q8_0: return 34;
    case GgufType::IQ4_NL: return 18;
    case GgufType::Q6_K: return 210;
    case GgufType::IQ3_S: return 110;
    case GgufType::IQ4_XS: return 136;
    }
    throw std::runtime_error("unsupported GGUF tensor type " + std::to_string(std::uint32_t(type)));
}

std::size_t row_bytes(GgufType type, std::int64_t elems) {
    const std::int64_t be = block_elems(type);
    if (elems % be != 0) throw std::runtime_error("row length is not a whole number of blocks");
    return std::size_t(elems / be) * block_bytes(type);
}

std::int64_t GgufTensor::elements() const {
    std::int64_t n = 1;
    for (std::int64_t d : shape) n *= d;
    return n;
}

class MappedFile {
public:
    explicit MappedFile(const std::string & path) : path_(path) {
#if defined(_WIN32)
        const int wlen = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
        std::wstring wpath(std::size_t(wlen), L'\0');
        MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wlen);
        file_ = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS, nullptr);
        if (file_ == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open " + path);
        LARGE_INTEGER size{};
        GetFileSizeEx(file_, &size);
        size_ = std::size_t(size.QuadPart);
        mapping_ = CreateFileMappingW(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!mapping_) throw std::runtime_error("cannot map " + path);
        base_ = static_cast<const std::uint8_t *>(MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0));
        if (!base_) throw std::runtime_error("cannot map a view of " + path);
#else
        fd_ = open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("cannot open " + path);
        struct stat st{};
        fstat(fd_, &st);
        size_ = std::size_t(st.st_size);
        void * p = mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
        if (p == MAP_FAILED) throw std::runtime_error("cannot map " + path);
        base_ = static_cast<const std::uint8_t *>(p);
#endif
    }
    ~MappedFile() {
#if defined(_WIN32)
        if (base_) UnmapViewOfFile(base_);
        if (mapping_) CloseHandle(mapping_);
        if (file_ != INVALID_HANDLE_VALUE) CloseHandle(file_);
#else
        if (base_) munmap(const_cast<std::uint8_t *>(base_), size_);
        if (fd_ >= 0) close(fd_);
#endif
    }
    const std::uint8_t * data() const { return base_; }
    std::size_t size() const { return size_; }
    const std::string & path() const { return path_; }

private:
    std::string path_;
    const std::uint8_t * base_ = nullptr;
    std::size_t size_ = 0;
#if defined(_WIN32)
    HANDLE file_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_ = nullptr;
#else
    int fd_ = -1;
#endif
};

namespace {

enum : std::uint32_t {
    T_UINT8 = 0, T_INT8 = 1, T_UINT16 = 2, T_INT16 = 3, T_UINT32 = 4, T_INT32 = 5, T_FLOAT32 = 6,
    T_BOOL = 7, T_STRING = 8, T_ARRAY = 9, T_UINT64 = 10, T_INT64 = 11, T_FLOAT64 = 12,
};

struct Cursor {
    const std::uint8_t * p;
    const std::uint8_t * end;
    void need(std::size_t n) const {
        if (std::size_t(end - p) < n) throw std::runtime_error("truncated GGUF header");
    }
    template <class T> T read() {
        need(sizeof(T));
        T v;
        std::memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }
    std::string str() {
        const auto n = read<std::uint64_t>();
        need(n);
        std::string s(reinterpret_cast<const char *>(p), std::size_t(n));
        p += n;
        return s;
    }
};

// Reads one scalar of the given GGUF type as int64/uint64/double/bool/string.
GgufValue read_scalar(Cursor & c, std::uint32_t type) {
    switch (type) {
    case T_UINT8: return std::uint64_t(c.read<std::uint8_t>());
    case T_INT8: return std::int64_t(c.read<std::int8_t>());
    case T_UINT16: return std::uint64_t(c.read<std::uint16_t>());
    case T_INT16: return std::int64_t(c.read<std::int16_t>());
    case T_UINT32: return std::uint64_t(c.read<std::uint32_t>());
    case T_INT32: return std::int64_t(c.read<std::int32_t>());
    case T_FLOAT32: return double(c.read<float>());
    case T_BOOL: return c.read<std::uint8_t>() != 0;
    case T_STRING: return c.str();
    case T_UINT64: return c.read<std::uint64_t>();
    case T_INT64: return c.read<std::int64_t>();
    case T_FLOAT64: return c.read<double>();
    }
    throw std::runtime_error("unknown GGUF value type " + std::to_string(type));
}

GgufValue read_value(Cursor & c, std::uint32_t type) {
    if (type != T_ARRAY) return read_scalar(c, type);
    const auto elem = c.read<std::uint32_t>();
    const auto n = c.read<std::uint64_t>();
    switch (elem) {
    case T_STRING: {
        std::vector<std::string> v;
        v.reserve(std::size_t(n));
        for (std::uint64_t i = 0; i < n; ++i) v.push_back(c.str());
        return v;
    }
    case T_FLOAT32:
    case T_FLOAT64: {
        std::vector<double> v;
        v.reserve(std::size_t(n));
        for (std::uint64_t i = 0; i < n; ++i) v.push_back(std::get<double>(read_scalar(c, elem)));
        return v;
    }
    case T_UINT8:
    case T_UINT16:
    case T_UINT32:
    case T_UINT64:
    case T_BOOL: {
        std::vector<std::uint64_t> v;
        v.reserve(std::size_t(n));
        for (std::uint64_t i = 0; i < n; ++i) {
            GgufValue s = read_scalar(c, elem);
            v.push_back(elem == T_BOOL ? std::uint64_t(std::get<bool>(s)) : std::get<std::uint64_t>(s));
        }
        return v;
    }
    default: {
        std::vector<std::int64_t> v;
        v.reserve(std::size_t(n));
        for (std::uint64_t i = 0; i < n; ++i) v.push_back(std::get<std::int64_t>(read_scalar(c, elem)));
        return v;
    }
    }
}

}  // namespace

GgufModel::GgufModel(const std::vector<std::string> & paths) {
    for (const auto & path : paths) {
        files_.push_back(std::make_unique<MappedFile>(path));
        const MappedFile & f = *files_.back();
        Cursor c{f.data(), f.data() + f.size()};
        if (c.read<std::uint32_t>() != 0x46554747u) throw std::runtime_error(path + " is not a GGUF file");
        const auto version = c.read<std::uint32_t>();
        if (version != 3) throw std::runtime_error(path + ": GGUF version " + std::to_string(version) + " is not supported");
        const auto n_tensors = c.read<std::uint64_t>();
        const auto n_kv = c.read<std::uint64_t>();
        std::uint64_t alignment = 32;
        for (std::uint64_t i = 0; i < n_kv; ++i) {
            std::string key = c.str();
            const auto type = c.read<std::uint32_t>();
            GgufValue value = read_value(c, type);
            if (key == "general.alignment") alignment = std::get<std::uint64_t>(value);
            kv_.emplace(std::move(key), std::move(value));  // the first shard's value wins
        }
        struct Pending { GgufTensor t; std::uint64_t offset; };
        std::vector<Pending> pending;
        for (std::uint64_t i = 0; i < n_tensors; ++i) {
            Pending pt;
            pt.t.name = c.str();
            const auto n_dims = c.read<std::uint32_t>();
            for (std::uint32_t d = 0; d < n_dims; ++d) pt.t.shape.push_back(std::int64_t(c.read<std::uint64_t>()));
            pt.t.type = GgufType(c.read<std::uint32_t>());
            pt.offset = c.read<std::uint64_t>();
            pending.push_back(std::move(pt));
        }
        const std::size_t header = std::size_t(c.p - f.data());
        const std::size_t data_start = (header + alignment - 1) / alignment * alignment;
        for (auto & pt : pending) {
            const std::int64_t row = pt.t.shape.empty() ? 1 : pt.t.shape[0];
            pt.t.bytes = row_bytes(pt.t.type, row) * std::size_t(pt.t.elements() / row);
            if (data_start + pt.offset + pt.t.bytes > f.size()) throw std::runtime_error(pt.t.name + " extends past the end of " + path);
            pt.t.data = f.data() + data_start + pt.offset;
            tensors_.emplace(pt.t.name, std::move(pt.t));
        }
    }
}

GgufModel::~GgufModel() = default;

namespace {
const GgufValue & lookup(const std::map<std::string, GgufValue> & kv, const std::string & key) {
    auto it = kv.find(key);
    if (it == kv.end()) throw std::runtime_error("GGUF key missing: " + key);
    return it->second;
}
}  // namespace

std::int64_t GgufModel::get_int(const std::string & key) const {
    const GgufValue & v = lookup(kv_, key);
    if (auto * i = std::get_if<std::int64_t>(&v)) return *i;
    if (auto * u = std::get_if<std::uint64_t>(&v)) return std::int64_t(*u);
    if (auto * b = std::get_if<bool>(&v)) return *b ? 1 : 0;
    throw std::runtime_error("GGUF key is not an integer: " + key);
}

double GgufModel::get_float(const std::string & key) const {
    const GgufValue & v = lookup(kv_, key);
    if (auto * d = std::get_if<double>(&v)) return *d;
    return double(get_int(key));
}

std::string GgufModel::get_string(const std::string & key) const {
    const GgufValue & v = lookup(kv_, key);
    if (auto * s = std::get_if<std::string>(&v)) return *s;
    throw std::runtime_error("GGUF key is not a string: " + key);
}

std::vector<std::int64_t> GgufModel::get_ints(const std::string & key) const {
    const GgufValue & v = lookup(kv_, key);
    if (auto * a = std::get_if<std::vector<std::int64_t>>(&v)) return *a;
    if (auto * a = std::get_if<std::vector<std::uint64_t>>(&v)) return {a->begin(), a->end()};
    return {get_int(key)};  // a scalar stands for a one-element array
}

std::vector<std::uint64_t> GgufModel::get_uints(const std::string & key) const {
    const GgufValue & v = lookup(kv_, key);
    if (auto * a = std::get_if<std::vector<std::uint64_t>>(&v)) return *a;
    if (auto * a = std::get_if<std::vector<std::int64_t>>(&v)) return {a->begin(), a->end()};
    return {std::uint64_t(get_int(key))};
}

std::vector<std::string> GgufModel::get_strings(const std::string & key) const {
    const GgufValue & v = lookup(kv_, key);
    if (auto * a = std::get_if<std::vector<std::string>>(&v)) return *a;
    throw std::runtime_error("GGUF key is not a string array: " + key);
}

const GgufTensor * GgufModel::find(const std::string & name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}

const GgufTensor & GgufModel::tensor(const std::string & name) const {
    if (const GgufTensor * t = find(name)) return *t;
    throw std::runtime_error("GGUF tensor missing: " + name);
}

}  // namespace ninfer::flashnext
