#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <optional>
#include <cstdio>
#include <mutex>

#include <lz4.h>
#include <zstd.h>

namespace coa::vfs {

enum class CompressionCodec : uint8_t {
    None = 0, // 未压缩直存 (音频或不可压缩块)
    LZ4  = 1, // 极速解压 (BLP 贴图 / M2 模型 / 动画)
    Zstd = 2  // 高压缩比 (ADT 地形 / WMO 建筑 / DBC 表)
};

#pragma pack(push, 1)
struct PakHeader {
    char     magic[8];   // 固定为 "COAPAK02"
    uint32_t fileCount;  // 包内文件总数
    uint32_t version;    // 格式版本 = 2
};

struct PakEntry {
    uint64_t offset;           // 在 .pak 中的绝对字节偏移
    uint32_t compressedSize;   // 磁盘存储字节数
    uint32_t uncompressedSize; // 解压后原始字节数
    uint8_t  codec;            // CompressionCodec 枚举值
    uint8_t  reserved[3];      // 字节对齐填充 (共 20 字节)
};
#pragma pack(pop)

static_assert(sizeof(PakHeader) == 16, "PakHeader must be 16 bytes");
static_assert(sizeof(PakEntry)  == 20, "PakEntry must be 20 bytes");

class VirtualFileSystem {
public:
    VirtualFileSystem() = default;

    // 1. 开发期挂载本地散文件目录
    bool MountLooseDirectory(const std::filesystem::path& physicalRootDir, int priority = 100) {
        std::error_code ec;
        if (!std::filesystem::exists(physicalRootDir, ec) || !std::filesystem::is_directory(physicalRootDir, ec)) {
            std::fprintf(stderr, "[VFS] Loose directory not found: %s\n", physicalRootDir.string().c_str());
            return false;
        }

        MountPoint mp;
        mp.type     = MountType::LooseDirectory;
        mp.priority = priority;
        mp.rootPath = std::filesystem::canonical(physicalRootDir, ec);

        m_mounts.push_back(std::move(mp));
        SortMountsByPriority();
        return true;
    }

    // 2. 发布期挂载混合压缩 .pak 归档
    bool MountPakArchive(const std::filesystem::path& pakFilePath, int priority = 500) {
        std::ifstream f(pakFilePath, std::ios::binary);
        if (!f.is_open()) return false;

        PakHeader hdr{};
        if (!f.read(reinterpret_cast<char*>(&hdr), sizeof(PakHeader)) ||
            std::memcmp(hdr.magic, "COAPAK02", 8) != 0) {
            return false;
        }

        MountPoint mp;
        mp.type     = MountType::PakArchive;
        mp.priority = priority;
        mp.rootPath = pakFilePath;
        mp.pakIndex.reserve(hdr.fileCount);

        for (uint32_t i = 0; i < hdr.fileCount; ++i) {
            uint16_t pathLen = 0;
            if (!f.read(reinterpret_cast<char*>(&pathLen), sizeof(uint16_t))) break;
            std::string vPath(pathLen, '\0');
            if (!f.read(vPath.data(), pathLen)) break;

            PakEntry entry{};
            if (!f.read(reinterpret_cast<char*>(&entry), sizeof(PakEntry))) break;

            mp.pakIndex.emplace(NormalizeVirtualPath(vPath), entry);
        }

        m_mounts.push_back(std::move(mp));
        SortMountsByPriority();
        return true;
    }

    // 3. 统一读取与自动分流解压 (None / LZ4 / Zstd)
    [[nodiscard]] std::optional<std::vector<uint8_t>> ReadFile(std::string_view virtualPath) const {
        std::string normPath = NormalizeVirtualPath(virtualPath);
        if (normPath.empty()) return std::nullopt;

        // 0. 优先命中后台预取的内存池：直接移出并擦除，完全不触发任何磁盘 I/O
        {
            std::lock_guard<std::mutex> lock(m_cacheMutex);
            auto cacheIt = m_ramCache.find(normPath);
            if (cacheIt != m_ramCache.end()) {
                std::vector<uint8_t> cachedBytes = std::move(cacheIt->second);
                m_ramCache.erase(cacheIt);
                return cachedBytes;
            }
        }

        for (const auto& mp : m_mounts) {
            if (mp.type == MountType::LooseDirectory) {
                std::filesystem::path fullPath = mp.rootPath / normPath;
                std::error_code ec;
                if (!std::filesystem::exists(fullPath, ec) || !std::filesystem::is_regular_file(fullPath, ec)) {
                    continue;
                }

                std::ifstream f(fullPath, std::ios::binary | std::ios::ate);
                if (!f.is_open()) continue;

                std::streamsize sz = f.tellg();
                if (sz <= 0) continue;
                f.seekg(0, std::ios::beg);

                std::vector<uint8_t> buf(static_cast<size_t>(sz));
                if (f.read(reinterpret_cast<char*>(buf.data()), sz)) {
                    return buf;
                }
            } else if (mp.type == MountType::PakArchive) {
                auto it = mp.pakIndex.find(normPath);
                if (it == mp.pakIndex.end()) continue;

                const PakEntry& entry = it->second;
                if (entry.uncompressedSize == 0) continue;

                std::ifstream f(mp.rootPath, std::ios::binary);
                if (!f.is_open()) continue;
                f.seekg(static_cast<std::streamoff>(entry.offset), std::ios::beg);

                auto codec = static_cast<CompressionCodec>(entry.codec);

                // 分支 A：未压缩数据，零中转直接读入输出缓冲区
                if (codec == CompressionCodec::None) {
                    std::vector<uint8_t> outBuf(entry.uncompressedSize);
                    if (f.read(reinterpret_cast<char*>(outBuf.data()), entry.uncompressedSize)) {
                        return outBuf;
                    }
                    continue;
                }

                // 读取压缩数据块
                std::vector<uint8_t> compBuf(entry.compressedSize);
                if (!f.read(reinterpret_cast<char*>(compBuf.data()), entry.compressedSize)) {
                    continue;
                }

                std::vector<uint8_t> outBuf(entry.uncompressedSize);

                // 分支 B：LZ4 极速解压
                if (codec == CompressionCodec::LZ4) {
                    int decBytes = LZ4_decompress_safe(
                        reinterpret_cast<const char*>(compBuf.data()),
                        reinterpret_cast<char*>(outBuf.data()),
                        static_cast<int>(entry.compressedSize),
                        static_cast<int>(entry.uncompressedSize)
                    );
                    if (decBytes == static_cast<int>(entry.uncompressedSize)) {
                        return outBuf;
                    }
                }
                // 分支 C：Zstd 高压缩比解压
                else if (codec == CompressionCodec::Zstd) {
                    size_t decBytes = ZSTD_decompress(
                        outBuf.data(),
                        entry.uncompressedSize,
                        compBuf.data(),
                        entry.compressedSize
                    );
                    if (!ZSTD_isError(decBytes) && decBytes == entry.uncompressedSize) {
                        return outBuf;
                    }
                }
            }
        }
        return std::nullopt;
    }

    // 4. 后台预取：把文件原始字节读入内存池，使主线程后续 ReadFile 零磁盘调用
    void PrefetchFile(std::string_view virtualPath) const {
        std::string normPath = NormalizeVirtualPath(virtualPath);
        if (normPath.empty()) return;

        {
            std::lock_guard<std::mutex> lock(m_cacheMutex);
            if (m_ramCache.size() >= 4096) return;
            if (m_ramCache.find(normPath) != m_ramCache.end()) return;
        }

        auto bytes = ReadFile(normPath);
        if (!bytes) return;

        std::lock_guard<std::mutex> lock(m_cacheMutex);
        if (m_ramCache.size() >= 4096) return;
        m_ramCache.try_emplace(std::move(normPath), std::move(*bytes));
    }

    [[nodiscard]] static std::string NormalizeVirtualPath(std::string_view rawPath) {
        std::string s(rawPath);
        for (char& c : s) {
            if (c == '\\') c = '/';
            else if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
        }
        while (!s.empty() && s.front() == '/') s.erase(0, 1);
        return s;
    }

private:
    enum class MountType { LooseDirectory, PakArchive };

    struct MountPoint {
        MountType                                 type;
        int                                       priority = 0;
        std::filesystem::path                     rootPath;
        std::unordered_map<std::string, PakEntry> pakIndex;
    };

    std::vector<MountPoint> m_mounts;

    // 后台预取池：ParseAdtCpu 工作线程写入，主线程 ReadFile 直接消费
    mutable std::mutex m_cacheMutex;
    mutable std::unordered_map<std::string, std::vector<uint8_t>> m_ramCache;

    void SortMountsByPriority() {
        std::stable_sort(m_mounts.begin(), m_mounts.end(), [](const MountPoint& a, const MountPoint& b) {
            return a.priority > b.priority;
        });
    }
};

} // namespace coa::vfs
