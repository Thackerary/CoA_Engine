#pragma once
#include <string>
#include <string_view>
#include <unordered_map>
#include <algorithm>
#include "../mpq/mpq_archive.hpp"
#include "blp_loader.hpp"

namespace coa::assets {

class AssetManager {
public:
    explicit AssetManager(const mpq::MpqFileSystem& vfs) : m_vfs(vfs) {}

    // 根据 MPQ 内部路径获取或加载 GPU 纹理（返回 0 表示加载失败）
    [[nodiscard]] GLuint GetOrLoadTexture(std::string_view mpqPath, bool repeatWrap = true) {
        std::string key = NormalizeKey(mpqPath);
        if (key.empty()) return 0;

        // 1. 若已在显存中，直接返回纹理句柄
        auto it = m_textureCache.find(key);
        if (it != m_textureCache.end()) {
            return it->second.Handle();
        }

        // 2. 从 MPQ 虚拟文件系统读取原始字节流
        auto fileBytes = m_vfs.ReadFile(key);
        if (!fileBytes) {
            return 0;
        }

        // 3. 交给 BlpLoader 解析并直传 DXT1/3/5 压缩块到 GPU
        auto texOpt = BlpLoader::LoadFromMemory(*fileBytes, repeatWrap);
        if (!texOpt) {
            return 0;
        }

        GLuint handle = texOpt->Handle();
        m_textureCache.emplace(std::move(key), std::move(*texOpt));
        return handle;
    }

private:
    const mpq::MpqFileSystem& m_vfs;
    std::unordered_map<std::string, rhi::GlTexture2D> m_textureCache;

    static std::string NormalizeKey(std::string_view path) {
        std::string s(path);
        std::replace(s.begin(), s.end(), '/', '\\');
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return s;
    }
};

} // namespace coa::assets