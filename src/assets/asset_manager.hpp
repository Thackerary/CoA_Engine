#pragma once
#include <string>
#include <string_view>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include "../vfs/vfs.hpp"
#include "blp_loader.hpp"

namespace coa::assets {

struct GpuM2Model;
struct GpuWmoModel;
class M2Loader;
class WmoLoader;

class AssetManager {
public:
    explicit AssetManager(const vfs::VirtualFileSystem& vfs) : m_vfs(vfs) {}

    // 1. 根据虚拟路径获取或加载 GPU BLP 纹理（返回 0 表示加载失败）
    [[nodiscard]] GLuint GetOrLoadTexture(std::string_view virtualPath, bool repeatWrap = true) {
        std::string key = vfs::VirtualFileSystem::NormalizeVirtualPath(virtualPath);
        if (key.empty()) return 0;

        auto it = m_textureCache.find(key);
        if (it != m_textureCache.end()) {
            return it->second.Handle();
        }

        auto fileBytes = m_vfs.ReadFile(key);
        if (!fileBytes) {
            return 0;
        }

        auto texOpt = BlpLoader::LoadFromMemory(*fileBytes, repeatWrap);
        if (!texOpt) {
            return 0;
        }

        GLuint handle = texOpt->Handle();
        m_textureCache.emplace(std::move(key), std::move(*texOpt));
        return handle;
    }

    // 2. 全局缓存获取或加载 M2 模型（跨 ADT 共享同一份 VBO/IBO 与动画轨）
    template <typename Loader = M2Loader>
    [[nodiscard]] std::shared_ptr<const GpuM2Model> GetOrLoadM2(
        std::string_view virtualPath,
        GLuint fallbackWhiteTex = 0)
    {
        std::string key = vfs::VirtualFileSystem::NormalizeVirtualPath(virtualPath);
        if (key.size() < 4) return nullptr;
        if (key.ends_with(".mdx") || key.ends_with(".mdl")) {
            key.replace(key.size() - 4, 4, ".m2");
        }

        if (auto it = m_m2Cache.find(key); it != m_m2Cache.end()) {
            return it->second;
        }
        if (m_failedModels.contains(key)) {
            return nullptr;
        }

        auto modelOpt = Loader::LoadModel(key, m_vfs, *this, fallbackWhiteTex);
        if (!modelOpt) {
            m_failedModels.insert(std::move(key));
            return nullptr;
        }

        auto ptr = std::make_shared<GpuM2Model>(std::move(*modelOpt));
        m_m2Cache.emplace(std::move(key), ptr);
        return ptr;
    }

    // 3. 全局缓存获取或加载 WMO 大型建筑模型（跨 ADT 共享子房间 VBO/IBO 与家具模板）
    template <typename Loader = WmoLoader>
    [[nodiscard]] std::shared_ptr<const GpuWmoModel> GetOrLoadWmo(
        std::string_view virtualPath,
        GLuint fallbackWhiteTex = 0)
    {
        std::string key = vfs::VirtualFileSystem::NormalizeVirtualPath(virtualPath);
        if (key.size() < 5 || !key.ends_with(".wmo")) return nullptr;

        if (auto it = m_wmoCache.find(key); it != m_wmoCache.end()) {
            return it->second;
        }
        if (m_failedModels.contains(key)) {
            return nullptr;
        }

        auto wmoOpt = Loader::LoadWmo(key, m_vfs, *this, fallbackWhiteTex);
        if (!wmoOpt) {
            m_failedModels.insert(std::move(key));
            return nullptr;
        }

        auto ptr = std::make_shared<GpuWmoModel>(std::move(*wmoOpt));
        m_wmoCache.emplace(std::move(key), ptr);
        return ptr;
    }

    [[nodiscard]] size_t CachedTextureCount() const noexcept { return m_textureCache.size(); }
    [[nodiscard]] size_t CachedM2Count()      const noexcept { return m_m2Cache.size(); }
    [[nodiscard]] size_t CachedWmoCount()     const noexcept { return m_wmoCache.size(); }

private:
    const vfs::VirtualFileSystem& m_vfs;
    std::unordered_map<std::string, rhi::GlTexture2D>                  m_textureCache;
    std::unordered_map<std::string, std::shared_ptr<const GpuM2Model>>  m_m2Cache;
    std::unordered_map<std::string, std::shared_ptr<const GpuWmoModel>> m_wmoCache;
    std::unordered_set<std::string>                                    m_failedModels;
};

} // namespace coa::assets