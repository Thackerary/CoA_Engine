#pragma once
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>
#include <span>
#include <optional>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "../rhi/rhi_gl.hpp"
#include "../vfs/vfs.hpp"
#include "asset_manager.hpp"

namespace coa::assets {

#pragma pack(push, 1)
// WMO Root 根文件头 MOHD (64 字节)
struct WmoRootHeader {
    uint32_t nMaterials;
    uint32_t nGroups;
    uint32_t nPortals;
    uint32_t nLights;
    uint32_t nDoodadNames;
    uint32_t nDoodadDefs;
    uint32_t nDoodadSets;
    uint32_t ambColor;
    uint32_t wmoId;
    glm::vec3 bboxMin;
    glm::vec3 bboxMax;
    uint32_t flags;
};
static_assert(sizeof(WmoRootHeader) == 64, "WmoRootHeader must be 64 bytes");

// WMO 材质定义 MOMT (3.3.5a v17 固定 64 字节)
struct WmoMaterial {
    uint32_t flags;
    uint32_t shader;
    uint32_t blendMode;        // 0 = Opaque, 1 = AlphaKey(Cutout)
    uint32_t texture1Ofs;      // 在 MOTX 字符串块中的字节偏移
    uint32_t emissiveColor;
    uint32_t frameEmissiveColor;
    uint32_t texture2Ofs;
    uint32_t diffColor;
    uint32_t groundType;
    float    texData[3];
    uint32_t runTimeData[4];
};
static_assert(sizeof(WmoMaterial) == 64, "WmoMaterial must be 64 bytes");

// WMO Group 绘制批次 MOBA (24 字节)
struct WmoBatch {
    int16_t  bbox[6];
    uint32_t firstIndex;
    uint16_t indexCount;
    uint16_t minVertex;
    uint16_t maxVertex;
    uint8_t  flags;
    uint8_t  materialId;       // 对应根文件 MOMT 数组下标
};
static_assert(sizeof(WmoBatch) == 24, "WmoBatch must be 24 bytes");

// WMO 内部家具摆设定义 MODD (40 字节)
struct WmoDoodadDef {
    uint32_t  nameIndexAndFlags; // 低 24 位 (nameIndexAndFlags & 0xFFFFFFu) 为在 MODN 中的字节偏移
    glm::vec3 pos;               // WMO 局部空间坐标 (Z-Up)
    float     quat[4];           // 四元数旋转 (x, y, z, w)
    float     scale;             // 缩放比例
    uint32_t  color;             // BGRA 颜色
};
static_assert(sizeof(WmoDoodadDef) == 40, "WmoDoodadDef must be 40 bytes");
// 在 WmoDoodadDef 下方新增 (约第 69 行):
struct WmoDoodadSet {
    char     name[20];
    uint32_t startIndex; // 在 MODD 数组中的起始下标
    uint32_t count;      // 家具数量
    uint32_t pad;
};
static_assert(sizeof(WmoDoodadSet) == 32, "WmoDoodadSet must be 32 bytes");
#pragma pack(pop)

// 与 TerrainVertex 严格一致的 32 字节顶点布局，直接复用 terrainVao
struct WmoVertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec2 uv;
};

struct GpuWmoSubBatch {
    uint32_t firstIndex = 0;
    uint32_t indexCount = 0;
    GLuint   textureId  = 0;
    uint32_t blendMode  = 0;
};

struct GpuWmoGroup {
    rhi::GlBuffer               vbo;
    rhi::GlBuffer               ibo;
    std::vector<GpuWmoSubBatch> batches;
    glm::vec3                   localBoundsMin{-1.0f};
    glm::vec3                   localBoundsMax{1.0f};
    // 保留 CPU 侧碰撞网格，供角色行走模式的竖直射线检测使用 (子房间局部空间)
    std::vector<glm::vec3>      cpuPositions;
    std::vector<uint16_t>       cpuIndices;
};

// WMO 内部家具模板 (M2 虚拟路径 + 已换算好的 WMO 局部空间变换矩阵)
struct WmoDoodadTemplate {
    std::string m2Path;
    glm::mat4   localMatrix{1.0f};
};

struct GpuWmoModel {
    std::vector<GpuWmoGroup>       groups;
    std::vector<WmoDoodadSet>      doodadSets; // 【新增】MODS 家具套装范围表
    std::vector<WmoDoodadTemplate> doodads;    // 室内家具/摆设 (MODD)
    glm::vec3                      localBoundsMin{-10.0f};
    glm::vec3                      localBoundsMax{10.0f};
};

class WmoLoader {
public:
    [[nodiscard]] static constexpr uint32_t FourCC(const char (&tag)[5]) noexcept {
        return (static_cast<uint32_t>(static_cast<uint8_t>(tag[0])) << 24) |
               (static_cast<uint32_t>(static_cast<uint8_t>(tag[1])) << 16) |
               (static_cast<uint32_t>(static_cast<uint8_t>(tag[2])) << 8)  |
               (static_cast<uint32_t>(static_cast<uint8_t>(tag[3])));
    }

    [[nodiscard]] static std::optional<GpuWmoModel> LoadWmo(
        std::string_view wmoVirtualPath,
        const vfs::VirtualFileSystem& vfs,
        AssetManager& assets,
        GLuint fallbackWhiteTex)
    {
        std::string normPath = vfs::VirtualFileSystem::NormalizeVirtualPath(wmoVirtualPath);
        if (normPath.size() < 5 || !normPath.ends_with(".wmo")) return std::nullopt;

        auto rootBytesOpt = vfs.ReadFile(normPath);
        if (!rootBytesOpt || rootBytesOpt->size() < 72) return std::nullopt;
        std::span<const uint8_t> rootBytes(*rootBytesOpt);

        // 1. 解析 Root .wmo 中的 MOHD, MOTX (贴图路径池)、MOMT (材质表)、MODN/MODD (内部家具)
        WmoRootHeader mohd{};
        std::span<const char>         motxBlob;
        std::span<const WmoMaterial>  momtMats;
        std::span<const char>         modnBlob;
        std::span<const WmoDoodadDef> moddDefs;
        GpuWmoModel wmoModel;
        size_t cursor = 0;
        while (cursor + 8 <= rootBytes.size()) {
            uint32_t magic = 0, size = 0;
            std::memcpy(&magic, rootBytes.data() + cursor, 4);
            std::memcpy(&size,  rootBytes.data() + cursor + 4, 4);
            size_t dataStart = cursor + 8;
            if (dataStart + size > rootBytes.size()) break;
            auto chunk = rootBytes.subspan(dataStart, size);

            if (magic == FourCC("MOHD") && size >= sizeof(WmoRootHeader)) {
                std::memcpy(&mohd, chunk.data(), sizeof(WmoRootHeader));
            } else if (magic == FourCC("MOTX")) {
                motxBlob = std::span<const char>(reinterpret_cast<const char*>(chunk.data()), chunk.size());
            } else if (magic == FourCC("MOMT")) {
                momtMats = std::span<const WmoMaterial>(reinterpret_cast<const WmoMaterial*>(chunk.data()),
                                                        chunk.size() / sizeof(WmoMaterial));
            } else if (magic == FourCC("MODN")) {
                modnBlob = std::span<const char>(reinterpret_cast<const char*>(chunk.data()), chunk.size());
            } else if (magic == FourCC("MODD")) {
                moddDefs = std::span<const WmoDoodadDef>(reinterpret_cast<const WmoDoodadDef*>(chunk.data()),
                                                         chunk.size() / sizeof(WmoDoodadDef));
			} else if (magic == FourCC("MODS")) {
                auto sets = std::span<const WmoDoodadSet>(reinterpret_cast<const WmoDoodadSet*>(chunk.data()),
                                                          chunk.size() / sizeof(WmoDoodadSet));
                wmoModel.doodadSets.assign(sets.begin(), sets.end());
            }
            cursor = dataStart + size;
        }

        if (mohd.nGroups == 0 || mohd.nGroups > 512) return std::nullopt;

        // 2. 加载 MOMT 材质引用的所有 BLP 纹理
        std::vector<GLuint> matTextures(momtMats.size(), fallbackWhiteTex);
        std::vector<uint32_t> matBlendModes(momtMats.size(), 0);
        for (size_t i = 0; i < momtMats.size(); ++i) {
            matBlendModes[i] = momtMats[i].blendMode;
            uint32_t ofs = momtMats[i].texture1Ofs;
            if (ofs < motxBlob.size()) {
                std::string_view texPath(motxBlob.data() + ofs);
                if (!texPath.empty()) {
                    GLuint h = assets.GetOrLoadTexture(texPath, true);
                    if (h != 0) matTextures[i] = h;
                }
            }
        }

        // 3. 解析 MODD 内部家具摆设，换算为 WMO 局部空间变换矩阵
        wmoModel.groups.reserve(mohd.nGroups);
        bool firstGroup = true;

        wmoModel.doodads.reserve(moddDefs.size());
        for (const auto& def : moddDefs) {
            uint32_t nameOfs = def.nameIndexAndFlags & 0xFFFFFFu;
            if (nameOfs >= modnBlob.size()) {
                wmoModel.doodads.push_back(WmoDoodadTemplate{}); // 保持与 MODD 原始下标 1:1 对齐
                continue;
            }

            std::string m2Path(modnBlob.data() + nameOfs);
            m2Path = vfs::VirtualFileSystem::NormalizeVirtualPath(m2Path);
            if (m2Path.empty()) {
                wmoModel.doodads.push_back(WmoDoodadTemplate{}); // 保持与 MODD 原始下标 1:1 对齐
                continue;
            }
            if (m2Path.size() >= 4 && (m2Path.ends_with(".mdx") || m2Path.ends_with(".mdl"))) {
                m2Path.replace(m2Path.size() - 4, 4, ".m2");
            }

            glm::mat4 localMat = glm::translate(glm::mat4(1.0f), def.pos);
            glm::quat q(def.quat[3], def.quat[0], def.quat[1], def.quat[2]); // GLM 构造顺序 (w, x, y, z)
            localMat *= glm::mat4_cast(glm::normalize(q));
            localMat = glm::scale(localMat, glm::vec3(def.scale));

            wmoModel.doodads.push_back(WmoDoodadTemplate{ std::move(m2Path), localMat });
        }

        // 若个别 WMO 未包含 MODS 块，生成默认全局 Set 0 兜底
        if (wmoModel.doodadSets.empty() && !wmoModel.doodads.empty()) {
            WmoDoodadSet defaultSet{};
            defaultSet.startIndex = 0;
            defaultSet.count = static_cast<uint32_t>(wmoModel.doodads.size());
            wmoModel.doodadSets.push_back(defaultSet);
        }

        // 4. 依次加载所有子房间文件 xxx_000.wmo ~ xxx_NNN.wmo
        std::string basePrefix = normPath.substr(0, normPath.size() - 4);
        for (uint32_t g = 0; g < mohd.nGroups; ++g) {
            char groupPath[512]{};
            std::snprintf(groupPath, sizeof(groupPath), "%s_%03u.wmo", basePrefix.c_str(), g);

            auto groupBytesOpt = vfs.ReadFile(groupPath);
            if (!groupBytesOpt) continue;

            if (auto grp = ParseGroupFile(*groupBytesOpt, matTextures, matBlendModes, fallbackWhiteTex)) {
                // 将所有子房间局部包围盒 union 到整栋建筑的父级包围盒
                if (firstGroup) {
                    wmoModel.localBoundsMin = grp->localBoundsMin;
                    wmoModel.localBoundsMax = grp->localBoundsMax;
                    firstGroup = false;
                } else {
                    wmoModel.localBoundsMin = glm::min(wmoModel.localBoundsMin, grp->localBoundsMin);
                    wmoModel.localBoundsMax = glm::max(wmoModel.localBoundsMax, grp->localBoundsMax);
                }
                wmoModel.groups.push_back(std::move(*grp));
            }
        }

        if (wmoModel.groups.empty()) return std::nullopt;
        return wmoModel;
    }

private:
    static std::optional<GpuWmoGroup> ParseGroupFile(
        std::span<const uint8_t> bytes,
        const std::vector<GLuint>& matTextures,
        const std::vector<uint32_t>& matBlendModes,
        GLuint fallbackWhiteTex)
    {
        // Group 文件结构：顶层 MVER (12字节) + MOGP 块 (8字节块头 + 68字节 MOGP 头 + 内部子块)
        size_t cursor = 0;
        std::span<const uint8_t> mogpSubChunks;
        while (cursor + 8 <= bytes.size()) {
            uint32_t magic = 0, size = 0;
            std::memcpy(&magic, bytes.data() + cursor, 4);
            std::memcpy(&size,  bytes.data() + cursor + 4, 4);
            size_t dataStart = cursor + 8;
            if (dataStart + size > bytes.size()) break;

            if (magic == FourCC("MOGP") && size > 68) {
                mogpSubChunks = bytes.subspan(dataStart + 68, size - 68);
                break;
            }
            cursor = dataStart + size;
        }
        if (mogpSubChunks.empty()) return std::nullopt;

        std::span<const uint16_t>  moviIndices;
        std::span<const glm::vec3> movtPositions;
        std::span<const glm::vec3> monrNormals;
        std::span<const glm::vec2> motvUVs;
        std::span<const WmoBatch>  mobaBatches;

        size_t subCur = 0;
        while (subCur + 8 <= mogpSubChunks.size()) {
            uint32_t magic = 0, size = 0;
            std::memcpy(&magic, mogpSubChunks.data() + subCur, 4);
            std::memcpy(&size,  mogpSubChunks.data() + subCur + 4, 4);
            size_t dataStart = subCur + 8;
            if (dataStart + size > mogpSubChunks.size()) break;
            auto sub = mogpSubChunks.subspan(dataStart, size);

            if (magic == FourCC("MOVI")) {
                moviIndices = std::span<const uint16_t>(reinterpret_cast<const uint16_t*>(sub.data()), sub.size() / sizeof(uint16_t));
            } else if (magic == FourCC("MOVT")) {
                movtPositions = std::span<const glm::vec3>(reinterpret_cast<const glm::vec3*>(sub.data()), sub.size() / sizeof(glm::vec3));
            } else if (magic == FourCC("MONR")) {
                monrNormals = std::span<const glm::vec3>(reinterpret_cast<const glm::vec3*>(sub.data()), sub.size() / sizeof(glm::vec3));
            } else if (magic == FourCC("MOTV") && motvUVs.empty()) { // 取第一套主 UV
                motvUVs = std::span<const glm::vec2>(reinterpret_cast<const glm::vec2*>(sub.data()), sub.size() / sizeof(glm::vec2));
            } else if (magic == FourCC("MOBA")) {
                mobaBatches = std::span<const WmoBatch>(reinterpret_cast<const WmoBatch*>(sub.data()), sub.size() / sizeof(WmoBatch));
            }
            subCur = dataStart + size;
        }

        if (movtPositions.empty() || moviIndices.empty() || mobaBatches.empty()) return std::nullopt;

        size_t vCount = movtPositions.size();
        std::vector<WmoVertex> verts(vCount);
        for (size_t i = 0; i < vCount; ++i) {
            verts[i].pos    = movtPositions[i];
            verts[i].normal = (i < monrNormals.size()) ? monrNormals[i] : glm::vec3(0.0f, 0.0f, 1.0f);
            verts[i].uv     = (i < motvUVs.size())     ? motvUVs[i]     : glm::vec2(0.0f);
        }

        // 统计 MOVT 顶点数组得到该子房间局部空间 AABB
        glm::vec3 gmin = movtPositions.front();
        glm::vec3 gmax = movtPositions.front();
        for (const auto& p : movtPositions) {
            gmin = glm::min(gmin, p);
            gmax = glm::max(gmax, p);
        }
        gmax = gmin + glm::max(gmax - gmin, glm::vec3(0.05f));

        GpuWmoGroup group;
        group.localBoundsMin = gmin;
        group.localBoundsMax = gmax;
        group.vbo = rhi::GlBuffer(verts.size() * sizeof(WmoVertex), verts.data(), 0);
        group.ibo = rhi::GlBuffer(moviIndices.size() * sizeof(uint16_t), moviIndices.data(), 0);

        // 保留原始 MOVT/MOVI 供地面高度射线检测，索引已从 0 开始 (baseVertex = 0)
        group.cpuPositions.assign(movtPositions.begin(), movtPositions.end());
        group.cpuIndices.assign(moviIndices.begin(), moviIndices.end());

        for (const auto& b : mobaBatches) {
            if (static_cast<size_t>(b.firstIndex) + b.indexCount > moviIndices.size()) continue;
            GpuWmoSubBatch sb{};
            sb.firstIndex = b.firstIndex;
            sb.indexCount = b.indexCount;
            sb.textureId  = (b.materialId < matTextures.size())   ? matTextures[b.materialId]   : fallbackWhiteTex;
            sb.blendMode  = (b.materialId < matBlendModes.size()) ? matBlendModes[b.materialId] : 0;
            group.batches.push_back(sb);
        }

        return group;
    }
};

} // namespace coa::assets
