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
#include "../rhi/rhi_gl.hpp"
#include "../vfs/vfs.hpp"
#include "asset_manager.hpp"

namespace coa::assets {

#pragma pack(push, 1)
struct M2Array {
    uint32_t count;
    uint32_t offset;
};

// 3.3.5a (Build 12340) 标准 48 字节 M2 顶点结构
struct M2Vertex {
    glm::vec3 pos;            // +0x00 (12 bytes): Z-Up 右手系局部坐标
    uint8_t   boneWeights[4]; // +0x0C (4 bytes) : 骨骼权重 (0~255)
    uint8_t   boneIndices[4]; // +0x10 (4 bytes) : 骨骼索引
    glm::vec3 normal;         // +0x14 (12 bytes): 顶点法线
    glm::vec2 texCoords[2];   // +0x20 (16 bytes): UV0 与 UV1
};
static_assert(sizeof(M2Vertex) == 48, "M2Vertex must be exactly 48 bytes");

// 3.3.5a MD20 (Version 264) 文件头核心段
struct M2HeaderV264 {
    char     magic[4];                // 0x00: "MD20"
    uint32_t version;                 // 0x04: 3.3.5a 固定为 264 (0x108)
    M2Array  name;                    // 0x08
    uint32_t globalFlags;             // 0x10
    M2Array  globalSequences;         // 0x14
    M2Array  animations;              // 0x1C
    M2Array  animationLookup;         // 0x24
    M2Array  bones;                   // 0x2C
    M2Array  keyBoneLookup;           // 0x34
    M2Array  vertices;                // 0x3C: 指向 M2Vertex 数组
    uint32_t numSkinProfiles;         // 0x44: 外部 .skin 文件数量
    M2Array  colors;                  // 0x48
    M2Array  textures;                // 0x50: 指向 M2TextureDef 数组
    M2Array  transparency;            // 0x58
    M2Array  textureTransforms;       // 0x60: 指向 UV 动画数组
    M2Array  texReplace;              // 0x68
    M2Array  renderFlags;             // 0x70: 指向 M2Material (flags + blendMode)
    M2Array  boneLookupTable;         // 0x78
    M2Array  texLookupTable;          // 0x80: 指向 uint16_t 纹理查找表
    M2Array  texUnitLookupTable;      // 0x88
    M2Array  transparencyLookupTable; // 0x90
    M2Array  texTransformLookupTable; // 0x98: 指向 uint16_t UV 动画查找表
};

struct M2TextureDef {
    uint32_t type;               // 0 = 硬编码贴图文件名; 非 0 = 角色皮肤/披风动态替换贴图
    uint32_t flags;              // 1 = WrapX, 2 = WrapY
    M2Array  filename;           // 指向以 '\0' 结尾的 BLP 路径字符串
};

struct M2Material {
    uint16_t flags;              // 0x01 = Unlit, 0x04 = TwoSided
    uint16_t blendMode;          // 0 = Opaque, 1 = AlphaKey(Cutout), 2 = AlphaBlend, 4 = Additive
};

// 外部 xxx00.skin 文件头
struct M2SkinHeader {
    char     magic[4];           // 0x00: "SKIN"
    M2Array  vertices;           // 0x04: uint16_t 局部顶点重映射表
    M2Array  indices;            // 0x0C: uint16_t 三角形索引表 (指向 skin.vertices)
    M2Array  bones;              // 0x14
    M2Array  submeshes;          // 0x1C: M2SkinSection 数组
    M2Array  batches;            // 0x24: M2Batch 材质批次数组
    uint32_t boneCountMax;       // 0x2C
};

struct M2SkinSection {
    uint16_t skinSectionId;      // Geoset ID (0 = 主体网格)
    uint16_t level;              // 索引高位偏移 (indexStart + (level << 16))
    uint16_t vertexStart;
    uint16_t vertexCount;
    uint16_t indexStart;
    uint16_t indexCount;
    uint16_t boneCount;
    uint16_t boneComboIndex;
    uint16_t boneInfluences;
    uint16_t centerBoneIndex;
    glm::vec3 centerPosition;
    glm::vec3 sortCenterPosition;
    float     sortRadius;
};
static_assert(sizeof(M2SkinSection) == 48, "M2SkinSection must be 48 bytes");

struct M2Batch {
    uint8_t  flags;
    int8_t   priorityPlane;
    uint16_t shaderId;
    uint16_t skinSectionIndex;   // 对应 submeshes 下标
    uint16_t geosetIndex;
    int16_t  colorIndex;
    uint16_t materialIndex;      // 对应 M2 renderFlags 下标
    uint16_t materialLayer;
    uint16_t textureCount;
    uint16_t textureComboIndex;  // 对应 M2 texLookupTable 下标
    uint16_t textureCoordComboIndex;
    uint16_t textureWeightComboIndex;
    uint16_t textureTransformComboIndex;
};
static_assert(sizeof(M2Batch) == 24, "M2Batch must be 24 bytes");
#pragma pack(pop)

// 上传到 GPU 后的 M2 模型渲染对象
struct GpuM2Submesh {
    uint32_t firstIndex  = 0;
    uint32_t indexCount  = 0;
    GLuint   textureId   = 0;
    uint16_t blendMode   = 0; // 0 = Opaque, 1 = AlphaCutout, 2 = AlphaBlend, 10 = UV Scroll Waterfall
    uint16_t geosetId    = 0;
};

struct GpuM2Model {
    rhi::GlBuffer               vbo;
    rhi::GlBuffer               ibo;
    std::vector<GpuM2Submesh>   submeshes;
    glm::vec3                   localBoundsMin{-1.0f};
    glm::vec3                   localBoundsMax{1.0f};

    // 创建全局共享的 M2 顶点格式 VAO (DSA 架构下所有 M2 模型共用同一个 VAO 即可)
    static GLuint CreateSharedVAO() {
        GLuint vao = 0;
        glCreateVertexArrays(1, &vao);

        // Attr 0: pos (vec3, offset 0)
        glEnableVertexArrayAttrib(vao, 0);
        glVertexArrayAttribFormat(vao, 0, 3, GL_FLOAT, GL_FALSE, offsetof(M2Vertex, pos));
        glVertexArrayAttribBinding(vao, 0, 0);

        // Attr 1: normal (vec3, offset 20)
        glEnableVertexArrayAttrib(vao, 1);
        glVertexArrayAttribFormat(vao, 1, 3, GL_FLOAT, GL_FALSE, offsetof(M2Vertex, normal));
        glVertexArrayAttribBinding(vao, 1, 0);

        // Attr 2: uv0 (vec2, offset 32)
        glEnableVertexArrayAttrib(vao, 2);
        glVertexArrayAttribFormat(vao, 2, 2, GL_FLOAT, GL_FALSE, offsetof(M2Vertex, texCoords[0]));
        glVertexArrayAttribBinding(vao, 2, 0);

        return vao;
    }
};

class M2Loader {
public:
    [[nodiscard]] static std::optional<GpuM2Model> LoadModel(
        std::string_view m2VirtualPath,
        const vfs::VirtualFileSystem& vfs,
        AssetManager& assets,
        GLuint fallbackWhiteTex = 0)
    {
        std::string normPath = vfs::VirtualFileSystem::NormalizeVirtualPath(m2VirtualPath);
        if (normPath.size() < 4) return std::nullopt;

        // 兼容魔兽地形/建筑内部把 .m2 写成 .mdx 或 .mdl 的历史遗留后缀
        if (normPath.ends_with(".mdx") || normPath.ends_with(".mdl")) {
            normPath.replace(normPath.size() - 4, 4, ".m2");
        }

        // 构造同目录下的 xxx00.skin 路径
        std::string skinPath = normPath.substr(0, normPath.size() - 3) + "00.skin";

        auto m2BytesOpt = vfs.ReadFile(normPath);
        if (!m2BytesOpt) {
            std::fprintf(stderr, "  -> [M2 Reason] Missing .m2 in VFS: %s\n", normPath.c_str());
            return std::nullopt;
        }
        auto skinBytesOpt = vfs.ReadFile(skinPath);
        if (!skinBytesOpt) {
            std::fprintf(stderr, "  -> [M2 Reason] Missing 00.skin in VFS: %s\n", skinPath.c_str());
            return std::nullopt;
        }

        std::span<const uint8_t> m2Bytes(*m2BytesOpt);
        std::span<const uint8_t> skinBytes(*skinBytesOpt);

        // 兼容 CoA 从高版本移植的 MD21 封装格式 (剥离 8 字节 'MD21' + size 块头后即为标准 MD20)
        if (m2Bytes.size() >= sizeof(M2HeaderV264) + 8 && std::memcmp(m2Bytes.data(), "MD21", 4) == 0) {
            m2Bytes = m2Bytes.subspan(8);
        }

        if (m2Bytes.size() < sizeof(M2HeaderV264) || skinBytes.size() < sizeof(M2SkinHeader)) {
            std::fprintf(stderr, "  -> [M2 Reason] File too small: %s\n", normPath.c_str());
            return std::nullopt;
        }

        M2HeaderV264 m2Hdr{};
        std::memcpy(&m2Hdr, m2Bytes.data(), sizeof(M2HeaderV264));
        if (std::memcmp(m2Hdr.magic, "MD20", 4) != 0) {
            std::fprintf(stderr, "  -> [M2 Reason] Bad magic in: %s\n", normPath.c_str());
            return std::nullopt;
        }
        if (m2Hdr.vertices.count == 0) {
            std::fprintf(stderr, "  -> [M2 Reason] 0 mesh vertices (particle-only M2): %s\n", normPath.c_str());
            return std::nullopt;
        }

        M2SkinHeader skinHdr{};
        std::memcpy(&skinHdr, skinBytes.data(), sizeof(M2SkinHeader));
        if (std::memcmp(skinHdr.magic, "SKIN", 4) != 0) {
            std::fprintf(stderr, "  -> [M2 Reason] Bad SKIN magic in: %s\n", skinPath.c_str());
            return std::nullopt;
        }

        // 1. 边界安全切片辅助函数
        auto getSpan = []<typename T>(std::span<const uint8_t> blob, M2Array arr) -> std::span<const T> {
            if (arr.count == 0 || static_cast<size_t>(arr.offset) + static_cast<size_t>(arr.count) * sizeof(T) > blob.size()) {
                return {};
            }
            return std::span<const T>(reinterpret_cast<const T*>(blob.data() + arr.offset), arr.count);
        };

        auto m2Verts    = getSpan.template operator()<M2Vertex>(m2Bytes, m2Hdr.vertices);
        auto m2TexDefs  = getSpan.template operator()<M2TextureDef>(m2Bytes, m2Hdr.textures);
        auto m2Mats     = getSpan.template operator()<M2Material>(m2Bytes, m2Hdr.renderFlags);
        auto m2TexLut   = getSpan.template operator()<uint16_t>(m2Bytes, m2Hdr.texLookupTable);
        auto uvAnimLut  = getSpan.template operator()<uint16_t>(m2Bytes, m2Hdr.texTransformLookupTable);

        auto skinVertLut   = getSpan.template operator()<uint16_t>(skinBytes, skinHdr.vertices);
        auto skinIndices   = getSpan.template operator()<uint16_t>(skinBytes, skinHdr.indices);
        auto skinSubmeshes = getSpan.template operator()<M2SkinSection>(skinBytes, skinHdr.submeshes);
        auto skinBatches   = getSpan.template operator()<M2Batch>(skinBytes, skinHdr.batches);

        if (m2Verts.empty() || skinVertLut.empty() || skinIndices.empty() || skinSubmeshes.empty()) {
            return std::nullopt;
        }

        // 1.5 统计 M2Vertex.pos 得到模型局部空间 AABB (供加载期世界空间包围盒预计算使用)
        glm::vec3 localMin = m2Verts.front().pos;
        glm::vec3 localMax = m2Verts.front().pos;
        for (const auto& v : m2Verts) {
            localMin = glm::min(localMin, v.pos);
            localMax = glm::max(localMax, v.pos);
        }
        // 施加最小厚度防止扁平/零体积几何体退化导致视锥误剔除
        localMax = localMin + glm::max(localMax - localMin, glm::vec3(0.05f));

        // 2. 解析 M2 内部引用的所有硬编码 BLP 贴图 (开启 repeatWrap = true 支持瀑布/长条构件 UV 平铺)
        std::vector<GLuint> loadedTexHandles(m2TexDefs.size(), fallbackWhiteTex);
        for (size_t i = 0; i < m2TexDefs.size(); ++i) {
            const auto& td = m2TexDefs[i];
            if (td.type == 0 && td.filename.count > 1 &&
                static_cast<size_t>(td.filename.offset) + td.filename.count <= m2Bytes.size()) {
                std::string_view texPath(reinterpret_cast<const char*>(m2Bytes.data() + td.filename.offset),
                                         td.filename.count - 1);
                GLuint h = assets.GetOrLoadTexture(texPath, true);
                if (h != 0) loadedTexHandles[i] = h;
            }
        }

        // 3. 将 .skin 的二级局部索引展开为直接指向 M2Vertex 的全局索引数组
        std::vector<uint16_t> globalIndices(skinIndices.size());
        for (size_t i = 0; i < skinIndices.size(); ++i) {
            uint16_t localIdx = skinIndices[i];
            globalIndices[i] = (localIdx < skinVertLut.size()) ? skinVertLut[localIdx] : 0;
        }

        // 4. 上传 VBO 与 IBO 到 GPU 显存 (不可变静态存储，0 flag)
        GpuM2Model gpuModel;
        gpuModel.vbo = rhi::GlBuffer(m2Verts.size_bytes(), m2Verts.data(), 0);
        gpuModel.ibo = rhi::GlBuffer(globalIndices.size() * sizeof(uint16_t), globalIndices.data(), 0);
        gpuModel.localBoundsMin = localMin;
        gpuModel.localBoundsMax = localMax;

        // 5. 将 M2Batch 材质与 M2SkinSection 子网格配对组装 (支持多层半透明/瀑布/发光叠加)
        for (size_t sIdx = 0; sIdx < skinSubmeshes.size(); ++sIdx) {
            const auto& sec = skinSubmeshes[sIdx];
            uint32_t firstIdx = static_cast<uint32_t>(sec.indexStart);
            if (globalIndices.size() > 65536u) {
                uint32_t extIdx = firstIdx + (static_cast<uint32_t>(sec.level) << 16);
                if (extIdx + sec.indexCount <= globalIndices.size()) {
                    firstIdx = extIdx;
                }
            }
            if (firstIdx + sec.indexCount > globalIndices.size()) continue;

            bool matchedBatch = false;
            for (const auto& batch : skinBatches) {
                if (batch.skinSectionIndex != sIdx) continue;

                GpuM2Submesh sub{};
                sub.firstIndex = firstIdx;
                sub.indexCount = sec.indexCount;
                sub.geosetId   = sec.skinSectionId;
                sub.textureId  = fallbackWhiteTex;
                sub.blendMode  = 0;

                if (batch.materialIndex < m2Mats.size()) {
                    sub.blendMode = m2Mats[batch.materialIndex].blendMode;
                }
                if (m2Hdr.textureTransforms.count > 0 && batch.textureTransformComboIndex < uvAnimLut.size()) {
                    if (uvAnimLut[batch.textureTransformComboIndex] < m2Hdr.textureTransforms.count) {
                        sub.blendMode = 10; // 10 = 启用 UV 动态滚动的瀑布/水幕层
                    }
                }
                if (batch.textureComboIndex < m2TexLut.size()) {
                    uint16_t texDefIdx = m2TexLut[batch.textureComboIndex];
                    if (texDefIdx < loadedTexHandles.size()) {
                        sub.textureId = loadedTexHandles[texDefIdx];
                    }
                }
                gpuModel.submeshes.push_back(sub);
                matchedBatch = true;
            }

            if (!matchedBatch) {
                GpuM2Submesh sub{};
                sub.firstIndex = firstIdx;
                sub.indexCount = sec.indexCount;
                sub.geosetId   = sec.skinSectionId;
                sub.textureId  = loadedTexHandles.empty() ? fallbackWhiteTex : loadedTexHandles.front();
                sub.blendMode  = 0;
                gpuModel.submeshes.push_back(sub);
            }
        }

        return gpuModel;
    }
};

} // namespace coa::assets