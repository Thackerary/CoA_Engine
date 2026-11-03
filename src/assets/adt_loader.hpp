#pragma once
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>
#include <span>
#include <optional>
#include <unordered_map>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "../rhi/rhi_gl.hpp"
#include "../vfs/vfs.hpp"
#include "asset_manager.hpp"
#include "m2_loader.hpp"
#include "wmo_loader.hpp"

namespace coa::assets {

#pragma pack(push, 1)
// 3.3.5a MCNK 地形子块 128 字节标准头 (位于 'MCNK' + size 8字节之后)
struct McnkHeader {
    uint32_t flags;             // 0x00
    uint32_t indexX;            // 0x04
    uint32_t indexY;            // 0x08
    uint32_t nLayers;           // 0x0C: 纹理层数 (1~4)
    uint32_t nDoodadRefs;       // 0x10
    uint32_t ofsHeight;         // 0x14: MCVT 相对于 MCNK 块起始处(含8字节头)的偏移
    uint32_t ofsNormal;         // 0x18: MCNR 相对于 MCNK 块起始处的偏移
    uint32_t ofsLayer;          // 0x1C: MCLY 相对于 MCNK 块起始处的偏移
    uint32_t ofsRefs;           // 0x20
    uint32_t ofsAlpha;          // 0x24: MCAL Alpha遮罩偏移
    uint32_t sizeAlpha;         // 0x28
    uint32_t ofsShadow;         // 0x2C
    uint32_t sizeShadow;        // 0x30
    uint32_t areaId;            // 0x34
    uint32_t nMapObjRefs;       // 0x38
    uint32_t holes;             // 0x3C: 低 16 位为 4x4 地形挖洞掩码 (矿洞/地下室入口)
    uint8_t  lowQualityMap[16]; // 0x40
    uint64_t noEffectDoodad;    // 0x50
    uint32_t ofsSndEmitters;    // 0x58
    uint32_t nSndEmitters;      // 0x5C
    uint32_t ofsLiquid;         // 0x60
    uint32_t sizeLiquid;        // 0x64
    glm::vec3 pos;              // 0x68: 该 MCNK 左上角(西北角)在 Z-Up 世界空间的绝对坐标!
    uint32_t ofsMCCV;           // 0x74
    uint32_t ofsMCLV;           // 0x78
    uint32_t unused;            // 0x7C
};
static_assert(sizeof(McnkHeader) == 128, "McnkHeader must be 128 bytes");

// MCLY 图层定义 (16 字节)
struct MclyEntry {
    uint32_t textureId; // 对应 MTEX 字符串表下标
    uint32_t flags;
    uint32_t ofsMCAL;
    int32_t  effectId;
};

// MDDF 装饰物 (M2 树木/石头/灌木) 摆放信息 (36 字节)
struct MddfEntry {
    uint32_t mmidIndex; // 对应 MMID 数组下标
    uint32_t uniqueId;  // 全局唯一 ID
    glm::vec3 pos;      // 编辑器坐标系位置 (需用 17066.66656 - pos 转换)
    glm::vec3 rot;      // 欧拉角 (单位：度)
    uint16_t scale;     // 缩放比例 (1024 = 1.0x)
    uint16_t flags;
};
// MODF 大型建筑 (WMO 教堂/酒馆/矿洞) 摆放信息 (64 字节)
struct ModfEntry {
    uint32_t  mwidIndex; // 对应 MWID 数组下标
    uint32_t  uniqueId;
    glm::vec3 pos;       // 编辑器坐标
    glm::vec3 rot;       // 欧拉角 (度)
    glm::vec3 bboxMin;
    glm::vec3 bboxMax;
    uint16_t  flags;
    uint16_t  doodadSet;
    uint16_t  nameSet;
    uint16_t  scale;
};
static_assert(sizeof(ModfEntry) == 64, "ModfEntry must be 64 bytes");

#pragma pack(pop)

// 单个地形顶点结构 (32 字节对齐)
struct TerrainVertex {
    glm::vec3 pos;    // 世界空间坐标 (Z-Up)
    glm::vec3 normal; // 归一化法线 (Z-Up)
    glm::vec2 uv;     // 地面贴图平铺坐标
};

struct GpuMcnkChunk {
    int32_t  baseVertex = 0; // 在合并 VBO 中的起始顶点 (chunkIdx * 145)
    uint32_t firstIndex = 0;
    uint32_t indexCount = 768;
    GLuint   layerTex[4]  = {0, 0, 0, 0}; // 【修改】将原来的 baseTexId 改为 4 层贴图句柄数组
    GLuint   alphaMapTex  = 0;            // 【新增】当前块的 64x64 RGB Alpha 遮罩纹理句柄
};

// 同种 M2 装饰物的实例化批次
struct GpuDoodadBatch {
    GpuM2Model model;
    uint32_t   baseInstance  = 0; // 在场景 SSBO 矩阵表中的起始下标
    uint32_t   instanceCount = 0; // 该种树木/石头在本 ADT 中的总数量
};
struct GpuWmoBatch {
    GpuWmoModel model;
    uint32_t    baseInstance  = 0;
    uint32_t    instanceCount = 0;
};
struct GpuAdtTile {
    rhi::GlBuffer               terrainVbo;
    rhi::GlBuffer               terrainIbo;
    std::vector<GpuMcnkChunk>   chunks;
    std::vector<GpuDoodadBatch> doodadBatches;
	std::vector<GpuWmoBatch>    wmoBatches; // 【新增】当前 ADT 上的所有 WMO 大型建筑批次
    std::vector<glm::mat4>      instanceMatrices; // 第 0 号固定为单位矩阵(供地形使用)，1..N 为所有 M2 实例矩阵
	std::vector<rhi::GlTexture2D> alphaTextures; // 【新增在 GpuAdtTile 内】托管所有 MCNK 的 64x64 遮罩纹理
    glm::vec3                   suggestedCamPos{0.0f};

    static GLuint CreateTerrainVAO() {
        GLuint vao = 0;
        glCreateVertexArrays(1, &vao);

        glEnableVertexArrayAttrib(vao, 0);
        glVertexArrayAttribFormat(vao, 0, 3, GL_FLOAT, GL_FALSE, offsetof(TerrainVertex, pos));
        glVertexArrayAttribBinding(vao, 0, 0);

        glEnableVertexArrayAttrib(vao, 1);
        glVertexArrayAttribFormat(vao, 1, 3, GL_FLOAT, GL_FALSE, offsetof(TerrainVertex, normal));
        glVertexArrayAttribBinding(vao, 1, 0);

        glEnableVertexArrayAttrib(vao, 2);
        glVertexArrayAttribFormat(vao, 2, 2, GL_FLOAT, GL_FALSE, offsetof(TerrainVertex, uv));
        glVertexArrayAttribBinding(vao, 2, 0);

        return vao;
    }
};

class AdtLoader {
public:
    static constexpr float kMapHalfSize = 32.0f * (1600.0f / 3.0f); // 17066.66656 码
    static constexpr float kChunkSize   = (1600.0f / 3.0f) / 16.0f; // 33.333333 码
    static constexpr float kUnitSize    = kChunkSize / 8.0f;        // 4.1666665 码

    [[nodiscard]] static constexpr uint32_t FourCC(const char (&tag)[5]) noexcept {
        // 匹配磁盘小端序读取后的Reversed FourCC
        return (static_cast<uint32_t>(static_cast<uint8_t>(tag[0])) << 24) |
               (static_cast<uint32_t>(static_cast<uint8_t>(tag[1])) << 16) |
               (static_cast<uint32_t>(static_cast<uint8_t>(tag[2])) << 8)  |
               (static_cast<uint32_t>(static_cast<uint8_t>(tag[3])));
    }

    [[nodiscard]] static std::optional<GpuAdtTile> LoadAdt(
        std::string_view adtVirtualPath,
        const vfs::VirtualFileSystem& vfs,
        AssetManager& assets,
        GLuint fallbackWhiteTex)
    {
        auto fileOpt = vfs.ReadFile(adtVirtualPath);
        if (!fileOpt || fileOpt->size() < 64) return std::nullopt;
        std::span<const uint8_t> bytes(*fileOpt);

        std::vector<std::string> mtexPaths;
        std::span<const char>    mmdxBlob;
        std::span<const uint32_t> mmidOffsets;
        std::span<const MddfEntry> mddfEntries;
		std::span<const char>      mwmoBlob;
        std::span<const uint32_t>  mwidOffsets;
        std::span<const ModfEntry> modfEntries;
        std::vector<TerrainVertex> allVerts;
        allVerts.reserve(256 * 145);

        std::vector<uint16_t> allIndices;
        // 预先在 allIndices[0..767] 生成一份标准无洞口 MCNK 的 768 个索引，供所有普通地形块复用！
        BuildChunkIndices(0, allIndices);

        GpuAdtTile tile;
        tile.chunks.reserve(256);
		tile.alphaTextures.reserve(256);
        tile.instanceMatrices.push_back(glm::mat4(1.0f)); // 槽位 0：留给地形的单位矩阵

        // 1. 扫描 ADT 顶层 IFF Chunks
        size_t cursor = 0;
        while (cursor + 8 <= bytes.size()) {
            uint32_t magic = 0, size = 0;
            std::memcpy(&magic, bytes.data() + cursor, 4);
            std::memcpy(&size,  bytes.data() + cursor + 4, 4);
            size_t dataStart = cursor + 8;
            if (dataStart + size > bytes.size()) break;

            auto chunkData = bytes.subspan(dataStart, size);

            if (magic == FourCC("MTEX")) {
                // 解析地表 BLP 贴图路径列表
                ParseNullTerminatedStrings(chunkData, mtexPaths);
            } else if (magic == FourCC("MMDX")) {
                mmdxBlob = std::span<const char>(reinterpret_cast<const char*>(chunkData.data()), chunkData.size());
            } else if (magic == FourCC("MMID")) {
                mmidOffsets = std::span<const uint32_t>(reinterpret_cast<const uint32_t*>(chunkData.data()),
                                                        chunkData.size() / sizeof(uint32_t));
            } else if (magic == FourCC("MDDF")) {
                mddfEntries = std::span<const MddfEntry>(reinterpret_cast<const MddfEntry*>(chunkData.data()),
                                                         chunkData.size() / sizeof(MddfEntry));
			} else if (magic == FourCC("MWMO")) {
                mwmoBlob = std::span<const char>(reinterpret_cast<const char*>(chunkData.data()), chunkData.size());
            } else if (magic == FourCC("MWID")) {
                mwidOffsets = std::span<const uint32_t>(reinterpret_cast<const uint32_t*>(chunkData.data()),
                                                        chunkData.size() / sizeof(uint32_t));
            } else if (magic == FourCC("MODF")) {
                modfEntries = std::span<const ModfEntry>(reinterpret_cast<const ModfEntry*>(chunkData.data()),
                                                         chunkData.size() / sizeof(ModfEntry));
            } else if (magic == FourCC("MCNK") && size >= sizeof(McnkHeader)) {
                // 解析单个 MCNK 地形块 (包含 8 字节头在内的起始位置为 cursor)
                auto mcnkFullSpan = bytes.subspan(cursor, size + 8);
                ParseMcnkChunk(mcnkFullSpan, mtexPaths, assets, fallbackWhiteTex, allVerts, allIndices, tile);
            }

            cursor = dataStart + size;
        }

        if (tile.chunks.empty()) return std::nullopt;

        // 2. 上传地形合并 VBO 与 IBO
        tile.terrainVbo = rhi::GlBuffer(allVerts.size() * sizeof(TerrainVertex), allVerts.data(), 0);
        tile.terrainIbo = rhi::GlBuffer(allIndices.size() * sizeof(uint16_t), allIndices.data(), 0);

        // 将初始相机设在整块 ADT 正中央的 MCNK (第 128 块) 上方 25 码处
        const auto& midVert = allVerts[(tile.chunks.size() / 2) * 145];
        tile.suggestedCamPos = midVert.pos + glm::vec3(0.0f, 0.0f, 25.0f);

        // 3. 按 M2 路径分组解析 MDDF 装饰物实例，合并为 Instanced Draw Batches
        std::unordered_map<std::string, std::vector<glm::mat4>> doodadGroups;
        for (const auto& entry : mddfEntries) {
            if (entry.mmidIndex >= mmidOffsets.size()) continue;
            uint32_t strOffset = mmidOffsets[entry.mmidIndex];
            if (strOffset >= mmdxBlob.size()) continue;

            std::string m2Path(mmdxBlob.data() + strOffset);
            m2Path = vfs::VirtualFileSystem::NormalizeVirtualPath(m2Path);
            if (m2Path.size() >= 4 && (m2Path.ends_with(".mdx") || m2Path.ends_with(".mdl"))) {
                m2Path.replace(m2Path.size() - 4, 4, ".m2");
            }

            // 将 MDDF 左上角编辑器坐标转换为魔兽 Z-Up 世界坐标
            glm::vec3 worldPos(
                kMapHalfSize - entry.pos.z,
                kMapHalfSize - entry.pos.x,
                entry.pos.y
            );
            float scale = static_cast<float>(entry.scale) / 1024.0f;

            glm::mat4 mat(1.0f);
            mat = glm::translate(mat, worldPos);
            // 魔兽 MDDF 欧拉角转 Z-Up 旋转矩阵
            mat = glm::rotate(mat, glm::radians(entry.rot.y - 180.0f), glm::vec3(0.0f, 0.0f, 1.0f));
            mat = glm::rotate(mat, glm::radians(-entry.rot.x),        glm::vec3(0.0f, 1.0f, 0.0f));
            mat = glm::rotate(mat, glm::radians(entry.rot.z),         glm::vec3(1.0f, 0.0f, 0.0f));
            mat = glm::scale(mat, glm::vec3(scale));

            doodadGroups[m2Path].push_back(mat);
        }

        // 加载视野内存在的 M2 模型并录入 SSBO 矩阵表
        for (auto& [m2Path, mats] : doodadGroups) {
            auto modelOpt = M2Loader::LoadModel(m2Path, vfs, assets, fallbackWhiteTex);
            if (!modelOpt) continue; // 若当前测试提取尚未提取该 M2，自动跳过

            GpuDoodadBatch batch;
            batch.model         = std::move(*modelOpt);
            batch.baseInstance  = static_cast<uint32_t>(tile.instanceMatrices.size());
            batch.instanceCount = static_cast<uint32_t>(mats.size());

            tile.instanceMatrices.insert(tile.instanceMatrices.end(), mats.begin(), mats.end());
            tile.doodadBatches.push_back(std::move(batch));
        }

        std::printf("[ADT Loader] Loaded %zu MCNK chunks, %zu unique M2 doodad types (%zu total instances)\n",
                    tile.chunks.size(), tile.doodadBatches.size(), tile.instanceMatrices.size() - 1);
		// 4. 解析 MODF 大型建筑 (WMO) 并录入 SSBO 矩阵表
        std::unordered_map<std::string, std::vector<glm::mat4>> wmoGroups;
        for (const auto& entry : modfEntries) {
            if (entry.mwidIndex >= mwidOffsets.size()) continue;
            uint32_t strOffset = mwidOffsets[entry.mwidIndex];
            if (strOffset >= mwmoBlob.size()) continue;

            std::string wmoPath = vfs::VirtualFileSystem::NormalizeVirtualPath(mwmoBlob.data() + strOffset);
            glm::vec3 worldPos(
                kMapHalfSize - entry.pos.z,
                kMapHalfSize - entry.pos.x,
                entry.pos.y
            );

            glm::mat4 mat(1.0f);
            mat = glm::translate(mat, worldPos);
            mat = glm::rotate(mat, glm::radians(entry.rot.y - 180.0f), glm::vec3(0.0f, 0.0f, 1.0f));
            mat = glm::rotate(mat, glm::radians(-entry.rot.x),        glm::vec3(0.0f, 1.0f, 0.0f));
            mat = glm::rotate(mat, glm::radians(entry.rot.z),         glm::vec3(1.0f, 0.0f, 0.0f));

            wmoGroups[wmoPath].push_back(mat);
        }

        for (auto& [wmoPath, mats] : wmoGroups) {
            auto wmoOpt = WmoLoader::LoadWmo(wmoPath, vfs, assets, fallbackWhiteTex);
            if (!wmoOpt) continue;

            GpuWmoBatch batch;
            batch.model         = std::move(*wmoOpt);
            batch.baseInstance  = static_cast<uint32_t>(tile.instanceMatrices.size());
            batch.instanceCount = static_cast<uint32_t>(mats.size());

            tile.instanceMatrices.insert(tile.instanceMatrices.end(), mats.begin(), mats.end());
            tile.wmoBatches.push_back(std::move(batch));
        }
        std::printf("[ADT Loader] Loaded %zu WMO building types\n", tile.wmoBatches.size());
        return tile;
    }

private:
    static void ParseNullTerminatedStrings(std::span<const uint8_t> blob, std::vector<std::string>& outList) {
        size_t start = 0;
        for (size_t i = 0; i < blob.size(); ++i) {
            if (blob[i] == 0) {
                if (i > start) {
                    outList.emplace_back(reinterpret_cast<const char*>(blob.data() + start), i - start);
                }
                start = i + 1;
            }
        }
    }

    static void ParseMcnkChunk(
        std::span<const uint8_t> mcnkFullSpan,
        const std::vector<std::string>& mtexPaths,
        AssetManager& assets,
        GLuint fallbackWhiteTex,
        std::vector<TerrainVertex>& allVerts,
        std::vector<uint16_t>& allIndices,
        GpuAdtTile& tile)
    {
        McnkHeader hdr{};
        std::memcpy(&hdr, mcnkFullSpan.data() + 8, sizeof(McnkHeader));

        // 校验 MCVT (145 个 float 相对高度) 与 MCNR (145 * 3 个 int8 法线)
        if (hdr.ofsHeight == 0 || hdr.ofsHeight + 8 + 145 * sizeof(float) > mcnkFullSpan.size()) return;
        if (hdr.ofsNormal == 0 || hdr.ofsNormal + 8 + 145 * 3 > mcnkFullSpan.size()) return;

        const auto* heights = reinterpret_cast<const float*>(mcnkFullSpan.data() + hdr.ofsHeight + 8);
        const auto* normals = reinterpret_cast<const int8_t*>(mcnkFullSpan.data() + hdr.ofsNormal + 8);

        int32_t baseVert = static_cast<int32_t>(allVerts.size());

        // 按 9 外 + 8 内 交错生成 145 个顶点
        size_t vIdx = 0;
        for (int r = 0; r < 9; ++r) {
            // 1. 生成当前行的 9 个外网格顶点
            for (int c = 0; c < 9; ++c) {
                TerrainVertex v{};
                v.pos.x = hdr.pos.x - static_cast<float>(r) * kUnitSize;
                v.pos.y = hdr.pos.y - static_cast<float>(c) * kUnitSize;
                v.pos.z = hdr.pos.z + heights[vIdx];

                v.normal = glm::normalize(glm::vec3(
                    static_cast<float>(normals[vIdx * 3 + 0]) / 127.0f,
                    static_cast<float>(normals[vIdx * 3 + 1]) / 127.0f,
                    static_cast<float>(normals[vIdx * 3 + 2]) / 127.0f
                ));
                v.uv = glm::vec2(static_cast<float>(c), static_cast<float>(r));
                allVerts.push_back(v);
                ++vIdx;
            }

            // 2. 生成当前行的 8 个内中心顶点 (最后一行 r=8 没有内顶点)
            if (r < 8) {
                for (int c = 0; c < 8; ++c) {
                    TerrainVertex v{};
                    v.pos.x = hdr.pos.x - (static_cast<float>(r) + 0.5f) * kUnitSize;
                    v.pos.y = hdr.pos.y - (static_cast<float>(c) + 0.5f) * kUnitSize;
                    v.pos.z = hdr.pos.z + heights[vIdx];

                    v.normal = glm::normalize(glm::vec3(
                        static_cast<float>(normals[vIdx * 3 + 0]) / 127.0f,
                        static_cast<float>(normals[vIdx * 3 + 1]) / 127.0f,
                        static_cast<float>(normals[vIdx * 3 + 2]) / 127.0f
                    ));
                    v.uv = glm::vec2(static_cast<float>(c) + 0.5f, static_cast<float>(r) + 0.5f);
                    allVerts.push_back(v);
                    ++vIdx;
                }
            }
        }
        // 解析 1~4 层 MCLY 纹理与 MCAL Alpha 遮罩
        GpuMcnkChunk gpuChunk{};
        gpuChunk.baseVertex = baseVert;
        for (int i = 0; i < 4; ++i) gpuChunk.layerTex[i] = fallbackWhiteTex;

        uint32_t layerCount = std::min<uint32_t>(hdr.nLayers, 4u);
        MclyEntry layers[4]{};
        if (layerCount > 0 && hdr.ofsLayer > 0 &&
            hdr.ofsLayer + 8 + layerCount * sizeof(MclyEntry) <= mcnkFullSpan.size())
        {
            std::memcpy(layers, mcnkFullSpan.data() + hdr.ofsLayer + 8, layerCount * sizeof(MclyEntry));
            for (uint32_t i = 0; i < layerCount; ++i) {
                if (layers[i].textureId < mtexPaths.size()) {
                    GLuint loaded = assets.GetOrLoadTexture(mtexPaths[layers[i].textureId], true);
                    if (loaded != 0) gpuChunk.layerTex[i] = loaded;
                }
            }
        }

        // 若层数 > 1，解码 MCAL 生成 64x64 RGBA8 遮罩纹理 (repeatWrap = false 即 GL_CLAMP_TO_EDGE)
        std::vector<uint8_t> alphaRgba(64 * 64 * 4, 0);
        if (layerCount > 1 && hdr.ofsAlpha > 0 && hdr.ofsAlpha + 8 < mcnkFullSpan.size()) {
            size_t maxMcal = mcnkFullSpan.size() - (hdr.ofsAlpha + 8);
            size_t mcalSize = (hdr.sizeAlpha >= 8) ? std::min<size_t>(hdr.sizeAlpha - 8, maxMcal) : maxMcal;
            auto mcalSpan = mcnkFullSpan.subspan(hdr.ofsAlpha + 8, mcalSize);

            for (uint32_t i = 1; i < layerCount; ++i) {
                if ((layers[i].flags & 0x100u) != 0) { // 0x100 = 启用 Alpha 遮罩
                    uint32_t curOfs = layers[i].ofsMCAL;
                    uint32_t nextOfs = static_cast<uint32_t>(mcalSpan.size());
                    if (i + 1 < layerCount && (layers[i + 1].flags & 0x100u) != 0) {
                        nextOfs = layers[i + 1].ofsMCAL;
                    }
                    size_t layerByteSize = (nextOfs > curOfs) ? (nextOfs - curOfs) : 0;
                    DecodeAlphaChannel(mcalSpan, hdr.flags, layers[i].flags, curOfs, layerByteSize, i - 1, alphaRgba);
                }
            }
        }

        rhi::GlTexture2D alphaTex(64, 64, 1, rhi::PixelFormat::RGBA8, false);
        alphaTex.UploadUncompressedMip(0, 64, 64, alphaRgba);
        gpuChunk.alphaMapTex = alphaTex.Handle();
        tile.alphaTextures.push_back(std::move(alphaTex));

        // 【补回被误删的地形洞口索引与 push_back】
        uint16_t holeMask = static_cast<uint16_t>(hdr.holes & 0xFFFFu);
        if (holeMask == 0) {
            gpuChunk.firstIndex = 0;
            gpuChunk.indexCount = 768;
        } else {
            gpuChunk.firstIndex = static_cast<uint32_t>(allIndices.size());
            BuildChunkIndices(holeMask, allIndices);
            gpuChunk.indexCount = static_cast<uint32_t>(allIndices.size()) - gpuChunk.firstIndex;
        }

        tile.chunks.push_back(gpuChunk);
    }

    // 根据右手系叉积 e_r x e_c = +Z 生成 8x8x4 = 256 个逆时针 (CCW) 三角形
    static void BuildChunkIndices(uint16_t holeMask, std::vector<uint16_t>& outIndices) {
        for (int r = 0; r < 8; ++r) {
            for (int c = 0; c < 8; ++c) {
                // 检查 4x4 地形洞口掩码
                if (holeMask != 0) {
                    int holeBit = (r / 2) * 4 + (c / 2);
                    if ((holeMask & (1u << holeBit)) != 0) continue;
                }

                uint16_t tl = static_cast<uint16_t>(r * 17 + c);
                uint16_t tr = static_cast<uint16_t>(r * 17 + c + 1);
                uint16_t bl = static_cast<uint16_t>((r + 1) * 17 + c);
                uint16_t br = static_cast<uint16_t>((r + 1) * 17 + c + 1);
                uint16_t center = static_cast<uint16_t>(r * 17 + 9 + c);

                // Top (北), Left (西), Bottom (南), Right (东) 四个扇形子三角形
                outIndices.insert(outIndices.end(), {
                    center, tr, tl,
                    center, tl, bl,
                    center, bl, br,
                    center, br, tr
                });
            }
        }
    }
	// 解码单层 64x64 MCAL Alpha 通道，写入 rgba64x64 的指定通道 (channel = 0:R, 1:G, 2:B)
    // 在参数列表中增加 size_t layerByteSize：
    static void DecodeAlphaChannel(
        std::span<const uint8_t> mcalSpan,
        uint32_t mcnkFlags,
        uint32_t mclyFlags,
        uint32_t ofsInMcal,
        size_t   layerByteSize, // <-- 【新增参数】当前层在 MCAL 中的真实字节数
        uint32_t channel,
        std::vector<uint8_t>& rgba64x64)
    {
        if (ofsInMcal >= mcalSpan.size() || layerByteSize == 0) return;
        const uint8_t* src = mcalSpan.data() + ofsInMcal;
        size_t avail = std::min<size_t>(layerByteSize, mcalSpan.size() - ofsInMcal);

        uint8_t raw64x64[4096]{};

        if ((mclyFlags & 0x200u) != 0) {
            // 1. RLE 行程压缩格式
            size_t inPos = 0, outPos = 0;
            while (outPos < 4096 && inPos < avail) {
                uint8_t ctrl = src[inPos++];
                uint32_t count = (ctrl & 0x7Fu) + 1u;
                if ((ctrl & 0x80u) != 0) {
                    if (inPos >= avail) break;
                    uint8_t val = src[inPos++];
                    for (uint32_t k = 0; k < count && outPos < 4096; ++k) raw64x64[outPos++] = val;
                } else {
                    for (uint32_t k = 0; k < count && outPos < 4096 && inPos < avail; ++k) {
                        raw64x64[outPos++] = src[inPos++];
                    }
                }
            }
        } else if (layerByteSize >= 4096) { // <-- 【核心修复】根据真实字节数判定 8-bit (4096字节)
            // 2. 未压缩 8-bit 格式 (4096 字节)
            size_t copyLen = std::min<size_t>(4096, avail);
            std::memcpy(raw64x64, src, copyLen);
        } else {
            // 3. 未压缩 4-bit 格式 (2048 字节, 每字节 2 像素)
            for (size_t i = 0; i < 2048 && i < avail; ++i) {
                raw64x64[i * 2 + 0] = static_cast<uint8_t>((src[i] & 0x0Fu) * 17u);
                raw64x64[i * 2 + 1] = static_cast<uint8_t>((src[i] >> 4u) * 17u);
            }
        }

        // 修补未开启 DoNotFixAlphaMap (0x8000) 的旧版 MCNK 边缘最后一行/列
        if ((mcnkFlags & 0x8000u) == 0) {
            for (int i = 0; i < 64; ++i) {
                raw64x64[63 * 64 + i] = raw64x64[62 * 64 + i];
                raw64x64[i * 64 + 63] = raw64x64[i * 64 + 62];
            }
        }

        for (size_t p = 0; p < 4096; ++p) {
            rgba64x64[p * 4 + channel] = raw64x64[p];
        }
    }
};

} // namespace coa::assets