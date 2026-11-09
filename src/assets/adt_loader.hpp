#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <string>
#include <string_view>
#include <vector>
#include <span>
#include <optional>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <SDL3/SDL.h>
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

// MH2O 水体块头 (12 字节)：每个 MCNK 索引对应一个
struct Mh2oChunkHeader {
    uint32_t ofsInstances;  // 相对于 MH2O 数据块起始处(跳过8字节头)的偏移
    uint32_t layerCount;    // 水体层数 (0 = 无水)
    uint32_t ofsAttributes;
};
static_assert(sizeof(Mh2oChunkHeader) == 12, "Mh2oChunkHeader must be 12 bytes");

// MH2O 单层水体实例 (24 字节)
struct Mh2oInstance {
    uint16_t liquidType;
    uint16_t liquidVertexFormat; // 0 = Height+Depth, 1 = Height+UV, 2 = DepthOnly, 3 = Height+UV+Depth
    float    minHeightLevel;
    float    maxHeightLevel;
    uint8_t  xOffset;            // 列偏移 c (0..7)
    uint8_t  yOffset;            // 行偏移 r (0..7)
    uint8_t  width;              // 列宽 (1..8)
    uint8_t  height;             // 行高 (1..8)
    uint32_t ofsExistsBitmap;    // 8x8 子格存在掩码偏移 (0 = 全部存在)
    uint32_t ofsVertexData;      // 顶点数据偏移 (0 = 恒取 minHeightLevel)
};
static_assert(sizeof(Mh2oInstance) == 24, "Mh2oInstance must be 24 bytes");

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
    glm::vec3 aabbMin{0.0f};
    glm::vec3 aabbMax{0.0f};
    // 方向 G：CPU 侧地形高度碰撞数据 (MCNK 西北角世界坐标 + 4x4 洞口掩码 + 145 个世界高度)
    glm::vec3 originPos{0.0f};
    uint16_t  holeMask = 0;
    float     worldHeights[145]{};
};

// 单个 MH2O 水体实例在合并液态 VBO/IBO 中的绘制区间
struct GpuLiquidChunk {
    uint32_t  firstIndex = 0;
    uint32_t  indexCount = 0;
    uint16_t  liquidType = 0;
    glm::vec3 aabbMin{0.0f};
    glm::vec3 aabbMax{0.0f};
};

// M2 装饰物单实例：加载期已预计算世界空间裁剪数据
struct M2Instance {
    glm::mat4 model{1.0f};
    glm::mat4 invModel{1.0f}; // 世界空间 -> M2 局部空间，用于地面射线拾取
    uint32_t  uniqueId = 0;   // MDDF 全局唯一 ID (由 MODD 派生的家具实例为 0)
    glm::vec3 worldBoundsMin{0.0f};
    glm::vec3 worldBoundsMax{0.0f};
    glm::vec3 worldCenter{0.0f};
    float     worldRadius = 0.0f;
};

// WMO 子房间在某个建筑实例下的世界空间裁剪数据
struct WmoGroupCull {
    glm::vec3 worldBoundsMin{0.0f};
    glm::vec3 worldBoundsMax{0.0f};
    glm::vec3 worldCenter{0.0f};
    float     worldRadius = 0.0f;
};

// WMO 大型建筑单实例：整栋包围盒 + 各子房间包围盒
struct WmoInstance {
    glm::mat4 model{1.0f};
    glm::mat4 invModel{1.0f}; // 世界空间 -> WMO 局部空间的逆变换，用于地板射线检测
    uint32_t  uniqueId = 0;   // MODF 全局唯一 ID
    glm::vec3 worldBoundsMin{0.0f};
    glm::vec3 worldBoundsMax{0.0f};
    glm::vec3 worldCenter{0.0f};
    float     worldRadius = 0.0f;
    std::vector<WmoGroupCull> groupBounds;
};

// 同种 M2 装饰物的实例化批次 (共享 AssetManager 缓存池中的 GpuM2Model)
struct GpuDoodadBatch {
    std::shared_ptr<const GpuM2Model> model;
    std::vector<M2Instance>           instances;
};
struct GpuWmoBatch {
    std::shared_ptr<const GpuWmoModel> model;
    std::vector<WmoInstance>           instances;
};
struct GpuAdtTile {
    rhi::GlBuffer               terrainVbo;
    rhi::GlBuffer               terrainIbo;
    std::vector<GpuMcnkChunk>   chunks;
    std::vector<GpuDoodadBatch> doodadBatches;
	std::vector<GpuWmoBatch>    wmoBatches; // 【新增】当前 ADT 上的所有 WMO 大型建筑批次
	std::vector<rhi::GlTexture2D> alphaTextures; // 【新增在 GpuAdtTile 内】托管所有 MCNK 的 64x64 遮罩纹理
    rhi::GlBuffer               liquidVbo;       // 【新增】MH2O 水面合并顶点缓冲
    rhi::GlBuffer               liquidIbo;       // 【新增】MH2O 水面合并索引缓冲
    std::vector<GpuLiquidChunk> liquidChunks;    // 【新增】所有水体绘制区间
    glm::vec3                   suggestedCamPos{0.0f};
    // 方向 I：网格坐标与去重 ID 归属记录，供动态流式加载 / 卸载使用
    int                         tileX = -1;      // ADT 网格 X 索引 (文件名的第一个数字)
    int                         tileY = -1;      // ADT 网格 Y 索引 (文件名的第二个数字)
    std::vector<uint32_t>       ownedUniqueIds;  // 本块占用的全局唯一 ID，卸载时归还 seenUniqueIds

    struct DebugPlacement {
        std::string path;
        glm::vec3   pos{0.0f};
        float       scale = 1.0f;
        bool        isWmo = false;
    };
    std::vector<DebugPlacement> debugPlacements;

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
    // ==================== 异步流式加载：后台 CPU 解包产物 ====================
    struct CpuAdtData {
        struct ChunkAlphaCpu {
            int32_t texIds[4]{ -1, -1, -1, -1 }; // 对应 MTEX 字符串表下标，-1 = 无贴图
            bool    hasAlpha = false;
            std::vector<uint8_t> rgba;           // 64x64x4 RGBA8 遮罩数据 (仅在 hasAlpha 时有效)
        };
        struct DoodadPlacement {
            glm::mat4 mat{ 1.0f };
            uint32_t  uniqueId = 0;
        };
        struct WmoPlacement {
            glm::mat4 mat{ 1.0f };
            glm::mat4 invMat{ 1.0f };
            uint16_t  doodadSet = 0;
            uint32_t  uniqueId  = 0;
        };

        int tileX = -1;
        int tileY = -1;
        GpuAdtTile tile;
        std::vector<TerrainVertex> allVerts;
        std::vector<TerrainVertex> liquidVerts;
        std::vector<uint16_t>      allIndices;
        std::vector<uint16_t>      liquidIndices;
        std::vector<std::string>   mtexPaths;
        std::vector<ChunkAlphaCpu> chunkAlphas;
        std::vector<std::string>   preloadTexPaths; // 去重后的 MTEX / MOTX / M2 贴图清单，供主线程逐张预上传

        std::unordered_map<std::string, std::vector<DoodadPlacement>>     doodadGroups;
        std::vector<std::pair<std::string, std::vector<WmoPlacement>>>    wmoList;
        std::vector<std::pair<std::string, std::vector<DoodadPlacement>>> m2List;

        int    stage  = 0;
        size_t cursor = 0;
    };

    static constexpr float kMapHalfSize = 32.0f * (1600.0f / 3.0f); // 17066.66656 码
    static constexpr float kChunkSize   = (1600.0f / 3.0f) / 16.0f; // 33.333333 码
    static constexpr float kUnitSize    = kChunkSize / 8.0f;        // 4.1666665 码
    static constexpr float kTileSize    = 1600.0f / 3.0f;           // 533.33333 码 (单块 ADT 边长)

    // 方向 I：世界坐标 -> ADT 网格索引 (与 azeroth_X_Y.adt 文件命名保持一致)
    [[nodiscard]] static glm::ivec2 WorldToAdtIndex(const glm::vec3& worldPos) noexcept {
        return glm::ivec2(
            static_cast<int>(std::floor((kMapHalfSize - worldPos.y) / kTileSize)),
            static_cast<int>(std::floor((kMapHalfSize - worldPos.x) / kTileSize))
        );
    }

    // 相机到指定 ADT 格子水平矩形的最近距离 (格子内为 0)，用于按真实距离决定加载 / 卸载优先级
    [[nodiscard]] static float DistanceToAdtTile(const glm::vec3& camPos, int tx, int ty) noexcept {
        const float maxX = kMapHalfSize - static_cast<float>(ty) * kTileSize;
        const float minX = maxX - kTileSize;
        const float maxY = kMapHalfSize - static_cast<float>(tx) * kTileSize;
        const float minY = maxY - kTileSize;
        const float dx = std::max({ minX - camPos.x, 0.0f, camPos.x - maxX });
        const float dy = std::max({ minY - camPos.y, 0.0f, camPos.y - maxY });
        return std::sqrt(dx * dx + dy * dy);
    }

    [[nodiscard]] static constexpr uint32_t FourCC(const char (&tag)[5]) noexcept {
        // 匹配磁盘小端序读取后的Reversed FourCC
        return (static_cast<uint32_t>(static_cast<uint8_t>(tag[0])) << 24) |
               (static_cast<uint32_t>(static_cast<uint8_t>(tag[1])) << 16) |
               (static_cast<uint32_t>(static_cast<uint8_t>(tag[2])) << 8)  |
               (static_cast<uint32_t>(static_cast<uint8_t>(tag[3])));
    }

    // Arvo 仿射包围盒变换：由局部 AABB 与模型矩阵直接求世界空间 AABB 及包围球
    static void TransformLocalAABB(const glm::vec3& bmin, const glm::vec3& bmax, const glm::mat4& m,
                                   glm::vec3& outMin, glm::vec3& outMax,
                                   glm::vec3& outCenter, float& outRadius) {
        glm::vec3 localCenter  = 0.5f * (bmin + bmax);
        glm::vec3 localExtents = 0.5f * (bmax - bmin);

        outCenter = glm::vec3(m * glm::vec4(localCenter, 1.0f));
        glm::vec3 worldExtents = glm::abs(glm::vec3(m[0])) * localExtents.x
                               + glm::abs(glm::vec3(m[1])) * localExtents.y
                               + glm::abs(glm::vec3(m[2])) * localExtents.z;

        outMin    = outCenter - worldExtents;
        outMax    = outCenter + worldExtents;
        outRadius = glm::length(worldExtents);
    }

    // ==================== 后台线程：纯 CPU ADT 解包 (线程安全) ====================
    // 【严格约束】本函数运行在 std::async 工作线程中，绝不调用任何 OpenGL / AssetManager 接口
    [[nodiscard]] static std::optional<CpuAdtData> ParseAdtCpu(
        std::string_view adtVirtualPath,
        const vfs::VirtualFileSystem& vfs,
        int tx = -1, int ty = -1)
    {
        auto fileOpt = vfs.ReadFile(adtVirtualPath);
        if (!fileOpt || fileOpt->size() < 64) return std::nullopt;
        std::span<const uint8_t> bytes(*fileOpt);

        CpuAdtData cpu;
        cpu.tileX = tx;
        cpu.tileY = ty;

        // 局部引用别名：解析阶段直接写入 CpuAdtData 字段，避免重复拷贝大块几何数据
        GpuAdtTile&                 tile      = cpu.tile;
        std::vector<TerrainVertex>& allVerts  = cpu.allVerts;
        std::vector<uint16_t>&      allIndices = cpu.allIndices;
        std::vector<std::string>&   mtexPaths = cpu.mtexPaths;
        allVerts.reserve(256 * 145);

        std::span<const char>    mmdxBlob;
        std::span<const uint32_t> mmidOffsets;
        std::span<const MddfEntry> mddfEntries;
		std::span<const char>      mwmoBlob;
        std::span<const uint32_t>  mwidOffsets;
        std::span<const ModfEntry> modfEntries;
        std::span<const uint8_t>   mh2oChunkData;

        // 预先在 allIndices[0..767] 生成一份标准无洞口 MCNK 的 768 个索引，供所有普通地形块复用！
        BuildChunkIndices(0, allIndices);

        tile.chunks.reserve(256);
        cpu.chunkAlphas.reserve(256);
		tile.alphaTextures.reserve(256);

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
            } else if (magic == FourCC("MH2O")) {
                mh2oChunkData = chunkData;
            } else if (magic == FourCC("MCNK") && size >= sizeof(McnkHeader)) {
                // 解析单个 MCNK 地形块 (包含 8 字节头在内的起始位置为 cursor)
                auto mcnkFullSpan = bytes.subspan(cursor, size + 8);
                ParseMcnkChunk(mcnkFullSpan, allVerts, allIndices, tile, cpu.chunkAlphas);
            }

            cursor = dataStart + size;
        }

        if (tile.chunks.empty()) return std::nullopt;
        if (cpu.chunkAlphas.size() != tile.chunks.size()) return std::nullopt;

        // 1.5 解析 MH2O 水体层 (必须放在全部 MCNK 顶点写入 allVerts 之后)
        if (!mh2oChunkData.empty()) {
            ParseMh2oChunk(mh2oChunkData, allVerts, tile, cpu.liquidVerts, cpu.liquidIndices);
        }

        // 将初始相机设在整块 ADT 正中央的 MCNK (第 128 块) 上方 25 码处
        const auto& midVert = allVerts[(tile.chunks.size() / 2) * 145];
        tile.suggestedCamPos = midVert.pos + glm::vec3(0.0f, 0.0f, 25.0f);

        // 3. 按 M2 路径分组解析 MDDF 装饰物实例 (仅记录变换，GPU 侧延后到主线程分步上传)
        for (const auto& entry : mddfEntries) {
            if (entry.mmidIndex >= mmidOffsets.size()) continue;
            uint32_t strOffset = mmidOffsets[entry.mmidIndex];
            if (strOffset >= mmdxBlob.size()) continue;

            std::string m2Path(mmdxBlob.data() + strOffset);
            m2Path = vfs::VirtualFileSystem::NormalizeVirtualPath(m2Path);
            if (m2Path.size() >= 4 && (m2Path.ends_with(".mdx") || m2Path.ends_with(".mdl"))) {
                m2Path.replace(m2Path.size() - 4, 4, ".m2");
            }

            // 每块 ADT 完整自包含：不再用 seenUniqueIds 跨块去重跳过，
            // 否则卸载相邻块时会连带让跨界建筑 (如 stormwind.wmo) 整体消失

            // 将 MDDF 左上角编辑器坐标转换为魔兽 Z-Up 世界坐标
            glm::vec3 worldPos(
                kMapHalfSize - entry.pos.z,
                kMapHalfSize - entry.pos.x,
                entry.pos.y
            );
            float scale = (entry.scale > 0) ? (static_cast<float>(entry.scale) / 1024.0f) : 1.0f;

            glm::mat4 mat(1.0f);
            mat = glm::translate(mat, worldPos);
            // 魔兽 MDDF 欧拉角转 Z-Up 旋转矩阵 (Noggit +Z 轴对应我方 -Y 轴，负负得正为 +entry.rot.x)
            mat = glm::rotate(mat, glm::radians(entry.rot.y - 180.0f), glm::vec3(0.0f, 0.0f, 1.0f));
            mat = glm::rotate(mat, glm::radians(entry.rot.x),          glm::vec3(0.0f, 1.0f, 0.0f));
            mat = glm::rotate(mat, glm::radians(entry.rot.z),          glm::vec3(1.0f, 0.0f, 0.0f));
            mat = glm::scale(mat, glm::vec3(scale));

            cpu.doodadGroups[m2Path].push_back(
                CpuAdtData::DoodadPlacement{mat, entry.uniqueId});
            tile.debugPlacements.push_back({m2Path, worldPos, scale, false});
        }

        // 4. 解析 MODF 大型建筑 (WMO) 并记录每个实例的变换矩阵、doodadSet 与唯一 ID
        std::unordered_map<std::string, std::vector<CpuAdtData::WmoPlacement>> wmoGroups;
        for (const auto& entry : modfEntries) {
            if (entry.mwidIndex >= mwidOffsets.size()) continue;
            uint32_t strOffset = mwidOffsets[entry.mwidIndex];
            if (strOffset >= mwmoBlob.size()) continue;

            std::string wmoPath = vfs::VirtualFileSystem::NormalizeVirtualPath(mwmoBlob.data() + strOffset);
            // 与 MDDF 一致：不再跨块去重跳过，唯一 ID 仅记录到实例上
            glm::vec3 worldPos(kMapHalfSize - entry.pos.z, kMapHalfSize - entry.pos.x, entry.pos.y);
            glm::mat4 mat(1.0f);
            mat = glm::translate(mat, worldPos);
            mat = glm::rotate(mat, glm::radians(entry.rot.y - 180.0f), glm::vec3(0.0f, 0.0f, 1.0f));
            mat = glm::rotate(mat, glm::radians(entry.rot.x),          glm::vec3(0.0f, 1.0f, 0.0f));
            mat = glm::rotate(mat, glm::radians(entry.rot.z),          glm::vec3(1.0f, 0.0f, 0.0f));

            wmoGroups[wmoPath].push_back(
                CpuAdtData::WmoPlacement{mat, glm::inverse(mat), entry.doodadSet, entry.uniqueId});
            tile.debugPlacements.push_back({wmoPath, worldPos, 1.0f, true});
        }

        // 将按路径分组的 WMO 摆放转换为有序队列，供主线程分步加载
        cpu.wmoList.reserve(wmoGroups.size());
        for (auto& [wmoPath, placements] : wmoGroups) {
            cpu.wmoList.emplace_back(wmoPath, std::move(placements));
        }

        // 5. 后台预读磁盘并提取全部贴图清单 (纯 CPU，绝不触碰 OpenGL / AssetManager)。
        //    主线程随后逐张预上传，使 WMO / M2 加载阶段不再触发突发贴图解码，杜绝跨越地点时的卡顿。
        {
            std::unordered_set<std::string> uniqueTex;
            uniqueTex.reserve(mtexPaths.size() + 128);

            // 5.1 地表 MTEX 贴图
            for (const auto& rawPath : mtexPaths) {
                std::string norm = vfs::VirtualFileSystem::NormalizeVirtualPath(rawPath);
                if (!norm.empty()) uniqueTex.insert(std::move(norm));
            }

            // 5.2 WMO：预读 root 文件，扫描顶层 IFF 块取 MOTX 内全部贴图路径
            for (const auto& wmoItem : cpu.wmoList) {
                auto wmoBytes = vfs.ReadFile(wmoItem.first);
                if (!wmoBytes || wmoBytes->size() < 8) continue;
                std::span<const uint8_t> wb(*wmoBytes);

                size_t wmoCursor = 0;
                while (wmoCursor + 8 <= wb.size()) {
                    uint32_t magic = 0, size = 0;
                    std::memcpy(&magic, wb.data() + wmoCursor, 4);
                    std::memcpy(&size,  wb.data() + wmoCursor + 4, 4);
                    const size_t dataStart = wmoCursor + 8;
                    if (dataStart + size > wb.size()) break;

                    if (magic == FourCC("MOTX")) {
                        std::vector<std::string> wmoTexPaths;
                        ParseNullTerminatedStrings(wb.subspan(dataStart, size), wmoTexPaths);
                        for (auto& rawPath : wmoTexPaths) {
                            std::string norm = vfs::VirtualFileSystem::NormalizeVirtualPath(rawPath);
                            if (norm.empty()) continue;
                            if (uniqueTex.insert(norm).second) {
                                vfs.ReadFile(norm); // 预热 OS 磁盘缓存
                            }
                        }
                    }
                    wmoCursor = dataStart + size;
                }
            }

            // 5.3 M2：预读模型与 00.skin，解析 MD20.textures 中的硬编码贴图 (type == 0)
            for (const auto& m2Item : cpu.doodadGroups) {
                const std::string& m2Path = m2Item.first;
                auto m2Bytes = vfs.ReadFile(m2Path);
                if (m2Path.size() >= 3) {
                    vfs.ReadFile(m2Path.substr(0, m2Path.size() - 3) + "00.skin");
                }
                if (!m2Bytes || m2Bytes->size() < sizeof(M2HeaderV264)) continue;
                std::span<const uint8_t> mb(*m2Bytes);

                // 兼容可选的 MD21 外层包装：跳过 8 字节头后再匹配 MD20
                size_t hdrOfs = 0;
                if (mb.size() >= 12 && std::memcmp(mb.data(), "MD21", 4) == 0) hdrOfs = 8;
                if (hdrOfs + sizeof(M2HeaderV264) > mb.size()) continue;
                if (std::memcmp(mb.data() + hdrOfs, "MD20", 4) != 0) continue;

                M2HeaderV264 hdr{};
                std::memcpy(&hdr, mb.data() + hdrOfs, sizeof(M2HeaderV264));
                if (hdr.textures.count == 0 || hdr.textures.count > 4096) continue;
                const size_t texBytes = static_cast<size_t>(hdr.textures.count) * sizeof(M2TextureDef);
                if (hdrOfs + hdr.textures.offset + texBytes > mb.size()) continue;

                const auto* defs = reinterpret_cast<const M2TextureDef*>(mb.data() + hdrOfs + hdr.textures.offset);
                for (uint32_t ti = 0; ti < hdr.textures.count; ++ti) {
                    if (defs[ti].type != 0 || defs[ti].filename.count == 0) continue;
                    const size_t strOfs = hdrOfs + defs[ti].filename.offset;
                    if (strOfs >= mb.size()) continue;

                    std::string norm = vfs::VirtualFileSystem::NormalizeVirtualPath(
                        reinterpret_cast<const char*>(mb.data() + strOfs));
                    if (norm.empty()) continue;
                    if (uniqueTex.insert(norm).second) {
                        vfs.ReadFile(norm); // 预热 OS 磁盘缓存
                    }
                }
            }

            cpu.preloadTexPaths.reserve(uniqueTex.size());
            for (auto& tp : uniqueTex) cpu.preloadTexPaths.push_back(std::move(tp));
        }

        return cpu;
    }

    // ==================== 主线程：分步 GPU 上传 ====================
    // 每帧最多消耗 budgetNs 纳秒；返回 true 表示整块 ADT 已可提交渲染
    static bool StepFinalizeAdtGpu(CpuAdtData& c, AssetManager& assets,
                                   GLuint fallbackWhiteTex, uint64_t budgetNs)
    {
        const uint64_t t0 = SDL_GetTicksNS();

        // ---- Stage 0: 一次性上传地形 / 水体合并缓冲，并释放 CPU 几何内存 ----
        if (c.stage == 0) {
            if (c.allVerts.empty() || c.allIndices.empty()) return false;

            c.tile.terrainVbo = rhi::GlBuffer(c.allVerts.size() * sizeof(TerrainVertex), c.allVerts.data(), 0);
            c.tile.terrainIbo = rhi::GlBuffer(c.allIndices.size() * sizeof(uint16_t), c.allIndices.data(), 0);

            if (!c.liquidVerts.empty() && !c.liquidIndices.empty()) {
                c.tile.liquidVbo = rhi::GlBuffer(c.liquidVerts.size() * sizeof(TerrainVertex),
                                                 c.liquidVerts.data(), 0);
                c.tile.liquidIbo = rhi::GlBuffer(c.liquidIndices.size() * sizeof(uint16_t),
                                                 c.liquidIndices.data(), 0);
            }

            std::vector<TerrainVertex>().swap(c.allVerts);
            std::vector<uint16_t>().swap(c.allIndices);
            std::vector<TerrainVertex>().swap(c.liquidVerts);
            std::vector<uint16_t>().swap(c.liquidIndices);

            c.stage  = 1;
            c.cursor = 0;
            return false; // Stage 0 已用掉本帧预算，贴图上传留到下一帧
        }

        // ---- Stage 1: 逐张预上传 BLP 贴图 (预算精确到单张贴图，杜绝集中解码卡顿) ----
        if (c.stage == 1) {
            while (c.cursor < c.preloadTexPaths.size()) {
                assets.GetOrLoadTexture(c.preloadTexPaths[c.cursor++], true);
                if ((SDL_GetTicksNS() - t0) >= budgetNs) return false;
            }
            std::vector<std::string>().swap(c.preloadTexPaths);
            c.stage  = 2;
            c.cursor = 0;
            return false; // 贴图已全部进缓存，地形遮罩上传留到下一帧
        }

        // ---- Stage 2: 分批解码 MCNK 地面贴图与 64x64 Alpha 遮罩 ----
        if (c.stage == 2) {
            const size_t chunkCount = std::min(c.tile.chunks.size(), c.chunkAlphas.size());
            while (c.cursor < chunkCount) {
                const size_t i = c.cursor;
                GpuMcnkChunk& ch = c.tile.chunks[i];
                CpuAdtData::ChunkAlphaCpu& ca = c.chunkAlphas[i];

                for (int l = 0; l < 4; ++l) {
                    ch.layerTex[l] = fallbackWhiteTex;
                    const int32_t id = ca.texIds[l];
                    if (id >= 0 && static_cast<size_t>(id) < c.mtexPaths.size()) {
                        const GLuint loaded = assets.GetOrLoadTexture(c.mtexPaths[static_cast<size_t>(id)], true);
                        if (loaded != 0) ch.layerTex[l] = loaded;
                    }
                }

                if (ca.hasAlpha && ca.rgba.size() >= 64u * 64u * 4u) {
                    rhi::GlTexture2D alphaTex(64, 64, 1, rhi::PixelFormat::RGBA8, false);
                    alphaTex.UploadUncompressedMip(0, 64, 64, ca.rgba);
                    ch.alphaMapTex = alphaTex.Handle();
                    c.tile.alphaTextures.push_back(std::move(alphaTex));
                } else {
                    ch.alphaMapTex = 0; // 单层地形无需遮罩，渲染期绑定全黑纹理
                }
                std::vector<uint8_t>().swap(ca.rgba);

                ++c.cursor;
                if ((SDL_GetTicksNS() - t0) >= budgetNs) return false;
            }
            c.stage  = 3;
            c.cursor = 0;
            return false; // Stage 2 已用掉本帧预算，WMO 加载留到下一帧
        }

        // ---- Stage 3: 分批加载 WMO 建筑，并展开其 MODD 家具到 M2 待加载队列 ----
        if (c.stage == 3) {
            while (c.cursor < c.wmoList.size()) {
                auto& item = c.wmoList[c.cursor];
                auto& placements = item.second;

                auto wmoPtr = assets.GetOrLoadWmo(item.first, fallbackWhiteTex);
                if (wmoPtr) {
                    GpuWmoBatch batch;
                    batch.model = std::move(wmoPtr);
                    batch.instances.reserve(placements.size());

                    // 每个建筑实例一次性预计算整栋 + 各子房间的世界空间 AABB
                    for (const auto& p : placements) {
                        WmoInstance inst{};
                        inst.model    = p.mat;
                        inst.invModel = p.invMat;
                        inst.uniqueId = p.uniqueId;
                        TransformLocalAABB(batch.model->localBoundsMin, batch.model->localBoundsMax, p.mat,
                                           inst.worldBoundsMin, inst.worldBoundsMax,
                                           inst.worldCenter, inst.worldRadius);

                        inst.groupBounds.resize(batch.model->groups.size());
                        for (size_t g = 0; g < batch.model->groups.size(); ++g) {
                            const auto& grp = batch.model->groups[g];
                            TransformLocalAABB(grp.localBoundsMin, grp.localBoundsMax, p.mat,
                                               inst.groupBounds[g].worldBoundsMin, inst.groupBounds[g].worldBoundsMax,
                                               inst.groupBounds[g].worldCenter, inst.groupBounds[g].worldRadius);
                        }
                        batch.instances.push_back(std::move(inst));
                    }

                    // 将 WMO 内部 MODD 家具按激活的 doodadSet 展开，并入统一的 M2 待加载队列
                    auto appendDoodadSet = [&](uint16_t setIdx, const glm::mat4& wmoWorldMat) {
                        if (setIdx >= batch.model->doodadSets.size()) return;
                        const auto& ds = batch.model->doodadSets[setIdx];
                        size_t endIdx = std::min<size_t>(ds.startIndex + ds.count, batch.model->doodads.size());
                        for (size_t i = ds.startIndex; i < endIdx; ++i) {
                            const auto& d = batch.model->doodads[i];
                            if (!d.m2Path.empty()) {
                                const glm::mat4 world = wmoWorldMat * d.localMatrix;
                                // WMO 内部 MODD 家具没有独立全局 ID，统一记为 0
                                c.doodadGroups[d.m2Path].push_back(
                                    CpuAdtData::DoodadPlacement{ world, 0u });
                            }
                        }
                    };
                    for (const auto& p : placements) {
                        appendDoodadSet(0, p.mat); // Set 0 (DefaultGlobal) 始终激活
                        if (p.doodadSet != 0) appendDoodadSet(p.doodadSet, p.mat);
                    }

                    c.tile.wmoBatches.push_back(std::move(batch));
                }

                ++c.cursor;
                if ((SDL_GetTicksNS() - t0) >= budgetNs) return false;
            }

            c.m2List.reserve(c.doodadGroups.size());
            for (auto& kv : c.doodadGroups) {
                c.m2List.emplace_back(kv.first, std::move(kv.second));
            }
            c.doodadGroups.clear();

            c.stage  = 4;
            c.cursor = 0;
            return false; // 绝不在同一帧穿透到 Stage 4
        }

        // ---- Stage 4: 分批加载 M2 装饰物，逐实例预计算世界空间 AABB ----
        if (c.stage == 4) {
            while (c.cursor < c.m2List.size()) {
                auto& item = c.m2List[c.cursor];

                auto modelPtr = assets.GetOrLoadM2(item.first, fallbackWhiteTex);
                if (modelPtr) {
                    GpuDoodadBatch batch;
                    batch.model = std::move(modelPtr);
                    batch.instances.reserve(item.second.size());

                    for (const auto& placement : item.second) {
                        M2Instance inst{};
                        inst.model    = placement.mat;
                        inst.uniqueId = placement.uniqueId;
                        // 仅带碰撞网格的模型才需要逆矩阵，省去 90% 的 glm::inverse 开销
                        if (!batch.model->collisionIndices.empty()) {
                            inst.invModel = glm::inverse(placement.mat);
                        }
                        TransformLocalAABB(batch.model->localBoundsMin, batch.model->localBoundsMax, placement.mat,
                                           inst.worldBoundsMin, inst.worldBoundsMax,
                                           inst.worldCenter, inst.worldRadius);
                        batch.instances.push_back(std::move(inst));
                    }
                    c.tile.doodadBatches.push_back(std::move(batch));
                }

                ++c.cursor;
                if ((SDL_GetTicksNS() - t0) >= budgetNs) return false;
            }

            c.tile.tileX = c.tileX;
            c.tile.tileY = c.tileY;
            c.stage = 5;
            return true;
        }

        return true;
    }

    // 保留同步版本，供启动时的首块 ADT 一次性加载
    [[nodiscard]] static std::optional<GpuAdtTile> LoadAdt(
        std::string_view adtVirtualPath,
        const vfs::VirtualFileSystem& vfs,
        AssetManager& assets,
        GLuint fallbackWhiteTex,
        std::unordered_set<uint32_t>& /*seenUniqueIds*/)
    {
        auto cpu = ParseAdtCpu(adtVirtualPath, vfs);
        if (!cpu) return std::nullopt;
        while (!StepFinalizeAdtGpu(*cpu, assets, fallbackWhiteTex, UINT64_MAX)) {}
        return std::move(cpu->tile);
    }

    // 方向 G：查询 (worldX, worldY) 处脚下最高可行走地面世界 Z
    // 1) MCNK 地形四角双线性插值 (跳过洞口子格)；2) WMO 室内地板 / 楼梯竖直射线求交
    [[nodiscard]] static bool SampleGroundHeight(const std::vector<GpuAdtTile>& tiles,
                                                 float worldX, float worldY,
                                                 float currentFootZ, float& outGroundZ)
    {
        float bestZ = -1e30f;

        // 1. MCNK 地形采样
        for (const auto& tile : tiles) {
            for (const auto& chunk : tile.chunks) {
                if (worldX > chunk.originPos.x || worldX < chunk.originPos.x - kChunkSize) continue;
                if (worldY > chunk.originPos.y || worldY < chunk.originPos.y - kChunkSize) continue;

                const float rf = (chunk.originPos.x - worldX) / kUnitSize;
                const float cf = (chunk.originPos.y - worldY) / kUnitSize;
                const int   r  = std::clamp(static_cast<int>(rf), 0, 7);
                const int   c  = std::clamp(static_cast<int>(cf), 0, 7);

                // 4x4 洞口掩码：被挖空的子格不提供地面
                if ((chunk.holeMask & (1u << ((r / 2) * 4 + (c / 2)))) != 0) continue;

                const float fr = std::clamp(rf - static_cast<float>(r), 0.0f, 1.0f);
                const float fc = std::clamp(cf - static_cast<float>(c), 0.0f, 1.0f);

                const float tl = chunk.worldHeights[r * 17 + c];
                const float tr = chunk.worldHeights[r * 17 + c + 1];
                const float bl = chunk.worldHeights[(r + 1) * 17 + c];
                const float br = chunk.worldHeights[(r + 1) * 17 + c + 1];

                const float zNorth = tl + (tr - tl) * fc;
                const float zSouth = bl + (br - bl) * fc;
                const float terrainZ = zNorth + (zSouth - zNorth) * fr;
                bestZ = std::max(bestZ, terrainZ);
            }
        }

        // 2. WMO 建筑地板射线检测：世界竖直向下射线变换到子房间局部空间做 Möller–Trumbore
        const glm::vec3 worldRayOrigin(worldX, worldY, currentFootZ + 1.8f);
        const glm::vec3 worldRayBottom(worldX, worldY, currentFootZ - 30.0f);
        std::unordered_set<uint32_t> testedWmoIds; // 跨界 WMO 同一 uniqueId 仅检测一次
        for (const auto& tile : tiles) {
            // 快速剪枝：跳过远离角色的 7 块 ADT，仅检测脚下附近的建筑
            if (tile.tileX >= 0 &&
                DistanceToAdtTile(glm::vec3(worldX, worldY, 0.0f), tile.tileX, tile.tileY) > 25.0f) continue;
            for (const auto& wmoBatch : tile.wmoBatches) {
                if (!wmoBatch.model) continue;
                for (const auto& inst : wmoBatch.instances) {
                    if (worldX < inst.worldBoundsMin.x || worldX > inst.worldBoundsMax.x) continue;
                    if (worldY < inst.worldBoundsMin.y || worldY > inst.worldBoundsMax.y) continue;
                    if (inst.uniqueId != 0 && !testedWmoIds.insert(inst.uniqueId).second) continue;
                    if (inst.groupBounds.size() != wmoBatch.model->groups.size()) continue;

                    const glm::vec3 localOrigin = glm::vec3(inst.invModel * glm::vec4(worldRayOrigin, 1.0f));
                    const glm::vec3 localDir    = glm::vec3(inst.invModel * glm::vec4(0.0f, 0.0f, -1.0f, 0.0f));

                    // 竖直射线有效高度区间 [foot-30, foot+1.8] 变换到局部空间，用于三角形级快速剔除
                    const glm::vec3 localTop = glm::vec3(inst.invModel * glm::vec4(worldRayOrigin, 1.0f));
                    const glm::vec3 localBot = glm::vec3(inst.invModel * glm::vec4(worldRayBottom, 1.0f));
                    const glm::vec3 rayMin = glm::min(localTop, localBot) - glm::vec3(0.05f);
                    const glm::vec3 rayMax = glm::max(localTop, localBot) + glm::vec3(0.05f);

                    for (size_t g = 0; g < wmoBatch.model->groups.size(); ++g) {
                        const auto& gc = inst.groupBounds[g];
                        if (worldX < gc.worldBoundsMin.x || worldX > gc.worldBoundsMax.x) continue;
                        if (worldY < gc.worldBoundsMin.y || worldY > gc.worldBoundsMax.y) continue;

                        const auto& grp = wmoBatch.model->groups[g];
                        if (grp.cpuPositions.empty() || grp.cpuIndices.size() < 3) continue;

                        for (size_t i = 0; i + 2 < grp.cpuIndices.size(); i += 3) {
                            const uint16_t i0 = grp.cpuIndices[i + 0];
                            const uint16_t i1 = grp.cpuIndices[i + 1];
                            const uint16_t i2 = grp.cpuIndices[i + 2];
                            if (i0 >= grp.cpuPositions.size() ||
                                i1 >= grp.cpuPositions.size() ||
                                i2 >= grp.cpuPositions.size()) continue;

                            const glm::vec3& v0 = grp.cpuPositions[i0];
                            const glm::vec3& v1 = grp.cpuPositions[i1];
                            const glm::vec3& v2 = grp.cpuPositions[i2];

                            // 三角形局部 AABB 快速剔除：绝大多数三角形不在射线包围盒内
                            if (std::max({v0.x, v1.x, v2.x}) < rayMin.x || std::min({v0.x, v1.x, v2.x}) > rayMax.x ||
                                std::max({v0.y, v1.y, v2.y}) < rayMin.y || std::min({v0.y, v1.y, v2.y}) > rayMax.y ||
                                std::max({v0.z, v1.z, v2.z}) < rayMin.z || std::min({v0.z, v1.z, v2.z}) > rayMax.z) continue;

                            const glm::vec3 e1   = v1 - v0;
                            const glm::vec3 e2   = v2 - v0;
                            const glm::vec3 pvec = glm::cross(localDir, e2);
                            const float det = glm::dot(e1, pvec);
                            if (std::fabs(det) < 1e-8f) continue;
                            const float invDet = 1.0f / det;

                            const glm::vec3 tvec = localOrigin - v0;
                            const float uu = glm::dot(tvec, pvec) * invDet;
                            if (uu < 0.0f || uu > 1.0f) continue;

                            const glm::vec3 qvec = glm::cross(tvec, e1);
                            const float vv = glm::dot(localDir, qvec) * invDet;
                            if (vv < 0.0f || uu + vv > 1.0f) continue;

                            const float tHit = glm::dot(e2, qvec) * invDet;
                            if (tHit < 0.0f) continue;

                            const glm::vec3 hitLocal = localOrigin + localDir * tHit;
                            const glm::vec3 hitWorld = glm::vec3(inst.model * glm::vec4(hitLocal, 1.0f));

                            // 仅接受头部以下 (地板/楼梯) 的交点，忽略头顶屋顶与过高装饰面
                            if (hitWorld.z > currentFootZ + 1.8f) continue;
                            if (hitWorld.z < currentFootZ - 30.0f) continue;
                            bestZ = std::max(bestZ, hitWorld.z);
                        }
                    }
                }
            }
        }

        // 3. M2 装饰物碰撞网格射线检测：渡口木板 / 桥梁 / 巨石 (依赖 MD20 boundingVertices/Triangles)
        std::unordered_set<uint32_t> testedM2Ids; // 跨界 M2 同一 uniqueId 仅检测一次
        for (const auto& tile : tiles) {
            // 快速剪枝：跳过远离角色的 7 块 ADT，仅检测脚下附近的装饰物
            if (tile.tileX >= 0 &&
                DistanceToAdtTile(glm::vec3(worldX, worldY, 0.0f), tile.tileX, tile.tileY) > 25.0f) continue;
            for (const auto& doodadBatch : tile.doodadBatches) {
                if (!doodadBatch.model) continue;
                const auto& mdl = *doodadBatch.model;
                if (mdl.collisionVerts.empty() || mdl.collisionIndices.size() < 3) continue;

                for (const auto& inst : doodadBatch.instances) {
                    if (worldX < inst.worldBoundsMin.x || worldX > inst.worldBoundsMax.x) continue;
                    if (worldY < inst.worldBoundsMin.y || worldY > inst.worldBoundsMax.y) continue;
                    if (inst.uniqueId != 0 && !testedM2Ids.insert(inst.uniqueId).second) continue;

                    const glm::vec3 localOrigin = glm::vec3(inst.invModel * glm::vec4(worldRayOrigin, 1.0f));
                    const glm::vec3 localDir    = glm::vec3(inst.invModel * glm::vec4(0.0f, 0.0f, -1.0f, 0.0f));

                    // 竖直射线有效高度区间 [foot-30, foot+1.8] 变换到局部空间，用于三角形级快速剔除
                    const glm::vec3 localTop = glm::vec3(inst.invModel * glm::vec4(worldRayOrigin, 1.0f));
                    const glm::vec3 localBot = glm::vec3(inst.invModel * glm::vec4(worldRayBottom, 1.0f));
                    const glm::vec3 rayMin = glm::min(localTop, localBot) - glm::vec3(0.05f);
                    const glm::vec3 rayMax = glm::max(localTop, localBot) + glm::vec3(0.05f);

                    for (size_t i = 0; i + 2 < mdl.collisionIndices.size(); i += 3) {
                        const uint16_t i0 = mdl.collisionIndices[i + 0];
                        const uint16_t i1 = mdl.collisionIndices[i + 1];
                        const uint16_t i2 = mdl.collisionIndices[i + 2];
                        if (i0 >= mdl.collisionVerts.size() ||
                            i1 >= mdl.collisionVerts.size() ||
                            i2 >= mdl.collisionVerts.size()) continue;

                        const glm::vec3& v0 = mdl.collisionVerts[i0];
                        const glm::vec3& v1 = mdl.collisionVerts[i1];
                        const glm::vec3& v2 = mdl.collisionVerts[i2];

                        // 三角形局部 AABB 快速剔除：绝大多数三角形不在射线包围盒内
                        if (std::max({v0.x, v1.x, v2.x}) < rayMin.x || std::min({v0.x, v1.x, v2.x}) > rayMax.x ||
                            std::max({v0.y, v1.y, v2.y}) < rayMin.y || std::min({v0.y, v1.y, v2.y}) > rayMax.y ||
                            std::max({v0.z, v1.z, v2.z}) < rayMin.z || std::min({v0.z, v1.z, v2.z}) > rayMax.z) continue;

                        const glm::vec3 e1   = v1 - v0;
                        const glm::vec3 e2   = v2 - v0;
                        const glm::vec3 pvec = glm::cross(localDir, e2);
                        const float det = glm::dot(e1, pvec);
                        if (std::fabs(det) < 1e-8f) continue;
                        const float invDet = 1.0f / det;

                        const glm::vec3 tvec = localOrigin - v0;
                        const float uu = glm::dot(tvec, pvec) * invDet;
                        if (uu < 0.0f || uu > 1.0f) continue;

                        const glm::vec3 qvec = glm::cross(tvec, e1);
                        const float vv = glm::dot(localDir, qvec) * invDet;
                        if (vv < 0.0f || uu + vv > 1.0f) continue;

                        const float tHit = glm::dot(e2, qvec) * invDet;
                        if (tHit < 0.0f) continue;

                        const glm::vec3 hitLocal = localOrigin + localDir * tHit;
                        const glm::vec3 hitWorld = glm::vec3(inst.model * glm::vec4(hitLocal, 1.0f));

                        // 仅接受头部以下 (木板 / 桥面) 的交点，忽略上方枝叶与装饰
                        if (hitWorld.z > currentFootZ + 1.8f) continue;
                        if (hitWorld.z < currentFootZ - 30.0f) continue;
                        bestZ = std::max(bestZ, hitWorld.z);
                    }
                }
            }
        }

        if (bestZ <= -1e29f) return false;
        outGroundZ = bestZ;
        return true;
    }

    // 方向 H：行走模式水平防穿墙检测与贴墙滑动 (Collide-and-Slide)
    // 返回解析后的水平位置；moveDelta 为本帧期望的水平位移增量
    [[nodiscard]] static glm::vec2 ResolveHorizontalCollision(
        const std::vector<GpuAdtTile>& tiles,
        const glm::vec2& startXY,
        const glm::vec2& moveDelta,
        float footZ)
    {
        if (glm::length(moveDelta) < 1e-5f) return startXY;

        const float kRadius = 0.55f;

        // 内部辅助：从 origXY 沿 dirXY 发射水平射线，命中陡峭墙壁时返回 true 并输出水平法线
        auto castHorizontalRay = [&](const glm::vec2& origXY, const glm::vec2& dirXY,
                                     float maxDist, glm::vec2& outHitNormal) -> bool {
            outHitNormal = glm::vec2(0.0f);
            if (maxDist <= 1e-5f) return false;

            // 在膝盖与胸口两个高度各发射一条水平射线，避免台阶与矮墙漏检
            const float heights[2] = { footZ + 0.65f, footZ + 1.45f };
            const glm::vec3 worldDir(dirXY.x, dirXY.y, 0.0f);
            const float expand = maxDist + kRadius;

            bool      found     = false;
            float     bestDist  = maxDist;
            glm::vec2 bestNormal(0.0f);

            auto boxNear = [&](const glm::vec3& bmin, const glm::vec3& bmax) {
                return origXY.x >= bmin.x - expand && origXY.x <= bmax.x + expand &&
                       origXY.y >= bmin.y - expand && origXY.y <= bmax.y + expand;
            };

            // 对某个实例的碰撞网格逐三角形做 Möller–Trumbore 水平射线求交
            auto testMesh = [&](const glm::mat4& model, const glm::mat4& invModel,
                                const std::vector<glm::vec3>& verts,
                                const std::vector<uint16_t>& indices) {
                if (verts.empty() || indices.size() < 3) return;

                for (float rayZ : heights) {
                    const glm::vec3 worldOrigin(origXY.x, origXY.y, rayZ);
                    const glm::vec3 localOrigin = glm::vec3(invModel * glm::vec4(worldOrigin, 1.0f));
                    const glm::vec3 localDir    = glm::vec3(invModel * glm::vec4(worldDir, 0.0f));

                    // 水平射线段在局部空间的 AABB，用于三角形级快速剔除
                    const glm::vec3 localEnd = localOrigin + localDir * maxDist;
                    const glm::vec3 rayMin   = glm::min(localOrigin, localEnd) - glm::vec3(0.05f);
                    const glm::vec3 rayMax   = glm::max(localOrigin, localEnd) + glm::vec3(0.05f);

                    for (size_t i = 0; i + 2 < indices.size(); i += 3) {
                        const uint16_t i0 = indices[i + 0];
                        const uint16_t i1 = indices[i + 1];
                        const uint16_t i2 = indices[i + 2];
                        if (i0 >= verts.size() || i1 >= verts.size() || i2 >= verts.size()) continue;

                        const glm::vec3& v0 = verts[i0];
                        const glm::vec3& v1 = verts[i1];
                        const glm::vec3& v2 = verts[i2];

                        if (std::max({v0.x, v1.x, v2.x}) < rayMin.x || std::min({v0.x, v1.x, v2.x}) > rayMax.x ||
                            std::max({v0.y, v1.y, v2.y}) < rayMin.y || std::min({v0.y, v1.y, v2.y}) > rayMax.y ||
                            std::max({v0.z, v1.z, v2.z}) < rayMin.z || std::min({v0.z, v1.z, v2.z}) > rayMax.z) continue;

                        const glm::vec3 e1   = v1 - v0;
                        const glm::vec3 e2   = v2 - v0;
                        const glm::vec3 pvec = glm::cross(localDir, e2);
                        const float det = glm::dot(e1, pvec);
                        if (std::fabs(det) < 1e-8f) continue;
                        const float invDet = 1.0f / det;

                        const glm::vec3 tvec = localOrigin - v0;
                        const float uu = glm::dot(tvec, pvec) * invDet;
                        if (uu < 0.0f || uu > 1.0f) continue;

                        const glm::vec3 qvec = glm::cross(tvec, e1);
                        const float vv = glm::dot(localDir, qvec) * invDet;
                        if (vv < 0.0f || uu + vv > 1.0f) continue;

                        const float tHit = glm::dot(e2, qvec) * invDet;
                        if (tHit < 0.0f) continue;

                        // 仅陡峭墙面 (|n.z| < 0.55) 视为水平阻挡，斜坡与地板不参与
                        const glm::vec3 nWorld =
                            glm::normalize(glm::mat3(model) * glm::cross(v1 - v0, v2 - v0));
                        if (std::fabs(nWorld.z) >= 0.55f) continue;

                        const glm::vec3 hitLocal = localOrigin + localDir * tHit;
                        const glm::vec3 hitWorld = glm::vec3(model * glm::vec4(hitLocal, 1.0f));
                        const float worldDist =
                            glm::length(glm::vec2(hitWorld.x, hitWorld.y) - origXY);
                        if (worldDist > maxDist || worldDist >= bestDist) continue;

                        glm::vec2 n2(nWorld.x, nWorld.y);
                        const float nLen = glm::length(n2);
                        if (nLen < 1e-5f) continue;
                        n2 /= nLen;
                        if (glm::dot(n2, dirXY) > 0.0f) n2 = -n2; // 法线须朝向射线来向

                        bestDist   = worldDist;
                        bestNormal = n2;
                        found      = true;
                    }
                }
            };

            // 1. WMO 建筑：整栋包围盒 + 逐子房间包围盒快速筛除
            std::unordered_set<uint32_t> testedWmoIds;
            for (const auto& tile : tiles) {
                // 快速剪枝：跳过远离角色的 7 块 ADT，仅检测脚下附近的建筑
                if (tile.tileX >= 0 &&
                    DistanceToAdtTile(glm::vec3(origXY.x, origXY.y, 0.0f), tile.tileX, tile.tileY) > 25.0f) continue;
                for (const auto& wmoBatch : tile.wmoBatches) {
                    if (!wmoBatch.model) continue;
                    for (const auto& inst : wmoBatch.instances) {
                        if (!boxNear(inst.worldBoundsMin, inst.worldBoundsMax)) continue;
                        if (inst.uniqueId != 0 && !testedWmoIds.insert(inst.uniqueId).second) continue;
                        const bool hasGroups =
                            (inst.groupBounds.size() == wmoBatch.model->groups.size());
                        for (size_t g = 0; g < wmoBatch.model->groups.size(); ++g) {
                            if (hasGroups) {
                                const auto& gc = inst.groupBounds[g];
                                if (!boxNear(gc.worldBoundsMin, gc.worldBoundsMax)) continue;
                            }
                            const auto& grp = wmoBatch.model->groups[g];
                            testMesh(inst.model, inst.invModel, grp.cpuPositions, grp.cpuIndices);
                        }
                    }
                }
            }

            // 2. M2 装饰物：仅测试带碰撞网格的模型 (木板 / 桥梁 / 巨石)
            std::unordered_set<uint32_t> testedM2Ids;
            for (const auto& tile : tiles) {
                // 快速剪枝：跳过远离角色的 7 块 ADT，仅检测脚下附近的装饰物
                if (tile.tileX >= 0 &&
                    DistanceToAdtTile(glm::vec3(origXY.x, origXY.y, 0.0f), tile.tileX, tile.tileY) > 25.0f) continue;
                for (const auto& batch : tile.doodadBatches) {
                    if (!batch.model || batch.model->collisionIndices.empty()) continue;
                    for (const auto& inst : batch.instances) {
                        if (!boxNear(inst.worldBoundsMin, inst.worldBoundsMax)) continue;
                        if (inst.uniqueId != 0 && !testedM2Ids.insert(inst.uniqueId).second) continue;
                        testMesh(inst.model, inst.invModel,
                                 batch.model->collisionVerts, batch.model->collisionIndices);
                    }
                }
            }

            if (!found) return false;
            outHitNormal = bestNormal;
            return true;
        };

        // 最多 2 次贴墙滑动迭代；第 2 次仍受阻即判定卡墙角并保持原地
        glm::vec2 remDelta = moveDelta;
        for (int iter = 0; iter < 2; ++iter) {
            const float stepLen = glm::length(remDelta);
            if (stepLen < 1e-5f) break;

            const glm::vec2 dir = remDelta / stepLen;
            const float angles[3] = { 0.0f, 0.5f, -0.5f };
            glm::vec2 hitNormal(0.0f);
            bool blocked = false;
            for (float a : angles) {
                const float cs = std::cos(a);
                const float sn = std::sin(a);
                const glm::vec2 rayDir(dir.x * cs - dir.y * sn, dir.x * sn + dir.y * cs);
                if (castHorizontalRay(startXY, rayDir, stepLen + kRadius, hitNormal)) {
                    blocked = true;
                    break;
                }
            }

            if (!blocked) return startXY + remDelta;

            if (iter == 0) {
                // 将剩余位移投影到墙面切线，沿墙滑行
                remDelta -= glm::dot(remDelta, hitNormal) * hitNormal;
            } else {
                return startXY;
            }
        }

        return startXY + remDelta;
    }

private:
    // 解析 MH2O 水体：还原每层子格水面高度，按河床高度计算精确水深，生成 Pass 4 几何
    static void ParseMh2oChunk(
        std::span<const uint8_t> mh2oChunkData,
        const std::vector<TerrainVertex>& allVerts,
        GpuAdtTile& tile,
        std::vector<TerrainVertex>& liquidVerts,
        std::vector<uint16_t>& liquidIndices)
    {
        if (mh2oChunkData.size() < 256 * sizeof(Mh2oChunkHeader)) return;

        const size_t chunkCount = std::min<size_t>(256, tile.chunks.size());
        for (size_t chunkIdx = 0; chunkIdx < chunkCount; ++chunkIdx) {
            Mh2oChunkHeader hdr{};
            std::memcpy(&hdr, mh2oChunkData.data() + chunkIdx * sizeof(Mh2oChunkHeader), sizeof(Mh2oChunkHeader));
            if (hdr.layerCount == 0) continue;
            if (hdr.ofsInstances + hdr.layerCount * sizeof(Mh2oInstance) > mh2oChunkData.size()) continue;

            // 该 MCNK 的 145 个地形顶点必须已存在于合并顶点数组中
            if ((chunkIdx + 1) * 145 > allVerts.size()) continue;

            for (uint32_t layer = 0; layer < hdr.layerCount; ++layer) {
                Mh2oInstance inst{};
                std::memcpy(&inst,
                            mh2oChunkData.data() + hdr.ofsInstances + layer * sizeof(Mh2oInstance),
                            sizeof(Mh2oInstance));

                if (inst.xOffset >= 8 || inst.yOffset >= 8) continue;
                if (inst.width < 1 || inst.height < 1) continue;
                if (inst.xOffset + inst.width > 8 || inst.yOffset + inst.height > 8) continue;

                const uint32_t wVerts = static_cast<uint32_t>(inst.width) + 1u;
                const uint32_t hVerts = static_cast<uint32_t>(inst.height) + 1u;
                const uint32_t vCount = wVerts * hVerts;

                // 可选的高度表 (格式 2 = DepthOnly 无高度数据)
                const float* heightTable = nullptr;
                if (inst.ofsVertexData != 0 && inst.liquidVertexFormat != 2 &&
                    static_cast<size_t>(inst.ofsVertexData) + static_cast<size_t>(vCount) * sizeof(float) <= mh2oChunkData.size())
                {
                    heightTable = reinterpret_cast<const float*>(mh2oChunkData.data() + inst.ofsVertexData);
                }

                const uint32_t vertexBase = static_cast<uint32_t>(liquidVerts.size());
                glm::vec3 aabbMin(0.0f);
                glm::vec3 aabbMax(0.0f);
                bool hasBounds = false;

                // 1. 逐顶点：水面高度 + 从河床高度求真实几何水深 (存入 normal.z)
                for (uint32_t dy = 0; dy < hVerts; ++dy) {
                    for (uint32_t dx = 0; dx < wVerts; ++dx) {
                        const uint32_t r = static_cast<uint32_t>(inst.yOffset) + dy;
                        const uint32_t c = static_cast<uint32_t>(inst.xOffset) + dx;
                        const glm::vec3& bedPos = allVerts[chunkIdx * 145 + r * 17 + c].pos;

                        float waterZ = inst.minHeightLevel;
                        if (heightTable != nullptr) {
                            float h = heightTable[dy * wVerts + dx];
                            if (!std::isfinite(h) || (h == 0.0f && std::fabs(inst.minHeightLevel) > 1e-4f)) {
                                waterZ = inst.minHeightLevel;
                            } else {
                                waterZ = h;
                            }
                        }

                        const float waterDepth = std::max(0.0f, waterZ - bedPos.z);

                        TerrainVertex v{};
                        v.pos    = glm::vec3(bedPos.x, bedPos.y, waterZ);
                        v.normal = glm::vec3(0.0f, 0.0f, waterDepth); // 借用 normal.z 传递精确几何水深
                        v.uv     = glm::vec2(static_cast<float>(c), static_cast<float>(r));
                        liquidVerts.push_back(v);

                        if (!hasBounds) {
                            aabbMin  = v.pos;
                            aabbMax  = v.pos;
                            hasBounds = true;
                        } else {
                            aabbMin = glm::min(aabbMin, v.pos);
                            aabbMax = glm::max(aabbMax, v.pos);
                        }
                    }
                }

                // 2. 逐子格生成 2 个逆时针三角形 (tl, bl, br) 与 (tl, br, tr)
                const uint32_t firstIndex = static_cast<uint32_t>(liquidIndices.size());
                for (uint32_t dy = 0; dy < inst.height; ++dy) {
                    for (uint32_t dx = 0; dx < inst.width; ++dx) {
                        // 8x8 子格存在掩码：bit = 1 才生成水面，否则为空洞 (如码头缝隙)
                        if (inst.ofsExistsBitmap != 0) {
                            const uint32_t bitIdx  = dy * static_cast<uint32_t>(inst.width) + dx;
                            const size_t   byteOfs = static_cast<size_t>(inst.ofsExistsBitmap) + (bitIdx >> 3);
                            if (byteOfs >= mh2oChunkData.size()) continue;
                            const uint8_t maskByte = mh2oChunkData[byteOfs];
                            if ((maskByte & (1u << (bitIdx & 7u))) == 0) continue;
                        }

                        const uint16_t tl = static_cast<uint16_t>(vertexBase + dy * wVerts + dx);
                        const uint16_t tr = static_cast<uint16_t>(tl + 1);
                        const uint16_t bl = static_cast<uint16_t>(vertexBase + (dy + 1) * wVerts + dx);
                        const uint16_t br = static_cast<uint16_t>(bl + 1);
                        liquidIndices.insert(liquidIndices.end(), { tl, bl, br, tl, br, tr });
                    }
                }

                const uint32_t indexCount = static_cast<uint32_t>(liquidIndices.size()) - firstIndex;
                if (indexCount == 0) {
                    // 所有子格都被掩码剔除，回收刚追加的顶点
                    liquidVerts.resize(vertexBase);
                    continue;
                }

                // 水面近乎水平，强制至少 0.5 码厚度，避免视锥 AABB 退化
                if (hasBounds && (aabbMax.z - aabbMin.z) < 0.5f) {
                    aabbMax.z = aabbMin.z + 0.5f;
                }

                GpuLiquidChunk liquidChunk{};
                liquidChunk.firstIndex = firstIndex;
                liquidChunk.indexCount = indexCount;
                liquidChunk.liquidType = inst.liquidType;
                liquidChunk.aabbMin    = aabbMin;
                liquidChunk.aabbMax    = aabbMax;
                tile.liquidChunks.push_back(liquidChunk);
            }
        }

        if (liquidVerts.empty() || liquidIndices.empty()) return;

        std::printf("[ADT Loader] Parsed %zu MH2O liquid chunks (CPU)\n", tile.liquidChunks.size());
    }

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
        std::vector<TerrainVertex>& allVerts,
        std::vector<uint16_t>& allIndices,
        GpuAdtTile& tile,
        std::vector<CpuAdtData::ChunkAlphaCpu>& chunkAlphas)
    {
        McnkHeader hdr{};
        std::memcpy(&hdr, mcnkFullSpan.data() + 8, sizeof(McnkHeader));

        // 校验 MCVT (145 个 float 相对高度) 与 MCNR (145 * 3 个 int8 法线)
        if (hdr.ofsHeight == 0 || hdr.ofsHeight + 8 + 145 * sizeof(float) > mcnkFullSpan.size()) return;
        if (hdr.ofsNormal == 0 || hdr.ofsNormal + 8 + 145 * 3 > mcnkFullSpan.size()) return;

        const auto* heights = reinterpret_cast<const float*>(mcnkFullSpan.data() + hdr.ofsHeight + 8);
        const auto* normals = reinterpret_cast<const int8_t*>(mcnkFullSpan.data() + hdr.ofsNormal + 8);

        int32_t baseVert = static_cast<int32_t>(allVerts.size());
		// 解析 1~4 层 MCLY 纹理与 MCAL Alpha 遮罩
        GpuMcnkChunk gpuChunk{};
        gpuChunk.baseVertex = baseVert;
        gpuChunk.originPos  = hdr.pos;
        gpuChunk.holeMask   = static_cast<uint16_t>(hdr.holes & 0xFFFFu);
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
                gpuChunk.worldHeights[vIdx] = v.pos.z;
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
                    gpuChunk.worldHeights[vIdx] = v.pos.z;
                    ++vIdx;
                }
            }
        }
        // 计算当前 MCNK 145 个顶点的世界空间 AABB
        size_t aabbStart = static_cast<size_t>(baseVert);
        glm::vec3 vMin = allVerts[aabbStart].pos;
        glm::vec3 vMax = allVerts[aabbStart].pos;
        for (size_t i = aabbStart + 1; i < aabbStart + 145; ++i) {
            vMin = glm::min(vMin, allVerts[i].pos);
            vMax = glm::max(vMax, allVerts[i].pos);
        }
        gpuChunk.aabbMin = vMin;
        gpuChunk.aabbMax = vMax;

        // CPU 阶段仅记录 MTEX 下标；贴图对象创建延后到主线程分步上传
        CpuAdtData::ChunkAlphaCpu chunkAlpha{};
        for (int i = 0; i < 4; ++i) gpuChunk.layerTex[i] = 0;

        uint32_t layerCount = std::min<uint32_t>(hdr.nLayers, 4u);
        MclyEntry layers[4]{};
        if (layerCount > 0 && hdr.ofsLayer > 0 &&
            hdr.ofsLayer + 8 + layerCount * sizeof(MclyEntry) <= mcnkFullSpan.size())
        {
            std::memcpy(layers, mcnkFullSpan.data() + hdr.ofsLayer + 8, layerCount * sizeof(MclyEntry));
            for (uint32_t i = 0; i < layerCount; ++i) {
                chunkAlpha.texIds[i] = static_cast<int32_t>(layers[i].textureId);
            }
        }

        // 若层数 > 1，解码 MCAL 得到 64x64 RGBA8 遮罩数据 (贴图创建延后到主线程)
        std::vector<uint8_t> alphaRgba(64 * 64 * 4, 0);
        bool decodedAlpha = false;
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
                    decodedAlpha = true;
                }
            }
        }

        gpuChunk.alphaMapTex = 0;
        if (decodedAlpha) {
            chunkAlpha.hasAlpha = true;
            chunkAlpha.rgba     = std::move(alphaRgba);
        }

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

        chunkAlphas.push_back(std::move(chunkAlpha));
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
