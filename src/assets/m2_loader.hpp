#pragma once
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <string>
#include <string_view>
#include <vector>
#include <span>
#include <optional>
#include <array>
#include <algorithm>
#include <type_traits>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/matrix_transform.hpp>
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

// 3.3.5a MD20 (Version 264) 文件头核心段 (160 字节)
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
    glm::vec3 boundingBoxMin;         // 0xA0
    glm::vec3 boundingBoxMax;         // 0xAC
    float     boundingBoxRadius;      // 0xB8
    glm::vec3 vertexBoxMin;           // 0xBC
    glm::vec3 vertexBoxMax;           // 0xC8
    float     vertexBoxRadius;        // 0xD4
    M2Array   boundingTriangles;      // 0xD8
    M2Array   boundingVertices;       // 0xE0
    M2Array   boundingNormals;        // 0xE8
    M2Array   attachments;            // 0xF0
    M2Array   attachmentLookupTable;  // 0xF8
    M2Array   events;                 // 0x100
    M2Array   lights;                 // 0x108
    M2Array   cameras;                // 0x110
    M2Array   cameraLookupTable;      // 0x118
    M2Array   ribbonEmitters;         // 0x120
    M2Array   particleEmitters;       // 0x128: 指向 476 字节 M2ParticleDefV264 数组
};
static_assert(sizeof(M2HeaderV264) == 304, "M2HeaderV264 must be 304 bytes");

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

// 3.3.5a (v264) M2Track 二级嵌套动画轨头 (20 字节)
struct M2TrackBase {
    uint16_t interpolationType;
    int16_t  globalSequence;
    M2Array  timestamps; // 外层为 M2Array<M2Array>，内层为 uint32_t 毫秒时间戳
    M2Array  values;     // 外层为 M2Array<M2Array>，内层为 T 关键帧值
};
static_assert(sizeof(M2TrackBase) == 20, "M2TrackBase must be 20 bytes");

struct M2Sequence {
    uint16_t animId;
    uint16_t subAnimId;
    uint32_t duration; // 动作循环时长 (毫秒)
    uint8_t  pad[56];
};
static_assert(sizeof(M2Sequence) == 64, "M2Sequence must be 64 bytes");

struct M2CompQuat {
    int16_t x, y, z, w;
};
static_assert(sizeof(M2CompQuat) == 8, "M2CompQuat must be 8 bytes");

struct M2CompBone {
    int32_t     keyBoneId;
    uint32_t    flags;
    int16_t     parentBone;
    uint16_t    submeshId;
    uint32_t    boneNameCRC;
    M2TrackBase translation; // glm::vec3
    M2TrackBase rotation;    // M2CompQuat (8 bytes)
    M2TrackBase scale;       // glm::vec3
    glm::vec3   pivot;
};
static_assert(sizeof(M2CompBone) == 88, "M2CompBone must be 88 bytes");

struct M2ColorDef {
    M2TrackBase color; // glm::vec3
    M2TrackBase alpha; // uint16_t (0..32767)
};
static_assert(sizeof(M2ColorDef) == 40, "M2ColorDef must be 40 bytes");

struct M2TexTransformDef {
    M2TrackBase translation; // glm::vec3
    M2TrackBase rotation;    // glm::quat (16 bytes)
    M2TrackBase scaling;     // glm::vec3
};
static_assert(sizeof(M2TexTransformDef) == 60, "M2TexTransformDef must be 60 bytes");

// 粒子颜色/透明度/尺寸动画块：时间戳为 uint16_t，缺失插值类型与全局序列号
struct M2FakeAnimBlock {
    M2Array timestamps; // uint16_t 数组
    M2Array values;     // T 数组
};
static_assert(sizeof(M2FakeAnimBlock) == 16, "M2FakeAnimBlock must be 16 bytes");

struct M2ParticleDefV264 {
    int32_t         particleId;          // 0x00
    uint32_t        flags;               // 0x04
    glm::vec3       position;            // 0x08: 模型局部空间位置
    uint16_t        bone;                // 0x14
    uint16_t        texture;             // 0x16: 低 5 位为贴图索引
    M2Array         modelFileName;       // 0x18
    M2Array         childEmitterName;    // 0x20
    uint8_t         blendingType;        // 0x28: 2=AlphaBlend, 3/4=Additive
    uint8_t         emitterType;         // 0x29: 1=Plane, 2=Sphere
    uint16_t        particleColorIndex;  // 0x2A
    uint8_t         particleType;        // 0x2C
    uint8_t         headOrTail;          // 0x2D
    int16_t         texTileRotation;     // 0x2E
    uint16_t        texRows;             // 0x30
    uint16_t        texCols;             // 0x32
    M2TrackBase     emissionSpeed;       // 0x34 (float)
    M2TrackBase     speedVariation;      // 0x48 (float)
    M2TrackBase     verticalRange;       // 0x5C (float)
    M2TrackBase     horizontalRange;     // 0x70 (float)
    M2TrackBase     gravity;             // 0x84 (float)
    M2TrackBase     lifespan;            // 0x98 (float)
    float           lifespanVary;        // 0xAC
    M2TrackBase     emissionRate;        // 0xB0 (float)
    float           emissionRateVary;    // 0xC4
    M2TrackBase     emissionAreaLength;  // 0xC8 (float)
    M2TrackBase     emissionAreaWidth;   // 0xDC (float)
    M2TrackBase     zSource;             // 0xF0 (float)
    M2FakeAnimBlock colorTrack;          // 0x104 (glm::vec3, 0..255)
    M2FakeAnimBlock alphaTrack;          // 0x114 (uint16_t, 0..32767)
    M2FakeAnimBlock scaleTrack;          // 0x124 (glm::vec2)
    uint8_t         tailPadding[168];    // 0x134..0x1DC
};
static_assert(sizeof(M2ParticleDefV264) == 476, "M2ParticleDefV264 must be 476 bytes");
#pragma pack(pop)

// CPU 侧已解包的单条关键帧动画轨
template <typename T>
struct M2AnimTrack {
    uint16_t              interpType    = 1;
    int16_t               globalSeq     = -1;
    uint32_t              seqDurationMs = 3000;
    std::vector<uint32_t> times;
    std::vector<T>        values;

    // 方向 J：多动作序列 (0=Stand/animId 0, 1=Walk/animId 4, 2=Run/animId 5, 3=Jump/animId 38 或 37)
    struct ActionSeq {
        uint32_t              durationMs = 1000;
        std::vector<uint32_t> times;
        std::vector<T>        values;
    };
    std::array<ActionSeq, 6> actionSeqs;

    [[nodiscard]] bool HasAnimation() const noexcept {
        if (values.size() > 1 && times.size() >= values.size()) return true;
        for (const auto& seq : actionSeqs) {
            if (seq.values.size() > 1 && seq.times.size() >= seq.values.size()) return true;
        }
        return false;
    }

    [[nodiscard]] uint32_t ResolveLoopMs(const std::vector<uint32_t>& globalSeqs) const noexcept {
        if (globalSeq >= 0 && static_cast<size_t>(globalSeq) < globalSeqs.size() && globalSeqs[globalSeq] > 0) {
            return globalSeqs[globalSeq];
        }
        if (!times.empty() && times.back() > seqDurationMs) {
            return times.back();
        }
        return (seqDurationMs > 0) ? seqDurationMs : (!times.empty() ? std::max(times.back(), 1u) : 1000u);
    }

    [[nodiscard]] T SampleLinear(float timeSec, const std::vector<uint32_t>& globalSeqs, const T& defVal) const noexcept {
        const size_t n = std::min(times.size(), values.size());
        if (n == 0) return defVal;
        if (n == 1 || interpType == 0) return values.front();

        const uint32_t loopMs = std::max(ResolveLoopMs(globalSeqs), 1u);
        const uint32_t tMs    = static_cast<uint32_t>( std::fmod(std::max(timeSec, 0.0f) * 1000.0f, static_cast<float>(loopMs)) );

        if (tMs <= times.front()) return values.front();
        if (tMs >= times[n - 1])  return values[n - 1];

        for (size_t i = 0; i + 1 < n; ++i) {
            if (tMs >= times[i] && tMs <= times[i + 1]) {
                uint32_t span = times[i + 1] - times[i];
                if (span == 0) return values[i];
                float a = static_cast<float>(tMs - times[i]) / static_cast<float>(span);
                return glm::mix(values[i], values[i + 1], a);
            }
        }
        return values.back();
    }

    [[nodiscard]] glm::quat SampleQuat(float timeSec, const std::vector<uint32_t>& globalSeqs) const noexcept {
        const size_t n = std::min(times.size(), values.size());
        if (n == 0) return glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
        if (n == 1 || interpType == 0) return values.front();

        const uint32_t loopMs = std::max(ResolveLoopMs(globalSeqs), 1u);
        const uint32_t tMs    = static_cast<uint32_t>( std::fmod(std::max(timeSec, 0.0f) * 1000.0f, static_cast<float>(loopMs)) );

        if (tMs <= times.front()) return values.front();
        if (tMs >= times[n - 1])  return values[n - 1];

        for (size_t i = 0; i + 1 < n; ++i) {
            if (tMs >= times[i] && tMs <= times[i + 1]) {
                uint32_t span = times[i + 1] - times[i];
                if (span == 0) return values[i];
                float a = static_cast<float>(tMs - times[i]) / static_cast<float>(span);
                return glm::normalize(glm::slerp(values[i], values[i + 1], a));
            }
        }
        return values.back();
    }

    // 按指定动作槽位采样标量 / 向量轨；无对应动作数据时回退到默认 Stand 采样
    [[nodiscard]] T SampleLinearAction(float timeSec, int actionIdx,
                                       const std::vector<uint32_t>& globalSeqs,
                                       const T& defVal) const noexcept {
        if (globalSeq < 0 && actionIdx >= 0 && actionIdx < 6) {
            const auto& seq = actionSeqs[actionIdx];
            const size_t n = std::min(seq.times.size(), seq.values.size());
            if (n > 0) {
                if (n == 1 || interpType == 0) return seq.values.front();

                const uint32_t loopMs = std::max(seq.durationMs > 0 ? seq.durationMs : seq.times[n - 1], 1u);
                const uint32_t tMs = static_cast<uint32_t>(
                    std::fmod(std::max(timeSec, 0.0f) * 1000.0f, static_cast<float>(loopMs)));

                if (tMs <= seq.times.front()) return seq.values.front();
                if (tMs >= seq.times[n - 1])  return seq.values[n - 1];

                for (size_t i = 0; i + 1 < n; ++i) {
                    if (tMs >= seq.times[i] && tMs <= seq.times[i + 1]) {
                        const uint32_t span = seq.times[i + 1] - seq.times[i];
                        if (span == 0) return seq.values[i];
                        const float a = static_cast<float>(tMs - seq.times[i]) / static_cast<float>(span);
                        return glm::mix(seq.values[i], seq.values[i + 1], a);
                    }
                }
                return seq.values.back();
            }
        }
        return SampleLinear(timeSec, globalSeqs, defVal);
    }

    // 按指定动作槽位采样四元数轨 (球面线性插值)；无对应动作数据时回退到默认 Stand 采样
    template <typename U = T>
        requires std::is_same_v<U, glm::quat>
    [[nodiscard]] glm::quat SampleQuatAction(float timeSec, int actionIdx,
                                             const std::vector<uint32_t>& globalSeqs) const noexcept {
        if (globalSeq < 0 && actionIdx >= 0 && actionIdx < 6) {
            const auto& seq = actionSeqs[actionIdx];
            const size_t n = std::min(seq.times.size(), seq.values.size());
            if (n > 0) {
                if (n == 1 || interpType == 0) return seq.values.front();

                const uint32_t loopMs = std::max(seq.durationMs > 0 ? seq.durationMs : seq.times[n - 1], 1u);
                const uint32_t tMs = static_cast<uint32_t>(
                    std::fmod(std::max(timeSec, 0.0f) * 1000.0f, static_cast<float>(loopMs)));

                if (tMs <= seq.times.front()) return seq.values.front();
                if (tMs >= seq.times[n - 1])  return seq.values[n - 1];

                for (size_t i = 0; i + 1 < n; ++i) {
                    if (tMs >= seq.times[i] && tMs <= seq.times[i + 1]) {
                        const uint32_t span = seq.times[i + 1] - seq.times[i];
                        if (span == 0) return seq.values[i];
                        const float a = static_cast<float>(tMs - seq.times[i]) / static_cast<float>(span);
                        return glm::normalize(glm::slerp(seq.values[i], seq.values[i + 1], a));
                    }
                }
                return seq.values.back();
            }
        }
        return SampleQuat(timeSec, globalSeqs);
    }
};

struct RuntimeM2Bone {
    int16_t                  parentBone = -1;
    glm::vec3                pivot{0.0f};
    M2AnimTrack<glm::vec3>   translation;
    M2AnimTrack<glm::quat>   rotation;
    M2AnimTrack<glm::vec3>   scale;
};

struct RuntimeM2Color {
    M2AnimTrack<glm::vec3> color;
    M2AnimTrack<float>     alpha;
};

// CPU 侧已解包的粒子发射器 (屋顶炊烟 / 火把火苗)
struct RuntimeM2ParticleEmitter {
    glm::vec3 localPos{0.0f};
    uint16_t  boneIndex    = 0;
    GLuint    textureId    = 0;
    uint8_t   blendingType = 2;
    uint8_t   emitterType  = 1;
    uint16_t  texRows      = 1;
    uint16_t  texCols      = 1;
    float     speed        = 2.0f;
    float     speedVary    = 0.2f;
    float     vertRange    = 0.2f;
    float     horizRange   = 0.2f;
    float     gravity      = 0.0f;
    float     lifespan     = 2.5f;
    float     emissionRate = 10.0f;
    float     areaLength   = 0.5f;
    float     areaWidth    = 0.5f;
    glm::vec3 colorStart{1.0f}, colorMid{1.0f}, colorEnd{1.0f};
    float     alphaStart = 1.0f, alphaMid = 0.7f, alphaEnd = 0.0f;
    float     scaleStart = 0.5f, scaleMid = 1.2f, scaleEnd = 2.0f;
};

// 上传到 GPU 后的 M2 子网格渲染对象
struct GpuM2Submesh {
    uint32_t firstIndex       = 0;
    uint32_t indexCount       = 0;
    GLuint   textureId        = 0;
    uint16_t blendMode        = 0;  // 0 = Opaque, 1 = AlphaCutout, 2 = AlphaBlend, 10 = UV Scroll Waterfall
    uint16_t materialFlags    = 0;  // 0x01 = Unlit, 0x04 = TwoSided
    uint16_t geosetId         = 0;
    int16_t  colorIndex       = -1; // 指向 GpuM2Model::colorTracks
    int16_t  transWeightIndex = -1; // 指向 GpuM2Model::transWeightTracks
    int16_t  uvTransIndex     = -1; // 指向 GpuM2Model::uvTransTracks
};

struct GpuM2Model {
    rhi::GlBuffer                       vbo;
    rhi::GlBuffer                       ibo;
    std::vector<GpuM2Submesh>           submeshes;
    std::vector<glm::vec3>              collisionVerts;    // M2 局部空间碰撞网格顶点 (MD20.boundingVertices)
    std::vector<uint16_t>               collisionIndices;  // 碰撞网格三角形索引 (MD20.boundingTriangles)
    glm::vec3                           localBoundsMin{-1.0f};
    glm::vec3                           localBoundsMax{1.0f};
    std::vector<uint32_t>               globalSequences;
    std::vector<RuntimeM2Bone>          bones;
    bool                                hasAnimatedBones = false;
    std::vector<RuntimeM2Color>         colorTracks;
    std::vector<M2AnimTrack<float>>       transWeightTracks;
    std::vector<M2AnimTrack<glm::vec3>>   uvTransTracks;
    std::vector<RuntimeM2ParticleEmitter> particleEmitters;

    // 评估当前时刻整棵骨骼树的枢轴空间蒙皮矩阵，追加到 visibleBoneMatrices 并返回起始下标
    // actionIdx: 0=Stand, 1=Walk, 2=Run, 3=Jump, 4=SwimIdle, 5=Swim
    uint32_t EvaluateBones(float timeSec, std::vector<glm::mat4>& outBoneBuffer, int actionIdx = 0) const {
        if (!hasAnimatedBones || bones.empty()) return 0;
        const uint32_t baseIdx = static_cast<uint32_t>(outBoneBuffer.size());
        outBoneBuffer.resize(baseIdx + bones.size(), glm::mat4(1.0f));

        for (size_t i = 0; i < bones.size(); ++i) {
            const auto& b = bones[i];
            glm::vec3 tr = b.translation.SampleLinearAction(timeSec, actionIdx, globalSequences, glm::vec3(0.0f));
            glm::quat rq = b.rotation.SampleQuatAction(timeSec, actionIdx, globalSequences);
            glm::vec3 sc = b.scale.SampleLinearAction(timeSec, actionIdx, globalSequences, glm::vec3(1.0f));

            glm::mat4 localMat = glm::translate(glm::mat4(1.0f), b.pivot + tr);
            localMat *= glm::mat4_cast(rq);
            localMat  = glm::scale(localMat, sc);
            localMat  = glm::translate(localMat, -b.pivot);

            if (b.parentBone >= 0 && static_cast<size_t>(b.parentBone) < i) {
                outBoneBuffer[baseIdx + i] = outBoneBuffer[baseIdx + static_cast<size_t>(b.parentBone)] * localMat;
            } else {
                outBoneBuffer[baseIdx + i] = localMat;
            }
        }
        return baseIdx;
    }

    // 创建全局共享的 M2 顶点格式 VAO (含 Attr 3 骨骼权重与 Attr 4 骨骼索引)
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

        // Attr 3: boneWeights (4 x uint8 normalized -> vec4, offset 12)
        glEnableVertexArrayAttrib(vao, 3);
        glVertexArrayAttribFormat(vao, 3, 4, GL_UNSIGNED_BYTE, GL_TRUE, offsetof(M2Vertex, boneWeights));
        glVertexArrayAttribBinding(vao, 3, 0);

        // Attr 4: boneIndices (4 x uint8 integer -> uvec4, offset 16)
        glEnableVertexArrayAttrib(vao, 4);
        glVertexArrayAttribIFormat(vao, 4, 4, GL_UNSIGNED_BYTE, offsetof(M2Vertex, boneIndices));
        glVertexArrayAttribBinding(vao, 4, 0);

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

        if (normPath.ends_with(".mdx") || normPath.ends_with(".mdl")) {
            normPath.replace(normPath.size() - 4, 4, ".m2");
        }

        std::string skinPath = normPath.substr(0, normPath.size() - 3) + "00.skin";

        auto m2BytesOpt = vfs.ReadFile(normPath);
        if (!m2BytesOpt) {
            std::fprintf(stderr, "  -> [M2 Reason] Missing .m2 in VFS: %s\n", normPath.c_str());
            return std::nullopt;
        }

        std::span<const uint8_t> m2Bytes(*m2BytesOpt);

        // 纯粒子 M2 (屋顶炊烟/火苗) 无外部 .skin，仅在存在实体网格时才延迟读取
        std::vector<uint8_t>     skinBytesStorage;
        std::span<const uint8_t> skinBytes{};

        if (m2Bytes.size() >= sizeof(M2HeaderV264) + 8 && std::memcmp(m2Bytes.data(), "MD21", 4) == 0) {
            m2Bytes = m2Bytes.subspan(8);
        }

        if (m2Bytes.size() < sizeof(M2HeaderV264)) {
            std::fprintf(stderr, "  -> [M2 Reason] File too small: %s\n", normPath.c_str());
            return std::nullopt;
        }

        M2HeaderV264 m2Hdr{};
        std::memcpy(&m2Hdr, m2Bytes.data(), sizeof(M2HeaderV264));
        if (std::memcmp(m2Hdr.magic, "MD20", 4) != 0) {
            std::fprintf(stderr, "  -> [M2 Reason] Bad magic in: %s\n", normPath.c_str());
            return std::nullopt;
        }
        const bool hasMesh      = (m2Hdr.vertices.count > 0);
        const bool hasParticles = (m2Hdr.particleEmitters.count > 0);
        if (!hasMesh && !hasParticles) {
            std::fprintf(stderr, "  -> [M2 Reason] 0 mesh vertices and 0 particle emitters: %s\n", normPath.c_str());
            return std::nullopt;
        }

        // 纯粒子 M2 不读取 .skin，直接跳过 VBO/IBO 与 submesh 构建流程
        if (hasMesh) {
            auto skinBytesOpt = vfs.ReadFile(skinPath);
            if (!skinBytesOpt) {
                std::fprintf(stderr, "  -> [M2 Reason] Missing 00.skin in VFS: %s\n", skinPath.c_str());
                return std::nullopt;
            }
            skinBytesStorage = std::move(*skinBytesOpt);
            skinBytes        = skinBytesStorage;
            if (skinBytes.size() < sizeof(M2SkinHeader)) {
                std::fprintf(stderr, "  -> [M2 Reason] Skin file too small: %s\n", skinPath.c_str());
                return std::nullopt;
            }
        }

        M2SkinHeader skinHdr{};
        if (hasMesh) {
            std::memcpy(&skinHdr, skinBytes.data(), sizeof(M2SkinHeader));
            if (std::memcmp(skinHdr.magic, "SKIN", 4) != 0) {
                std::fprintf(stderr, "  -> [M2 Reason] Bad SKIN magic in: %s\n", skinPath.c_str());
                return std::nullopt;
            }
        }

        auto getSpan = []<typename T>(std::span<const uint8_t> blob, M2Array arr) -> std::span<const T> {
            if (arr.count == 0 || static_cast<size_t>(arr.offset) + static_cast<size_t>(arr.count) * sizeof(T) > blob.size()) {
                return {};
            }
            return std::span<const T>(reinterpret_cast<const T*>(blob.data() + arr.offset), arr.count);
        };

        auto m2Verts     = getSpan.template operator()<M2Vertex>(m2Bytes, m2Hdr.vertices);
        auto m2BoundVerts = getSpan.template operator()<glm::vec3>(m2Bytes, m2Hdr.boundingVertices);
        auto m2BoundTris  = getSpan.template operator()<uint16_t>(m2Bytes, m2Hdr.boundingTriangles);
        auto m2TexDefs   = getSpan.template operator()<M2TextureDef>(m2Bytes, m2Hdr.textures);
        auto m2Mats      = getSpan.template operator()<M2Material>(m2Bytes, m2Hdr.renderFlags);
        auto m2TexLut    = getSpan.template operator()<uint16_t>(m2Bytes, m2Hdr.texLookupTable);
        auto transLut    = getSpan.template operator()<uint16_t>(m2Bytes, m2Hdr.transparencyLookupTable);
        auto uvAnimLut   = getSpan.template operator()<uint16_t>(m2Bytes, m2Hdr.texTransformLookupTable);
        auto globalSeqs  = getSpan.template operator()<uint32_t>(m2Bytes, m2Hdr.globalSequences);
        auto sequences   = getSpan.template operator()<M2Sequence>(m2Bytes, m2Hdr.animations);
        auto compBones   = getSpan.template operator()<M2CompBone>(m2Bytes, m2Hdr.bones);
        auto colorDefs   = getSpan.template operator()<M2ColorDef>(m2Bytes, m2Hdr.colors);
        auto transDefs   = getSpan.template operator()<M2TrackBase>(m2Bytes, m2Hdr.transparency);
        auto uvTransDefs = getSpan.template operator()<M2TexTransformDef>(m2Bytes, m2Hdr.textureTransforms);

        auto skinVertLut   = getSpan.template operator()<uint16_t>(skinBytes, skinHdr.vertices);
        auto skinIndices   = getSpan.template operator()<uint16_t>(skinBytes, skinHdr.indices);
        auto skinSubmeshes = getSpan.template operator()<M2SkinSection>(skinBytes, skinHdr.submeshes);
        auto skinBatches   = getSpan.template operator()<M2Batch>(skinBytes, skinHdr.batches);

        if (hasMesh && (m2Verts.empty() || skinVertLut.empty() || skinIndices.empty() || skinSubmeshes.empty())) {
            return std::nullopt;
        }

        // 统计 M2Vertex.pos 得到模型局部空间 AABB；纯粒子模型使用保守包围盒
        glm::vec3 localMin(-4.0f, -4.0f, -1.0f);
        glm::vec3 localMax( 4.0f,  4.0f, 12.0f);
        if (hasMesh) {
            localMin = m2Verts.front().pos;
            localMax = m2Verts.front().pos;
            for (const auto& v : m2Verts) {
                localMin = glm::min(localMin, v.pos);
                localMax = glm::max(localMax, v.pos);
            }
            localMax = localMin + glm::max(localMax - localMin, glm::vec3(0.05f));
        }

        // 校验顶点骨骼索引不越界，避免非法索引访问 SSBO
        std::vector<M2Vertex> safeVerts(m2Verts.begin(), m2Verts.end());
        const uint8_t maxBoneIdx = compBones.empty() ? 0 : static_cast<uint8_t>(std::min<size_t>(compBones.size() - 1, 255));
        for (auto& v : safeVerts) {
            for (int k = 0; k < 4; ++k) {
                if (v.boneIndices[k] > maxBoneIdx) {
                    v.boneIndices[k] = 0;
                    v.boneWeights[k] = 0;
                }
            }
        }

        // 加载 M2 引用的所有 BLP 贴图
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

        std::vector<uint16_t> globalIndices(skinIndices.size());
        for (size_t i = 0; i < skinIndices.size(); ++i) {
            uint16_t localIdx = skinIndices[i];
            globalIndices[i] = (localIdx < skinVertLut.size()) ? skinVertLut[localIdx] : 0;
        }

        GpuM2Model gpuModel;
        if (hasMesh) {
            gpuModel.vbo = rhi::GlBuffer(safeVerts.size() * sizeof(M2Vertex), safeVerts.data(), 0);
            gpuModel.ibo = rhi::GlBuffer(globalIndices.size() * sizeof(uint16_t), globalIndices.data(), 0);
        }
        gpuModel.localBoundsMin = localMin;
        gpuModel.localBoundsMax = localMax;
        gpuModel.globalSequences.assign(globalSeqs.begin(), globalSeqs.end());

        // 碰撞网格：仅使用 MD20 内嵌的 boundingVertices / boundingTriangles (渡口木板 / 桥梁 / 巨石)。
        // 不再用实体网格兜底，避免把花草 / 蜡烛 / 书本等无碰撞模型也变成高模碰撞体。
        gpuModel.collisionVerts.assign(m2BoundVerts.begin(), m2BoundVerts.end());
        gpuModel.collisionIndices.assign(m2BoundTris.begin(), m2BoundTris.end());

        // 解析骨骼、颜色、透明度与 UV 变换关键帧轨道
        gpuModel.bones.reserve(compBones.size());
        for (const auto& cb : compBones) {
            RuntimeM2Bone rb{};
            rb.parentBone  = cb.parentBone;
            rb.pivot       = cb.pivot;
            rb.translation = ExtractTrack<glm::vec3, glm::vec3>(m2Bytes, cb.translation, sequences,
                                [](const glm::vec3& v) { return v; });
            rb.rotation    = ExtractTrack<M2CompQuat, glm::quat>(m2Bytes, cb.rotation, sequences, DecompressQuat);
            rb.scale       = ExtractTrack<glm::vec3, glm::vec3>(m2Bytes, cb.scale, sequences,
                                [](const glm::vec3& v) { return v; });

            if (rb.translation.HasAnimation() || rb.rotation.HasAnimation() || rb.scale.HasAnimation()) {
                gpuModel.hasAnimatedBones = true;
            }
            gpuModel.bones.push_back(std::move(rb));
        }

        gpuModel.colorTracks.reserve(colorDefs.size());
        for (const auto& cd : colorDefs) {
            RuntimeM2Color rc{};
            rc.color = ExtractTrack<glm::vec3, glm::vec3>(m2Bytes, cd.color, sequences,
                          [](const glm::vec3& v) { return glm::clamp(v, glm::vec3(0.0f), glm::vec3(2.0f)); });
            rc.alpha = ExtractTrack<uint16_t, float>(m2Bytes, cd.alpha, sequences,
                          [](uint16_t a) { return glm::clamp(static_cast<float>(a) / 32767.0f, 0.0f, 1.0f); });
            gpuModel.colorTracks.push_back(std::move(rc));
        }

        gpuModel.transWeightTracks.reserve(transDefs.size());
        for (const auto& td : transDefs) {
            gpuModel.transWeightTracks.push_back(
                ExtractTrack<uint16_t, float>(m2Bytes, td, sequences,
                    [](uint16_t w) { return glm::clamp(static_cast<float>(w) / 32767.0f, 0.0f, 1.0f); })
            );
        }

        gpuModel.uvTransTracks.reserve(uvTransDefs.size());
        for (const auto& ut : uvTransDefs) {
            gpuModel.uvTransTracks.push_back(
                ExtractTrack<glm::vec3, glm::vec3>(m2Bytes, ut.translation, sequences,
                    [](const glm::vec3& v) { return v; })
            );
        }

        // 将 M2Batch 材质与 M2SkinSection 子网格配对组装
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
                    sub.blendMode     = m2Mats[batch.materialIndex].blendMode;
                    sub.materialFlags = m2Mats[batch.materialIndex].flags;
                }
                if (batch.colorIndex >= 0 && static_cast<size_t>(batch.colorIndex) < gpuModel.colorTracks.size()) {
                    sub.colorIndex = batch.colorIndex;
                }
                if (batch.textureWeightComboIndex < transLut.size()) {
                    uint16_t twIdx = transLut[batch.textureWeightComboIndex];
                    if (twIdx < gpuModel.transWeightTracks.size()) {
                        sub.transWeightIndex = static_cast<int16_t>(twIdx);
                    }
                }
                if (m2Hdr.textureTransforms.count > 0 && batch.textureTransformComboIndex < uvAnimLut.size()) {
                    uint16_t uvIdx = uvAnimLut[batch.textureTransformComboIndex];
                    if (uvIdx < gpuModel.uvTransTracks.size()) {
                        sub.uvTransIndex = static_cast<int16_t>(uvIdx);
                        sub.blendMode    = 10; // 启用 UV 动画的瀑布/特效层
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

        // ==================== 解析 3.3.5a 粒子发射器 (476 字节 M2ParticleDefV264) ====================
        // 光柱 / 光束类模型 (lightshaft / lightray) 的叠加粒子会铺满整个屏幕，直接跳过
        const bool isLightBeamModel = normPath.find("lightshaft") != std::string::npos ||
                                      normPath.find("lightray")  != std::string::npos;

        auto particleDefs = getSpan.template operator()<M2ParticleDefV264>(m2Bytes, m2Hdr.particleEmitters);
        if (!particleDefs.empty() && !isLightBeamModel) {
            // 浮点轨统一取前 4 帧均值，抑制异常单帧；无关键帧才回落到 fallback，
            // 真实值 0 必须保留，用于识别 Stand 待机状态下的休眠发射器
            auto sampleTrackFloat = [&](const M2TrackBase& track, float fallback) -> float {
                auto anim = ExtractTrack<float, float>(m2Bytes, track, sequences, [](float v) { return v; });
                if (anim.values.empty()) return fallback;
                const size_t n = std::min<size_t>(anim.values.size(), 4);
                float sum = 0.0f;
                for (size_t k = 0; k < n; ++k) sum += anim.values[k];
                return sum / static_cast<float>(n);
            };
            // M2FakeAnimBlock 的值数组存在两种落盘布局：
            //   1) 一维直接存储: values 直接指向 DiskT[kCount]
            //   2) 单元素二次间接: values 指向 1 个 M2Array，该内层 M2Array 再指向 DiskT[kCount]
            // 统一按 DiskT 读出为连续的 std::vector<DiskT>，并做严格越界校验
            auto extractFakeValues = [&]<typename DiskT>(const M2FakeAnimBlock& blk) -> std::vector<DiskT> {
                if (blk.values.count == 0) return std::vector<DiskT>{};

                if (blk.timestamps.count == 1 && blk.values.count == 1 &&
                    static_cast<size_t>(blk.values.offset) + sizeof(M2Array) <= m2Bytes.size()) {
                    M2Array innerVal{};
                    std::memcpy(&innerVal, m2Bytes.data() + blk.values.offset, sizeof(M2Array));
                    if (innerVal.count > 0 && innerVal.count <= 32 &&
                        static_cast<size_t>(innerVal.offset) + static_cast<size_t>(innerVal.count) * sizeof(DiskT) <= m2Bytes.size()) {
                        const auto* p = reinterpret_cast<const DiskT*>(m2Bytes.data() + innerVal.offset);
                        return std::vector<DiskT>(p, p + innerVal.count);
                    }
                }

                auto spanVals = getSpan.template operator()<DiskT>(m2Bytes, blk.values);
                return std::vector<DiskT>(spanVals.begin(), spanVals.end());
            };
            auto applyColorTriple = [](const std::vector<glm::vec3>& vals, RuntimeM2ParticleEmitter& em) {
                if (vals.empty()) return;
                glm::vec3 src[3] = {vals.front(), vals[vals.size() / 2], vals.back()};
                for (auto& c : src) {
                    if (c.r > 2.0f || c.g > 2.0f || c.b > 2.0f) c /= 255.0f;
                    c = glm::clamp(c, glm::vec3(0.0f), glm::vec3(1.0f));
                }
                em.colorStart = src[0];
                em.colorMid   = src[1];
                em.colorEnd   = src[2];
            };
            auto applyAlphaTriple = [](const std::vector<float>& vals, RuntimeM2ParticleEmitter& em) {
                if (vals.empty()) return;
                em.alphaStart = glm::clamp(vals.front(), 0.0f, 1.0f);
                em.alphaMid   = glm::clamp(vals[vals.size() / 2], 0.0f, 1.0f);
                em.alphaEnd   = glm::clamp(vals.back(), 0.0f, 1.0f);
            };
            auto applyScaleTriple = [](const std::vector<glm::vec2>& vals, RuntimeM2ParticleEmitter& em) {
                if (vals.empty()) return;
                em.scaleStart = vals.front().x;
                em.scaleMid   = vals[vals.size() / 2].x;
                em.scaleEnd   = vals.back().x;
            };

            gpuModel.particleEmitters.reserve(particleDefs.size());
            for (const auto& pDef : particleDefs) {
                RuntimeM2ParticleEmitter em{};
                em.localPos     = pDef.position;
                em.boneIndex    = pDef.bone;
                em.blendingType = pDef.blendingType;
                em.emitterType  = pDef.emitterType;
                em.texRows      = (pDef.texRows > 0) ? pDef.texRows : 1;
                em.texCols      = (pDef.texCols > 0) ? pDef.texCols : 1;

                const uint16_t texIdx = static_cast<uint16_t>(pDef.texture & 0x1Fu);
                em.textureId = (texIdx < loadedTexHandles.size()) ? loadedTexHandles[texIdx] : fallbackWhiteTex;

                // Stand 待机状态下 emissionRate 为 0 的发射器视为休眠，直接跳过 (避免静止马车冒烟)
                em.emissionRate = sampleTrackFloat(pDef.emissionRate, 0.0f);
                if (em.emissionRate <= 0.05f) continue;

                em.speed      = sampleTrackFloat(pDef.emissionSpeed, 0.0f); // 保留灯笼光晕的 0 速度
                em.speedVary  = sampleTrackFloat(pDef.speedVariation, 0.2f);
                em.vertRange  = sampleTrackFloat(pDef.verticalRange,  0.2f);
                em.horizRange = sampleTrackFloat(pDef.horizontalRange, 0.2f);
                em.gravity    = sampleTrackFloat(pDef.gravity,        0.0f);
                em.lifespan   = std::max(sampleTrackFloat(pDef.lifespan, 2.0f), 0.25f);
                em.areaLength = sampleTrackFloat(pDef.emissionAreaLength, 0.5f);
                em.areaWidth  = sampleTrackFloat(pDef.emissionAreaWidth,  0.5f);

                applyColorTriple(
                    extractFakeValues.template operator()<glm::vec3>(pDef.colorTrack), em);

                const auto alphaRaw = extractFakeValues.template operator()<uint16_t>(pDef.alphaTrack);
                uint16_t maxA = 0;
                for (uint16_t a : alphaRaw) maxA = std::max(maxA, a);
                // 3.3.5a 磁盘 alpha 存在 0..255 与 0..32767 两种量纲，按实测最大值自适应选择标度
                const float alphaScale = (maxA > 0 && maxA <= 255) ? (1.0f / 255.0f) : (1.0f / 32767.0f);
                std::vector<float> alphaNorm;
                alphaNorm.reserve(alphaRaw.size());
                for (uint16_t a : alphaRaw) {
                    alphaNorm.push_back(std::clamp(static_cast<float>(a) * alphaScale, 0.0f, 1.0f));
                }
                applyAlphaTriple(alphaNorm, em);

                applyScaleTriple(
                    extractFakeValues.template operator()<glm::vec2>(pDef.scaleTrack), em);

                // 长寿命烟雾 (housesmoke.m2 等)：略微提升上升速度、中段透明度与发射率，增强可见度
                if (em.blendingType <= 2 && em.lifespan >= 5.0f) {
                    em.speed        = std::max(em.speed, 1.2f);
                    em.alphaMid     = std::max(em.alphaMid, 0.42f);
                    em.emissionRate = std::max(em.emissionRate, 1.5f);
                }

                // 可见性兜底：动画数据缺失/全零时给出柔和且合理的三段默认曲线
                if (std::max({em.alphaStart, em.alphaMid, em.alphaEnd}) < 0.05f) {
                    em.alphaStart = 0.2f;
                    em.alphaMid   = 0.65f;
                    em.alphaEnd   = 0.0f;
                }
                if (std::max({em.scaleStart, em.scaleMid, em.scaleEnd}) <= 1e-4f) {
                    em.scaleStart = 0.15f;
                    em.scaleMid   = 0.30f;
                    em.scaleEnd   = 0.45f;
                }
                if (glm::length(em.colorStart) + glm::length(em.colorMid) < 0.05f) {
                    em.colorStart = glm::vec3(0.75f);
                    em.colorMid   = glm::vec3(0.75f);
                    em.colorEnd   = glm::vec3(0.75f);
                }

                gpuModel.particleEmitters.push_back(em);
            }

            // 含粒子模型扩大保守包围盒，防止 housesmoke 等纯特效层被视锥裁剪
            if (!gpuModel.particleEmitters.empty()) {
                gpuModel.localBoundsMin = glm::min(gpuModel.localBoundsMin, glm::vec3(-6.0f, -6.0f, -2.0f));
                gpuModel.localBoundsMax = glm::max(gpuModel.localBoundsMax, glm::vec3( 6.0f,  6.0f, 18.0f));
            }

            // 静默加载：避免主线程分步上传 M2 时被 Windows 控制台同步输出阻塞
        }

        // particleemitters/ 目录下的模型自身仅含编辑器占位白盒网格：
        // 仅剔除不透明 (blendMode == 0) 占位面，保留 fountainparticles.m2 的半透明水流柱网格
        if (normPath.find("particleemitters/") != std::string::npos &&
            !gpuModel.particleEmitters.empty()) {
            std::erase_if(gpuModel.submeshes, [](const GpuM2Submesh& s) { return s.blendMode == 0; });
        }

        return gpuModel;
    }

private:
    static glm::quat DecompressQuat(const M2CompQuat& cq) noexcept {
        auto conv = [](int16_t v) -> float {
            return static_cast<float>(v < 0 ? (static_cast<int32_t>(v) + 32768)
                                            : (static_cast<int32_t>(v) - 32767)) / 32767.0f;
        };
        glm::quat q(conv(cq.w), conv(cq.x), conv(cq.y), conv(cq.z));
        float len2 = glm::dot(q, q);
        return (len2 > 1e-8f) ? (q * (1.0f / std::sqrt(len2))) : glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
    }

    template <typename DiskT, typename OutT, typename ConvFn>
    static M2AnimTrack<OutT> ExtractTrack(
        std::span<const uint8_t> m2Bytes,
        const M2TrackBase& raw,
        std::span<const M2Sequence> sequences,
        ConvFn conv)
    {
        M2AnimTrack<OutT> out{};
        out.interpType = raw.interpolationType;
        out.globalSeq  = raw.globalSequence;

        if (raw.timestamps.count == 0 || raw.values.count == 0) return out;
        const size_t outerCount = std::min(raw.timestamps.count, raw.values.count);
        if (static_cast<size_t>(raw.timestamps.offset) + outerCount * sizeof(M2Array) > m2Bytes.size()) return out;
        if (static_cast<size_t>(raw.values.offset)     + outerCount * sizeof(M2Array) > m2Bytes.size()) return out;

        const auto* tsOuter  = reinterpret_cast<const M2Array*>(m2Bytes.data() + raw.timestamps.offset);
        const auto* valOuter = reinterpret_cast<const M2Array*>(m2Bytes.data() + raw.values.offset);

        // 仅在 Stand (animId == 0) 待机序列中挑选多关键帧轨道；
        // 严禁误选 Open/Destroy/Death 等一次性动作序列，避免木桶、吊桥等物件异常摆动
        size_t chosenSeq = 0;
        if (raw.globalSequence < 0 && !sequences.empty()) {
            const size_t seqLimit      = std::min(outerCount, sequences.size());
            size_t       firstStandSeq = seqLimit; // 首个 animId == 0 的序列
            size_t       multiStandSeq = seqLimit; // 首个含多关键帧的 Stand 序列
            for (size_t s = 0; s < seqLimit; ++s) {
                if (sequences[s].animId != 0) continue;
                if (firstStandSeq == seqLimit) firstStandSeq = s;
                if (tsOuter[s].count > 1 && valOuter[s].count > 1) {
                    multiStandSeq = s;
                    break;
                }
            }
            if (multiStandSeq != seqLimit)      chosenSeq = multiStandSeq;
            else if (firstStandSeq != seqLimit) chosenSeq = firstStandSeq;
            else                                chosenSeq = 0;
        } else {
            for (size_t s = 0; s < outerCount; ++s) {
                if (tsOuter[s].count > 1 && valOuter[s].count > 1) {
                    chosenSeq = s;
                    break;
                }
            }
        }

        if (chosenSeq < sequences.size() && sequences[chosenSeq].duration > 0) {
            out.seqDurationMs = sequences[chosenSeq].duration;
        }

        const M2Array tsArr  = tsOuter[chosenSeq];
        const M2Array valArr = valOuter[chosenSeq];
        const size_t  kCount = std::min(tsArr.count, valArr.count);
        if (kCount == 0) return out;

        if (static_cast<size_t>(tsArr.offset)  + kCount * sizeof(uint32_t) > m2Bytes.size()) return out;
        if (static_cast<size_t>(valArr.offset) + kCount * sizeof(DiskT)    > m2Bytes.size()) return out;

        const auto* tsPtr  = reinterpret_cast<const uint32_t*>(m2Bytes.data() + tsArr.offset);
        const auto* valPtr = reinterpret_cast<const DiskT*>(m2Bytes.data() + valArr.offset);

        out.times.assign(tsPtr, tsPtr + kCount);
        out.values.reserve(kCount);
        for (size_t k = 0; k < kCount; ++k) {
            out.values.push_back(conv(valPtr[k]));
        }

        // 额外抽取多动作骨骼序列：Stand(animId 0) / Walk(4) / Run(5) / Jump(38 或 37)
        // 以及 SwimIdle(41) / Swim(42)
        if (raw.globalSequence < 0 && !sequences.empty()) {
            auto fillAction = [&](size_t seqIdx, int actionSlot) {
                if (actionSlot < 0 || actionSlot >= 6 || seqIdx >= outerCount) return;

                const M2Array tsArrA  = tsOuter[seqIdx];
                const M2Array valArrA = valOuter[seqIdx];
                const size_t  kCountA = std::min(tsArrA.count, valArrA.count);
                if (kCountA == 0) return;
                if (static_cast<size_t>(tsArrA.offset)  + kCountA * sizeof(uint32_t) > m2Bytes.size()) return;
                if (static_cast<size_t>(valArrA.offset) + kCountA * sizeof(DiskT)    > m2Bytes.size()) return;

                auto& slot = out.actionSeqs[actionSlot];
                const auto* tsA  = reinterpret_cast<const uint32_t*>(m2Bytes.data() + tsArrA.offset);
                const auto* valA = reinterpret_cast<const DiskT*>(m2Bytes.data() + valArrA.offset);
                slot.times.assign(tsA, tsA + kCountA);
                slot.values.reserve(kCountA);
                for (size_t k = 0; k < kCountA; ++k) slot.values.push_back(conv(valA[k]));
                slot.durationMs = (seqIdx < sequences.size() && sequences[seqIdx].duration > 0)
                                ? sequences[seqIdx].duration
                                : slot.durationMs;
            };

            const size_t seqLimit = std::min(outerCount, sequences.size());
            size_t standSeq = seqLimit, walkSeq = seqLimit, runSeq = seqLimit, jumpSeq = seqLimit;
            size_t swimIdleSeq = seqLimit, swimSeq = seqLimit;
            for (size_t s = 0; s < seqLimit; ++s) {
                const uint16_t aid = sequences[s].animId;
                if (aid == 0 && standSeq == seqLimit)                       standSeq = s;
                else if (aid == 4 && walkSeq == seqLimit)                   walkSeq  = s;
                else if (aid == 5 && runSeq == seqLimit)                    runSeq   = s;
                else if ((aid == 38 || aid == 37) && jumpSeq == seqLimit)   jumpSeq  = s;
                else if (aid == 41 && swimIdleSeq == seqLimit)              swimIdleSeq = s;
                else if (aid == 42 && swimSeq == seqLimit)                  swimSeq  = s;
            }
            if (standSeq != seqLimit) fillAction(standSeq, 0);
            if (walkSeq  != seqLimit) fillAction(walkSeq,  1);
            if (runSeq   != seqLimit) fillAction(runSeq,   2);
            if (jumpSeq  != seqLimit) fillAction(jumpSeq,  3);
            if (swimIdleSeq != seqLimit)   fillAction(swimIdleSeq, 4);
            else if (standSeq != seqLimit) fillAction(standSeq,    4); // 无 SwimIdle (41) 时回退 Stand 踩水
            if (swimSeq  != seqLimit)      fillAction(swimSeq,     5);
        }

        return out;
    }
};

} // namespace coa::assets
