#include <SDL3/SDL.h>
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <cstdio>
#include <cmath>
#include <vector>
#include <array>
#include <algorithm>
#include <unordered_set>
#include "rhi/rhi_gl.hpp"
#include "vfs/vfs.hpp"
#include "assets/asset_manager.hpp"
#include "assets/m2_loader.hpp"
#include "assets/adt_loader.hpp"
#include "assets/dbc_loader.hpp"

void GLAPIENTRY GlDebugCallback(GLenum /*source*/, GLenum /*type*/, GLuint id, GLenum severity,
                                GLsizei /*length*/, const GLchar* message, const void* /*userParam*/) {
    if (severity == GL_DEBUG_SEVERITY_NOTIFICATION) return;
    std::fprintf(stderr, "[GL Debug] (ID:%u) %s\n", id, message);
}

glm::mat4 MakeInfiniteReverseZProj(float fovRadians, float aspect, float zNear) {
    float f = 1.0f / std::tan(fovRadians * 0.5f);
    glm::mat4 m(0.0f);
    m[0][0] = f / aspect;
    m[1][1] = f;
    m[2][3] = -1.0f;
    m[3][2] = zNear;
    return m;
}

struct alignas(16) FrameUniforms {
    glm::mat4 view;
    glm::mat4 proj;
    glm::mat4 viewProj;
    glm::vec4 cameraPos;          // xyz = 相机位置, w = 累计运行秒数
    glm::vec4 sunDirAndIntensity; // xyz = 太阳/月亮方向, w = 强度
    glm::vec4 sunColor;           // rgb = 太阳主光颜色
    glm::vec4 ambientSky;         // rgb = 天空环境光
    glm::vec4 ambientGround;      // rgb = 地面反射环境光
    glm::vec4 fogColorAndParams;  // rgb = 雾色(或水下滤镜色), w = 雾起始距离
    glm::vec4 skyTopAndDensity;   // rgb = 天顶色, w = 雾浓度系数
    glm::vec4 skyMid;             // rgb = 中空色
    glm::vec4 skyBand1;           // rgb = 中低空过渡色
    glm::vec4 skyHorizon;         // rgb = 地平线色
    glm::vec4 waterShallow;       // rgb = 动态浅水色
    glm::vec4 waterDeep;          // rgb = 动态深水色
};

// 单个公告板粒子的 GPU 实例数据 (48 字节，std430 紧凑排布，对应 kParticleVertShader)
struct alignas(16) GpuParticleQuad {
    glm::vec4 worldPosAndSize; // xyz = 世界坐标, w = 半径尺寸
    glm::vec4 color;           // rgba = 颜色与透明度
    glm::vec4 uvRect;          // xy = UV 起点, zw = UV 缩放 (子图格)
};
static_assert(sizeof(GpuParticleQuad) == 48, "GpuParticleQuad must be 48 bytes");

struct ParticleBatchCmd {
    GLuint   texId     = 0;
    uint8_t  blendType = 2;
    uint32_t firstQuad = 0;
    uint32_t quadCount = 0;
};

struct Frustum {
    glm::vec4 planes[6];

    static Frustum ExtractFromViewProj(const glm::mat4& vp) {
        // 转置列主序矩阵，使 m[0]..m[3] 直接对应 vp 的第 0..3 行向量 (r0, r1, r2, r3)
        const glm::mat4 m = glm::transpose(vp);
        Frustum f;
        f.planes[0] = m[3] + m[0]; // Left   : c_x >= -c_w
        f.planes[1] = m[3] - m[0]; // Right  : c_x <=  c_w
        f.planes[2] = m[3] + m[1]; // Bottom : c_y >= -c_w
        f.planes[3] = m[3] - m[1]; // Top    : c_y <=  c_w
        f.planes[4] = m[3] - m[2]; // Near   : c_z <=  c_w (Reverse-Z 近平面为 1.0)
        f.planes[5] = m[2];        // Far    : c_z >=  0.0 (Reverse-Z 无穷远平面为 0.0)

        for (int i = 0; i < 6; ++i) {
            float len = glm::length(glm::vec3(f.planes[i]));
            if (len > 1e-8f) {
                f.planes[i] /= len;
            }
        }
        return f;
    }

    bool IntersectsAABB(const glm::vec3& bMin, const glm::vec3& bMax) const {
        glm::vec3 center  = (bMin + bMax) * 0.5f;
        glm::vec3 extents = (bMax - bMin) * 0.5f;
        for (const auto& p : planes) {
            glm::vec3 n(p);
            glm::vec3 absN(std::fabs(p.x), std::fabs(p.y), std::fabs(p.z));
            float r = glm::dot(extents, absN);
            float d = glm::dot(n, center) + p.w;
            if (d < -r) return false;
        }
        return true;
    }
};

// 自适应距离剔除 + Reverse-Z 视锥体裁剪
inline bool IsInstanceVisible(const Frustum& frustum, const glm::vec3& camPos,
                              const glm::vec3& bmin, const glm::vec3& bmax,
                              const glm::vec3& center, float radius) {
    float maxDist = glm::clamp(80.0f + radius * 35.0f, 100.0f, 950.0f);
    glm::vec3 d = center - camPos;
    if (glm::dot(d, d) > maxDist * maxDist) return false;
    return frustum.IntersectsAABB(bmin, bmax);
}

inline bool IsCameraUnderwater(const glm::vec3& camPos, const std::vector<coa::assets::GpuAdtTile>& tiles) {
    for (const auto& tile : tiles) {
        for (const auto& lc : tile.liquidChunks) {
            const float surfaceZ = std::max(lc.aabbMin.z, lc.aabbMax.z - 0.45f);
            if (camPos.x >= lc.aabbMin.x && camPos.x <= lc.aabbMax.x &&
                camPos.y >= lc.aabbMin.y && camPos.y <= lc.aabbMax.y &&
                camPos.z < surfaceZ && camPos.z > lc.aabbMin.z - 35.0f) {
                return true;
            }
        }
    }
    return false;
}

// 全屏天空渐变：利用逆视图旋转与投影对角线重建世界空间视线方向
constexpr std::string_view kSkyVertShader = R"GLSL(
#version 460 core
void main() {
    vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
)GLSL";

constexpr std::string_view kSkyFragShader = R"GLSL(
#version 460 core
layout(location = 0) uniform vec2 uScreenSize;

layout(std140, binding = 0) uniform FrameBlock {
    mat4 uView;
    mat4 uProj;
    mat4 uViewProj;
    vec4 uCameraPos;
    vec4 uSunDirAndIntensity;
    vec4 uSunColor;
    vec4 uAmbientSky;
    vec4 uAmbientGround;
    vec4 uFogColorAndParams;
    vec4 uSkyTopAndDensity;
    vec4 uSkyMid;
    vec4 uSkyBand1;
    vec4 uSkyHorizon;
    vec4 uWaterShallow;
    vec4 uWaterDeep;
};

out vec4 FragColor;

void main() {
    vec2 ndc = (gl_FragCoord.xy / max(uScreenSize, vec2(1.0))) * 2.0 - 1.0;
    vec3 viewRay = vec3(ndc.x / uProj[0][0], ndc.y / uProj[1][1], -1.0);
    vec3 dir = normalize(transpose(mat3(uView)) * viewRay);

    vec3 cTop     = uSkyTopAndDensity.rgb;
    vec3 cMid     = uSkyMid.rgb;
    vec3 cBand1   = uSkyBand1.rgb;
    vec3 cHorizon = uSkyHorizon.rgb;
    vec3 cLow     = mix(uFogColorAndParams.rgb, cHorizon, 0.45);
    vec3 cUpper   = mix(cMid, cTop, 0.55);

    // 6 段渐变天空：地平线雾 -> 地平线 -> 中低空 -> 中空 -> 高空混合 -> 天顶
    float h = clamp(dir.z, -0.1, 1.0);
    vec3 sky;
    if (h < 0.08)      sky = mix(cLow, cHorizon, smoothstep(-0.1, 0.08, h));
    else if (h < 0.20) sky = mix(cHorizon, cBand1, smoothstep(0.08, 0.20, h));
    else if (h < 0.36) sky = mix(cBand1, cMid, smoothstep(0.20, 0.36, h));
    else if (h < 0.55) sky = mix(cMid, cUpper, smoothstep(0.36, 0.55, h));
    else if (h < 0.78) sky = mix(cUpper, cTop, smoothstep(0.55, 0.78, h));
    else               sky = cTop;

    vec3 sunDir = normalize(uSunDirAndIntensity.xyz);
    float sunAmt = pow(max(dot(dir, sunDir), 0.0), 96.0);
    sky += uSunColor.rgb * sunAmt * 0.75 * smoothstep(-0.05, 0.12, dir.z);

    if (uFogColorAndParams.w < 1.0) {
        // 水下视角：将远景背景混入水下深蓝绿体积雾色，仅天顶保留微弱天光透射
        float upTransmit = smoothstep(0.15, 0.95, dir.z) * 0.35;
        sky = mix(uFogColorAndParams.rgb, sky, upTransmit);
    }

    FragColor = vec4(sky, 1.0);
}
)GLSL";

// 顶点着色器：支持 SSBO 硬件实例化 (binding = 0) + M2 骨骼蒙皮 (binding = 1) + UV 动画偏移
constexpr std::string_view kWorldVertShader = R"GLSL(
#version 460 core
layout(location = 0) in vec3  aPos;
layout(location = 1) in vec3  aNormal;
layout(location = 2) in vec2  aUV;
layout(location = 3) in vec4  aBoneWeights;
layout(location = 4) in uvec4 aBoneIndices;

layout(location = 2) uniform int  uBoneBaseIndex;
layout(location = 3) uniform vec2 uUvOffset;

layout(std140, binding = 0) uniform FrameBlock {
    mat4 uView;
    mat4 uProj;
    mat4 uViewProj;
    vec4 uCameraPos;
    vec4 uSunDirAndIntensity;
    vec4 uSunColor;
    vec4 uAmbientSky;
    vec4 uAmbientGround;
    vec4 uFogColorAndParams;
    vec4 uSkyTopAndDensity;
    vec4 uSkyMid;
    vec4 uSkyBand1;
    vec4 uSkyHorizon;
    vec4 uWaterShallow;
    vec4 uWaterDeep;
};

layout(std430, binding = 0) readonly buffer InstanceBuffer {
    mat4 uInstanceMatrices[];
};

layout(std430, binding = 1) readonly buffer BoneBuffer {
    mat4 uBoneMatrices[];
};

out vec3 vWorldPos;
out vec3 vNormal;
out vec2 vUV;

void main() {
    vec3 localPos = aPos;
    vec3 localNrm = aNormal;

    if (uBoneBaseIndex > 0) {
        float wSum = aBoneWeights.x + aBoneWeights.y + aBoneWeights.z + aBoneWeights.w;
        if (wSum > 0.001) {
            mat4 skinMat = (aBoneWeights.x * uBoneMatrices[uBoneBaseIndex + int(aBoneIndices.x)]
                          + aBoneWeights.y * uBoneMatrices[uBoneBaseIndex + int(aBoneIndices.y)]
                          + aBoneWeights.z * uBoneMatrices[uBoneBaseIndex + int(aBoneIndices.z)]
                          + aBoneWeights.w * uBoneMatrices[uBoneBaseIndex + int(aBoneIndices.w)]) / wSum;
            localPos = (skinMat * vec4(aPos, 1.0)).xyz;
            localNrm = mat3(skinMat) * aNormal;
        }
    }

    mat4 modelMat = uInstanceMatrices[gl_BaseInstance + gl_InstanceID];
    vec4 worldPos = modelMat * vec4(localPos, 1.0);
    vWorldPos = worldPos.xyz;
    vNormal   = normalize(mat3(modelMat) * localNrm);
    vUV       = aUV + uUvOffset;
    gl_Position = uViewProj * worldPos;
}
)GLSL";

// 片段着色器：支持 TwoSided 背面法线翻转、Unlit 自发光、M2 批次颜色/透明度动画与指数距离雾
constexpr std::string_view kWorldFragShader = R"GLSL(
#version 460 core
in vec3 vWorldPos;
in vec3 vNormal;
in vec2 vUV;

layout(binding = 0) uniform sampler2D uTexAlbedo;   // Layer 0 (或 M2 主贴图)
layout(binding = 1) uniform sampler2D uTexLayer1;   // 地形 Layer 1
layout(binding = 2) uniform sampler2D uTexLayer2;   // 地形 Layer 2
layout(binding = 3) uniform sampler2D uTexLayer3;   // 地形 Layer 3
layout(binding = 4) uniform sampler2D uTexAlphaMap; // 64x64 RGB Alpha 遮罩 (M2 模型绑定全黑纹理，alpha=0)

layout(location = 0) uniform int  uAlphaCutout;   // 0 = Opaque, 1 = Cutout, 2 = Blend, 3 = UV Scroll
layout(location = 1) uniform int  uMaterialFlags; // bit 0 = TwoSided, bit 1 = Unlit
layout(location = 4) uniform vec4 uBatchColor;    // M2Track 颜色与透明度调制

layout(std140, binding = 0) uniform FrameBlock {
    mat4 uView;
    mat4 uProj;
    mat4 uViewProj;
    vec4 uCameraPos;
    vec4 uSunDirAndIntensity;
    vec4 uSunColor;
    vec4 uAmbientSky;
    vec4 uAmbientGround;
    vec4 uFogColorAndParams;
    vec4 uSkyTopAndDensity;
    vec4 uSkyMid;
    vec4 uSkyBand1;
    vec4 uSkyHorizon;
    vec4 uWaterShallow;
    vec4 uWaterDeep;
};

out vec4 FragColor;

float GetLuma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

vec3 HeightBlend(vec3 baseCol, vec3 overCol, float alpha) {
    float hBase = GetLuma(baseCol);
    float hOver = GetLuma(overCol);
    float w = clamp((alpha * 1.4 - 0.2) + (hOver - hBase) * 0.35, 0.0, 1.0);
    return mix(baseCol, overCol, smoothstep(0.0, 1.0, w));
}

void main() {
    vec4 texColor = texture(uTexAlbedo, vUV);
    vec3 alphaWeights = texture(uTexAlphaMap, clamp(vUV / 8.0, 0.0, 1.0)).rgb;
    if (dot(alphaWeights, vec3(1.0)) > 0.001) {
        vec3 c1 = texture(uTexLayer1, vUV).rgb;
        vec3 c2 = texture(uTexLayer2, vUV).rgb;
        vec3 c3 = texture(uTexLayer3, vUV).rgb;
        texColor.rgb = HeightBlend(texColor.rgb, c1, alphaWeights.r);
        texColor.rgb = HeightBlend(texColor.rgb, c2, alphaWeights.g);
        texColor.rgb = HeightBlend(texColor.rgb, c3, alphaWeights.b);
    }
    if (uAlphaCutout == 1 && (texColor.a * uBatchColor.a) < 0.5) {
        discard;
    }

    vec3 litColor;
    if ((uMaterialFlags & 2) != 0) {
        // Unlit 自发光材质 (火盆火焰/窗户夜景/瀑布水花)：不受背光阴影影响
        litColor = texColor.rgb * uBatchColor.rgb;
    } else {
        vec3 N = normalize(vNormal);
        // TwoSided 双面材质 (树冠/灌木/旗帜)：当观察背面时翻转法线，避免向光面树叶背光死黑
        if ((uMaterialFlags & 1) != 0 && !gl_FrontFacing) {
            N = -N;
        }

        float height = GetLuma(texColor.rgb);
        vec3 dPdx = dFdx(vWorldPos);
        vec3 dPdy = dFdy(vWorldPos);
        vec3 r1 = cross(dPdy, N);
        vec3 r2 = cross(N, dPdx);
        float det = dot(dPdx, r1);
        vec3 surfGrad = (abs(det) > 1e-8) ? ((r1 * dFdx(height) + r2 * dFdy(height)) / det) : vec3(0.0);
        N = normalize(N - surfGrad * 0.35);

        vec3 L = normalize(uSunDirAndIntensity.xyz);
        vec3 V = normalize(uCameraPos.xyz - vWorldPos);
        vec3 H = normalize(L + V);

        float ndotl = max(dot(N, L), 0.0);
        float halfLambert = ndotl * 0.65 + 0.35;
        float spec = pow(max(dot(N, H), 0.0), 24.0) * height * 0.28 * ndotl;

        vec3 skyAmbient    = uAmbientSky.rgb;
        vec3 groundAmbient = uAmbientGround.rgb;
        vec3 ambient = mix(groundAmbient, skyAmbient, N.z * 0.5 + 0.5);

        vec3 sunColor = uSunColor.rgb * uSunDirAndIntensity.w;
        litColor = (texColor.rgb * uBatchColor.rgb) * (ambient + sunColor * halfLambert) + sunColor * spec;
    }

    float dist = length(vWorldPos - uCameraPos.xyz);
    float fogFactor = clamp(exp(-max(dist - uFogColorAndParams.w, 0.0) * uSkyTopAndDensity.w), 0.0, 1.0);
    vec3 fogColor = uFogColorAndParams.rgb;

    float outAlpha = (uAlphaCutout >= 2) ? clamp(texColor.a * uBatchColor.a * 0.85, 0.0, 0.95) : 1.0;
    FragColor = vec4(mix(fogColor, litColor, fogFactor), outAlpha);
}
)GLSL";

// 水体顶点着色器：复用 TerrainVertex 布局，aNormal.z 即加载期预计算的精确几何水深
constexpr std::string_view kWaterVertShader = R"GLSL(
#version 460 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec2 aUV;

layout(std140, binding = 0) uniform FrameBlock {
    mat4 uView;
    mat4 uProj;
    mat4 uViewProj;
    vec4 uCameraPos;
    vec4 uSunDirAndIntensity;
    vec4 uSunColor;
    vec4 uAmbientSky;
    vec4 uAmbientGround;
    vec4 uFogColorAndParams;
    vec4 uSkyTopAndDensity;
    vec4 uSkyMid;
    vec4 uSkyBand1;
    vec4 uSkyHorizon;
    vec4 uWaterShallow;
    vec4 uWaterDeep;
};

layout(std430, binding = 0) readonly buffer InstanceBuffer {
    mat4 uInstanceMatrices[];
};

out vec3  vWorldPos;
out vec2  vUV;
out float vWaterDepth;

void main() {
    mat4 modelMat = uInstanceMatrices[gl_BaseInstance + gl_InstanceID];
    vec4 worldPos = modelMat * vec4(aPos, 1.0);
    vWorldPos   = worldPos.xyz;
    vUV         = aUV;
    vWaterDepth = aNormal.z;
    gl_Position = uViewProj * worldPos;
}
)GLSL";

// 水体片段着色器：解析正弦波法线 + Schlick 菲涅尔 + Beer-Lambert 深度透射 + 阳光高光
constexpr std::string_view kWaterFragShader = R"GLSL(
#version 460 core
in vec3  vWorldPos;
in vec2  vUV;
in float vWaterDepth;

layout(std140, binding = 0) uniform FrameBlock {
    mat4 uView;
    mat4 uProj;
    mat4 uViewProj;
    vec4 uCameraPos;          // xyz = 相机位置, w = 累计运行秒数
    vec4 uSunDirAndIntensity;
    vec4 uSunColor;
    vec4 uAmbientSky;
    vec4 uAmbientGround;
    vec4 uFogColorAndParams;
    vec4 uSkyTopAndDensity;
    vec4 uSkyMid;
    vec4 uSkyBand1;
    vec4 uSkyHorizon;
    vec4 uWaterShallow;
    vec4 uWaterDeep;
};

out vec4 FragColor;

void main() {
    float time = uCameraPos.w;
    vec3  V    = normalize(uCameraPos.xyz - vWorldPos);
    vec3  L    = normalize(uSunDirAndIntensity.xyz);
    vec3  H    = normalize(L + V);

    // 1. 4 组低频交叉相位波函数：避免规则正弦网格，得到柔和自然的动态水面
    vec2 p  = vWorldPos.xy;
    vec2 k0 = vec2( 0.11,  0.06);
    vec2 k1 = vec2(-0.07,  0.13);
    vec2 k2 = vec2( 0.16, -0.09);
    vec2 k3 = vec2(-0.13, -0.15);
    float a0 = 0.12, a1 = 0.09, a2 = 0.07, a3 = 0.05;

    float ph0 = dot(p, k0) + time * 0.45;
    float ph1 = dot(p, k1) + time * 0.62 + 0.7;
    float ph2 = dot(p, k2) + time * 0.78 + 1.3;
    float ph3 = dot(p, k3) + time * 0.94 + 2.1;

    // 交叉相位调制：用相邻波相位轻微扰动当前波，消除生硬感
    ph0 += sin(ph1) * 0.35;
    ph1 += sin(ph2) * 0.30;
    ph2 += sin(ph3) * 0.28;
    ph3 += sin(ph0) * 0.25;

    vec2 grad = a0 * k0 * cos(ph0)
              + a1 * k1 * cos(ph1)
              + a2 * k2 * cos(ph2)
              + a3 * k3 * cos(ph3);

    vec3 N = normalize(vec3(-grad.x, -grad.y, 1.0));

    // 2. Schlick 菲涅尔：掠射角反射上升
    float fresnel = 0.02 + 0.98 * pow(1.0 - max(dot(N, V), 0.0), 5.0);

    // 3. Beer-Lambert 深度透射：浅水青绿 -> 深水暗蓝
    float depthT      = clamp(vWaterDepth / 6.0, 0.0, 1.0);
    vec3  shallowColor = uWaterShallow.rgb;
    vec3  deepColor    = uWaterDeep.rgb;
    vec3  waterColor   = mix(shallowColor, deepColor, depthT);

    vec3  skyReflect = uSkyHorizon.rgb;
    vec3  sunColor   = uSunColor.rgb * uSunDirAndIntensity.w;
    float spec       = pow(max(dot(N, H), 0.0), 128.0) * 1.6;

    vec3 waterLitColor = mix(waterColor, skyReflect, fresnel * 0.65) + sunColor * spec;

    // 4. 岸边柔化与整体透明度
    float shoreAlpha = smoothstep(0.0, 0.55, vWaterDepth);
    float alpha = clamp(shoreAlpha * mix(0.45, 0.92,
                    clamp(vWaterDepth * 0.25 + fresnel * 0.6, 0.0, 1.0)), 0.0, 0.95);

    // 5. 与 kWorldFragShader 完全一致的指数距离雾，使远处水面融入天际
    float dist = length(vWorldPos - uCameraPos.xyz);
    float fogFactor = clamp(exp(-max(dist - uFogColorAndParams.w, 0.0) * uSkyTopAndDensity.w), 0.0, 1.0);
    vec3  fogColor  = uFogColorAndParams.rgb;

    FragColor = vec4(mix(fogColor, waterLitColor, fogFactor), alpha);
}
)GLSL";

// 粒子顶点着色器：gl_VertexID 展开 6 顶点公告板四边形，从 SSBO (binding = 2) 读取实例数据
constexpr std::string_view kParticleVertShader = R"GLSL(
#version 460 core
struct GpuParticleQuad {
    vec4 worldPosAndSize; // xyz = 世界坐标, w = 半径尺寸
    vec4 color;           // rgba = 颜色与透明度
    vec4 uvRect;          // xy = UV 起点, zw = UV 缩放 (子图格)
};

layout(std430, binding = 2) readonly buffer ParticleBuffer {
    GpuParticleQuad uParticles[];
};

layout(std140, binding = 0) uniform FrameBlock {
    mat4 uView;
    mat4 uProj;
    mat4 uViewProj;
    vec4 uCameraPos;
    vec4 uSunDirAndIntensity;
    vec4 uSunColor;
    vec4 uAmbientSky;
    vec4 uAmbientGround;
    vec4 uFogColorAndParams;
    vec4 uSkyTopAndDensity;
    vec4 uSkyMid;
    vec4 uSkyBand1;
    vec4 uSkyHorizon;
    vec4 uWaterShallow;
    vec4 uWaterDeep;
};

out vec2 vUV;
out vec2 vCorner;
out vec4 vColor;
out vec3 vWorldPos;

const vec2 kQuadCorners[6] = vec2[6](
    vec2(-0.5, -0.5), vec2( 0.5, -0.5), vec2( 0.5,  0.5),
    vec2(-0.5, -0.5), vec2( 0.5,  0.5), vec2(-0.5,  0.5)
);

void main() {
    vec2 corner = kQuadCorners[gl_VertexID % 6];
    vec2 uv     = vec2(corner.x + 0.5, 1.0 - (corner.y + 0.5));

    GpuParticleQuad p = uParticles[gl_BaseInstance + gl_InstanceID];

    // 视图矩阵前两列即相机右向量与上向量 (世界空间)，使公告板始终正对镜头
    vec3 camRight = vec3(uView[0][0], uView[1][0], uView[2][0]);
    vec3 camUp    = vec3(uView[0][1], uView[1][1], uView[2][1]);

    vec3 worldPos = p.worldPosAndSize.xyz
                  + (camRight * corner.x + camUp * corner.y) * p.worldPosAndSize.w;

    vUV       = p.uvRect.xy + uv * p.uvRect.zw;
    vCorner   = corner;
    vColor    = p.color;
    vWorldPos = worldPos;
    gl_Position = uViewProj * vec4(worldPos, 1.0);
}
)GLSL";

// 粒子片段着色器：径向软边衰减 + 指数距离雾 (Additive 仅对 rgb 乘 fogFactor)
constexpr std::string_view kParticleFragShader = R"GLSL(
#version 460 core
in vec2 vUV;
in vec2 vCorner;
in vec4 vColor;
in vec3 vWorldPos;

layout(binding = 0) uniform sampler2D uTexAlbedo;
layout(location = 0) uniform int uAdditive; // 1 = 叠加混合, 0 = 常规 AlphaBlend

layout(std140, binding = 0) uniform FrameBlock {
    mat4 uView;
    mat4 uProj;
    mat4 uViewProj;
    vec4 uCameraPos;
    vec4 uSunDirAndIntensity;
    vec4 uSunColor;
    vec4 uAmbientSky;
    vec4 uAmbientGround;
    vec4 uFogColorAndParams;
    vec4 uSkyTopAndDensity;
    vec4 uSkyMid;
    vec4 uSkyBand1;
    vec4 uSkyHorizon;
    vec4 uWaterShallow;
    vec4 uWaterDeep;
};

out vec4 FragColor;

void main() {
    vec4 c = texture(uTexAlbedo, vUV) * vColor;

    // 无贴图 / 纯白贴图时避免硬边方块，保证炊烟与火苗边缘柔和
    float radial = 1.0 - smoothstep(0.35, 0.5, length(vCorner));
    c.a *= radial;
    if (c.a < 0.004) discard;

    float dist      = length(vWorldPos - uCameraPos.xyz);
    float fogFactor = clamp(exp(-max(dist - uFogColorAndParams.w, 0.0) * uSkyTopAndDensity.w), 0.0, 1.0);

    vec3 litRgb = (uAdditive != 0) ? (c.rgb * fogFactor)
                                   : mix(uFogColorAndParams.rgb, c.rgb, fogFactor);
    FragColor = vec4(litRgb, c.a);
}
)GLSL";

int main(int /*argc*/, char* /*argv*/[]) {
    SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS);

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 6);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_DEBUG_FLAG);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 32);

    int winW = 1920, winH = 1080;
    SDL_Window* window = SDL_CreateWindow("CoA Custom Engine - Elwynn Forest ADT", winW, winH,
                                          SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
    SDL_GLContext glCtx = SDL_GL_CreateContext(window);
    gladLoadGLLoader(reinterpret_cast<GLADloadproc>(SDL_GL_GetProcAddress));
    SDL_GL_SetSwapInterval(1);

    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    glDebugMessageCallback(GlDebugCallback, nullptr);

    glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_GREATER);
    glClearDepth(0.0);
    glEnable(GL_MULTISAMPLE);

    {
        coa::rhi::GlBuffer frameUbo(sizeof(FrameUniforms));
        frameUbo.BindAsUniformBuffer(0);

        coa::rhi::GlShaderProgram worldShader(kWorldVertShader, kWorldFragShader);
        coa::rhi::GlShaderProgram waterShader(kWaterVertShader, kWaterFragShader);
        coa::rhi::GlShaderProgram skyShader(kSkyVertShader, kSkyFragShader);
        coa::rhi::GlShaderProgram particleShader(kParticleVertShader, kParticleFragShader);
        GLuint sharedM2Vao = coa::assets::GpuM2Model::CreateSharedVAO();
        GLuint terrainVao  = coa::assets::GpuAdtTile::CreateTerrainVAO();
        GLuint skyVao      = 0;
        glCreateVertexArrays(1, &skyVao);

        // 每帧动态上传可见实例矩阵的 SSBO (binding = 0) 与 M2 骨骼蒙皮矩阵的 SSBO (binding = 1)
        coa::rhi::GlBuffer instanceSsbo(sizeof(glm::mat4), nullptr, GL_DYNAMIC_STORAGE_BIT);
        instanceSsbo.BindAsStorageBuffer(0);
        coa::rhi::GlBuffer boneSsbo(sizeof(glm::mat4), nullptr, GL_DYNAMIC_STORAGE_BIT);
        boneSsbo.BindAsStorageBuffer(1);
        // binding = 2: 本帧全部公告板粒子实例数据
        coa::rhi::GlBuffer particleSsbo(sizeof(GpuParticleQuad), nullptr, GL_DYNAMIC_STORAGE_BIT);
        particleSsbo.BindAsStorageBuffer(2);

        coa::rhi::GlTexture2D whiteFallbackTex(1, 1, 1, coa::rhi::PixelFormat::RGBA8, true);
        const uint8_t whitePixel[4] = {180, 210, 140, 255}; // 浅草绿兜底色
        whiteFallbackTex.UploadUncompressedMip(0, 1, 1, whitePixel);
		coa::rhi::GlTexture2D blackAlphaTex(1, 1, 1, coa::rhi::PixelFormat::RGBA8, false);
        const uint8_t blackPixel[4] = {0, 0, 0, 0};
        blackAlphaTex.UploadUncompressedMip(0, 1, 1, blackPixel);
		
        coa::vfs::VirtualFileSystem vfs;
        vfs.MountLooseDirectory("CoA_Assets/wotlk_base", 100);
        vfs.MountLooseDirectory("CoA_Assets/coa_patches", 200);

        coa::assets::AssetManager assets(vfs);

        coa::assets::LightDbcManager lightMgr;
        lightMgr.Load(vfs, 0);

        // 多 ADT 无缝拼接：同时加载北郡(32_48)与闪金镇(32_49)
        std::vector<coa::assets::GpuAdtTile> adtTiles;
        std::unordered_set<uint32_t> seenUniqueIds;

        auto tile1 = coa::assets::AdtLoader::LoadAdt(
            "world/maps/azeroth/azeroth_32_48.adt",
            vfs, assets, whiteFallbackTex.Handle(), seenUniqueIds
        );
        if (tile1) adtTiles.push_back(std::move(*tile1));

        auto tile2 = coa::assets::AdtLoader::LoadAdt(
            "world/maps/azeroth/azeroth_32_49.adt",
            vfs, assets, whiteFallbackTex.Handle(), seenUniqueIds
        );
        if (tile2) adtTiles.push_back(std::move(*tile2));

        // 跨 ADT 全局 M2 合批桶：将所有已加载 ADT 中指向同一个 GpuM2Model 的实例列表聚合到一起
        struct GlobalM2Bucket {
            const coa::assets::GpuM2Model* model = nullptr;
            std::vector<const std::vector<coa::assets::M2Instance>*> instanceLists;
        };
        std::vector<GlobalM2Bucket> globalM2Buckets;
        {
            std::unordered_map<const coa::assets::GpuM2Model*, size_t> bucketMap;
            for (const auto& tile : adtTiles) {
                for (const auto& batch : tile.doodadBatches) {
                    const auto* modelPtr = batch.model.get();
                    if (!modelPtr) continue;
                    auto [it, inserted] = bucketMap.try_emplace(modelPtr, globalM2Buckets.size());
                    if (inserted) {
                        globalM2Buckets.push_back(GlobalM2Bucket{modelPtr, {}});
                    }
                    globalM2Buckets[it->second].instanceLists.push_back(&batch.instances);
                }
            }
            std::printf("[Global Batcher] Merged M2 across %zu ADTs into %zu unique buckets (VRAM Cache: %zu M2, %zu WMO, %zu BLP)\n",
                        adtTiles.size(), globalM2Buckets.size(),
                        assets.CachedM2Count(), assets.CachedWmoCount(), assets.CachedTextureCount());
        }

        // 自动将摄像机瞬移至首个成功加载的 ADT 地形中心上空
        glm::vec3 camPos = (!adtTiles.empty()) ? adtTiles.front().suggestedCamPos : glm::vec3(0.0f, 0.0f, 20.0f);
        float camYaw   = 0.0f;
        float camPitch = -0.25f;
        float gameTimeHalfMin = 1560.0f; // 默认 13:00，金色阳光
        bool rightMouseDown = false;
        uint64_t lastTicks = SDL_GetTicksNS();

        int visibleChunks = 0;
        int totalChunks = 0;
        int visibleWmoGroups = 0;
        int visibleM2Instances = 0;
        int visibleWaterChunks = 0;
        float titleTimer = 0.0f;

        std::vector<coa::rhi::DrawCommand> renderQueue;
        std::vector<glm::mat4> visibleMatrices;
        std::vector<glm::mat4> visibleBoneMatrices;
        std::vector<GpuParticleQuad>  visibleParticles;
        std::vector<ParticleBatchCmd> particleBatches;
        bool running = true;

        while (running) {
            uint64_t nowTicks = SDL_GetTicksNS();
            float dt = static_cast<float>(nowTicks - lastTicks) * 1e-9f;
            lastTicks = nowTicks;

            SDL_Event ev;
            while (SDL_PollEvent(&ev)) {
                if (ev.type == SDL_EVENT_QUIT) {
                    running = false;
                } else if (ev.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED) {
                    winW = ev.window.data1;
                    winH = ev.window.data2;
                    glViewport(0, 0, winW, winH);
                } else if (ev.type == SDL_EVENT_KEY_DOWN && ev.key.scancode == SDL_SCANCODE_P) {
                    std::printf("\n=== [Scene Inspector] Camera Pos: (%.1f, %.1f, %.1f) ===\n",
                                camPos.x, camPos.y, camPos.z);
                    for (size_t t = 0; t < adtTiles.size(); ++t) {
                        for (const auto& dp : adtTiles[t].debugPlacements) {
                            float dist = glm::length(dp.pos - camPos);
                            if (dist <= 120.0f) {
                                std::printf("  [Tile %zu] [%s] dist=%5.1f scale=%.2f path=%s\n",
                                            t, dp.isWmo ? "WMO" : "M2 ", dist, dp.scale, dp.path.c_str());
                            }
                        }
                    }
                    std::printf("========================================================\n\n");
                } else if (ev.type == SDL_EVENT_MOUSE_BUTTON_DOWN && ev.button.button == SDL_BUTTON_RIGHT) {
                    rightMouseDown = true;
                    SDL_SetWindowRelativeMouseMode(window, true);
                } else if (ev.type == SDL_EVENT_MOUSE_BUTTON_UP && ev.button.button == SDL_BUTTON_RIGHT) {
                    rightMouseDown = false;
                    SDL_SetWindowRelativeMouseMode(window, false);
                } else if (ev.type == SDL_EVENT_MOUSE_MOTION && rightMouseDown) {
                    camYaw   -= ev.motion.xrel * 0.0025f;
                    camPitch  = std::clamp(camPitch - ev.motion.yrel * 0.0025f, -1.5f, 1.5f);
                }
            }

            glm::vec3 forward(
                std::cos(camPitch) * std::cos(camYaw),
                std::cos(camPitch) * std::sin(camYaw),
                std::sin(camPitch)
            );
            glm::vec3 worldUp(0.0f, 0.0f, 1.0f);
            glm::vec3 right = glm::normalize(glm::cross(forward, worldUp));

            const bool* keys = SDL_GetKeyboardState(nullptr);
            // 按住 Shift 高速飞行 (120码/秒)，方便跨越 533x533 码的 ADT 大地图
            float moveSpeed = (keys[SDL_SCANCODE_LSHIFT] ? 120.0f : 25.0f) * dt;
            if (keys[SDL_SCANCODE_W]) camPos += forward * moveSpeed;
            if (keys[SDL_SCANCODE_S]) camPos -= forward * moveSpeed;
            if (keys[SDL_SCANCODE_D]) camPos += right * moveSpeed;
            if (keys[SDL_SCANCODE_A]) camPos -= right * moveSpeed;
            if (keys[SDL_SCANCODE_SPACE]) camPos += worldUp * moveSpeed;
            if (keys[SDL_SCANCODE_LCTRL]) camPos -= worldUp * moveSpeed;

            if (keys[SDL_SCANCODE_LEFTBRACKET])  gameTimeHalfMin -= 360.0f * dt;
            if (keys[SDL_SCANCODE_RIGHTBRACKET]) gameTimeHalfMin += 360.0f * dt;
            gameTimeHalfMin += dt; // 默认缓慢流逝：1 半分钟 / 秒
            while (gameTimeHalfMin < 0.0f)    gameTimeHalfMin += 2880.0f;
            while (gameTimeHalfMin >= 2880.0f) gameTimeHalfMin -= 2880.0f;

            const bool underwater = IsCameraUnderwater(camPos, adtTiles);
            const coa::assets::EvaluatedLight lightState =
                lightMgr.Evaluate(camPos, gameTimeHalfMin, underwater);

            constexpr float kTwoPi = 6.28318530718f;
            constexpr float kHalfPi = 1.57079632679f;
            const float sunElev = std::sin((gameTimeHalfMin / 2880.0f) * kTwoPi - kHalfPi);
            const float dayFactor = glm::smoothstep(-0.08f, 0.25f, sunElev);
            const float lightIntensity = glm::mix(0.18f, 1.15f, dayFactor);

            float aspect = (winH > 0) ? (static_cast<float>(winW) / static_cast<float>(winH)) : 1.777f;
            FrameUniforms uboData{};
            uboData.view               = glm::lookAt(camPos, camPos + forward, worldUp);
            uboData.proj               = MakeInfiniteReverseZProj(glm::radians(60.0f), aspect, 0.2f);
            uboData.viewProj           = uboData.proj * uboData.view;
            uboData.cameraPos          = glm::vec4(camPos, static_cast<float>(nowTicks) * 1e-9f);
            uboData.sunDirAndIntensity = glm::vec4(glm::normalize(lightState.sunDir), lightIntensity);
            uboData.sunColor           = glm::vec4(lightState.directColor, 1.0f);
            uboData.ambientSky         = glm::vec4(lightState.ambientColor, 1.0f);
            uboData.ambientGround      = glm::vec4(lightState.ambientColor * 0.55f, 1.0f);
            uboData.fogColorAndParams  = glm::vec4(lightState.fogColor, underwater ? 0.0f : 120.0f);
            uboData.skyTopAndDensity   = glm::vec4(lightState.skyTop, underwater ? 0.028f : 0.0022f);
            uboData.skyMid             = glm::vec4(lightState.skyMiddle, 1.0f);
            uboData.skyBand1           = glm::vec4(lightState.skyBand1, 1.0f);
            uboData.skyHorizon         = glm::vec4(lightState.skyHorizon, 1.0f);
            uboData.waterShallow       = glm::vec4(lightState.riverShallow, 1.0f);
            uboData.waterDeep          = glm::vec4(lightState.riverDeep, 1.0f);
            frameUbo.UpdateSubData(0, sizeof(FrameUniforms), &uboData);

            glClearColor(lightState.skyHorizon.r, lightState.skyHorizon.g, lightState.skyHorizon.b, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            // 天空盒先于所有不透明几何绘制：关闭深度测试与写入，只填充背景渐变
            glDisable(GL_DEPTH_TEST);
            glDepthMask(GL_FALSE);
            glDisable(GL_BLEND);
            glUseProgram(skyShader.Handle());
            glUniform2f(0, static_cast<float>(winW), static_cast<float>(winH));
            glBindVertexArray(skyVao);
            glDrawArrays(GL_TRIANGLES, 0, 3);
            glDepthMask(GL_TRUE);
            glEnable(GL_DEPTH_TEST);

            renderQueue.clear();
            visibleParticles.clear();
            particleBatches.clear();
            visibleMatrices.clear();
            visibleMatrices.push_back(glm::mat4(1.0f)); // 下标 0 固定为单位阵，供 MCNK 地形复用
            visibleBoneMatrices.clear();
            visibleBoneMatrices.push_back(glm::mat4(1.0f)); // 下标 0 固定为单位阵，供静态无骨骼网格占位
            const float animTimeSec = static_cast<float>(nowTicks) * 1e-9f;

            // 每帧提取 Reverse-Z 视锥体
            Frustum camFrustum = Frustum::ExtractFromViewProj(uboData.viewProj);
            visibleChunks = 0;
            totalChunks = 0;
            visibleWmoGroups = 0;
            visibleM2Instances = 0;
            visibleWaterChunks = 0;

            for (const auto& tile : adtTiles) {
                // 1. 提交 MCNK 地形块，先按 AABB 进行视锥裁剪
                for (const auto& chunk : tile.chunks) {
                    ++totalChunks;
                    if (!camFrustum.IntersectsAABB(chunk.aabbMin, chunk.aabbMax)) {
                        continue;
                    }
                    ++visibleChunks;

                    coa::rhi::DrawCommand cmd{};
                    cmd.sortKey       = coa::rhi::DrawCommand::MakeKey(2, 1, static_cast<uint16_t>(chunk.layerTex[0]), 0);
                    cmd.program       = worldShader.Handle();
                    cmd.vao           = terrainVao;
                    cmd.vbo           = tile.terrainVbo.Handle();
                    cmd.ibo           = tile.terrainIbo.Handle();
                    cmd.stride        = sizeof(coa::assets::TerrainVertex);
                    cmd.firstIndex    = chunk.firstIndex;
                    cmd.indexCount    = chunk.indexCount;
                    cmd.baseVertex    = chunk.baseVertex;
                    cmd.instanceCount = 1;
                    cmd.baseInstance  = 0; // 复用下标 0 的单位阵
                    cmd.texAlbedo     = chunk.layerTex[0];
                    cmd.texLayers[0]  = chunk.layerTex[1];
                    cmd.texLayers[1]  = chunk.layerTex[2];
                    cmd.texLayers[2]  = chunk.layerTex[3];
                    cmd.texAlphaMap   = chunk.alphaMapTex;
                    renderQueue.push_back(cmd);
                }
            }

            // 2. 跨 ADT 全局 M2 单次合批硬件实例化 + 全局单次骨骼/UV/颜色关键帧评估
            for (const auto& bucket : globalM2Buckets) {
                const auto& model = *bucket.model;
                uint32_t firstInstance = static_cast<uint32_t>(visibleMatrices.size());
                uint32_t visibleCount  = 0;

                for (const auto* instList : bucket.instanceLists) {
                    for (const auto& inst : *instList) {
                        if (!IsInstanceVisible(camFrustum, camPos, inst.worldBoundsMin, inst.worldBoundsMax,
                                               inst.worldCenter, inst.worldRadius)) {
                            continue;
                        }
                        visibleMatrices.push_back(inst.model);
                        ++visibleCount;
                    }
                }
                if (visibleCount == 0) continue;
                visibleM2Instances += static_cast<int>(visibleCount);

                // ---- 粒子发射器：为每个可见实例生成确定性公告板粒子 (屋顶炊烟 / 火把火苗) ----
                if (!model.particleEmitters.empty()) {
                    auto hash01 = [](uint32_t n) {
                        n ^= n >> 16; n *= 2246822519u;
                        n ^= n >> 13; n *= 3266489917u;
                        n ^= n >> 16;
                        return static_cast<float>(n & 0xFFFFFFu) / 16777216.0f;
                    };

                    const glm::mat4* visMats = visibleMatrices.data() + firstInstance;
                    for (uint32_t k = 0; k < visibleCount; ++k) {
                        const glm::mat4& instMat = visMats[k];
                        const glm::vec3  instWorld(instMat[3].x, instMat[3].y, instMat[3].z);
                        if (glm::distance(instWorld, camPos) > 220.0f) continue;

                        const float instHash = std::fabs(std::fmod(
                            instWorld.x * 0.137f + instWorld.y * 0.311f + instWorld.z * 0.733f, 1.0f));

                        for (size_t emi = 0; emi < model.particleEmitters.size(); ++emi) {
                            const auto& em    = model.particleEmitters[emi];
                            const float life  = std::max(em.lifespan, 0.25f);
                            int count = std::clamp(static_cast<int>(em.emissionRate * life), 6, 32);
                            // 静止灯笼光晕 (speed≈0)：限制层数，避免多层半透明叠加过曝
                            if (em.speed < 0.05f) {
                                count = std::min(count, 2);
                            }
                            const int   rows  = std::max(1, static_cast<int>(em.texRows));
                            const int   cols  = std::max(1, static_cast<int>(em.texCols));
                            const int   frames = rows * cols;
                            const uint32_t firstQuad = static_cast<uint32_t>(visibleParticles.size());

                            for (int p = 0; p < count; ++p) {
                                const uint32_t seed = static_cast<uint32_t>(p) * 9781u
                                                    + k * 6271u
                                                    + static_cast<uint32_t>(emi) * 4099u;
                                const float r1 = hash01(seed + 11u);
                                const float r2 = hash01(seed + 29u);
                                const float r3 = hash01(seed + 47u);

                                float u = std::fmod(animTimeSec / life
                                                    + static_cast<float>(p) / static_cast<float>(count)
                                                    + instHash, 1.0f);
                                if (u < 0.0f) u += 1.0f;

                                // 发射点：发射器局部位置 + 发射面积抖动
                                const glm::vec3 localStart = em.localPos + glm::vec3(
                                    (r1 - 0.5f) * std::max(em.areaLength, 0.0f),
                                    (r2 - 0.5f) * std::max(em.areaWidth, 0.0f),
                                    0.0f);
                                const glm::vec4 start4 = instMat * glm::vec4(localStart, 1.0f);
                                const glm::vec3 startWorld(start4.x, start4.y, start4.z);

                                // 初速度：以向上为主 + 横向扩散，受速度变化率扰动
                                const float spd = em.speed * (1.0f - em.speedVary * 0.5f + em.speedVary * r3);
                                // 横向分量按 spd 缩放，避免低速光晕出现水平喷雾
                                const glm::vec4 vel4 = instMat * glm::vec4(
                                    (r1 - 0.5f) * em.horizRange * spd,
                                    (r2 - 0.5f) * em.horizRange * spd,
                                    spd, 0.0f);
                                const glm::vec3 velWorld(vel4.x, vel4.y, vel4.z);

                                const float t = u * life;
                                glm::vec3 worldPos = startWorld + velWorld * t;
                                // 重力方向向下：Z-Up 世界空间需取负
                                worldPos.z -= 0.5f * em.gravity * t * t;

                                // 三段插值：起始 -> 中段 -> 结束
                                glm::vec3 col; float alp; float scl;
                                if (u < 0.35f) {
                                    const float a = u / 0.35f;
                                    col = glm::mix(em.colorStart, em.colorMid, a);
                                    alp = glm::mix(em.alphaStart, em.alphaMid, a);
                                    scl = glm::mix(em.scaleStart, em.scaleMid, a);
                                } else {
                                    const float a = (u - 0.35f) / 0.65f;
                                    col = glm::mix(em.colorMid, em.colorEnd, a);
                                    alp = glm::mix(em.alphaMid, em.alphaEnd, a);
                                    scl = glm::mix(em.scaleMid, em.scaleEnd, a);
                                }
                                // 静止灯笼光晕：压低单层透明度，防止多层叠加过曝
                                if (em.speed < 0.05f) {
                                    alp *= 0.25f;
                                }
                                if (alp < 0.004f) continue;

                                // texRows x texCols 子图格：随寿命推进帧
                                const int   frame = std::min(frames - 1, static_cast<int>(u * static_cast<float>(frames)));
                                const float cellW = 1.0f / static_cast<float>(cols);
                                const float cellH = 1.0f / static_cast<float>(rows);
                                const int   colIdx = frame % cols;
                                const int   rowIdx = frame / cols;

                                GpuParticleQuad quad{};
                                quad.worldPosAndSize = glm::vec4(worldPos, std::max(scl, 0.05f));
                                quad.color           = glm::vec4(col, alp);
                                quad.uvRect          = glm::vec4(colIdx * cellW, rowIdx * cellH, cellW, cellH);
                                visibleParticles.push_back(quad);
                            }

                            const uint32_t emitted = static_cast<uint32_t>(visibleParticles.size()) - firstQuad;
                            if (emitted > 0) {
                                particleBatches.push_back(ParticleBatchCmd{
                                    em.textureId != 0 ? em.textureId : whiteFallbackTex.Handle(),
                                    em.blendingType,
                                    firstQuad,
                                    emitted
                                });
                            }
                        }
                    }
                }

                // 全场景同一种 M2 模型每帧仅评估 1 次骨骼矩阵栈
                uint32_t boneBaseIndex = 0;
                if (model.hasAnimatedBones) {
                    boneBaseIndex = model.EvaluateBones(animTimeSec, visibleBoneMatrices);
                }

                // 跨所有已加载 ADT 的同种 M2 子网格仅生成 1 条实例化 DrawCommand
                for (const auto& sub : model.submeshes) {
                    glm::vec4 batchColor(1.0f);
                    if (sub.colorIndex >= 0 && static_cast<size_t>(sub.colorIndex) < model.colorTracks.size()) {
                        const auto& ct = model.colorTracks[sub.colorIndex];
                        glm::vec3 rgb = ct.color.SampleLinear(animTimeSec, model.globalSequences, glm::vec3(1.0f));
                        float     alp = ct.alpha.SampleLinear(animTimeSec, model.globalSequences, 1.0f);
                        batchColor = glm::vec4(rgb, alp);
                    }
                    if (sub.transWeightIndex >= 0 && static_cast<size_t>(sub.transWeightIndex) < model.transWeightTracks.size()) {
                        const auto& tw = model.transWeightTracks[sub.transWeightIndex];
                        batchColor.a *= tw.SampleLinear(animTimeSec, model.globalSequences, 1.0f);
                    }
                    if (batchColor.a < 0.01f) continue;

                    glm::vec2 uvOffset(0.0f);
                    if (sub.uvTransIndex >= 0 && static_cast<size_t>(sub.uvTransIndex) < model.uvTransTracks.size()) {
                        const auto& ut = model.uvTransTracks[sub.uvTransIndex];
                        if (ut.HasAnimation()) {
                            glm::vec3 uv3 = ut.SampleLinear(animTimeSec, model.globalSequences, glm::vec3(0.0f));
                            uvOffset = glm::vec2(uv3.x, uv3.y);
                        } else {
                            uvOffset = glm::vec2(0.0f, -0.75f * animTimeSec);
                        }
                    } else if (sub.blendMode == 10) {
                        uvOffset = glm::vec2(0.0f, -0.75f * animTimeSec);
                    }

                    coa::rhi::DrawCommand cmd{};
                    uint8_t pass = 2;
                    if (sub.blendMode == 1) {
                        pass = 3; // Alpha Cutout (树叶/栅栏)
                    } else if (sub.blendMode == 2) {
                        pass = 5; // Alpha Blend (半透明水幕/薄纱)
                    } else if (sub.blendMode == 10) {
                        pass = 7; // UV 动态滚动的瀑布水幕
                    } else if (sub.blendMode >= 3) {
                        pass = 6; // Additive / Mod (火盆光晕/魔法水花)
                    }
                    cmd.sortKey       = coa::rhi::DrawCommand::MakeKey(pass, 1, static_cast<uint16_t>(sub.textureId), 0);
                    cmd.program       = worldShader.Handle();
                    cmd.vao           = sharedM2Vao;
                    cmd.vbo           = model.vbo.Handle();
                    cmd.ibo           = model.ibo.Handle();
                    cmd.stride        = sizeof(coa::assets::M2Vertex);
                    cmd.texAlbedo     = sub.textureId;
                    cmd.firstIndex    = sub.firstIndex;
                    cmd.indexCount    = sub.indexCount;
                    cmd.baseVertex    = 0;
                    cmd.instanceCount = visibleCount;
                    cmd.baseInstance  = firstInstance;
                    cmd.boneBaseIndex = boneBaseIndex;
                    cmd.isTwoSided    = ((sub.materialFlags & 0x04u) != 0) ? 1 : 0;
                    cmd.isUnlit       = ((sub.materialFlags & 0x01u) != 0 || pass >= 6) ? 1 : 0;
                    cmd.uvOffset      = uvOffset;
                    cmd.batchColor    = batchColor;
                    cmd.texAlphaMap   = blackAlphaTex.Handle();
                    renderQueue.push_back(cmd);
                }
            }

            // 3. WMO 两级层次化裁剪：先整栋建筑，再逐个子房间
            for (const auto& tile : adtTiles) {
                for (const auto& wmoBatch : tile.wmoBatches) {
                    if (!wmoBatch.model) continue;
                    const auto& model = *wmoBatch.model;
                    for (const auto& inst : wmoBatch.instances) {
                        // 3.1 整栋建筑不可见 -> 直接跳过其所有子房间
                        if (!IsInstanceVisible(camFrustum, camPos, inst.worldBoundsMin, inst.worldBoundsMax,
                                               inst.worldCenter, inst.worldRadius)) {
                            continue;
                        }
                        if (inst.groupBounds.size() != model.groups.size()) continue;

                        uint32_t wmoBaseInstance = UINT32_MAX;
                        // 3.2 逐个子房间测试世界空间包围盒
                        for (size_t g = 0; g < model.groups.size(); ++g) {
                            const auto& grp = model.groups[g];
                            const auto& gc  = inst.groupBounds[g];
                            if (!IsInstanceVisible(camFrustum, camPos, gc.worldBoundsMin, gc.worldBoundsMax,
                                                   gc.worldCenter, gc.worldRadius)) {
                                continue;
                            }
                            ++visibleWmoGroups;

                            if (wmoBaseInstance == UINT32_MAX) {
                                wmoBaseInstance = static_cast<uint32_t>(visibleMatrices.size());
                                visibleMatrices.push_back(inst.model);
                            }

                            for (const auto& sb : grp.batches) {
                                coa::rhi::DrawCommand cmd{};
                                uint8_t pass      = (sb.blendMode == 1) ? 3 : 2;
                                cmd.sortKey       = coa::rhi::DrawCommand::MakeKey(pass, 1, static_cast<uint16_t>(sb.textureId), 0);
                                cmd.program       = worldShader.Handle();
                                cmd.vao           = terrainVao;
                                cmd.vbo           = grp.vbo.Handle();
                                cmd.ibo           = grp.ibo.Handle();
                                cmd.stride        = sizeof(coa::assets::WmoVertex);
                                cmd.texAlbedo     = sb.textureId;
                                cmd.texAlphaMap   = blackAlphaTex.Handle();
                                cmd.firstIndex    = sb.firstIndex;
                                cmd.indexCount    = sb.indexCount;
                                cmd.baseVertex    = 0;
                                cmd.instanceCount = 1;
                                cmd.baseInstance  = wmoBaseInstance;
                                renderQueue.push_back(cmd);
                            }
                        }
                    }
                }
            }

            // 4. MH2O 半透明水体 (Pass 4)：按水面 AABB 视锥裁剪后，复用下标 0 的单位阵实例绘制
            for (const auto& tile : adtTiles) {
                if (tile.liquidChunks.empty() || tile.liquidVbo.Handle() == 0) continue;
                for (const auto& lchunk : tile.liquidChunks) {
                    if (!camFrustum.IntersectsAABB(lchunk.aabbMin, lchunk.aabbMax)) {
                        continue;
                    }
                    ++visibleWaterChunks;

                    coa::rhi::DrawCommand cmd{};
                    cmd.sortKey       = coa::rhi::DrawCommand::MakeKey(4, 2, static_cast<uint16_t>(lchunk.liquidType), 0);
                    cmd.program       = waterShader.Handle();
                    cmd.vao           = terrainVao;
                    cmd.vbo           = tile.liquidVbo.Handle();
                    cmd.ibo           = tile.liquidIbo.Handle();
                    cmd.stride        = sizeof(coa::assets::TerrainVertex);
                    cmd.texAlbedo     = whiteFallbackTex.Handle();
                    cmd.firstIndex    = lchunk.firstIndex;
                    cmd.indexCount    = lchunk.indexCount;
                    cmd.baseVertex    = 0;
                    cmd.instanceCount = 1;
                    cmd.baseInstance  = 0; // 复用下标 0 的单位阵
                    renderQueue.push_back(cmd);
                }
            }

            // 将本帧可见实例的模型矩阵上传至 SSBO (binding = 0)，必要时按需扩容重建
            size_t neededBytes = visibleMatrices.size() * sizeof(glm::mat4);
            if (neededBytes > instanceSsbo.Size()) {
                instanceSsbo = coa::rhi::GlBuffer(neededBytes, nullptr, GL_DYNAMIC_STORAGE_BIT);
                instanceSsbo.BindAsStorageBuffer(0);
            }
            instanceSsbo.UpdateSubData(0, neededBytes, visibleMatrices.data());

            // 将本帧活动 M2 骨骼矩阵上传至 SSBO (binding = 1)
            size_t neededBoneBytes = visibleBoneMatrices.size() * sizeof(glm::mat4);
            if (neededBoneBytes > boneSsbo.Size()) {
                boneSsbo = coa::rhi::GlBuffer(neededBoneBytes, nullptr, GL_DYNAMIC_STORAGE_BIT);
                boneSsbo.BindAsStorageBuffer(1);
            }
            boneSsbo.UpdateSubData(0, neededBoneBytes, visibleBoneMatrices.data());

            // 每 0.25 秒更新一次窗口标题，观察视锥裁剪统计
            titleTimer += dt;
            if (titleTimer >= 0.25f) {
                int totalMinutes = static_cast<int>(gameTimeHalfMin * 0.5f) % (24 * 60);
                int hh = totalMinutes / 60;
                int mm = totalMinutes % 60;

                char titleBuf[320];
                std::snprintf(titleBuf, sizeof(titleBuf),
                              "CoA Engine | Time: %02d:%02d | %s | Visible MCNK: %d/%d | Visible WMO Groups: %d | Visible M2 Instances: %d | Water: %d | Batched DrawCalls: %zu",
                              hh, mm, underwater ? "UNDERWATER" : "Surface",
                              visibleChunks, totalChunks, visibleWmoGroups, visibleM2Instances,
                              visibleWaterChunks, renderQueue.size());
                SDL_SetWindowTitle(window, titleBuf);
                titleTimer = 0.0f;
            }

            coa::rhi::SubmitDrawCommands(renderQueue);

            // ---- 粒子公告板渲染：AlphaBlend 炊烟与 Additive 火苗，硬件实例化单次提交 ----
            if (!visibleParticles.empty()) {
                const size_t particleBytes = visibleParticles.size() * sizeof(GpuParticleQuad);
                if (particleBytes > particleSsbo.Size()) {
                    particleSsbo = coa::rhi::GlBuffer(particleBytes, nullptr, GL_DYNAMIC_STORAGE_BIT);
                    particleSsbo.BindAsStorageBuffer(2);
                }
                particleSsbo.UpdateSubData(0, particleBytes, visibleParticles.data());

                glUseProgram(particleShader.Handle());
                glBindVertexArray(skyVao);
                glEnable(GL_BLEND);
                glDepthMask(GL_FALSE);
                glDisable(GL_CULL_FACE);

                GLuint lastParticleTex   = 0;
                int    lastParticleBlend = -1;
                for (const auto& pb : particleBatches) {
                    if (lastParticleBlend != static_cast<int>(pb.blendType)) {
                        const bool additive = (pb.blendType >= 3);
                        if (additive) {
                            glBlendFunc(GL_SRC_ALPHA, GL_ONE);
                        } else {
                            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
                        }
                        glUniform1i(0, additive ? 1 : 0);
                        lastParticleBlend = static_cast<int>(pb.blendType);
                    }
                    if (pb.texId != lastParticleTex) {
                        glBindTextureUnit(0, pb.texId);
                        lastParticleTex = pb.texId;
                    }
                    glDrawArraysInstancedBaseInstance(
                        GL_TRIANGLES, 0, 6,
                        static_cast<GLsizei>(pb.quadCount),
                        pb.firstQuad
                    );
                }

                glDepthMask(GL_TRUE);
                glDisable(GL_BLEND);
            }

            SDL_GL_SwapWindow(window);
        }

        glDeleteVertexArrays(1, &terrainVao);
        glDeleteVertexArrays(1, &sharedM2Vao);
        glDeleteVertexArrays(1, &skyVao);
    }

    SDL_GL_DestroyContext(glCtx);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
