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
#include <future>
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
    glm::mat4 lightViewProj;      // 近景级联阴影光视图投影矩阵 (58 码 / 2048)
    glm::mat4 lightViewProjFar;   // 远景级联阴影光视图投影矩阵 (320 码 / 1024，每 6 帧缓存更新)
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

// 单个 2D HUD 方框的 GPU 实例数据 (32 字节，std430 紧凑排布，对应 kUiVertShader)
struct alignas(16) GpuUiQuad {
    glm::vec4 rectPx; // xy = 屏幕左上角像素坐标, zw = 宽高像素
    glm::vec4 color;  // rgba
};
static_assert(sizeof(GpuUiQuad) == 32, "GpuUiQuad must be 32 bytes");

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
    mat4 uLightViewProj;
    mat4 uLightViewProjFar;
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
    mat4 uLightViewProj;
    mat4 uLightViewProjFar;
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
layout(binding = 5) uniform sampler2DShadow uShadowMap;    // 近景级联 2048x2048 实时阴影 (硬件比较 + 双线性)
layout(binding = 6) uniform sampler2DShadow uShadowMapFar; // 远景级联 1024x1024 缓存阴影 (5 点十字采样)

layout(location = 0) uniform int  uAlphaCutout;   // 0 = Opaque, 1 = Cutout, 2 = Blend, 3 = UV Scroll
layout(location = 1) uniform int  uMaterialFlags; // bit 0 = TwoSided, bit 1 = Unlit
layout(location = 4) uniform vec4 uBatchColor;    // M2Track 颜色与透明度调制
layout(location = 2) uniform int  uBoneBaseIndex; // > 0 表示该批次为骨骼蒙皮角色

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
    mat4 uLightViewProj;
    mat4 uLightViewProjFar;
};

out vec4 FragColor;

float GetLuma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

vec3 HeightBlend(vec3 baseCol, vec3 overCol, float alpha) {
    float hBase = GetLuma(baseCol);
    float hOver = GetLuma(overCol);
    float w = clamp((alpha * 1.4 - 0.2) + (hOver - hBase) * 0.35, 0.0, 1.0);
    return mix(baseCol, overCol, smoothstep(0.0, 1.0, w));
}

// 双级联软阴影：近景 2048 高清 3x3 PCF + 远景 1024 极速 5 点十字采样，44~56 码平滑混合
float CalcShadow(vec3 worldPos, float ndotl) {
    float dXY = length(worldPos.xy - uCameraPos.xy);

    // 260~320 码淡出：远景阴影盒半径 320 码，末端留 60 码缓冲带与远景雾无缝衔接
    float fade = 1.0 - smoothstep(260.0, 320.0, dXY);
    if (fade <= 0.0) return 1.0;

    // ---- 近景级联：58 码半径 / 2048 贴图，3x3 PCF ----
    float visNear = 1.0;
    if (dXY < 56.0) {
        vec4  lc    = uLightViewProj * vec4(worldPos, 1.0);
        vec2  uv    = lc.xy * 0.5 + 0.5;
        float curD  = lc.z;
        if (uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0 && curD > 0.0 && curD < 1.0) {
            float bias = max(0.0015 * (1.0 - ndotl), 0.0004);
            float vis = 0.0;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    vis += texture(uShadowMap, vec3(uv + vec2(dx, dy) / 2048.0, curD - bias));
                }
            }
            visNear = vis / 9.0;
        }
    }

    // ---- 远景级联：320 码半径 / 1024 贴图，仅中心 + 上下左右共 5 次硬件比较 ----
    float visFar = 1.0;
    if (dXY > 44.0) {
        vec4  lc    = uLightViewProjFar * vec4(worldPos, 1.0);
        vec2  uv    = lc.xy * 0.5 + 0.5;
        float curD  = lc.z;
        if (uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0 && curD > 0.0 && curD < 1.0) {
            const float bias = 0.0025;
            const vec2  off  = vec2(1.5 / 1024.0, 0.0);
            float vis = texture(uShadowMapFar, vec3(uv, curD - bias));
            vis += texture(uShadowMapFar, vec3(uv + off,  curD - bias));
            vis += texture(uShadowMapFar, vec3(uv - off,  curD - bias));
            vis += texture(uShadowMapFar, vec3(uv + off.yx, curD - bias));
            vis += texture(uShadowMapFar, vec3(uv - off.yx, curD - bias));
            visFar = vis / 5.0;
        }
    }

    float cascadeBlend = smoothstep(44.0, 56.0, dXY);
    float vis = (dXY <= 44.0) ? visNear : ((dXY >= 56.0) ? visFar : mix(visNear, visFar, cascadeBlend));
    return mix(1.0, vis, fade * 0.75);
}

void main() {
    // 远景景深 (DoF)：超过 55 码后逐级提升 Mip 采样偏置，模拟空气透视并抹掉远景高频闪烁
    float dist    = length(vWorldPos - uCameraPos.xyz);
    float dofBlur = smoothstep(55.0, 210.0, dist);
    float lodBias = dofBlur * 2.6;

    vec4 texColor = texture(uTexAlbedo, vUV, lodBias);
    vec3 alphaWeights = texture(uTexAlphaMap, clamp(vUV / 8.0, 0.0, 1.0)).rgb;
    if (dot(alphaWeights, vec3(1.0)) > 0.001) {
        vec3 c1 = texture(uTexLayer1, vUV, lodBias).rgb;
        vec3 c2 = texture(uTexLayer2, vUV, lodBias).rgb;
        vec3 c3 = texture(uTexLayer3, vUV, lodBias).rgb;
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
        // 统一双面光照：观察背面时翻转法线，避免树冠/旗帜/羽毛背光死黑
        if (!gl_FrontFacing) {
            N = -N;
        }

        float height = GetLuma(texColor.rgb);
        vec3 dPdx = dFdx(vWorldPos);
        vec3 dPdy = dFdy(vWorldPos);
        vec3 r1 = cross(dPdy, N);
        vec3 r2 = cross(N, dPdx);
        float det = dot(dPdx, r1);
        vec3 surfGrad = (abs(det) > 1e-8) ? ((r1 * dFdx(height) + r2 * dFdy(height)) / det) : vec3(0.0);
        // 远景随景深淡出法线扰动，避免远处像素导数噪点在模糊后仍然闪烁
        N = normalize(N - surfGrad * 0.04 * (1.0 - dofBlur));

        vec3 L = normalize(uSunDirAndIntensity.xyz);
        vec3 V = normalize(uCameraPos.xyz - vWorldPos);
        vec3 H = normalize(L + V);

        float ndotl = max(dot(N, L), 0.0);
        float halfLambert = ndotl * 0.65 + 0.35;
        float spec = pow(max(dot(N, H), 0.0), 24.0) * height * 0.04 * ndotl * (1.0 - dofBlur);

        vec3 skyAmbient    = uAmbientSky.rgb;
        vec3 groundAmbient = uAmbientGround.rgb;
        vec3 ambient = mix(groundAmbient, skyAmbient, N.z * 0.5 + 0.5);

        float shadowFactor = CalcShadow(vWorldPos, ndotl);

        vec3 sunColor = uSunColor.rgb * uSunDirAndIntensity.w;
        litColor = (texColor.rgb * uBatchColor.rgb) * (ambient + sunColor * halfLambert * shadowFactor)
                 + sunColor * spec * shadowFactor;
    }

    // 水下躯干遮挡：仅对骨骼蒙皮角色生效，避免污染河床地形与岸边造成深色轮廓闪烁
    if (uBoneBaseIndex > 0) {
        float subDepth = uWaterShallow.w - vWorldPos.z;
        if (subDepth > 0.0) {
            float uwFade = exp(-subDepth * 1.45);
            vec3 uwTint = mix(uWaterDeep.rgb * 0.42, uWaterShallow.rgb * 0.72, uwFade);
            litColor = mix(uwTint, litColor * vec3(0.52, 0.75, 0.85), smoothstep(0.0, 0.25, subDepth) * (1.0 - uwFade * 0.35));
        }
    }

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
    mat4 uLightViewProj;
    mat4 uLightViewProjFar;
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
    mat4 uLightViewProj;
    mat4 uLightViewProjFar;
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
    float shoreAlpha = smoothstep(0.0, 0.16, vWaterDepth);
    float alpha = clamp(shoreAlpha * mix(0.76, 0.95,
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
    mat4 uLightViewProj;
    mat4 uLightViewProjFar;
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

    // w < 0 表示水平贴地光环：沿世界 X/Y 平面展开，半径取 |w|；否则为面片公告板
    vec3 worldPos;
    if (p.worldPosAndSize.w < 0.0) {
        float ringRadius = -p.worldPosAndSize.w;
        worldPos = p.worldPosAndSize.xyz
                 + (vec3(1.0, 0.0, 0.0) * corner.x + vec3(0.0, 1.0, 0.0) * corner.y) * ringRadius;
    } else {
        worldPos = p.worldPosAndSize.xyz
                 + (camRight * corner.x + camUp * corner.y) * p.worldPosAndSize.w;
    }

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
    mat4 uLightViewProj;
    mat4 uLightViewProjFar;
};

out vec4 FragColor;

void main() {
    // vUV.x < -0.5：程序化选中光环 (不采样贴图)，双层翠绿圆环 + 中心柔和填充
    if (vUV.x < -0.5) {
        float r    = length(vCorner) * 2.0;
        float ring = smoothstep(0.72, 0.88, r) * (1.0 - smoothstep(0.88, 0.98, r))
                   + (1.0 - r * r) * 0.22;
        FragColor = vec4(vec3(0.2, 1.0, 0.45) * 1.4, ring * vColor.a);
        return;
    }

    // 粒子贴图普遍以 RGB 亮度携带形状、alpha 恒为 1：按亮度重建透明度以抠除黑底
    vec4  tex      = texture(uTexAlbedo, vUV);
    float luma     = max(tex.r, max(tex.g, tex.b));
    float texAlpha = (tex.a < 0.96) ? tex.a : luma;
    vec4  c        = vec4(tex.rgb * vColor.rgb, texAlpha * vColor.a);

    // 高斯云团衰减：中心实、边缘柔和消散，避免炊烟出现硬边白块
    float r = clamp(length(vCorner) * 2.0, 0.0, 1.0);
    float radial = pow(1.0 - r * r, 2.2);
    c.a *= radial;
    if (c.a < 0.004) discard;

    float dist      = length(vWorldPos - uCameraPos.xyz);
    float fogFactor = clamp(exp(-max(dist - uFogColorAndParams.w, 0.0) * uSkyTopAndDensity.w), 0.0, 1.0);

    vec3 litRgb = (uAdditive != 0) ? (c.rgb * fogFactor)
                                   : mix(uFogColorAndParams.rgb, c.rgb, fogFactor);
    FragColor = vec4(litRgb, c.a);
}
)GLSL";

// 2D HUD 顶点着色器：gl_VertexID 展开 6 顶点矩形，按左上角像素原点换算到 NDC
constexpr std::string_view kUiVertShader = R"GLSL(
#version 460 core
struct GpuUiQuad {
    vec4 rectPx; // xy = 屏幕左上角像素坐标, zw = 宽高
    vec4 color;  // rgba
};

layout(std430, binding = 3) readonly buffer UiBuffer {
    GpuUiQuad uQuads[];
};

layout(location = 0) uniform vec2 uScreenSize;

out vec4 vColor;
out vec2 vLocal;

const vec2 kQuadCorners[6] = vec2[6](
    vec2(0.0, 0.0), vec2(1.0, 0.0), vec2(1.0, 1.0),
    vec2(0.0, 0.0), vec2(1.0, 1.0), vec2(0.0, 1.0)
);

void main() {
    GpuUiQuad q  = uQuads[gl_BaseInstance + gl_InstanceID];
    vec2 corner  = kQuadCorners[gl_VertexID % 6];
    vec2 px      = q.rectPx.xy + corner * q.rectPx.zw;
    vec2 ndc     = vec2(px.x / max(uScreenSize.x, 1.0),
                        1.0 - px.y / max(uScreenSize.y, 1.0)) * 2.0 - 1.0;
    gl_Position  = vec4(ndc, 0.0, 1.0);
    vColor       = q.color;
    vLocal       = corner;
}
)GLSL";

// 2D HUD 片段着色器：纯色填充 + 约 1 像素的暗边描边，让方框具备立体层次
constexpr std::string_view kUiFragShader = R"GLSL(
#version 460 core
in vec4 vColor;
in vec2 vLocal;

out vec4 FragColor;

void main() {
    float edge   = min(min(vLocal.x, 1.0 - vLocal.x), min(vLocal.y, 1.0 - vLocal.y));
    float border = smoothstep(0.0, 0.12, edge);
    FragColor = vec4(vColor.rgb * mix(0.55, 1.0, border), vColor.a);
}
)GLSL";

// 阴影深度顶点着色器：复用世界顶点布局 + 骨骼蒙皮 + 硬件实例化，仅输出光空间深度
constexpr std::string_view kShadowVertShader = R"GLSL(
#version 460 core
layout(location = 0) in vec3  aPos;
layout(location = 1) in vec3  aNormal;
layout(location = 2) in vec2  aUV;
layout(location = 3) in vec4  aBoneWeights;
layout(location = 4) in uvec4 aBoneIndices;

layout(location = 2) uniform int uBoneBaseIndex;

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
    mat4 uLightViewProj;
    mat4 uLightViewProjFar;
};

layout(std430, binding = 0) readonly buffer InstanceBuffer {
    mat4 uInstanceMatrices[];
};

layout(std430, binding = 1) readonly buffer BoneBuffer {
    mat4 uBoneMatrices[];
};

out vec2 vUV;

void main() {
    vec3 localPos = aPos;

    if (uBoneBaseIndex > 0) {
        float wSum = aBoneWeights.x + aBoneWeights.y + aBoneWeights.z + aBoneWeights.w;
        if (wSum > 0.001) {
            mat4 skinMat = (aBoneWeights.x * uBoneMatrices[uBoneBaseIndex + int(aBoneIndices.x)]
                          + aBoneWeights.y * uBoneMatrices[uBoneBaseIndex + int(aBoneIndices.y)]
                          + aBoneWeights.z * uBoneMatrices[uBoneBaseIndex + int(aBoneIndices.z)]
                          + aBoneWeights.w * uBoneMatrices[uBoneBaseIndex + int(aBoneIndices.w)]) / wSum;
            localPos = (skinMat * vec4(aPos, 1.0)).xyz;
        }
    }

    mat4 modelMat = uInstanceMatrices[gl_BaseInstance + gl_InstanceID];
    vec4 worldPos = modelMat * vec4(localPos, 1.0);
    gl_Position   = uLightViewProj * worldPos;
    vUV           = aUV;
}
)GLSL";

// 阴影深度片段着色器：树叶 Alpha Cutout 镂空阴影，无颜色输出
constexpr std::string_view kShadowFragShader = R"GLSL(
#version 460 core
in vec2 vUV;

layout(binding = 0) uniform sampler2D uTexAlbedo;
layout(location = 0) uniform int uAlphaCutout;

void main() {
    if (uAlphaCutout == 1 && texture(uTexAlbedo, vUV).a < 0.5) {
        discard;
    }
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
    glDepthFunc(GL_GEQUAL);
    glClearDepth(0.0);
    glEnable(GL_MULTISAMPLE);

    {
        coa::rhi::GlBuffer frameUbo(sizeof(FrameUniforms));
        frameUbo.BindAsUniformBuffer(0);

        coa::rhi::GlShaderProgram worldShader(kWorldVertShader, kWorldFragShader);
        coa::rhi::GlShaderProgram waterShader(kWaterVertShader, kWaterFragShader);
        coa::rhi::GlShaderProgram skyShader(kSkyVertShader, kSkyFragShader);
        coa::rhi::GlShaderProgram particleShader(kParticleVertShader, kParticleFragShader);
        coa::rhi::GlShaderProgram uiShader(kUiVertShader, kUiFragShader);
        coa::rhi::GlShaderProgram shadowShader(kShadowVertShader, kShadowFragShader);
        GLuint sharedM2Vao = coa::assets::GpuM2Model::CreateSharedVAO();
        GLuint terrainVao  = coa::assets::GpuAdtTile::CreateTerrainVAO();
        GLuint skyVao      = 0;
        glCreateVertexArrays(1, &skyVao);

        // ---- 双级联实时阴影贴图 (仅深度)：近景 2048 写实 + 远景 1024 极低开销 ----
        auto CreateShadowTexture = [](GLuint& tex, GLuint& fbo, GLsizei size) {
            glCreateTextures(GL_TEXTURE_2D, 1, &tex);
            glTextureStorage2D(tex, 1, GL_DEPTH_COMPONENT24, size, size);
            glTextureParameteri(tex, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTextureParameteri(tex, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glTextureParameteri(tex, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
            glTextureParameteri(tex, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
            // 硬件深度比较 + 双线性过滤：每次采样即 2x2 纹素软过渡，消除阴影边缘方块锯齿
            glTextureParameteri(tex, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_REF_TO_TEXTURE);
            glTextureParameteri(tex, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL);
            const float border[4] = {1.0f, 1.0f, 1.0f, 1.0f};
            glTextureParameterfv(tex, GL_TEXTURE_BORDER_COLOR, border);

            glCreateFramebuffers(1, &fbo);
            glNamedFramebufferTexture(fbo, GL_DEPTH_ATTACHMENT, tex, 0);
            glNamedFramebufferDrawBuffer(fbo, GL_NONE);
            glNamedFramebufferReadBuffer(fbo, GL_NONE);
        };

        GLuint shadowTex    = 0, shadowTexFar = 0;
        GLuint shadowFbo    = 0, shadowFboFar = 0;
        CreateShadowTexture(shadowTex, shadowFbo, 2048);
        CreateShadowTexture(shadowTexFar, shadowFboFar, 1024);

        // 远景级联在首个可用刷新帧之前先清成「全亮」(深度 1.0)，避免采样到未定义深度而整屏变黑
        glClearDepth(1.0);
        glBindFramebuffer(GL_FRAMEBUFFER, shadowFboFar);
        glClear(GL_DEPTH_BUFFER_BIT);
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glClearDepth(0.0); // 恢复反向 Z 主渲染的默认清屏值

        // 每帧动态上传可见实例矩阵的 SSBO (binding = 0) 与 M2 骨骼蒙皮矩阵的 SSBO (binding = 1)
        coa::rhi::GlBuffer instanceSsbo(sizeof(glm::mat4), nullptr, GL_DYNAMIC_STORAGE_BIT);
        instanceSsbo.BindAsStorageBuffer(0);
        coa::rhi::GlBuffer boneSsbo(sizeof(glm::mat4), nullptr, GL_DYNAMIC_STORAGE_BIT);
        boneSsbo.BindAsStorageBuffer(1);
        // binding = 2: 本帧全部公告板粒子实例数据
        coa::rhi::GlBuffer particleSsbo(sizeof(GpuParticleQuad), nullptr, GL_DYNAMIC_STORAGE_BIT);
        particleSsbo.BindAsStorageBuffer(2);
        // binding = 3: 本帧全部 2D HUD 方框实例数据
        coa::rhi::GlBuffer uiSsbo(sizeof(GpuUiQuad), nullptr, GL_DYNAMIC_STORAGE_BIT);
        uiSsbo.BindAsStorageBuffer(3);

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

        // 多 ADT 无缝拼接：3x2 共 6 块艾尔文森林与暴风城周边区域
        // 32_48 保持首位，使初始出生点仍位于北郡
        // 方向 I：基于相机坐标的 3x3 动态 ADT 流式加载与卸载
        std::vector<coa::assets::GpuAdtTile> adtTiles;
        adtTiles.reserve(32);                  // 预留容量：降低 vector 扩容使实例指针失效的频率
        std::unordered_set<uint32_t> seenUniqueIds;
        std::unordered_set<int> failedAdtKeys; // 键值 = (tileX << 8) | tileY，避免反复重试缺失文件
        float streamCooldown = 0.25f;          // 相邻两次流式加载之间的冷却时间，避免瞬移时卡顿

        // 方向 I：后台线程纯 CPU 解包 + 主线程每帧 2ms 时间片分步上传 GPU
        std::future<std::optional<coa::assets::AdtLoader::CpuAdtData>> asyncAdtFuture;
        std::optional<coa::assets::AdtLoader::CpuAdtData>              pendingAdtCpu;
        int loadingTx = -1;
        int loadingTy = -1;

        // 跨 ADT 全局 M2 合批桶：按 uniqueId 去重后直接持有实例指针，避免跨界 ADT 重复绘制同一棵树
        struct GlobalM2Bucket {
            const coa::assets::GpuM2Model* model = nullptr;
            std::vector<const coa::assets::M2Instance*> instances;
        };
        std::vector<GlobalM2Bucket> globalM2Buckets;

        // 流式加载 / 卸载后重建全局 M2 合批桶 (实例指针可能已失效)
        auto rebuildGlobalM2Buckets = [&]() {
            globalM2Buckets.clear();
            std::unordered_map<const coa::assets::GpuM2Model*, size_t> bucketMap;
            std::unordered_set<uint32_t> seenM2Ids; // 跨界 ADT 的同一 uniqueId 仅保留一份
            // 预分配容量：避免每次重建桶时反复 rehash 造成瞬时卡顿
            globalM2Buckets.reserve(assets.CachedM2Count() + 16);
            bucketMap.reserve(assets.CachedM2Count() + 16);
            seenM2Ids.reserve(16384);
            for (const auto& tile : adtTiles) {
                for (const auto& batch : tile.doodadBatches) {
                    const auto* modelPtr = batch.model.get();
                    if (!modelPtr) continue;
                    auto [it, inserted] = bucketMap.try_emplace(modelPtr, globalM2Buckets.size());
                    if (inserted) {
                        globalM2Buckets.push_back(GlobalM2Bucket{modelPtr, {}});
                    }
                    auto& bucket = globalM2Buckets[it->second];
                    for (const auto& inst : batch.instances) {
                        // uniqueId == 0 为 WMO 内部 MODD 家具，没有全局唯一性，全部保留
                        if (inst.uniqueId != 0 && !seenM2Ids.insert(inst.uniqueId).second) {
                            continue;
                        }
                        bucket.instances.push_back(&inst);
                    }
                }
            }
        };

        // 初始仅加载出生点所在的北郡 (32, 48)，其余区块由主循环按 3x3 视野动态流式加载
        {
            auto tile = coa::assets::AdtLoader::LoadAdt(
                "world/maps/azeroth/azeroth_32_48.adt", vfs, assets,
                whiteFallbackTex.Handle(), seenUniqueIds
            );
            if (tile) {
                tile->tileX = 32;
                tile->tileY = 48;
                adtTiles.push_back(std::move(*tile));
            }
        }
        rebuildGlobalM2Buckets();

        // 方向 J：第三人称角色模型 (优先阿尔萨斯，回退农民 / 狮鹫)
        std::shared_ptr<const coa::assets::GpuM2Model> playerModel =
            assets.GetOrLoadM2("creature/arthas/arthas.m2", whiteFallbackTex.Handle());
        if (!playerModel || playerModel->submeshes.empty()) {
            playerModel = assets.GetOrLoadM2("creature/peasant/peasant.m2", whiteFallbackTex.Handle());
        }
        if (!playerModel || playerModel->submeshes.empty()) {
            playerModel = assets.GetOrLoadM2(
                "world/generic/human/passive doodads/gryphonroost/gryphonroost01.m2",
                whiteFallbackTex.Handle());
        }

        // 自动将摄像机瞬移至首个成功加载的 ADT 地形中心上空
        glm::vec3 camPos = (!adtTiles.empty()) ? adtTiles.front().suggestedCamPos : glm::vec3(0.0f, 0.0f, 20.0f);
        float camYaw   = 0.0f;
        float camPitch = -0.25f;
        float gameTimeHalfMin = 1560.0f; // 默认 13:00，金色阳光
        bool rightMouseDown = false;
        // 方向 G：角色贴地行走 / 自由飞行双模式
        bool  walkMode    = false;
        float verticalVel = 0.0f;
        bool  isGrounded  = false;
        bool  isSwimming  = false; // 方向 L：游泳 / 潜泳状态
        constexpr float kEyeHeight = 1.85f;
        // 方向 J：第三人称角色状态 (playerPos.z 直接代表脚底高度)
        glm::vec3 playerPos = camPos - glm::vec3(0.0f, 0.0f, kEyeHeight);
        float     playerYaw = 0.0f;
        float     camDist   = 4.5f;
        int       playerAnimAction = 0;
        uint64_t lastTicks = SDL_GetTicksNS();

        int visibleChunks = 0;
        int totalChunks = 0;
        int visibleWmoGroups = 0;
        int visibleM2Instances = 0;
        int visibleWaterChunks = 0;
        float titleTimer = 0.0f;

        std::vector<coa::rhi::DrawCommand> renderQueue;
        // 独立轻量阴影队列：不含 MCNK 地形，仅收纳阴影半径内近处 M2/角色/WMO 批次
        std::vector<coa::rhi::DrawCommand> shadowQueue;
        // 远景级联阴影队列：仅大尺寸投射物，且每 6 帧才真正渲染一次
        std::vector<coa::rhi::DrawCommand> shadowFarQueue;
        // M2 实例按阴影级联距离分片排列的复用缓冲，跨帧保留容量避免反复分配
        std::vector<glm::mat4> nearMats;
        std::vector<glm::mat4> farMats;
        std::vector<glm::mat4> viewOnlyMats;
        glm::mat4 lastLightViewProjFar(1.0f); // 远景级联矩阵缓存，非刷新帧直接沿用
        uint32_t  frameCounter = 0;
        std::vector<glm::mat4> visibleMatrices;
        std::vector<glm::mat4> visibleBoneMatrices;
        std::vector<GpuParticleQuad>  visibleParticles;
        std::vector<ParticleBatchCmd> particleBatches;
        std::vector<GpuUiQuad>        uiQuads;
        // 方向 M：鼠标左键射线拾取的当前选中目标 (ESC 清空)
        coa::assets::AdtLoader::PickResult currentPick{};

        // 方向 M/N：各地标前放置可点击选中、可自主巡逻并拥有生命值的 NPC 实体
        struct LandmarkNpc {
            glm::vec3 spawn{0.0f};       // 巡逻起点 (Z 每帧按地形吸附)
            glm::vec3 worldPos{0.0f};    // 每帧更新后的脚底世界坐标
            bool      visible  = false;
            bool      initialized   = false; // 是否已完成首次地面吸附初始化
            bool      patrolForward = true;  // 当前巡逻方向：true = +X, false = -X
            float     pauseTimer    = 0.0f;  // 折返后或起步时的原地停顿时长
            float     yaw      = 0.0f;
            int       animAction = 0;     // 0=Stand, 1=Walk
            float     hp       = 100.0f;
            float     maxHp    = 100.0f;
        };
        std::vector<LandmarkNpc> landmarkNpcs = {
            { glm::vec3(-8942.0f, -165.0f,  83.0f), glm::vec3(0.0f), false }, // F1 北郡修道院南侧开阔大道
            { glm::vec3(-9450.0f,   62.0f,  60.0f), glm::vec3(0.0f), false }, // F2 闪金镇旅馆门前大道
            { glm::vec3(-9050.0f,  435.0f,  96.0f), glm::vec3(0.0f), false }, // F3 暴风城英雄谷大桥桥面正中央
            { glm::vec3(-8528.0f,  845.0f, 110.0f), glm::vec3(0.0f), false }, // F4 暴风城大教堂广场中央
            { glm::vec3(-9258.0f, -185.0f,  64.0f), glm::vec3(0.0f), false }, // F5 水晶湖北岸开阔草地
        };

        // 各 NPC 起步停顿错开，避免整排 NPC 同步齐步走
        for (size_t i = 0; i < landmarkNpcs.size(); ++i) {
            landmarkNpcs[i].pauseTimer = static_cast<float>(i) * 0.35f;
        }

        // 方向 N：玩家攻击动作计时、施法读条计时与魔法飞弹状态
        float attackAnimTimer = 0.0f;
        float castBarTimer    = 0.0f;
        struct SpellProjectile {
            bool      active    = false;
            glm::vec3 pos{0.0f};
            size_t    targetNpc = 0;
        };
        SpellProjectile activeProj{};

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
                } else if (ev.type == SDL_EVENT_KEY_DOWN && ev.key.scancode == SDL_SCANCODE_TAB) {
                    walkMode    = !walkMode;
                    verticalVel = 0.0f;
                    isGrounded  = false;
                    isSwimming  = false;
                    if (walkMode) {
                        // 进入第三人称模式：角色出生在相机下方地面，朝向沿用当前水平视角
                        playerPos = camPos - glm::vec3(0.0f, 0.0f, kEyeHeight);
                        playerYaw = camYaw;
                    }
                } else if (ev.type == SDL_EVENT_KEY_DOWN && ev.key.scancode == SDL_SCANCODE_ESCAPE) {
                    // 方向 M：ESC 取消当前选中目标并移除脚下光环
                    currentPick = {};
                } else if (ev.type == SDL_EVENT_KEY_DOWN &&
                           (ev.key.scancode == SDL_SCANCODE_1 || ev.key.scancode == SDL_SCANCODE_2)) {
                    // 方向 N：对着当前选中的 NPC 施放 1/2 号法术，触发攻击动作与魔法飞弹
                    if (currentPick.hit && currentPick.uniqueId >= 1000) {
                        const size_t idx = static_cast<size_t>(currentPick.uniqueId - 1000);
                        if (idx < landmarkNpcs.size() && landmarkNpcs[idx].visible) {
                            const glm::vec3& npcPos = landmarkNpcs[idx].worldPos;
                            const glm::vec3  castFrom = walkMode ? playerPos : camPos;
                            if (walkMode) {
                                playerYaw = std::atan2(npcPos.y - playerPos.y, npcPos.x - playerPos.x);
                            } else {
                                camYaw = std::atan2(npcPos.y - camPos.y, npcPos.x - camPos.x);
                            }
                            attackAnimTimer = 0.65f;
                            castBarTimer    = 0.45f;
                            activeProj.active    = true;
                            activeProj.pos       = castFrom + glm::vec3(0.0f, 0.0f, 1.3f);
                            activeProj.targetNpc = idx;
                            std::printf("[Spell] Cast %s at NPC #%zu (hp:%.0f)\n",
                                        (ev.key.scancode == SDL_SCANCODE_1) ? "Firebolt" : "Frostbolt",
                                        idx, landmarkNpcs[idx].hp);
                        }
                    }
                } else if (ev.type == SDL_EVENT_MOUSE_BUTTON_DOWN && ev.button.button == SDL_BUTTON_LEFT) {
                    // 方向 M：鼠标左键发射世界空间射线，仅精确拾取场景 NPC 实体
                    // (背景树木 / 建筑不参与拾取，避免选中光环错位到树冠或整栋建筑)
                    const float mx = ev.button.x;
                    const float my = ev.button.y;
                    const float ndcX = (2.0f * mx) / static_cast<float>(std::max(1, winW)) - 1.0f;
                    const float ndcY = 1.0f - (2.0f * my) / static_cast<float>(std::max(1, winH));
                    const float tanHalfFov = std::tan(glm::radians(60.0f) * 0.5f);
                    const float pickAspect = (winH > 0)
                        ? (static_cast<float>(winW) / static_cast<float>(winH)) : 1.777f;

                    // 相机基向量 (与主循环 forward / right / up 完全一致)
                    const glm::vec3 pickFwd(
                        std::cos(camPitch) * std::cos(camYaw),
                        std::cos(camPitch) * std::sin(camYaw),
                        std::sin(camPitch));
                    const glm::vec3 pickRight =
                        glm::normalize(glm::cross(pickFwd, glm::vec3(0.0f, 0.0f, 1.0f)));
                    const glm::vec3 pickUp = glm::normalize(glm::cross(pickRight, pickFwd));

                    const glm::vec3 rayDir = glm::normalize(
                        pickFwd
                        + pickRight * (ndcX * tanHalfFov * pickAspect)
                        + pickUp * (ndcY * tanHalfFov));

                    currentPick = {};

                    // 地标 NPC 拾取：局部盒 [-0.8,-0.8,0] ~ [0.8,0.8,2.1]，取沿射线最近的命中
                    for (size_t npcIdx = 0; npcIdx < landmarkNpcs.size(); ++npcIdx) {
                        const auto& npc = landmarkNpcs[npcIdx];
                        if (!npc.visible) continue;
                        const glm::vec3 bmin = npc.worldPos + glm::vec3(-0.8f, -0.8f, 0.0f);
                        const glm::vec3 bmax = npc.worldPos + glm::vec3( 0.8f,  0.8f, 2.1f);
                        float tNear = -1e30f;
                        float tFar  =  1e30f;
                        bool  hitBox = true;
                        for (int i = 0; i < 3 && hitBox; ++i) {
                            if (std::fabs(rayDir[i]) < 1e-8f) {
                                if (camPos[i] < bmin[i] || camPos[i] > bmax[i]) hitBox = false;
                            } else {
                                const float invD = 1.0f / rayDir[i];
                                float t1 = (bmin[i] - camPos[i]) * invD;
                                float t2 = (bmax[i] - camPos[i]) * invD;
                                if (t1 > t2) std::swap(t1, t2);
                                tNear = std::max(tNear, t1);
                                tFar  = std::min(tFar,  t2);
                                if (tNear > tFar) hitBox = false;
                            }
                        }
                        if (!hitBox || tNear <= 0.05f || tNear > 250.0f) continue;
                        if (tNear >= currentPick.distance) continue;

                        currentPick.hit       = true;
                        currentPick.isWmo     = false;
                        currentPick.distance  = tNear;
                        currentPick.uniqueId  = static_cast<uint32_t>(npcIdx + 1000);
                        currentPick.center    = npc.worldPos + glm::vec3(0.0f, 0.0f, 1.05f);
                        currentPick.groundPos = npc.worldPos;
                        currentPick.radius    = 2.0f;
                        currentPick.modelPath = "NPC: Stormwind Guard / Hero";
                    }

                    if (currentPick.hit) {
                        std::printf("[Picker] Selected NPC (dist:%.1f, radius:%.1f): %s\n",
                                    currentPick.distance, currentPick.radius,
                                    currentPick.modelPath.c_str());
                    } else {
                        std::printf("[Picker] No NPC hit (max 250 yards).\n");
                    }
                } else if (ev.type == SDL_EVENT_KEY_DOWN &&
                           ev.key.scancode >= SDL_SCANCODE_F1 &&
                           ev.key.scancode <= SDL_SCANCODE_F5) {
                    // F1~F5 地标瞬移热键
                    static const glm::vec3 kLandmarks[5] = {
                        glm::vec3(-8952.0f, -165.0f,  83.0f), // F1 北郡修道院南侧开阔大道
                        glm::vec3(-9460.0f,   62.0f,  60.0f), // F2 闪金镇
                        glm::vec3(-9065.0f,  435.0f,  96.0f), // F3 暴风城英雄谷大门
                        glm::vec3(-8540.0f,  845.0f, 110.0f), // F4 暴风城大教堂广场
                        glm::vec3(-9268.0f, -185.0f,  64.0f), // F5 水晶湖北岸开阔草地
                    };
                    const int idx = static_cast<int>(ev.key.scancode) - static_cast<int>(SDL_SCANCODE_F1);
                    camPos      = kLandmarks[idx];
                    verticalVel = 0.0f;
                    isGrounded  = false;
                    isSwimming  = false;
                    playerPos   = camPos - glm::vec3(0.0f, 0.0f, kEyeHeight);
                    std::printf("[Teleport] Landmark F%d -> (%.1f, %.1f, %.1f)\n",
                                idx + 1, camPos.x, camPos.y, camPos.z);
                } else if (ev.type == SDL_EVENT_MOUSE_BUTTON_DOWN && ev.button.button == SDL_BUTTON_RIGHT) {
                    rightMouseDown = true;
                    SDL_SetWindowRelativeMouseMode(window, true);
                } else if (ev.type == SDL_EVENT_MOUSE_BUTTON_UP && ev.button.button == SDL_BUTTON_RIGHT) {
                    rightMouseDown = false;
                    SDL_SetWindowRelativeMouseMode(window, false);
                } else if (ev.type == SDL_EVENT_MOUSE_MOTION && rightMouseDown) {
                    camYaw   -= ev.motion.xrel * 0.0025f;
                    camPitch  = std::clamp(camPitch - ev.motion.yrel * 0.0025f, -1.5f, 1.5f);
                } else if (ev.type == SDL_EVENT_MOUSE_WHEEL && walkMode) {
                    // 方向 J：滚轮缩放第三人称弹簧臂相机距离
                    camDist = std::clamp(camDist - ev.wheel.y * 0.6f, 0.0f, 12.0f);
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
            if (walkMode) {
                // ---- 第三人称角色移动：水平碰撞 + 跳跃 / 游泳 + 重力 + 地面吸附 (playerPos.z 即脚底高度) ----
                glm::vec3 fwdFlat(forward.x, forward.y, 0.0f);
                if (glm::length(fwdFlat) > 1e-4f) fwdFlat = glm::normalize(fwdFlat);
                glm::vec3 flatRight = glm::normalize(glm::cross(fwdFlat, worldUp));

                // 方向 L：采样脚下地面高度与水面高度，判定游泳 / 潜泳状态
                constexpr float kSubmergeOffset = 1.45f; // 吃水线：水面没过角色胸口至肩膀
                float groundZ = playerPos.z;
                const bool hasGround = coa::assets::AdtLoader::SampleGroundHeight(
                    adtTiles, playerPos.x, playerPos.y, playerPos.z, groundZ);

                float waterZ = -1e30f;
                const bool hasWater = coa::assets::AdtLoader::SampleWaterHeight(
                    adtTiles, playerPos.x, playerPos.y, waterZ);

                // 深水且脚底已沉入吃水线以下 -> 进入游泳；浅水 / 离水 -> 立刻退出
                if (hasWater && (waterZ - groundZ) > 1.55f &&
                    playerPos.z <= waterZ - kSubmergeOffset + 0.15f) {
                    isSwimming = true;
                }
                if (!hasWater || (waterZ - groundZ) <= 1.25f) {
                    isSwimming = false;
                }

                const float walkSpeed = (keys[SDL_SCANCODE_LSHIFT] ? 14.0f : 7.0f) * dt;
                const float swimSpeed = (keys[SDL_SCANCODE_LSHIFT] ? 11.0f : 6.0f) * dt;
                const float moveSpeed = isSwimming ? swimSpeed : walkSpeed;

                // 方向 H：先累积期望水平位移，再交给 Collide-and-Slide 解析
                glm::vec2 moveDelta(0.0f);
                if (keys[SDL_SCANCODE_W]) moveDelta += glm::vec2(fwdFlat.x, fwdFlat.y) * moveSpeed;
                if (keys[SDL_SCANCODE_S]) moveDelta -= glm::vec2(fwdFlat.x, fwdFlat.y) * moveSpeed;
                if (keys[SDL_SCANCODE_D]) moveDelta += glm::vec2(flatRight.x, flatRight.y) * moveSpeed;
                if (keys[SDL_SCANCODE_A]) moveDelta -= glm::vec2(flatRight.x, flatRight.y) * moveSpeed;

                // 方向 L：游泳时 W/S 沿视线俯仰分量潜泳 / 上浮，SPACE 上浮，LCTRL 下潜
                float swimVertical = 0.0f;
                if (isSwimming) {
                    if (keys[SDL_SCANCODE_W])     swimVertical += forward.z * swimSpeed;
                    if (keys[SDL_SCANCODE_S])     swimVertical -= forward.z * swimSpeed;
                    if (keys[SDL_SCANCODE_SPACE]) swimVertical += 5.5f * dt;
                    if (keys[SDL_SCANCODE_LCTRL]) swimVertical -= 5.5f * dt;
                }

                const float curFootZ = playerPos.z;
                const glm::vec2 resolvedXY = coa::assets::AdtLoader::ResolveHorizontalCollision(
                    adtTiles, glm::vec2(playerPos.x, playerPos.y), moveDelta, curFootZ);

                // 陡坡防穿模保护：贴地状态下前方地面骤升过高 (超过 60 度陡崖) 时保持原地
                float nextGroundZ = curFootZ;
                const bool hasNextGround = coa::assets::AdtLoader::SampleGroundHeight(
                    adtTiles, resolvedXY.x, resolvedXY.y, curFootZ, nextGroundZ);

                if (isGrounded && hasNextGround && (nextGroundZ - curFootZ) > 1.15f) {
                    // 目标点地面过高：拒绝该水平位移
                } else {
                    playerPos.x = resolvedXY.x;
                    playerPos.y = resolvedXY.y;
                }

                if (isSwimming) {
                    // 游泳 / 潜泳：水面吃水线吸附 + 湖底限制，垂直速度归零避免与重力叠加冲突
                    float newZ = playerPos.z + swimVertical;
                    if (swimVertical >= 0.0f && playerPos.z >= waterZ - kSubmergeOffset - 0.1f) {
                        newZ = waterZ - kSubmergeOffset; // 水面附近且未下潜 -> 稳定吸附在吃水线
                    }
                    newZ = std::min(newZ, waterZ - kSubmergeOffset); // 不允许浮出水面之上
                    if (hasGround) newZ = std::max(newZ, groundZ);   // 不允许穿透湖底
                    playerPos.z = newZ;
                    verticalVel = 0.0f;
                    isGrounded  = false;
                } else {
                    if (keys[SDL_SCANCODE_SPACE] && isGrounded) {
                        verticalVel = 8.2f;
                        isGrounded  = false;
                    }

                    verticalVel -= 22.0f * dt;
                    float footZ = playerPos.z + verticalVel * dt;
                    groundZ = playerPos.z; // 水平位移后重新采样落脚点地面高度

                    if (coa::assets::AdtLoader::SampleGroundHeight(adtTiles, playerPos.x, playerPos.y,
                                                                  playerPos.z, groundZ)) {
                        if (footZ <= groundZ + 0.08f && verticalVel <= 0.0f) {
                            footZ       = groundZ;
                            verticalVel = 0.0f;
                            isGrounded  = true;
                        } else {
                            isGrounded = false;
                        }
                    } else {
                        isGrounded = false;
                    }

                    playerPos.z = footZ;
                }

                // 角色朝向与骨骼动作状态机：Stand(0) / Walk(1) / Run(2) / Jump(3) / SwimIdle(4) / Swim(5)
                if (glm::length(moveDelta) > 1e-4f) {
                    playerYaw = std::atan2(moveDelta.y, moveDelta.x);
                }
                if (isSwimming) {
                    // 划水 (5) 与踩水待机 (4)：只要有水平/垂直推进输入即视为划水
                    playerAnimAction = (glm::length(moveDelta) > 1e-4f ||
                                        keys[SDL_SCANCODE_SPACE] ||
                                        keys[SDL_SCANCODE_LCTRL]) ? 5 : 4;
                } else if (!isGrounded) {
                    playerAnimAction = 3;
                } else if (glm::length(moveDelta) > 1e-4f) {
                    playerAnimAction = keys[SDL_SCANCODE_LSHIFT] ? 2 : 1;
                } else {
                    playerAnimAction = 0;
                }

                // 第三人称弹簧臂相机：从头顶枢轴沿视线后退 camDist，并防止镜头穿入地面
                const glm::vec3 pivotPos = playerPos + glm::vec3(0.0f, 0.0f, kEyeHeight);
                camPos = pivotPos - forward * camDist;
                if (camDist > 0.2f) {
                    float camGroundZ = camPos.z;
                    if (coa::assets::AdtLoader::SampleGroundHeight(adtTiles, camPos.x, camPos.y,
                                                                  camPos.z, camGroundZ) &&
                        camGroundZ + 0.35f > camPos.z) {
                        camPos.z = camGroundZ + 0.35f;
                    }
                }
            } else {
                // ---- 自由飞行模式：按住 Shift 高速飞行 (120 码/秒)，方便跨越 533x533 码的 ADT 大地图 ----
                float moveSpeed = (keys[SDL_SCANCODE_LSHIFT] ? 120.0f : 25.0f) * dt;
                if (keys[SDL_SCANCODE_W]) camPos += forward * moveSpeed;
                if (keys[SDL_SCANCODE_S]) camPos -= forward * moveSpeed;
                if (keys[SDL_SCANCODE_D]) camPos += right * moveSpeed;
                if (keys[SDL_SCANCODE_A]) camPos -= right * moveSpeed;
                if (keys[SDL_SCANCODE_SPACE]) camPos += worldUp * moveSpeed;
                if (keys[SDL_SCANCODE_LCTRL]) camPos -= worldUp * moveSpeed;
            }

            // 方向 N：攻击动作与施法读条计时，攻击期间强制覆盖骨骼动作为 Attack/Spell(6)
            if (attackAnimTimer > 0.0f) {
                attackAnimTimer -= dt;
                if (walkMode) playerAnimAction = 6;
            }
            if (castBarTimer > 0.0f) {
                castBarTimer -= dt;
            }

            // ---- 方向 I：基于真实距离的 ADT 流式加载与卸载 ----
            {
                const glm::ivec2 curTile = coa::assets::AdtLoader::WorldToAdtIndex(camPos);

                // 冷却计时：相邻两次流式加载之间至少间隔一段时间，避免瞬移时集中解码造成卡顿
                streamCooldown -= dt;
                if (streamCooldown < 0.0f) streamCooldown = 0.0f;

                // 本帧是否已执行过"卸载 / 并入"操作：两者互斥，避免单帧叠加过多工作
                bool didUnloadOrMerge = false;

                // 1. 渐进式卸载超出 580 码的区块：每帧最多释放 32 张遮罩纹理，或彻底卸载 1 个区块，
                //    避免单帧析构大量 GL 纹理与合批桶重建叠加造成明显卡顿
                for (auto it = adtTiles.begin(); it != adtTiles.end();) {
                    const bool valid = (it->tileX >= 0 && it->tileY >= 0);
                    const float dist = valid
                        ? coa::assets::AdtLoader::DistanceToAdtTile(camPos, it->tileX, it->tileY)
                        : 0.0f;
                    if (valid && dist > 580.0f) {
                        if (it->alphaTextures.size() > 32) {
                            // 仅弹出末尾 32 张遮罩纹理，纹理析构开销摊薄到多帧
                            it->alphaTextures.resize(it->alphaTextures.size() - 32);
                            break;
                        }
                        for (uint32_t uid : it->ownedUniqueIds) seenUniqueIds.erase(uid);
                        adtTiles.erase(it);
                        // 立即在当前帧重建合批桶：erase 会使其后所有 GpuAdtTile 前移，
                        // globalM2Buckets 中的实例指针必须马上刷新，绝不跨帧悬空
                        rebuildGlobalM2Buckets();
                        didUnloadOrMerge = true;
                        break; // 每帧最多彻底卸载 1 个区块
                    }
                    ++it;
                }

                // 2. 主线程分步 GPU 上传：每帧最多消耗 0.45ms，避免贴图解码 / 建缓冲造成掉帧
                if (didUnloadOrMerge) {
                    // 本帧已执行卸载 / 并入：跳过后续 GPU 上传与后台轮询，避免单帧叠加过多工作
                } else if (pendingAdtCpu.has_value()) {
                    if (coa::assets::AdtLoader::StepFinalizeAdtGpu(
                            *pendingAdtCpu, assets, whiteFallbackTex.Handle(), 450'000ULL)) {
                        adtTiles.push_back(std::move(pendingAdtCpu->tile));
                        pendingAdtCpu.reset();
                        loadingTx = loadingTy = -1;
                        // 立即在当前帧重建合批桶：push_back 可能触发 vector 扩容，
                        // 旧实例指针必须马上刷新，否则渲染阶段会解引用悬空地址闪退
                        rebuildGlobalM2Buckets();
                    }
                } else if (asyncAdtFuture.valid()) {
                    // 2.1 非阻塞轮询后台 CPU 解包任务
                    if (asyncAdtFuture.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                        auto parsed = asyncAdtFuture.get();
                        if (parsed) {
                            pendingAdtCpu = std::move(*parsed);
                        } else {
                            if (loadingTx >= 0 && loadingTy >= 0) {
                                failedAdtKeys.insert((loadingTx << 8) | loadingTy);
                            }
                            loadingTx = loadingTy = -1;
                        }
                    }
                } else if (streamCooldown <= 0.0f) {
                    // 3. 启动后台 CPU 解包：查找 380 码内距离最近、未加载且不在加载中的区块
                    int   bestTx = -1;
                    int   bestTy = -1;
                    float bestDist = 380.0f;

                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            const int tx = curTile.x + dx;
                            const int ty = curTile.y + dy;
                            if (tx < 0 || tx > 63 || ty < 0 || ty > 63) continue;
                            if (failedAdtKeys.count((tx << 8) | ty) != 0) continue;
                            if (tx == loadingTx && ty == loadingTy) continue;

                            bool alreadyLoaded = false;
                            for (const auto& t : adtTiles) {
                                if (t.tileX == tx && t.tileY == ty) {
                                    alreadyLoaded = true;
                                    break;
                                }
                            }
                            if (alreadyLoaded) continue;

                            const float dist = coa::assets::AdtLoader::DistanceToAdtTile(camPos, tx, ty);
                            if (dist > 380.0f) continue;
                            if (dist < bestDist) {
                                bestDist = dist;
                                bestTx   = tx;
                                bestTy   = ty;
                            }
                        }
                    }

                    if (bestTx >= 0 && bestTy >= 0) {
                        loadingTx      = bestTx;
                        loadingTy      = bestTy;
                        streamCooldown = 0.15f;

                        char adtPath[96];
                        std::snprintf(adtPath, sizeof(adtPath),
                                      "world/maps/azeroth/azeroth_%d_%d.adt", bestTx, bestTy);

                        // 后台线程仅做纯 CPU 解包，绝不触碰 OpenGL / AssetManager
                        asyncAdtFuture = std::async(std::launch::async,
                            [adtPathStr = std::string(adtPath), &vfs, bestTx, bestTy]() {
                                return coa::assets::AdtLoader::ParseAdtCpu(adtPathStr, vfs, bestTx, bestTy);
                            });
                    }
                }

            }

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
            // 仅当角色处于行走/游泳模式且脚底真正没入水面超过 0.25 码时，才写入水面高度；
            // 否则保持 -1e30，使世界着色器跳过水下遮挡，避免飞行 / 岸边站立被误染色
            float curWaterZ = -1e30f;
            if (walkMode) {
                float rawWaterZ = -1e30f;
                if (coa::assets::AdtLoader::SampleWaterHeight(adtTiles, playerPos.x, playerPos.y, rawWaterZ) &&
                    rawWaterZ > playerPos.z + 0.25f) {
                    curWaterZ = rawWaterZ;
                }
            }
            uboData.waterShallow       = glm::vec4(lightState.riverShallow, curWaterZ);
            uboData.waterDeep          = glm::vec4(lightState.riverDeep, 1.0f);

            // 方向 K：双级联太阳方向阴影正交投影
            // 近景级联：58 码半径 + 2048 分辨率 (2*58/2048 ≈ 0.0566 码/纹素)，每帧刷新保证清晰度
            // 远景级联：320 码半径 + 1024 分辨率，每 6 帧才重算矩阵并重绘一次，开销趋近于零
            // 两级联均在光空间内对锚点做纹素级吸附，消除相机平移时的阴影蠕动
            {
                glm::vec3 sunDir = glm::normalize(glm::vec3(
                    lightState.sunDir.x, lightState.sunDir.y, std::max(lightState.sunDir.z, 0.28f)));

                ++frameCounter;
                // 远景级联刷新节奏：首两帧立即构建，其后每 6 帧一次；ADT 分步上传期间跳过以免抢帧
                const bool updateFarShadow = ((frameCounter % 6 == 0) || (frameCounter <= 2)) &&
                                             !pendingAdtCpu.has_value();

                const glm::vec3 anchor   = walkMode ? playerPos : camPos;
                const glm::mat4 lightRot = glm::lookAt(sunDir, glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, 1.0f));

                // 在光空间内把锚点吸附到纹素格点，避免移动时阴影边缘随纹素栅格抖动
                auto snapAnchorToTexels = [&](float rad, float resolution) {
                    const float texel = (rad * 2.0f) / resolution;
                    glm::vec3 lv = glm::vec3(lightRot * glm::vec4(anchor, 1.0f));
                    lv.x = std::floor(lv.x / texel) * texel;
                    lv.y = std::floor(lv.y / texel) * texel;
                    return glm::vec3(glm::transpose(glm::mat3(lightRot)) * lv);
                };

                // ---- 近景级联：每帧刷新 ----
                const float     kShadowRad = 58.0f;
                const glm::vec3 nearAnchor = snapAnchorToTexels(kShadowRad, 2048.0f);
                const glm::mat4 nearView =
                    glm::lookAt(nearAnchor + sunDir * 300.0f, nearAnchor, glm::vec3(0.0f, 0.0f, 1.0f));
                const glm::mat4 nearProj =
                    glm::orthoRH_ZO(-kShadowRad, kShadowRad, -kShadowRad, kShadowRad, 10.0f, 380.0f);
                uboData.lightViewProj = nearProj * nearView;

                // ---- 远景级联：仅刷新帧重算矩阵，其余帧沿用缓存 ----
                if (updateFarShadow) {
                    const float     kFarRad   = 320.0f;
                    const glm::vec3 farAnchor = snapAnchorToTexels(kFarRad, 1024.0f);
                    const glm::mat4 farView =
                        glm::lookAt(farAnchor + sunDir * 400.0f, farAnchor, glm::vec3(0.0f, 0.0f, 1.0f));
                    const glm::mat4 farProj =
                        glm::orthoRH_ZO(-kFarRad, kFarRad, -kFarRad, kFarRad, 10.0f, 750.0f);
                    lastLightViewProjFar = farProj * farView;
                }
                uboData.lightViewProjFar = lastLightViewProjFar;
            }
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
            shadowQueue.clear();
            shadowFarQueue.clear();
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
                    cmd.texAlphaMap   = (chunk.alphaMapTex != 0) ? chunk.alphaMapTex : blackAlphaTex.Handle();
                    renderQueue.push_back(cmd);
                }
            }

            // 2. 跨 ADT 全局 M2 单次合批硬件实例化 + 全局单次骨骼/UV/颜色关键帧评估
            for (const auto& bucket : globalM2Buckets) {
                const auto& model = *bucket.model;
                uint32_t firstInstance = static_cast<uint32_t>(visibleMatrices.size());
                uint32_t visibleCount  = 0;
                float    minDistSq     = 1e30f; // 该桶可见实例到相机的最近中心距离平方

                // 实例按阴影级联距离分片排列：近景 (<=58 码) -> 远景 (58~320 码) -> 仅渲染 (>320 码)。
                // 这样阴影 Pass 只需绘制前 N 个实例，远处实例永远不会占用近景 2048 贴图的开销。
                nearMats.clear();
                farMats.clear();
                viewOnlyMats.clear();

                for (const auto* instPtr : bucket.instances) {
                    const auto& inst = *instPtr;
                    const float centerDist = glm::distance(inst.worldCenter, camPos);
                    const float edgeDist   = std::max(0.0f, centerDist - inst.worldRadius);
                    const bool  inView     = IsInstanceVisible(camFrustum, camPos,
                                                               inst.worldBoundsMin, inst.worldBoundsMax,
                                                               inst.worldCenter, inst.worldRadius);
                    // 镜头背后但足够近的大物件仍需投射阴影，避免玩家转身时树影突兀消失
                    const bool isCaster = (inst.worldRadius >= 1.2f);
                    if (!inView && !(isCaster && edgeDist <= 58.0f)) {
                        continue;
                    }
                    if (isCaster && edgeDist <= 58.0f) {
                        nearMats.push_back(inst.model);
                    } else if (isCaster && edgeDist <= 320.0f) {
                        farMats.push_back(inst.model);
                    } else {
                        viewOnlyMats.push_back(inst.model);
                    }
                    minDistSq = std::min(minDistSq, centerDist * centerDist);
                }

                const uint32_t nearCount = static_cast<uint32_t>(nearMats.size());
                const uint32_t farCount  = static_cast<uint32_t>(farMats.size());
                visibleCount = nearCount + farCount + static_cast<uint32_t>(viewOnlyMats.size());

                visibleMatrices.insert(visibleMatrices.end(), nearMats.begin(), nearMats.end());
                visibleMatrices.insert(visibleMatrices.end(), farMats.begin(), farMats.end());
                visibleMatrices.insert(visibleMatrices.end(), viewOnlyMats.begin(), viewOnlyMats.end());

                if (visibleCount == 0) continue;
                visibleM2Instances += static_cast<int>(visibleCount);

                // ---- 粒子发射器：为每个可见实例生成确定性公告板粒子 (屋顶炊烟 / 火把火苗) ----
                if (!model.particleEmitters.empty() && minDistSq <= 220.0f * 220.0f) {
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
                            const auto& em  = model.particleEmitters[emi];
                            // 长寿命 AlphaBlend 发射器视为炊烟：固定模拟周期并采用蓬松扩散的柔和烟雾形态
                            const bool  isSmoke = (em.blendingType <= 2 && em.lifespan >= 4.0f);
                            const float life    = isSmoke ? 4.2f : std::max(em.lifespan, 0.25f);
                            int count = isSmoke
                                      ? 28
                                      : std::clamp(static_cast<int>(em.emissionRate * life), 6, 32);
                            // 静止灯笼光晕 (speed≈0)：限制层数，避免多层半透明叠加过曝
                            if (!isSmoke && em.speed < 0.05f) {
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
                                if (isSmoke) {
                                    // 炊烟：以 u 驱动的平滑上升 + 微风摆动替代重力抛物线，避免横向喷雾与瞬移感
                                    worldPos = startWorld + glm::vec3(
                                        std::sin(u * 6.28f + r1 * 6.28f) * (0.35f * u),
                                        std::cos(u * 6.28f + r2 * 6.28f) * (0.35f * u),
                                        u * 7.5f);
                                    // 尺寸随高度自然蓬松放大
                                    scl = glm::mix(1.1f, 3.4f, u);
                                    // 透明度两头淡入淡出
                                    alp = std::sin(u * 3.14159f) * 0.32f;
                                    // 烟色柔和偏暖灰
                                    col = glm::mix(glm::vec3(0.82f), glm::vec3(0.62f), u);
                                }
                                if (!isSmoke) {
                                    // 非炊烟粒子 (火星/水花/光晕) 随实例缩放折算尺寸，并限制最大半径避免白团过曝
                                    scl = glm::clamp(scl * glm::length(glm::vec3(instMat[0])), 0.05f, 0.35f);
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

                // 全场景同一种 M2 模型每帧仅评估 1 次骨骼矩阵栈 (远处实例无需骨骼姿态)
                uint32_t boneBaseIndex = 0;
                if (model.hasAnimatedBones && minDistSq <= 180.0f * 180.0f) {
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
                    // 近景级联：只绘制离相机 58 码内的实例，2048 贴图的纹素精度全部留给近处
                    if (pass <= 3 && nearCount > 0) {
                        coa::rhi::DrawCommand sCmd = cmd;
                        sCmd.instanceCount = nearCount;
                        shadowQueue.push_back(sCmd);
                    }
                    // 远景级联：仅大尺寸模型 (树冠/建筑) 投入，并强制 pass = 2 跳过 alpha discard
                    if (pass <= 3 && (nearCount + farCount) > 0 &&
                        glm::length(model.localBoundsMax - model.localBoundsMin) >= 5.0f) {
                        coa::rhi::DrawCommand fCmd = cmd;
                        fCmd.instanceCount = nearCount + farCount;
                        fCmd.sortKey       = coa::rhi::DrawCommand::MakeKey(2, 1, 0, 0);
                        shadowFarQueue.push_back(fCmd);
                    }
                }
            }

            // 2.5 第三人称角色实体：独立骨骼姿态，不参与跨 ADT 合批
            if (walkMode && playerModel && camDist > 0.4f && !playerModel->submeshes.empty()) {
                const uint32_t playerInstIdx = static_cast<uint32_t>(visibleMatrices.size());
                const glm::mat4 playerMat =
                    glm::translate(glm::mat4(1.0f), playerPos) *
                    glm::rotate(glm::mat4(1.0f), playerYaw, glm::vec3(0.0f, 0.0f, 1.0f));
                visibleMatrices.push_back(playerMat);

                const uint32_t playerBoneBase =
                    playerModel->EvaluateBones(animTimeSec, visibleBoneMatrices, playerAnimAction);

                for (const auto& sub : playerModel->submeshes) {
                    glm::vec4 batchColor(1.0f);
                    if (sub.colorIndex >= 0 &&
                        static_cast<size_t>(sub.colorIndex) < playerModel->colorTracks.size()) {
                        const auto& ct = playerModel->colorTracks[sub.colorIndex];
                        glm::vec3 rgb = ct.color.SampleLinear(animTimeSec, playerModel->globalSequences,
                                                              glm::vec3(1.0f));
                        float     alp = ct.alpha.SampleLinear(animTimeSec, playerModel->globalSequences, 1.0f);
                        batchColor = glm::vec4(rgb, alp);
                    }
                    if (sub.transWeightIndex >= 0 &&
                        static_cast<size_t>(sub.transWeightIndex) < playerModel->transWeightTracks.size()) {
                        const auto& tw = playerModel->transWeightTracks[sub.transWeightIndex];
                        batchColor.a *= tw.SampleLinear(animTimeSec, playerModel->globalSequences, 1.0f);
                    }
                    if (batchColor.a < 0.01f) continue;

                    glm::vec2 uvOffset(0.0f);
                    if (sub.uvTransIndex >= 0 &&
                        static_cast<size_t>(sub.uvTransIndex) < playerModel->uvTransTracks.size()) {
                        const auto& ut = playerModel->uvTransTracks[sub.uvTransIndex];
                        if (ut.HasAnimation()) {
                            glm::vec3 uv3 = ut.SampleLinear(animTimeSec, playerModel->globalSequences,
                                                            glm::vec3(0.0f));
                            uvOffset = glm::vec2(uv3.x, uv3.y);
                        } else {
                            uvOffset = glm::vec2(0.0f, -0.75f * animTimeSec);
                        }
                    } else if (sub.blendMode == 10) {
                        uvOffset = glm::vec2(0.0f, -0.75f * animTimeSec);
                    }

                    uint8_t pass = 2;
                    if (sub.blendMode == 1)       pass = 3;
                    else if (sub.blendMode == 2)  pass = 5;
                    else if (sub.blendMode == 10) pass = 7;
                    else if (sub.blendMode >= 3)  pass = 6;

                    coa::rhi::DrawCommand cmd{};
                    cmd.sortKey       = coa::rhi::DrawCommand::MakeKey(pass, 1, static_cast<uint16_t>(sub.textureId), 0);
                    cmd.program       = worldShader.Handle();
                    cmd.vao           = sharedM2Vao;
                    cmd.vbo           = playerModel->vbo.Handle();
                    cmd.ibo           = playerModel->ibo.Handle();
                    cmd.stride        = sizeof(coa::assets::M2Vertex);
                    cmd.texAlbedo     = sub.textureId;
                    cmd.firstIndex    = sub.firstIndex;
                    cmd.indexCount    = sub.indexCount;
                    cmd.baseVertex    = 0;
                    cmd.instanceCount = 1;
                    cmd.baseInstance  = playerInstIdx;
                    cmd.boneBaseIndex = playerBoneBase;
                    cmd.isTwoSided    = ((sub.materialFlags & 0x04u) != 0) ? 1 : 0;
                    cmd.isUnlit       = ((sub.materialFlags & 0x01u) != 0 || pass >= 6) ? 1 : 0;
                    cmd.uvOffset      = uvOffset;
                    cmd.batchColor    = batchColor;
                    cmd.texAlphaMap   = blackAlphaTex.Handle();
                    renderQueue.push_back(cmd);
                    if (pass <= 3) {
                        shadowQueue.push_back(cmd);
                    }
                }
            }

            // 2.6 方向 M/N：地标 NPC 实体 (自主巡逻 + 面向玩家驻足 + 独立动作姿态 + 生命值)
            if (playerModel && !playerModel->submeshes.empty()) {
                const glm::vec3 viewerPos = walkMode ? playerPos : camPos;
                std::vector<glm::mat4> npcMats(landmarkNpcs.size(), glm::mat4(1.0f));
                std::vector<uint32_t>  npcBoneBases(landmarkNpcs.size(), 0u);

                for (size_t i = 0; i < landmarkNpcs.size(); ++i) {
                    auto& npc = landmarkNpcs[i];
                    npc.visible = false;
                    if (glm::distance(npc.spawn, camPos) > 180.0f) continue;

                    // 首次进入视距时初始化巡逻起点：以出生海拔作为射线起点，使 WMO 桥面
                    // (暴风城大桥约 95 码) 能被射线命中，而不再穿透到桥下护城河地形。
                    // 关键修复：若该 NPC 所在 ADT 尚未流式加载完成，则本帧直接跳过，
                    // 绝不回退为 0.0f 或提前标记 initialized，避免瞬移后 NPC 卡在地下。
                    if (!npc.initialized) {
                        float initGroundZ = npc.spawn.z;
                        if (!coa::assets::AdtLoader::SampleGroundHeight(adtTiles, npc.spawn.x, npc.spawn.y,
                                                                       npc.spawn.z, initGroundZ)) {
                            continue;
                        }
                        npc.worldPos    = glm::vec3(npc.spawn.x, npc.spawn.y, initGroundZ);
                        npc.initialized = true;
                    }

                    const bool selected = currentPick.hit &&
                                          currentPick.uniqueId == static_cast<uint32_t>(i + 1000);
                    if (selected && glm::distance(npc.worldPos, viewerPos) < 12.0f) {
                        // 被选中且玩家已靠近：原地驻足并转身面向玩家
                        npc.animAction = 0;
                        npc.yaw = std::atan2(viewerPos.y - npc.worldPos.y, viewerPos.x - npc.worldPos.x);
                    } else if (npc.pauseTimer > 0.0f) {
                        // 折返 / 起步停顿：原地站立，避免瞬间转向抖动
                        npc.pauseTimer -= dt;
                        npc.animAction = 0;
                    } else {
                        // 物理巡逻步进：Collide-and-Slide 解析墙体与陡坡，受阻或越界即自动折返
                        const glm::vec2 dir = npc.patrolForward ? glm::vec2(1.0f, 0.0f)
                                                                : glm::vec2(-1.0f, 0.0f);
                        const glm::vec2 step = dir * (2.6f * dt);
                        const glm::vec2 curXY(npc.worldPos.x, npc.worldPos.y);
                        const glm::vec2 resolvedXY = coa::assets::AdtLoader::ResolveHorizontalCollision(
                            adtTiles, curXY, step, npc.worldPos.z);

                        // 从当前脚下上方 2 码向下探测：取到的是前方真正的地面而非脚下缓存高度
                        float nextZ = npc.worldPos.z;
                        const bool hasNextGround = coa::assets::AdtLoader::SampleGroundHeight(
                            adtTiles, resolvedXY.x, resolvedXY.y, npc.worldPos.z + 2.0f, nextZ);
                        // 前方无地面 (已走出已加载地形边缘) 视为受阻
                        if (!hasNextGround) {
                            npc.patrolForward = !npc.patrolForward;
                            npc.pauseTimer    = 2.5f;
                            npc.animAction    = 0;
                            npc.visible       = true;
                            continue;
                        }

                        // 前方入水检测：水面明显高于落脚地面时视为受阻，杜绝陆地 NPC 走进湖中
                        float waterZ = -1e30f;
                        const bool inWater = coa::assets::AdtLoader::SampleWaterHeight(
                            adtTiles, resolvedXY.x, resolvedXY.y, waterZ) &&
                            waterZ > nextZ + 0.35f;

                        const bool blocked = glm::length(resolvedXY - curXY) < glm::length(step) * 0.5f ||
                                             std::fabs(nextZ - npc.worldPos.z) > 0.8f ||
                                             inWater;
                        const bool outOfRange =
                            glm::distance(resolvedXY, glm::vec2(npc.spawn.x, npc.spawn.y)) >= 9.0f;
                        if (blocked || outOfRange) {
                            npc.patrolForward = !npc.patrolForward;
                            npc.pauseTimer    = 2.5f;
                            npc.animAction    = 0;
                        } else {
                            npc.worldPos   = glm::vec3(resolvedXY.x, resolvedXY.y, nextZ);
                            npc.yaw        = std::atan2(dir.y, dir.x);
                            npc.animAction = 1; // Walk
                        }
                    }

                    npc.visible = true;

                    // 每个 NPC 动作不同，必须独立评估一次骨骼栈，不能共用同一 baseIndex
                    npcBoneBases[i] = playerModel->EvaluateBones(animTimeSec, visibleBoneMatrices,
                                                                 npc.animAction);
                    npcMats[i] = glm::translate(glm::mat4(1.0f), npc.worldPos) *
                                 glm::rotate(glm::mat4(1.0f), npc.yaw, glm::vec3(0.0f, 0.0f, 1.0f));

                    // 选中目标脚下的绿圈实时跟随移动中的 NPC
                    if (selected) {
                        currentPick.groundPos = npc.worldPos;
                    }
                }

                for (size_t i = 0; i < landmarkNpcs.size(); ++i) {
                    const auto& npc = landmarkNpcs[i];
                    if (!npc.visible) continue;
                    const uint32_t npcInstIdx = static_cast<uint32_t>(visibleMatrices.size());
                    visibleMatrices.push_back(npcMats[i]);

                    for (const auto& sub : playerModel->submeshes) {
                        glm::vec4 batchColor(1.0f);
                        if (sub.colorIndex >= 0 &&
                            static_cast<size_t>(sub.colorIndex) < playerModel->colorTracks.size()) {
                            const auto& ct = playerModel->colorTracks[sub.colorIndex];
                            batchColor = glm::vec4(
                                ct.color.SampleLinear(animTimeSec, playerModel->globalSequences,
                                                      glm::vec3(1.0f)),
                                ct.alpha.SampleLinear(animTimeSec, playerModel->globalSequences, 1.0f));
                        }
                        if (sub.transWeightIndex >= 0 &&
                            static_cast<size_t>(sub.transWeightIndex) < playerModel->transWeightTracks.size()) {
                            batchColor.a *= playerModel->transWeightTracks[sub.transWeightIndex]
                                                .SampleLinear(animTimeSec, playerModel->globalSequences, 1.0f);
                        }
                        if (batchColor.a < 0.01f) continue;

                        uint8_t pass = 2;
                        if (sub.blendMode == 1)       pass = 3;
                        else if (sub.blendMode == 2)  pass = 5;
                        else if (sub.blendMode == 10) pass = 7;
                        else if (sub.blendMode >= 3)  pass = 6;

                        coa::rhi::DrawCommand cmd{};
                        cmd.sortKey       = coa::rhi::DrawCommand::MakeKey(pass, 1, static_cast<uint16_t>(sub.textureId), 0);
                        cmd.program       = worldShader.Handle();
                        cmd.vao           = sharedM2Vao;
                        cmd.vbo           = playerModel->vbo.Handle();
                        cmd.ibo           = playerModel->ibo.Handle();
                        cmd.stride        = sizeof(coa::assets::M2Vertex);
                        cmd.texAlbedo     = sub.textureId;
                        cmd.firstIndex    = sub.firstIndex;
                        cmd.indexCount    = sub.indexCount;
                        cmd.baseVertex    = 0;
                        cmd.instanceCount = 1;
                        cmd.baseInstance  = npcInstIdx;
                        cmd.boneBaseIndex = npcBoneBases[i];
                        cmd.isTwoSided    = ((sub.materialFlags & 0x04u) != 0) ? 1 : 0;
                        cmd.isUnlit       = ((sub.materialFlags & 0x01u) != 0 || pass >= 6) ? 1 : 0;
                        cmd.uvOffset      = glm::vec2(0.0f);
                        cmd.batchColor    = batchColor;
                        cmd.texAlphaMap   = blackAlphaTex.Handle();
                        renderQueue.push_back(cmd);
                        if (pass <= 3) {
                            shadowQueue.push_back(cmd);
                        }
                    }
                }
            }

            // 2.7 方向 N：魔法飞弹飞行、发光尾迹与命中判定
            if (activeProj.active && activeProj.targetNpc < landmarkNpcs.size()) {
                auto& tgt = landmarkNpcs[activeProj.targetNpc];
                const glm::vec3 aimPos = tgt.worldPos + glm::vec3(0.0f, 0.0f, 1.35f);
                const glm::vec3 toAim  = aimPos - activeProj.pos;
                const float     dist   = glm::length(toAim);
                const glm::vec3 dir    = (dist > 1e-4f) ? (toAim / dist) : glm::vec3(0.0f, 0.0f, 1.0f);
                const float     step   = 28.0f * dt;

                if (dist <= std::max(step, 0.9f)) {
                    // 命中：扣血并在归零后重置为满血，随后销毁飞弹
                    tgt.hp -= 25.0f;
                    if (tgt.hp <= 0.0f) tgt.hp = tgt.maxHp;
                    activeProj.active = false;
                } else {
                    activeProj.pos += dir * step;
                }

                // 发光尾迹：沿飞行路径压入叠加混合的球状光点
                const uint32_t trailFirst = static_cast<uint32_t>(visibleParticles.size());
                for (int k = 0; k < 3; ++k) {
                    const float back = static_cast<float>(k) * 0.42f;
                    const float fade = static_cast<float>(k) / 3.0f;
                    GpuParticleQuad trail{};
                    trail.worldPosAndSize = glm::vec4(activeProj.pos - dir * back,
                                                     std::max(0.55f - 0.14f * static_cast<float>(k), 0.18f));
                    trail.color           = glm::vec4(glm::mix(glm::vec3(1.0f, 0.72f, 0.28f),
                                                               glm::vec3(0.45f, 0.72f, 1.0f), fade),
                                                      0.85f - 0.2f * static_cast<float>(k));
                    trail.uvRect          = glm::vec4(0.0f, 0.0f, 1.0f, 1.0f);
                    visibleParticles.push_back(trail);
                }
                particleBatches.push_back(ParticleBatchCmd{
                    whiteFallbackTex.Handle(), 3, trailFirst, 3u });
            }

            // 3. WMO 两级层次化裁剪：先整栋建筑，再逐个子房间
            std::unordered_set<uint32_t> drawnWmoIds; // 跨界 WMO 的同一 uniqueId 每帧仅绘制一次
            for (const auto& tile : adtTiles) {
                for (const auto& wmoBatch : tile.wmoBatches) {
                    if (!wmoBatch.model) continue;
                    const auto& model = *wmoBatch.model;
                    for (const auto& inst : wmoBatch.instances) {
                        if (inst.uniqueId != 0 && !drawnWmoIds.insert(inst.uniqueId).second) {
                            continue;
                        }
                        // 3.1 整栋建筑剔除：使用宽松的固定距离阈值 (2500 码) + 视锥测试。
                        // 不使用 IsInstanceVisible 的 80 + radius*35 自适应距离，否则高空俯瞰时
                        // 大型主城 WMO 会被整体提前剔除，导致街道与室内小房间消失。
                        if (glm::distance(inst.worldCenter, camPos) > 2500.0f + inst.worldRadius) {
                            continue;
                        }
                        // 近距建筑即使完全位于镜头背后（视锥外）也放行，交由 3.2 的距离判断
                        // 决定是否进入阴影队列，避免玩家背对建筑时阴影突然消失
                        const bool wmoNearShadow =
                            (glm::distance(inst.worldCenter, camPos) <= 175.0f + inst.worldRadius);
                        if (!wmoNearShadow &&
                            !camFrustum.IntersectsAABB(inst.worldBoundsMin, inst.worldBoundsMax)) {
                            continue;
                        }
                        if (inst.groupBounds.size() != model.groups.size()) continue;

                        uint32_t wmoBaseInstance = UINT32_MAX;
                        // 3.2 逐个子房间：视锥测试决定是否进入主渲染队列；
                        //     距离测试 (包含镜头背后、视锥外的房间) 决定是否进入阴影队列，避免背后建筑漏光
                        for (size_t g = 0; g < model.groups.size(); ++g) {
                            const auto& grp = model.groups[g];
                            const auto& gc  = inst.groupBounds[g];

                            const bool  inView        = camFrustum.IntersectsAABB(gc.worldBoundsMin, gc.worldBoundsMax);
                            const float grpDist       = glm::distance(gc.worldCenter, camPos);
                            // 远景级联覆盖 320 码：房间即使完全在视锥外，只要在范围内仍参与阴影
                            const bool  inShadowRange = (grpDist <= 320.0f + gc.worldRadius);
                            // 近景级联只覆盖 65 码，保证 2048 贴图精度不被远处房间稀释
                            const bool  inNearShadow  = (grpDist <= 65.0f + gc.worldRadius);
                            if (!inView && !inShadowRange) continue;

                            auto ensureBaseInstance = [&]() -> uint32_t {
                                if (wmoBaseInstance == UINT32_MAX) {
                                    wmoBaseInstance = static_cast<uint32_t>(visibleMatrices.size());
                                    visibleMatrices.push_back(inst.model);
                                }
                                return wmoBaseInstance;
                            };

                            if (inView) ++visibleWmoGroups;

                            for (const auto& sb : grp.batches) {
                                const uint8_t pass = (sb.blendMode == 1) ? 3 : 2;

                                if (inShadowRange && pass <= 3) {
                                    coa::rhi::DrawCommand sCmd{};
                                    sCmd.sortKey       = coa::rhi::DrawCommand::MakeKey(pass, 1, static_cast<uint16_t>(sb.textureId), 0);
                                    sCmd.program       = worldShader.Handle();
                                    sCmd.vao           = terrainVao;
                                    sCmd.vbo           = grp.vbo.Handle();
                                    sCmd.ibo           = grp.ibo.Handle();
                                    sCmd.stride        = sizeof(coa::assets::WmoVertex);
                                    sCmd.texAlbedo     = sb.textureId;
                                    sCmd.texAlphaMap   = blackAlphaTex.Handle();
                                    sCmd.firstIndex    = sb.firstIndex;
                                    sCmd.indexCount    = sb.indexCount;
                                    sCmd.baseVertex    = 0;
                                    sCmd.instanceCount = 1;
                                    sCmd.baseInstance  = ensureBaseInstance();
                                    if (inNearShadow) {
                                        shadowQueue.push_back(sCmd);
                                    }
                                    // 远景级联强制 pass = 2：跳过着色器 alpha discard，换取 6 帧一次的极低开销
                                    sCmd.sortKey = coa::rhi::DrawCommand::MakeKey(2, 1, 0, 0);
                                    shadowFarQueue.push_back(sCmd);
                                }
                                if (!inView) continue;

                                coa::rhi::DrawCommand cmd{};
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
                                cmd.baseInstance  = ensureBaseInstance();
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

            // 方向 M：选中目标脚下翡翠绿光环 (负半径标识水平贴地展开，Additive 叠加)
            if (currentPick.hit) {
                const uint32_t ringFirst = static_cast<uint32_t>(visibleParticles.size());

                GpuParticleQuad ringQuad{};
                ringQuad.worldPosAndSize = glm::vec4(
                    currentPick.groundPos + glm::vec3(0.0f, 0.0f, 0.15f),
                    -glm::clamp(currentPick.radius, 1.5f, 4.5f));
                ringQuad.color  = glm::vec4(0.2f, 1.0f, 0.45f, 0.85f);
                ringQuad.uvRect = glm::vec4(-2.0f, -2.0f, 0.0f, 0.0f); // 负数 UV 起点标记程序化光环
                visibleParticles.push_back(ringQuad);

                particleBatches.push_back(ParticleBatchCmd{
                    whiteFallbackTex.Handle(),
                    3,        // Additive
                    ringFirst,
                    1u
                });
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

            // ---- 方向 K：双级联阴影贴图 Pass (仅 Pass 2/3，先构建深度再恢复反向 Z 主渲染) ----
            glDepthFunc(GL_LESS);
            glClearDepth(1.0);

            // 近景级联：2048x2048，每帧刷新，覆盖相机 58 码内的全部阴影
            glBindFramebuffer(GL_FRAMEBUFFER, shadowFbo);
            glViewport(0, 0, 2048, 2048);
            glClear(GL_DEPTH_BUFFER_BIT);
            coa::rhi::SubmitShadowCommands(shadowQueue, shadowShader.Handle());

            // 远景级联：1024x1024，每 6 帧刷新一次；非刷新帧直接复用上一次的深度内容
            if (((frameCounter % 6 == 0) || (frameCounter <= 2)) && !pendingAdtCpu.has_value()) {
                // 阴影顶点着色器统一读取 uLightViewProj，这里临时换成远景矩阵，画完立刻还原
                const glm::mat4 savedLightViewProj = uboData.lightViewProj;
                uboData.lightViewProj = uboData.lightViewProjFar;
                frameUbo.UpdateSubData(0, sizeof(FrameUniforms), &uboData);

                glBindFramebuffer(GL_FRAMEBUFFER, shadowFboFar);
                glViewport(0, 0, 1024, 1024);
                glClear(GL_DEPTH_BUFFER_BIT);
                coa::rhi::SubmitShadowCommands(shadowFarQueue, shadowShader.Handle());

                uboData.lightViewProj = savedLightViewProj;
                frameUbo.UpdateSubData(0, sizeof(FrameUniforms), &uboData);
            }

            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glViewport(0, 0, winW, winH);
            glDepthFunc(GL_GEQUAL);
            glClearDepth(0.0);
            glBindTextureUnit(5, shadowTex);
            glBindTextureUnit(6, shadowTexFar);

            // 每 0.25 秒更新一次窗口标题，观察视锥裁剪统计
            titleTimer += dt;
            if (titleTimer >= 0.25f) {
                int totalMinutes = static_cast<int>(gameTimeHalfMin * 0.5f) % (24 * 60);
                int hh = totalMinutes / 60;
                int mm = totalMinutes % 60;

                const glm::ivec2 curTileTitle = coa::assets::AdtLoader::WorldToAdtIndex(camPos);
                char titleBuf[384];
                std::snprintf(titleBuf, sizeof(titleBuf),
                              "CoA Engine | Time: %02d:%02d | %s | [Mode: %s] | Tile:(%d,%d) [%zu ADTs] | Visible MCNK: %d/%d | Visible WMO Groups: %d | Visible M2 Instances: %d | Water: %d | Batched DrawCalls: %zu",
                              hh, mm, underwater ? "UNDERWATER" : "Surface",
                              walkMode ? (isSwimming ? "SWIM" : "WALK") : "FLY",
                              curTileTitle.x, curTileTitle.y, adtTiles.size(),
                              visibleChunks, totalChunks, visibleWmoGroups, visibleM2Instances,
                              visibleWaterChunks, renderQueue.size());

                std::string titleStr = titleBuf;
                if (currentPick.hit) {
                    std::string shortName = currentPick.modelPath;
                    const size_t slashPos = shortName.find_last_of("/\\");
                    if (slashPos != std::string::npos) shortName = shortName.substr(slashPos + 1);
                    char pickBuf[160];
                    std::snprintf(pickBuf, sizeof(pickBuf), " | Target: %s (%.1fm)",
                                  shortName.c_str(), currentPick.distance);
                    titleStr += pickBuf;
                }
                SDL_SetWindowTitle(window, titleStr.c_str());
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

            // ---- 方向 N：2D HUD 框体、施法条与 3D 头顶浮动血条 ----
            uiQuads.clear();

            // 1. 左上角玩家框体：深色底框 + 绿色生命条 + 蓝色法力条
            uiQuads.push_back(GpuUiQuad{ glm::vec4(20.0f, 20.0f, 210.0f, 48.0f),
                                         glm::vec4(0.03f, 0.05f, 0.07f, 0.85f) });
            uiQuads.push_back(GpuUiQuad{ glm::vec4(26.0f, 26.0f, 198.0f, 16.0f),
                                         glm::vec4(0.08f, 0.09f, 0.09f, 0.90f) });
            uiQuads.push_back(GpuUiQuad{ glm::vec4(26.0f, 26.0f, 198.0f, 16.0f),
                                         glm::vec4(0.18f, 0.78f, 0.26f, 0.95f) });
            uiQuads.push_back(GpuUiQuad{ glm::vec4(26.0f, 46.0f, 198.0f, 14.0f),
                                         glm::vec4(0.08f, 0.09f, 0.09f, 0.90f) });
            uiQuads.push_back(GpuUiQuad{ glm::vec4(26.0f, 46.0f, 198.0f, 14.0f),
                                         glm::vec4(0.22f, 0.42f, 0.95f, 0.95f) });

            // 2. 目标框体：按 npc.hp / npc.maxHp 比例绘制红色血条
            const bool hasNpcTarget = currentPick.hit && currentPick.uniqueId >= 1000 &&
                                      static_cast<size_t>(currentPick.uniqueId - 1000) < landmarkNpcs.size();
            if (hasNpcTarget) {
                const auto& tgt    = landmarkNpcs[static_cast<size_t>(currentPick.uniqueId - 1000)];
                const float hpFrac = (tgt.maxHp > 0.0f) ? glm::clamp(tgt.hp / tgt.maxHp, 0.0f, 1.0f) : 0.0f;
                uiQuads.push_back(GpuUiQuad{ glm::vec4(250.0f, 20.0f, 210.0f, 48.0f),
                                             glm::vec4(0.03f, 0.05f, 0.07f, 0.85f) });
                uiQuads.push_back(GpuUiQuad{ glm::vec4(256.0f, 26.0f, 198.0f, 16.0f),
                                             glm::vec4(0.09f, 0.08f, 0.08f, 0.90f) });
                uiQuads.push_back(GpuUiQuad{ glm::vec4(256.0f, 26.0f, 198.0f * hpFrac, 16.0f),
                                             glm::vec4(0.85f, 0.18f, 0.16f, 0.95f) });
                uiQuads.push_back(GpuUiQuad{ glm::vec4(256.0f, 46.0f, 198.0f, 14.0f),
                                             glm::vec4(0.05f, 0.06f, 0.07f, 0.90f) });
            }

            // 3. 施法读条：屏幕底部中央的金色进度条
            if (castBarTimer > 0.0f) {
                const float castFrac = glm::clamp(castBarTimer / 0.45f, 0.0f, 1.0f);
                const float barW     = 320.0f;
                const float barX     = (static_cast<float>(winW) - barW) * 0.5f;
                const float barY     = static_cast<float>(winH) - 96.0f;
                uiQuads.push_back(GpuUiQuad{ glm::vec4(barX - 2.0f, barY - 2.0f, barW + 4.0f, 22.0f),
                                             glm::vec4(0.02f, 0.03f, 0.04f, 0.90f) });
                uiQuads.push_back(GpuUiQuad{ glm::vec4(barX, barY, barW * castFrac, 18.0f),
                                             glm::vec4(0.95f, 0.78f, 0.22f, 0.95f) });
            }

            // 4. NPC 头顶浮动血条：世界坐标投影到屏幕像素
            for (const auto& npc : landmarkNpcs) {
                if (!npc.visible) continue;
                const glm::vec4 clip = uboData.viewProj *
                    glm::vec4(npc.worldPos + glm::vec3(0.0f, 0.0f, 2.35f), 1.0f);
                if (clip.w <= 0.0f) continue;

                const float sx = (clip.x / clip.w * 0.5f + 0.5f) * static_cast<float>(winW);
                const float sy = (1.0f - (clip.y / clip.w * 0.5f + 0.5f)) * static_cast<float>(winH);
                if (sx < -90.0f || sx > static_cast<float>(winW) + 90.0f ||
                    sy < -24.0f || sy > static_cast<float>(winH) + 24.0f) continue;

                const float hpFrac = (npc.maxHp > 0.0f) ? glm::clamp(npc.hp / npc.maxHp, 0.0f, 1.0f) : 0.0f;
                uiQuads.push_back(GpuUiQuad{ glm::vec4(sx - 42.0f, sy - 8.0f, 84.0f, 14.0f),
                                             glm::vec4(0.02f, 0.02f, 0.03f, 0.75f) });
                uiQuads.push_back(GpuUiQuad{ glm::vec4(sx - 40.0f, sy - 6.0f, 80.0f * hpFrac, 10.0f),
                                             glm::vec4(0.85f - 0.45f * hpFrac,
                                                       0.22f + 0.50f * hpFrac, 0.18f, 0.90f) });
            }

            if (!uiQuads.empty()) {
                const size_t uiBytes = uiQuads.size() * sizeof(GpuUiQuad);
                if (uiBytes > uiSsbo.Size()) {
                    uiSsbo = coa::rhi::GlBuffer(uiBytes, nullptr, GL_DYNAMIC_STORAGE_BIT);
                    uiSsbo.BindAsStorageBuffer(3);
                }
                uiSsbo.UpdateSubData(0, uiBytes, uiQuads.data());

                glDisable(GL_DEPTH_TEST);
                glDepthMask(GL_FALSE);
                glDisable(GL_CULL_FACE);
                glEnable(GL_BLEND);
                glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

                glUseProgram(uiShader.Handle());
                glUniform2f(0, static_cast<float>(winW), static_cast<float>(winH));
                glBindVertexArray(skyVao);
                glDrawArraysInstancedBaseInstance(GL_TRIANGLES, 0, 6,
                                                  static_cast<GLsizei>(uiQuads.size()), 0);

                glDisable(GL_BLEND);
                glDepthMask(GL_TRUE);
                glEnable(GL_DEPTH_TEST);
            }

            SDL_GL_SwapWindow(window);
        }

        glDeleteVertexArrays(1, &terrainVao);
        glDeleteVertexArrays(1, &sharedM2Vao);
        glDeleteVertexArrays(1, &skyVao);
        glDeleteFramebuffers(1, &shadowFbo);
        glDeleteFramebuffers(1, &shadowFboFar);
        glDeleteTextures(1, &shadowTex);
        glDeleteTextures(1, &shadowTexFar);
    }

    SDL_GL_DestroyContext(glCtx);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
