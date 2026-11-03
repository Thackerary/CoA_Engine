#include <SDL3/SDL.h>
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <cstdio>
#include <cmath>
#include <vector>
#include "rhi/rhi_gl.hpp"
#include "vfs/vfs.hpp"
#include "assets/asset_manager.hpp"
#include "assets/m2_loader.hpp"
#include "assets/adt_loader.hpp"

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
    glm::vec4 cameraPos;
    glm::vec4 sunDirAndIntensity;
};
// 顶点着色器：通过 gl_BaseInstance + gl_InstanceID 从 1 号 SSBO 提取实例世界变换矩阵
// 标准 GLSL 4.60 着色器：支持魔兽 Z-Up 半兰伯特光照 + 三色天光环境反射 + Alpha Cutout 树叶镂空
constexpr std::string_view kWorldVertShader = R"GLSL(
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
};

layout(std430, binding = 1) readonly buffer InstanceBlock {
    mat4 uModelMatrices[];
};

out vec3 vWorldPos;
out vec3 vNormal;
out vec2 vUV;

void main() {
    mat4 modelMat = uModelMatrices[gl_BaseInstance + gl_InstanceID];
    vec4 worldPos = modelMat * vec4(aPos, 1.0);
    vWorldPos = worldPos.xyz;
    vNormal   = mat3(modelMat) * aNormal;
    vUV       = aUV;
    gl_Position = uViewProj * worldPos;
}
)GLSL";

// 片段着色器：半兰伯特光照 + 双色天光 + 远景大气指数雾
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

layout(std140, binding = 0) uniform FrameBlock {
    mat4 uView;
    mat4 uProj;
    mat4 uViewProj;
    vec4 uCameraPos;
    vec4 uSunDirAndIntensity;
};

out vec4 FragColor;

// 辅助函数：提取像素感知亮度作为微表面高度，并用平滑混合替换线性 mix
float GetLuma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

vec3 HeightBlend(vec3 baseCol, vec3 overCol, float alpha) {
    float hBase = GetLuma(baseCol);
    float hOver = GetLuma(overCol);
    // 让上层材质优先填充下层材质的暗部缝隙 (0.25 控制咬合锐度)
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
    if (texColor.a < 0.5) {
        discard;
    }

    // 1. 利用屏幕空间偏导数 (dFdx/dFdy) 从贴图亮度实时重建凹凸法线
    vec3 N = normalize(vNormal);
    float height = GetLuma(texColor.rgb);
    vec3 dPdx = dFdx(vWorldPos);
    vec3 dPdy = dFdy(vWorldPos);
    vec3 r1 = cross(dPdy, N);
    vec3 r2 = cross(N, dPdx);
    float det = dot(dPdx, r1);
    vec3 surfGrad = ( abs(det) > 1e-8 ) ? ((r1 * dFdx(height) + r2 * dFdy(height)) / det) : vec3(0.0);
    N = normalize(N - surfGrad * 0.35); // 0.35 为凹凸强度，可按喜好调节

    // 2. 半兰伯特漫反射 + Blinn-Phong 阳光微表面高光
    vec3 L = normalize(uSunDirAndIntensity.xyz);
    vec3 V = normalize(uCameraPos.xyz - vWorldPos);
    vec3 H = normalize(L + V);

    float ndotl = max(dot(N, L), 0.0);
    float halfLambert = ndotl * 0.65 + 0.35;
    float spec = pow(max(dot(N, H), 0.0), 24.0) * height * 0.28 * ndotl;

    vec3 skyAmbient    = vec3(0.45, 0.52, 0.62);
    vec3 groundAmbient = vec3(0.24, 0.22, 0.18);
    vec3 ambient = mix(groundAmbient, skyAmbient, N.z * 0.5 + 0.5);

    vec3 sunColor = vec3(1.0, 0.95, 0.85) * uSunDirAndIntensity.w;
    vec3 litColor = texColor.rgb * (ambient + sunColor * halfLambert) + sunColor * spec;

    float dist = length(vWorldPos - uCameraPos.xyz);
    float fogFactor = clamp(exp(-max(dist - 120.0, 0.0) * 0.0022), 0.0, 1.0);
    vec3 fogColor = vec3(0.53, 0.72, 0.92);

    FragColor = vec4(mix(fogColor, litColor, fogFactor), texColor.a);
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
        GLuint sharedM2Vao = coa::assets::GpuM2Model::CreateSharedVAO();
        GLuint terrainVao  = coa::assets::GpuAdtTile::CreateTerrainVAO();

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

        // 加载艾尔文森林闪金镇 ADT 地形与森林装饰物
        auto adtTile = coa::assets::AdtLoader::LoadAdt(
            "world/maps/azeroth/azeroth_32_48.adt",
            vfs, assets, whiteFallbackTex.Handle()
        );

        // 将 ADT 中所有物体的世界变换矩阵一次性上传至 1 号 SSBO
        std::vector<glm::mat4> ssboMats = {glm::mat4(1.0f)};
        if (adtTile && !adtTile->instanceMatrices.empty()) {
            ssboMats = adtTile->instanceMatrices;
        }
        coa::rhi::GlBuffer instanceSsbo(ssboMats.size() * sizeof(glm::mat4), ssboMats.data(), 0);
        instanceSsbo.BindAsStorageBuffer(1);

        // 自动将摄像机瞬移至当前加载的 ADT 地形中心上空
        glm::vec3 camPos = adtTile ? adtTile->suggestedCamPos : glm::vec3(0.0f, 0.0f, 20.0f);
        float camYaw   = 0.0f;
        float camPitch = -0.25f;
        bool rightMouseDown = false;
        uint64_t lastTicks = SDL_GetTicksNS();

        std::vector<coa::rhi::DrawCommand> renderQueue;
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

            float aspect = (winH > 0) ? (static_cast<float>(winW) / static_cast<float>(winH)) : 1.777f;
            FrameUniforms uboData{};
            uboData.view               = glm::lookAt(camPos, camPos + forward, worldUp);
            uboData.proj               = MakeInfiniteReverseZProj(glm::radians(60.0f), aspect, 0.2f);
            uboData.viewProj           = uboData.proj * uboData.view;
            uboData.cameraPos          = glm::vec4(camPos, 1.0f);
            uboData.sunDirAndIntensity = glm::vec4(glm::normalize(glm::vec3(-0.5f, -0.4f, 0.75f)), 1.1f);
            frameUbo.UpdateSubData(0, sizeof(FrameUniforms), &uboData);

            glClearColor(0.53f, 0.72f, 0.92f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            renderQueue.clear();

            if (adtTile) {
                // 1. 提交 256 个 MCNK 地形块绘制指令 (baseInstance = 0 指向单位矩阵)
                for (const auto& chunk : adtTile->chunks) {
                    coa::rhi::DrawCommand cmd{};
                    cmd.sortKey       = coa::rhi::DrawCommand::MakeKey(2, 1, static_cast<uint16_t>(chunk.layerTex[0]), 0);
                    cmd.program       = worldShader.Handle();
                    cmd.vao           = terrainVao;
                    cmd.vbo           = adtTile->terrainVbo.Handle();
                    cmd.ibo           = adtTile->terrainIbo.Handle();
                    cmd.stride        = sizeof(coa::assets::TerrainVertex);
                    cmd.firstIndex    = chunk.firstIndex;
                    cmd.indexCount    = chunk.indexCount;
                    cmd.baseVertex    = chunk.baseVertex;
                    cmd.instanceCount = 1;
                    cmd.baseInstance  = 0;
					cmd.texAlbedo    = chunk.layerTex[0];
                    cmd.texLayers[0] = chunk.layerTex[1];
                    cmd.texLayers[1] = chunk.layerTex[2];
                    cmd.texLayers[2] = chunk.layerTex[3];
                    cmd.texAlphaMap  = chunk.alphaMapTex;
                    renderQueue.push_back(cmd);
                }

                // 2. 批量提交整片森林的 M2 装饰物硬件实例化指令
                for (const auto& batch : adtTile->doodadBatches) {
                    for (const auto& sub : batch.model.submeshes) {
                        coa::rhi::DrawCommand cmd{};
                        uint8_t pass      = (sub.blendMode == 1) ? 3 : 2;
                        cmd.sortKey       = coa::rhi::DrawCommand::MakeKey(pass, 1, static_cast<uint16_t>(sub.textureId), 0);
                        cmd.program       = worldShader.Handle();
                        cmd.vao           = sharedM2Vao;
                        cmd.vbo           = batch.model.vbo.Handle();
                        cmd.ibo           = batch.model.ibo.Handle();
                        cmd.stride        = sizeof(coa::assets::M2Vertex);
                        cmd.texAlbedo     = sub.textureId;
                        cmd.firstIndex    = sub.firstIndex;
                        cmd.indexCount    = sub.indexCount;
                        cmd.baseVertex    = 0;
                        cmd.instanceCount = batch.instanceCount;
                        cmd.baseInstance  = batch.baseInstance;
						cmd.texAlphaMap  = blackAlphaTex.Handle(); // M2 模型权重为 0，仅使用 texAlbedo
                        renderQueue.push_back(cmd);
                    }
                }
				// 3. 批量提交 WMO 大型建筑 (教堂/酒馆/矿洞) 绘制指令 (复用 terrainVao)
                for (const auto& wmoBatch : adtTile->wmoBatches) {
                    for (const auto& grp : wmoBatch.model.groups) {
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
                            cmd.instanceCount = wmoBatch.instanceCount;
                            cmd.baseInstance  = wmoBatch.baseInstance;
                            renderQueue.push_back(cmd);
                        }
                    }
                }
            }

            coa::rhi::SubmitDrawCommands(renderQueue);
            SDL_GL_SwapWindow(window);
        }

        glDeleteVertexArrays(1, &terrainVao);
        glDeleteVertexArrays(1, &sharedM2Vao);
    }

    SDL_GL_DestroyContext(glCtx);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}