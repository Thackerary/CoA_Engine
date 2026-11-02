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

// 标准 GLSL 4.60 着色器：支持魔兽 Z-Up 半兰伯特光照 + 三色天光环境反射 + Alpha Cutout 树叶镂空
constexpr std::string_view kM2VertShader = R"GLSL(
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

out vec3 vNormal;
out vec2 vUV;

void main() {
    vNormal = aNormal;
    vUV = aUV;
    gl_Position = uViewProj * vec4(aPos, 1.0);
}
)GLSL";

constexpr std::string_view kM2FragShader = R"GLSL(
#version 460 core
in vec3 vNormal;
in vec2 vUV;

layout(binding = 0) uniform sampler2D uTexAlbedo;

layout(std140, binding = 0) uniform FrameBlock {
    mat4 uView;
    mat4 uProj;
    mat4 uViewProj;
    vec4 uCameraPos;
    vec4 uSunDirAndIntensity;
};

out vec4 FragColor;

void main() {
    vec4 texColor = texture(uTexAlbedo, vUV);
    // Alpha Cutout：裁剪魔兽树叶、草丛、栅栏的透明边缘 (配合 4x MSAA 极其平滑)
    if (texColor.a < 0.5) {
        discard;
    }

    vec3 N = normalize(vNormal);
    vec3 L = normalize(uSunDirAndIntensity.xyz);

    // 半兰伯特漫反射 + 天空/地面双色半球环境光 (还原暴雪手绘贴图通透感)
    float ndotl = max(dot(N, L), 0.0);
    float halfLambert = ndotl * 0.65 + 0.35;
    vec3 skyAmbient    = vec3(0.42, 0.48, 0.58);
    vec3 groundAmbient = vec3(0.22, 0.20, 0.16);
    vec3 ambient = mix(groundAmbient, skyAmbient, N.z * 0.5 + 0.5);

    vec3 sunColor = vec3(1.0, 0.95, 0.85) * uSunDirAndIntensity.w;
    vec3 lighting = ambient + sunColor * halfLambert;

    FragColor = vec4(texColor.rgb * lighting, texColor.a);
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
    SDL_Window* window = SDL_CreateWindow("CoA Custom Engine (GL 4.6 DSA)", winW, winH,
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

        coa::rhi::GlShaderProgram m2Shader(kM2VertShader, kM2FragShader);
        GLuint sharedM2Vao = coa::assets::GpuM2Model::CreateSharedVAO();

        // 生成一张 1x1 纯白兜底纹理（当本地暂未放置对应 BLP 时也能看清 3D 模型光影轮廓）
        coa::rhi::GlTexture2D whiteFallbackTex(1, 1, 1, coa::rhi::PixelFormat::RGBA8, true);
        const uint8_t whitePixel[4] = {230, 230, 230, 255};
        whiteFallbackTex.UploadUncompressedMip(0, 1, 1, whitePixel);

        coa::vfs::VirtualFileSystem vfs;
        vfs.MountLooseDirectory("D:/CoA_Assets/wotlk_base", 100);
        vfs.MountLooseDirectory("D:/CoA_Assets/coa_patches", 200);

        coa::assets::AssetManager assets(vfs);
        GLuint grassTex = assets.GetOrLoadTexture("Tileset\\Elwynn\\ElwynnGrassBase.blp");
        if (grassTex == 0) grassTex = whiteFallbackTex.Handle();

        // 创建一块 40x40 码的 Z-Up 测试地面 (复用 M2Vertex 48 字节格式)
        const coa::assets::M2Vertex groundVerts[4] = {
            {{-20.0f, -20.0f, 0.0f}, {255,0,0,0}, {0,0,0,0}, {0.0f, 0.0f, 1.0f}, {{0.0f, 0.0f}, {0.0f, 0.0f}}},
            {{ 20.0f, -20.0f, 0.0f}, {255,0,0,0}, {0,0,0,0}, {0.0f, 0.0f, 1.0f}, {{8.0f, 0.0f}, {0.0f, 0.0f}}},
            {{ 20.0f,  20.0f, 0.0f}, {255,0,0,0}, {0,0,0,0}, {0.0f, 0.0f, 1.0f}, {{8.0f, 8.0f}, {0.0f, 0.0f}}},
            {{-20.0f,  20.0f, 0.0f}, {255,0,0,0}, {0,0,0,0}, {0.0f, 0.0f, 1.0f}, {{0.0f, 8.0f}, {0.0f, 0.0f}}},
        };
        const uint16_t groundIndices[6] = {0, 1, 2, 0, 2, 3};
        coa::rhi::GlBuffer groundVbo(sizeof(groundVerts), groundVerts, 0);
        coa::rhi::GlBuffer groundIbo(sizeof(groundIndices), groundIndices, 0);

        // 尝试加载一个魔兽 M2 模型 (可替换为你解压目录中实际存在的任意 .m2 路径)
        auto testM2 = coa::assets::M2Loader::LoadModel(
            "World\\Azeroth\\Elwynn\\PassiveDoodads\\Trees\\ElwynnTree01.m2",
            vfs, assets, whiteFallbackTex.Handle()
        );

        // Z-Up 摄像机状态 (+X 朝前, +Y 朝左, +Z 朝上)
        glm::vec3 camPos(-12.0f, 0.0f, 4.0f);
        float camYaw   = 0.0f;   // 绕 Z 轴水平偏航角
        float camPitch = -0.15f; // 俯仰角
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

            // 计算 Z-Up 坐标系下的摄像机前向与右向向量
            glm::vec3 forward(
                std::cos(camPitch) * std::cos(camYaw),
                std::cos(camPitch) * std::sin(camYaw),
                std::sin(camPitch)
            );
            glm::vec3 worldUp(0.0f, 0.0f, 1.0f);
            glm::vec3 right = glm::normalize(glm::cross(forward, worldUp));

            const bool* keys = SDL_GetKeyboardState(nullptr);
            float moveSpeed = (keys[SDL_SCANCODE_LSHIFT] ? 25.0f : 8.0f) * dt;
            if (keys[SDL_SCANCODE_W]) camPos += forward * moveSpeed;
            if (keys[SDL_SCANCODE_S]) camPos -= forward * moveSpeed;
            if (keys[SDL_SCANCODE_D]) camPos += right * moveSpeed;
            if (keys[SDL_SCANCODE_A]) camPos -= right * moveSpeed;
            if (keys[SDL_SCANCODE_SPACE]) camPos += worldUp * moveSpeed;
            if (keys[SDL_SCANCODE_LCTRL]) camPos -= worldUp * moveSpeed;

            // 更新全局 0 号 UBO
            float aspect = (winH > 0) ? (static_cast<float>(winW) / static_cast<float>(winH)) : 1.777f;
            FrameUniforms uboData{};
            uboData.view               = glm::lookAt(camPos, camPos + forward, worldUp);
            uboData.proj               = MakeInfiniteReverseZProj(glm::radians(60.0f), aspect, 0.1f);
            uboData.viewProj           = uboData.proj * uboData.view;
            uboData.cameraPos          = glm::vec4(camPos, 1.0f);
            uboData.sunDirAndIntensity = glm::vec4(glm::normalize(glm::vec3(-0.5f, -0.4f, 0.75f)), 1.1f);
            frameUbo.UpdateSubData(0, sizeof(FrameUniforms), &uboData);

            glClearColor(0.53f, 0.72f, 0.92f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            renderQueue.clear();

            // 1. 提交地面网格绘制指令
            coa::rhi::DrawCommand groundCmd{};
            groundCmd.sortKey    = coa::rhi::DrawCommand::MakeKey(2, 1, static_cast<uint16_t>(grassTex), 0);
            groundCmd.program    = m2Shader.Handle();
            groundCmd.vao        = sharedM2Vao;
            groundCmd.vbo        = groundVbo.Handle();
            groundCmd.ibo        = groundIbo.Handle();
            groundCmd.stride     = sizeof(coa::assets::M2Vertex);
            groundCmd.texAlbedo  = grassTex;
            groundCmd.indexCount = 6;
            renderQueue.push_back(groundCmd);

            // 2. 提交 M2 模型的所有子网格绘制指令
            if (testM2) {
                for (const auto& sub : testM2->submeshes) {
                    coa::rhi::DrawCommand cmd{};
                    uint8_t pass = (sub.blendMode == 1) ? 3 : 2; // 2=Opaque, 3=AlphaCutout
                    cmd.sortKey    = coa::rhi::DrawCommand::MakeKey(pass, 1, static_cast<uint16_t>(sub.textureId), 0);
                    cmd.program    = m2Shader.Handle();
                    cmd.vao        = sharedM2Vao;
                    cmd.vbo        = testM2->vbo.Handle();
                    cmd.ibo        = testM2->ibo.Handle();
                    cmd.stride     = sizeof(coa::assets::M2Vertex);
                    cmd.texAlbedo  = sub.textureId;
                    cmd.firstIndex = sub.firstIndex;
                    cmd.indexCount = sub.indexCount;
                    renderQueue.push_back(cmd);
                }
            }

            coa::rhi::SubmitDrawCommands(renderQueue);
            SDL_GL_SwapWindow(window);
        }

        glDeleteVertexArrays(1, &sharedM2Vao);
    }

    SDL_GL_DestroyContext(glCtx);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}