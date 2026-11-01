#include <SDL3/SDL.h>
#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <cstdio>
#include "rhi/rhi_gl.hpp"

// OpenGL 4.6 驱动级实时报错回调
void GLAPIENTRY GlDebugCallback(GLenum source, GLenum type, GLuint id, GLenum severity,
                                GLsizei /*length*/, const GLchar* message, const void* /*userParam*/) {
    if (severity == GL_DEBUG_SEVERITY_NOTIFICATION) return; // 忽略普通提示
    std::fprintf(stderr, "[GL Debug] (ID:%u) %s\n", id, message);
}

// 构造无穷远平面的 Reverse-Z 透视投影矩阵（近平面映射到 1.0，无穷远映射到 0.0）
glm::mat4 MakeInfiniteReverseZProj(float fovRadians, float aspect, float zNear) {
    float f = 1.0f / std::tan(fovRadians * 0.5f);
    glm::mat4 m(0.0f);
    m[0][0] = f / aspect;
    m[1][1] = f;
    m[2][3] = -1.0f; // 观察空间右手系 (-Z 朝向屏幕内)
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

int main(int /*argc*/, char* /*argv*/[]) {
    SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS);

    // 申请 OpenGL 4.6 Core Profile + 硬件 4x MSAA + Debug 上下文
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 6);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_DEBUG_FLAG);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
    SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 4);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 32);

    SDL_Window* window = SDL_CreateWindow("CoA Custom Engine (GL 4.6 DSA)", 1920, 1080,
                                          SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
    SDL_GLContext glCtx = SDL_GL_CreateContext(window);
    gladLoadGLLoader(reinterpret_cast<GLADloadproc>(SDL_GL_GetProcAddress));
    SDL_GL_SetSwapInterval(1); // 开启垂直同步

    // 1. 开启驱动同步调试输出
    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    glDebugMessageCallback(GlDebugCallback, nullptr);

    // 2. 启用 Reverse-Z ([0, 1] 深度映射 + 深度清除值为 0.0 + GL_GREATER 测试)
    glClipControl(GL_LOWER_LEFT, GL_ZERO_TO_ONE);
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_GREATER);
    glClearDepth(0.0);
    glEnable(GL_MULTISAMPLE);

    // 3. 创建全局 0 号 UBO 存放每帧摄像机与光照矩阵
    coa::rhi::GlBuffer frameUbo(sizeof(FrameUniforms));
    frameUbo.BindAsUniformBuffer(0);

    bool running = true;
    while (running) {
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            if (ev.type == SDL_EVENT_QUIT) running = false;
        }

        glClearColor(0.53f, 0.72f, 0.92f, 1.0f); // 艾尔文森林天空底色
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // 此处收集可见的 M2 / WMO / ADT 地形批次并调用 SubmitDrawCommands(queue)

        SDL_GL_SwapWindow(window);
    }

    SDL_GL_DestroyContext(glCtx);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}