#pragma once
#include <glad/glad.h>
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <span>
#include <string_view>
#include <vector>
#include <algorithm>

namespace coa::rhi {

// ==================== 1. 显存缓冲区 (VBO / IBO / UBO / SSBO) ====================
class GlBuffer {
public:
    GlBuffer() = default;
    GlBuffer(std::size_t sizeBytes, const void* initialData = nullptr, GLbitfield storageFlags = GL_DYNAMIC_STORAGE_BIT)
        : m_size(sizeBytes) {
        glCreateBuffers(1, &m_handle);
        glNamedBufferStorage(m_handle, static_cast<GLsizeiptr>(sizeBytes), initialData, storageFlags);
    }

    ~GlBuffer() {
        if (m_handle) glDeleteBuffers(1, &m_handle);
    }

    GlBuffer(const GlBuffer&) = delete;
    GlBuffer& operator=(const GlBuffer&) = delete;

    GlBuffer(GlBuffer&& other) noexcept : m_handle(other.m_handle), m_size(other.m_size) {
        other.m_handle = 0;
        other.m_size   = 0;
    }
    GlBuffer& operator=(GlBuffer&& other) noexcept {
        if (this != &other) {
            if (m_handle) glDeleteBuffers(1, &m_handle);
            m_handle       = other.m_handle;
            m_size         = other.m_size;
            other.m_handle = 0;
            other.m_size   = 0;
        }
        return *this;
    }

    void UpdateSubData(std::size_t offset, std::size_t sizeBytes, const void* data) const {
        glNamedBufferSubData(m_handle, static_cast<GLintptr>(offset), static_cast<GLsizeiptr>(sizeBytes), data);
    }

    void BindAsUniformBuffer(uint32_t bindingSlot) const {
        glBindBufferBase(GL_UNIFORM_BUFFER, bindingSlot, m_handle);
    }

    void BindAsStorageBuffer(uint32_t bindingSlot) const {
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, bindingSlot, m_handle);
    }

    [[nodiscard]] GLuint Handle() const noexcept { return m_handle; }
    [[nodiscard]] std::size_t Size() const noexcept { return m_size; }

private:
    GLuint      m_handle = 0;
    std::size_t m_size   = 0;
};

// ==================== 2. 2D 纹理与 BLP 压缩纹理直传封装 ====================
enum class PixelFormat : GLenum {
    RGBA8    = GL_RGBA8,
    BC1_RGBA = GL_COMPRESSED_RGBA_S3TC_DXT1_EXT,
    BC2_RGBA = GL_COMPRESSED_RGBA_S3TC_DXT3_EXT,
    BC3_RGBA = GL_COMPRESSED_RGBA_S3TC_DXT5_EXT
};

class GlTexture2D {
public:
    GlTexture2D() = default;
    GlTexture2D(uint32_t width, uint32_t height, uint32_t mipLevels, PixelFormat format, bool repeatWrap = true)
        : m_width(width), m_height(height), m_mipLevels(mipLevels), m_format(format) {
        glCreateTextures(GL_TEXTURE_2D, 1, &m_handle);
        glTextureStorage2D(m_handle, static_cast<GLsizei>(mipLevels), static_cast<GLenum>(format),
                           static_cast<GLsizei>(width), static_cast<GLsizei>(height));

        glTextureParameteri(m_handle, GL_TEXTURE_MIN_FILTER, mipLevels > 1 ? GL_LINEAR_MIPMAP_LINEAR : GL_LINEAR);
        glTextureParameteri(m_handle, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        GLint wrapMode = repeatWrap ? GL_REPEAT : GL_CLAMP_TO_EDGE;
        glTextureParameteri(m_handle, GL_TEXTURE_WRAP_S, wrapMode);
        glTextureParameteri(m_handle, GL_TEXTURE_WRAP_T, wrapMode);
        glTextureParameterf(m_handle, GL_TEXTURE_MAX_ANISOTROPY, 8.0f);
    }

    ~GlTexture2D() {
        if (m_handle) glDeleteTextures(1, &m_handle);
    }

    GlTexture2D(const GlTexture2D&) = delete;
    GlTexture2D& operator=(const GlTexture2D&) = delete;

    GlTexture2D(GlTexture2D&& other) noexcept
        : m_handle(other.m_handle), m_width(other.m_width), m_height(other.m_height),
          m_mipLevels(other.m_mipLevels), m_format(other.m_format) {
        other.m_handle = 0;
    }
    GlTexture2D& operator=(GlTexture2D&& other) noexcept {
        if (this != &other) {
            if (m_handle) glDeleteTextures(1, &m_handle);
            m_handle       = other.m_handle;
            m_width        = other.m_width;
            m_height       = other.m_height;
            m_mipLevels    = other.m_mipLevels;
            m_format       = other.m_format;
            other.m_handle = 0;
        }
        return *this;
    }

    void UploadCompressedMip(uint32_t mipLevel, uint32_t mipW, uint32_t mipH, std::span<const uint8_t> compressedBytes) const {
        glCompressedTextureSubImage2D(
            m_handle, static_cast<GLint>(mipLevel), 0, 0,
            static_cast<GLsizei>(mipW), static_cast<GLsizei>(mipH),
            static_cast<GLenum>(m_format),
            static_cast<GLsizei>(compressedBytes.size()), compressedBytes.data()
        );
    }

    void UploadUncompressedMip(uint32_t mipLevel, uint32_t mipW, uint32_t mipH, std::span<const uint8_t> rgbaBytes) const {
        glTextureSubImage2D(
            m_handle, static_cast<GLint>(mipLevel), 0, 0,
            static_cast<GLsizei>(mipW), static_cast<GLsizei>(mipH),
            GL_RGBA, GL_UNSIGNED_BYTE, rgbaBytes.data()
        );
    }

    void BindToUnit(uint32_t unit) const {
        glBindTextureUnit(unit, m_handle);
    }

    [[nodiscard]] GLuint Handle() const noexcept { return m_handle; }

private:
    GLuint      m_handle    = 0;
    uint32_t    m_width     = 0;
    uint32_t    m_height    = 0;
    uint32_t    m_mipLevels = 1;
    PixelFormat m_format    = PixelFormat::RGBA8;
};

// ==================== 3. GLSL 4.60 着色器程序封装 ====================
class GlShaderProgram {
public:
    GlShaderProgram() = default;
    GlShaderProgram(std::string_view vertSrc, std::string_view fragSrc) {
        GLuint vs = CompileStage(GL_VERTEX_SHADER, vertSrc);
        GLuint fs = CompileStage(GL_FRAGMENT_SHADER, fragSrc);
        if (vs && fs) {
            m_handle = glCreateProgram();
            glAttachShader(m_handle, vs);
            glAttachShader(m_handle, fs);
            glLinkProgram(m_handle);

            GLint linked = GL_FALSE;
            glGetProgramiv(m_handle, GL_LINK_STATUS, &linked);
            if (linked == GL_FALSE) {
                char log[1024]{};
                glGetProgramInfoLog(m_handle, sizeof(log), nullptr, log);
                std::fprintf(stderr, "[Shader Link Error] %s\n", log);
                glDeleteProgram(m_handle);
                m_handle = 0;
            }
            glDetachShader(m_handle, vs);
            glDetachShader(m_handle, fs);
        }
        if (vs) glDeleteShader(vs);
        if (fs) glDeleteShader(fs);
    }

    ~GlShaderProgram() {
        if (m_handle) glDeleteProgram(m_handle);
    }

    GlShaderProgram(const GlShaderProgram&) = delete;
    GlShaderProgram& operator=(const GlShaderProgram&) = delete;

    GlShaderProgram(GlShaderProgram&& other) noexcept : m_handle(other.m_handle) {
        other.m_handle = 0;
    }
    GlShaderProgram& operator=(GlShaderProgram&& other) noexcept {
        if (this != &other) {
            if (m_handle) glDeleteProgram(m_handle);
            m_handle = other.m_handle;
            other.m_handle = 0;
        }
        return *this;
    }

    [[nodiscard]] GLuint Handle() const noexcept { return m_handle; }

private:
    GLuint m_handle = 0;

    static GLuint CompileStage(GLenum type, std::string_view src) {
        GLuint s = glCreateShader(type);
        const char* ptr = src.data();
        auto len = static_cast<GLint>(src.size());
        glShaderSource(s, 1, &ptr, &len);
        glCompileShader(s);

        GLint compiled = GL_FALSE;
        glGetShaderiv(s, GL_COMPILE_STATUS, &compiled);
        if (compiled == GL_FALSE) {
            char log[1024]{};
            glGetShaderInfoLog(s, sizeof(log), nullptr, log);
            std::fprintf(stderr, "[Shader Compile Error] %s\n", log);
            glDeleteShader(s);
            return 0;
        }
        return s;
    }
};

// ==================== 4. 64 位 Sort-Key 渲染队列指令 (支持硬件 Instancing) ====================
struct DrawCommand {
    uint64_t sortKey       = 0;
    GLuint   program       = 0;
    GLuint   vao           = 0;
    GLuint   vbo           = 0;
    GLuint   ibo           = 0;
    GLsizei  stride        = 0;
    GLuint   texAlbedo     = 0;          // 纹理单元 0: 主贴图 / 地形 Layer 0
    GLuint   texLayers[3]  = {0, 0, 0};  // 【新增】纹理单元 1..3: 地形 Layer 1..3
    GLuint   texAlphaMap   = 0;          // 【新增】纹理单元 4: 地形 64x64 RGB Alpha 遮罩图
    uint32_t indexCount    = 0;
    uint32_t firstIndex    = 0;
    int32_t  baseVertex    = 0;
    uint32_t instanceCount = 1; // 实例化绘制数量
    uint32_t baseInstance  = 0; // 在 1 号 SSBO 矩阵数组中的起始下标

    static uint64_t MakeKey(uint8_t pass, uint16_t progId, uint16_t texId, uint32_t depthBits) noexcept {
        return (static_cast<uint64_t>(pass & 0xF) << 60) |
               (static_cast<uint64_t>(progId & 0xFFF) << 48) |
               (static_cast<uint64_t>(texId & 0xFFFF) << 32) |
               static_cast<uint64_t>(depthBits);
    }
};

inline void SubmitDrawCommands(std::vector<DrawCommand>& cmds) {
    std::sort(cmds.begin(), cmds.end(), [](const DrawCommand& a, const DrawCommand& b) {
        return a.sortKey < b.sortKey;
    });

    GLuint lastProg = 0, lastVao = 0, lastVbo = 0, lastIbo = 0, lastTex0 = 0;
    uint8_t lastPass = 0xFF; // 0xFF = 尚未向当前 program 下发过 Cutout 开关
    for (const auto& c : cmds) {
        if (c.program != lastProg) {
            glUseProgram(c.program);
            lastProg = c.program;
            lastPass = 0xFF; // 新着色器必须重新下发一次 uniform
        }
        // sortKey 高 4 位即 Pass ID (2 = 不透明, 3 = Alpha Cutout)
        uint8_t pass = static_cast<uint8_t>(c.sortKey >> 60);
        if (pass != lastPass) {
            glUniform1i(0, (pass == 3) ? 1 : 0);
            lastPass = pass;
        }
        if (c.vao != lastVao) {
            glBindVertexArray(c.vao);
            lastVao = c.vao;
            lastVbo = 0;
            lastIbo = 0;
        }
        if (c.vbo != lastVbo) {
            glVertexArrayVertexBuffer(c.vao, 0, c.vbo, 0, c.stride);
            lastVbo = c.vbo;
        }
        if (c.ibo != lastIbo) {
            glVertexArrayElementBuffer(c.vao, c.ibo);
            lastIbo = c.ibo;
        }
        if (c.texAlbedo != lastTex0) {
            glBindTextureUnit(0, c.texAlbedo);
            lastTex0 = c.texAlbedo;
        }
		// 【新增】若该指令携带了地形多层遮罩图，则绑定 Layer 1..3 与 Alpha 遮罩到单元 1..4
        if (c.texAlphaMap != 0) {
            glBindTextureUnit(1, c.texLayers[0]);
            glBindTextureUnit(2, c.texLayers[1]);
            glBindTextureUnit(3, c.texLayers[2]);
            glBindTextureUnit(4, c.texAlphaMap);
        }
        const void* indexOffsetPtr = reinterpret_cast<const void*>(static_cast<uintptr_t>(c.firstIndex) * sizeof(uint16_t));
        // 使用 OpenGL 4.2+ BaseInstance 指令，让着色器通过 gl_BaseInstance + gl_InstanceID 直接寻址 SSBO
        glDrawElementsInstancedBaseVertexBaseInstance(
            GL_TRIANGLES,
            static_cast<GLsizei>(c.indexCount),
            GL_UNSIGNED_SHORT,
            indexOffsetPtr,
            static_cast<GLsizei>(c.instanceCount),
            c.baseVertex,
            c.baseInstance
        );
    }
}

} // namespace coa::rhi
