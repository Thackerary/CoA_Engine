#pragma once
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>
#include <algorithm>
#include <bit>
#include <optional>
#include "../rhi/rhi_gl.hpp"

namespace coa::assets {

#pragma pack(push, 1)
struct Blp2Header {
    uint8_t  magic[4];         // 固定为 'B', 'L', 'P', '2'
    uint32_t version;          // 固定为 1
    uint8_t  compression;      // 1 = 256色调色板, 2 = DXTC (BC1/BC2/BC3), 3 = 未压缩 BGRA8
    uint8_t  alphaDepth;       // Alpha 位数: 0, 1, 4, 8
    uint8_t  alphaEncoding;    // DXTC 模式下: 0 = DXT1, 1 = DXT3, 7 = DXT5
    uint8_t  hasMips;          // 0 = 仅 1 层, 非 0 = 包含 Mipmap 链
    uint32_t width;            // 基礎宽度 (2 的幂)
    uint32_t height;           // 基础高度 (2 的幂)
    uint32_t mipOffsets[16];   // 各级 Mipmap 相对于文件开头的字节偏移
    uint32_t mipSizes[16];     // 各级 Mipmap 的字节长度
    uint32_t palette[256];     // 256 个 BGRA8 调色板颜色项 (共 1024 字节)
};
#pragma pack(pop)

static_assert(sizeof(Blp2Header) == 1172, "Blp2Header size must be exactly 1172 bytes");

class BlpLoader {
public:
    // 从内存字节流直接创建并上传 OpenGL 4.6 DSA 纹理
    [[nodiscard]] static std::optional<rhi::GlTexture2D> LoadFromMemory(
        std::span<const uint8_t> fileBytes,
        bool repeatWrap = true)
    {
        if (fileBytes.size() < sizeof(Blp2Header)) {
            return std::nullopt;
        }

        Blp2Header hdr{};
        std::memcpy(&hdr, fileBytes.data(), sizeof(Blp2Header));

        if (std::memcmp(hdr.magic, "BLP2", 4) != 0 || hdr.version != 1 || hdr.width == 0 || hdr.height == 0) {
            return std::nullopt;
        }

        // 1. 判定目标 GPU 像素格式与是否为硬件压缩块
        bool isCompressed = (hdr.compression == 2);
        rhi::PixelFormat gpuFormat = rhi::PixelFormat::RGBA8;
        uint32_t blockBytes = 0;

        if (isCompressed) {
            if (hdr.alphaEncoding == 0) {
                gpuFormat  = rhi::PixelFormat::BC1_RGBA; // DXT1
                blockBytes = 8;
            } else if (hdr.alphaEncoding == 1) {
                gpuFormat  = rhi::PixelFormat::BC2_RGBA; // DXT3
                blockBytes = 16;
            } else if (hdr.alphaEncoding == 7) {
                gpuFormat  = rhi::PixelFormat::BC3_RGBA; // DXT5
                blockBytes = 16;
            } else {
                return std::nullopt;
            }
        }

        // 2. 计算理论最大 Mipmap 层数，并统计文件中实际有效且未越界的 Mipmap 层数
        uint32_t maxDim = std::max(hdr.width, hdr.height);
        uint32_t maxTheoreticalMips = std::bit_width(maxDim); // floor(log2(maxDim)) + 1
        uint32_t maxCheckMips = (hdr.hasMips == 0) ? 1u : std::min<uint32_t>(16u, maxTheoreticalMips);

        uint32_t validMipCount = 0;
        for (uint32_t i = 0; i < maxCheckMips; ++i) {
            uint32_t offset = hdr.mipOffsets[i];
            uint32_t size   = hdr.mipSizes[i];
            if (offset == 0 || size == 0 || static_cast<size_t>(offset) + size > fileBytes.size()) {
                break;
            }

            uint32_t mipW = std::max(1u, hdr.width >> i);
            uint32_t mipH = std::max(1u, hdr.height >> i);

            if (isCompressed) {
                uint32_t blocksX = std::max(1u, (mipW + 3u) / 4u);
                uint32_t blocksY = std::max(1u, (mipH + 3u) / 4u);
                uint32_t expectedBytes = blocksX * blocksY * blockBytes;
                if (size < expectedBytes) {
                    break; // 截断的尾端 Mipmap 直接丢弃，防止 glCompressedTextureSubImage2D 报错
                }
            }
            ++validMipCount;
        }

        if (validMipCount == 0) {
            return std::nullopt;
        }

        // 3. 使用 DSA 一次性分配不可变纹理存储
        rhi::GlTexture2D texture(hdr.width, hdr.height, validMipCount, gpuFormat, repeatWrap);

        // 4. 分支 A：DXT1 / DXT3 / DXT5 零解压直传 GPU 显存
        if (isCompressed) {
            for (uint32_t i = 0; i < validMipCount; ++i) {
                uint32_t mipW = std::max(1u, hdr.width >> i);
                uint32_t mipH = std::max(1u, hdr.height >> i);
                uint32_t blocksX = std::max(1u, (mipW + 3u) / 4u);
                uint32_t blocksY = std::max(1u, (mipH + 3u) / 4u);
                uint32_t exactBytes = blocksX * blocksY * blockBytes;

                auto mipSpan = fileBytes.subspan(hdr.mipOffsets[i], exactBytes);
                texture.UploadCompressedMip(i, mipW, mipH, mipSpan);
            }
            return texture;
        }

        // 5. 分支 B：兜底处理调色板贴图 (compression == 1) 与未压缩 BGRA8 (compression == 3)
        std::vector<uint8_t> rgbaBuf;
        for (uint32_t i = 0; i < validMipCount; ++i) {
            uint32_t mipW = std::max(1u, hdr.width >> i);
            uint32_t mipH = std::max(1u, hdr.height >> i);
            uint32_t pixelCount = mipW * mipH;
            auto rawMip = fileBytes.subspan(hdr.mipOffsets[i], hdr.mipSizes[i]);

            rgbaBuf.resize(static_cast<size_t>(pixelCount) * 4u);

            if (hdr.compression == 1) {
                if (!DecodePalettizedMip(hdr, rawMip, pixelCount, rgbaBuf)) {
                    break;
                }
            } else if (hdr.compression == 3) {
                if (rawMip.size() < static_cast<size_t>(pixelCount) * 4u) {
                    break;
                }
                // BGRA8 转 RGBA8
                for (uint32_t p = 0; p < pixelCount; ++p) {
                    rgbaBuf[p * 4 + 0] = rawMip[p * 4 + 2]; // R
                    rgbaBuf[p * 4 + 1] = rawMip[p * 4 + 1]; // G
                    rgbaBuf[p * 4 + 2] = rawMip[p * 4 + 0]; // B
                    rgbaBuf[p * 4 + 3] = rawMip[p * 4 + 3]; // A
                }
            }
            texture.UploadUncompressedMip(i, mipW, mipH, rgbaBuf);
        }

        return texture;
    }

private:
    // 解码 256 色调色板 + 0/1/4/8-bit 独立 Alpha 通道
    static bool DecodePalettizedMip(
        const Blp2Header& hdr,
        std::span<const uint8_t> rawMip,
        uint32_t pixelCount,
        std::vector<uint8_t>& outRgba)
    {
        if (rawMip.size() < pixelCount) return false;
        const uint8_t* indices  = rawMip.data();
        const uint8_t* alphaPtr = rawMip.data() + pixelCount;
        size_t alphaBytesLeft   = rawMip.size() - pixelCount;

        for (uint32_t p = 0; p < pixelCount; ++p) {
            uint32_t bgra = hdr.palette[indices[p]];
            outRgba[p * 4 + 0] = static_cast<uint8_t>((bgra >> 16) & 0xFF); // R
            outRgba[p * 4 + 1] = static_cast<uint8_t>((bgra >> 8)  & 0xFF); // G
            outRgba[p * 4 + 2] = static_cast<uint8_t>((bgra >> 0)  & 0xFF); // B

            uint8_t a = 255;
            if (hdr.alphaDepth == 8 && p < alphaBytesLeft) {
                a = alphaPtr[p];
            } else if (hdr.alphaDepth == 4 && (p / 2) < alphaBytesLeft) {
                uint8_t nibble = (p & 1) ? (alphaPtr[p / 2] >> 4) : (alphaPtr[p / 2] & 0x0F);
                a = static_cast<uint8_t>(nibble * 17);
            } else if (hdr.alphaDepth == 1 && (p / 8) < alphaBytesLeft) {
                a = ((alphaPtr[p / 8] >> (p & 7)) & 1) ? 255 : 0;
            }
            outRgba[p * 4 + 3] = a;
        }
        return true;
    }
};

} // namespace coa::assets