#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <vector>
#include <array>
#include <unordered_map>
#include <algorithm>

#include <glm/glm.hpp>

#include "vfs/vfs.hpp"

namespace coa::assets {

#pragma pack(push, 1)
struct WdbcHeader {
    char     magic[4];          // "WDBC"
    uint32_t recordCount;       // 记录数
    uint32_t fieldCount;        // 每条记录字段数
    uint32_t recordSize;        // 每条记录字节数
    uint32_t stringBlockSize;   // 字符串块字节数
};
#pragma pack(pop)

static_assert(sizeof(WdbcHeader) == 20, "WdbcHeader must be 20 bytes");

struct ColorTrack {
    std::vector<uint32_t> times;   // 半分钟单位 [0, 2880)
    std::vector<glm::vec3> colors;

    [[nodiscard]] glm::vec3 Sample(float timeHalfMinutes, const glm::vec3& fallback) const {
        const size_t n = std::min(times.size(), colors.size());
        if (n == 0) return fallback;
        if (n == 1) return colors[0];

        float t = std::fmod(timeHalfMinutes, 2880.0f);
        if (t < 0.0f) t += 2880.0f;

        for (size_t i = 0; i < n; ++i) {
            const size_t j = (i + 1) % n;
            float t0 = static_cast<float>(times[i]);
            float t1 = static_cast<float>(times[j]);
            if (j == 0) t1 += 2880.0f;

            float tt = t;
            if (tt < t0) tt += 2880.0f;

            if (tt >= t0 && tt <= t1) {
                const float span = std::max(t1 - t0, 1e-4f);
                const float a = glm::clamp((tt - t0) / span, 0.0f, 1.0f);
                return glm::mix(colors[i], colors[j], a);
            }
        }
        return colors.back();
    }
};

struct EvaluatedLight {
    glm::vec3 directColor{1.0f, 0.95f, 0.85f};   // Band 0: 太阳/月亮漫反射光
    glm::vec3 ambientColor{0.45f, 0.52f, 0.62f};  // Band 1: 环境光
    glm::vec3 skyTop{0.22f, 0.45f, 0.78f};        // Band 2: 天顶色
    glm::vec3 skyMiddle{0.35f, 0.58f, 0.86f};     // Band 3: 中空色
    glm::vec3 skyBand1{0.48f, 0.68f, 0.90f};      // Band 4: 中低空色
    glm::vec3 skyBand2{0.60f, 0.76f, 0.92f};      // Band 5: 地平线上方色
    glm::vec3 skyHorizon{0.70f, 0.82f, 0.94f};    // Band 6: 地平线色
    glm::vec3 fogColor{0.53f, 0.72f, 0.92f};      // Band 7: 远景雾色
    glm::vec3 riverShallow{0.16f, 0.50f, 0.48f};  // Band 16: 湖河浅水色
    glm::vec3 riverDeep{0.03f, 0.14f, 0.25f};     // Band 17: 湖河深水色
    glm::vec3 sunDir{-0.5f, -0.4f, 0.75f};
};

class LightDbcManager {
public:
    bool Load(const vfs::VirtualFileSystem& vfs, uint32_t targetMapId = 0) {
        m_lights.clear();
        m_bands.clear();

        auto lightBytes = vfs.ReadFile("dbfilesclient/light.dbc");
        auto bandBytes  = vfs.ReadFile("dbfilesclient/lightintband.dbc");

        if (!lightBytes || !bandBytes) {
            std::fprintf(stderr, "[LightDbc] DBC not found, using built-in Elwynn Forest fallback.\n");
            BuildFallback();
            return true;
        }

        const bool bandsOk  = ParseBands(*bandBytes);
        const bool lightsOk = ParseLights(*lightBytes, targetMapId);

        if (!bandsOk || !lightsOk || m_lights.empty()) {
            std::fprintf(stderr, "[LightDbc] Parse failed or no map %u light, using built-in fallback.\n", targetMapId);
            BuildFallback();
            return true;
        }

        std::printf("[LightDbc] Loaded %zu light points and %zu color tracks for map %u.\n",
                    m_lights.size(), m_bands.size(), targetMapId);
        return true;
    }

    [[nodiscard]] EvaluatedLight Evaluate(const glm::vec3& camPos, float timeHalfMinutes, bool isUnderwater) const {
        EvaluatedLight out{};

        const glm::vec3 defDirect   = out.directColor;
        const glm::vec3 defAmbient  = out.ambientColor;
        const glm::vec3 defSkyTop   = out.skyTop;
        const glm::vec3 defSkyMid   = out.skyMiddle;
        const glm::vec3 defSkyBand1 = out.skyBand1;
        const glm::vec3 defSkyBand2 = out.skyBand2;
        const glm::vec3 defHorizon  = out.skyHorizon;
        const glm::vec3 defFog      = out.fogColor;
        const glm::vec3 defShallow  = out.riverShallow;
        const glm::vec3 defDeep     = out.riverDeep;

        glm::vec3 sumDirect(0.0f);
        glm::vec3 sumAmbient(0.0f);
        glm::vec3 sumSkyTop(0.0f);
        glm::vec3 sumSkyMid(0.0f);
        glm::vec3 sumSkyBand1(0.0f);
        glm::vec3 sumSkyBand2(0.0f);
        glm::vec3 sumHorizon(0.0f);
        glm::vec3 sumFog(0.0f);
        glm::vec3 sumShallow(0.0f);
        glm::vec3 sumDeep(0.0f);

        float totalWeight = 0.0f;

        for (const auto& lp : m_lights) {
            float weight = 0.0f;

            if (lp.isGlobal) {
                weight = 1.0f;
            } else {
                const float dist = glm::length(camPos - lp.worldPos);
                const float outer = std::max(lp.rOuter, lp.rInner + 1e-3f);
                weight = 1.0f - glm::smoothstep(lp.rInner, outer, dist);
            }

            if (weight <= 1e-4f) continue;

            uint32_t paramsId = isUnderwater ? lp.underwaterParamsId : lp.clearParamsId;
            if (paramsId == 0) paramsId = lp.clearParamsId;
            if (paramsId == 0) continue;

            sumDirect   += SampleBand(paramsId, 0,  defDirect)   * weight;
            sumAmbient  += SampleBand(paramsId, 1,  defAmbient)  * weight;
            sumSkyTop   += SampleBand(paramsId, 2,  defSkyTop)   * weight;
            sumSkyMid   += SampleBand(paramsId, 3,  defSkyMid)   * weight;
            sumSkyBand1 += SampleBand(paramsId, 4,  defSkyBand1) * weight;
            sumSkyBand2 += SampleBand(paramsId, 5,  defSkyBand2) * weight;
            sumHorizon  += SampleBand(paramsId, 6,  defHorizon)  * weight;
            sumFog      += SampleBand(paramsId, 7,  defFog)      * weight;
            sumShallow  += SampleBand(paramsId, 16, defShallow)  * weight;
            sumDeep     += SampleBand(paramsId, 17, defDeep)     * weight;

            totalWeight += weight;
        }

        if (totalWeight <= 1e-4f) {
            sumDirect   = SampleBand(1, 0,  defDirect);
            sumAmbient  = SampleBand(1, 1,  defAmbient);
            sumSkyTop   = SampleBand(1, 2,  defSkyTop);
            sumSkyMid   = SampleBand(1, 3,  defSkyMid);
            sumSkyBand1 = SampleBand(1, 4,  defSkyBand1);
            sumSkyBand2 = SampleBand(1, 5,  defSkyBand2);
            sumHorizon  = SampleBand(1, 6,  defHorizon);
            sumFog      = SampleBand(1, 7,  defFog);
            sumShallow  = SampleBand(1, 16, defShallow);
            sumDeep     = SampleBand(1, 17, defDeep);
            totalWeight = 1.0f;
        }

        const float invW = 1.0f / totalWeight;

        out.directColor  = sumDirect   * invW;
        out.ambientColor = sumAmbient  * invW;
        out.skyTop       = sumSkyTop   * invW;
        out.skyMiddle    = sumSkyMid   * invW;
        out.skyBand1     = sumSkyBand1 * invW;
        out.skyBand2     = sumSkyBand2 * invW;
        out.skyHorizon   = sumHorizon  * invW;
        out.fogColor     = sumFog      * invW;
        out.riverShallow = sumShallow  * invW;
        out.riverDeep    = sumDeep     * invW;

        constexpr float kTwoPi = 6.28318530718f;
        constexpr float kHalfPi = 1.57079632679f;
        const float phase = (timeHalfMinutes / 2880.0f) * kTwoPi;
        const float elev  = std::sin(phase - kHalfPi);
        const float horiz = std::cos(phase - kHalfPi);

        const glm::vec3 sun = glm::normalize(glm::vec3(horiz * 0.75f, horiz * 0.35f, elev));
        out.sunDir = (elev >= 0.0f) ? sun : glm::normalize(glm::vec3(-sun.x, -sun.y, -sun.z));

        if (isUnderwater) {
            out.fogColor = glm::mix(out.riverDeep, out.riverShallow, 0.35f);
        }

        return out;
    }

private:
    struct LightPoint {
        bool      isGlobal = false;
        glm::vec3 worldPos{0.0f};
        float     rInner = 0.0f;
        float     rOuter = 0.0f;
        uint32_t  clearParamsId = 0;
        uint32_t  underwaterParamsId = 0;
    };

    std::vector<LightPoint> m_lights;
    std::unordered_map<uint32_t, ColorTrack> m_bands;

    [[nodiscard]] glm::vec3 SampleBand(uint32_t paramsId, uint32_t k, const glm::vec3& fallback) const {
        if (paramsId == 0) return fallback;

        const uint32_t bandId = (paramsId - 1) * 18 + 1 + k;
        auto it = m_bands.find(bandId);
        if (it == m_bands.end()) return fallback;
        return it->second.Sample(0.0f, fallback);
    }

    static glm::vec3 DecodeColor(uint32_t c) {
        return glm::vec3(
            static_cast<float>((c >> 16) & 0xFF) / 255.0f,
            static_cast<float>((c >> 8) & 0xFF) / 255.0f,
            static_cast<float>(c & 0xFF) / 255.0f
        );
    }

    static uint32_t ReadU32(const uint8_t* rec, uint32_t fieldIndex) {
        uint32_t v = 0;
        std::memcpy(&v, rec + static_cast<size_t>(fieldIndex) * 4, sizeof(v));
        return v;
    }

    static float ReadF32(const uint8_t* rec, uint32_t fieldIndex) {
        float v = 0.0f;
        std::memcpy(&v, rec + static_cast<size_t>(fieldIndex) * 4, sizeof(v));
        return v;
    }

    bool ParseBands(const std::vector<uint8_t>& bytes) {
        if (bytes.size() < sizeof(WdbcHeader)) return false;

        WdbcHeader hdr{};
        std::memcpy(&hdr, bytes.data(), sizeof(hdr));
        if (std::memcmp(hdr.magic, "WDBC", 4) != 0) return false;
        if (hdr.recordSize < 34 * 4) return false;

        const size_t recordsBytes = static_cast<size_t>(hdr.recordCount) * hdr.recordSize;
        if (bytes.size() < sizeof(WdbcHeader) + recordsBytes) return false;

        const uint8_t* records = bytes.data() + sizeof(WdbcHeader);

        for (uint32_t r = 0; r < hdr.recordCount; ++r) {
            const uint8_t* rec = records + static_cast<size_t>(r) * hdr.recordSize;

            const uint32_t id = ReadU32(rec, 0);
            const uint32_t count = std::min(ReadU32(rec, 1), 16u);
            if (count == 0) continue;

            ColorTrack track;
            track.times.reserve(count);
            track.colors.reserve(count);

            for (uint32_t i = 0; i < count; ++i) {
                track.times.push_back(ReadU32(rec, 2 + i));
                track.colors.push_back(DecodeColor(ReadU32(rec, 18 + i)));
            }

            m_bands.emplace(id, std::move(track));
        }

        return !m_bands.empty();
    }

    bool ParseLights(const std::vector<uint8_t>& bytes, uint32_t targetMapId) {
        if (bytes.size() < sizeof(WdbcHeader)) return false;

        WdbcHeader hdr{};
        std::memcpy(&hdr, bytes.data(), sizeof(hdr));
        if (std::memcmp(hdr.magic, "WDBC", 4) != 0) return false;
        if (hdr.recordSize < 9 * 4) return false;

        const size_t recordsBytes = static_cast<size_t>(hdr.recordCount) * hdr.recordSize;
        if (bytes.size() < sizeof(WdbcHeader) + recordsBytes) return false;

        const uint8_t* records = bytes.data() + sizeof(WdbcHeader);

        for (uint32_t r = 0; r < hdr.recordCount; ++r) {
            const uint8_t* rec = records + static_cast<size_t>(r) * hdr.recordSize;

            const uint32_t mapId = ReadU32(rec, 1);
            if (mapId != targetMapId) continue;

            const float dbcX = ReadF32(rec, 2);
            const float dbcY = ReadF32(rec, 3);
            const float dbcZ = ReadF32(rec, 4);

            LightPoint lp;
            lp.clearParamsId      = ReadU32(rec, 7);
            lp.underwaterParamsId = ReadU32(rec, 8);

            if (std::fabs(dbcX) < 1e-3f && std::fabs(dbcY) < 1e-3f && std::fabs(dbcZ) < 1e-3f) {
                lp.isGlobal = true;
            } else {
                const float falloffStart = ReadF32(rec, 5);
                const float falloffEnd   = ReadF32(rec, 6);

                lp.worldPos = glm::vec3(
                    17066.66656f - dbcZ / 36.0f,
                    17066.66656f - dbcX / 36.0f,
                    dbcY / 36.0f
                );
                lp.rInner = falloffStart / 36.0f;
                lp.rOuter = falloffEnd / 36.0f;
            }

            m_lights.push_back(lp);
        }

        return !m_lights.empty();
    }

    void BuildFallback() {
        m_lights.clear();
        m_bands.clear();

        const std::array<uint32_t, 4> times = {0, 660, 1440, 2220};

        auto addTrack = [&](uint32_t id, const std::array<glm::vec3, 4>& colors) {
            ColorTrack track;
            track.times.assign(times.begin(), times.end());
            track.colors.assign(colors.begin(), colors.end());
            m_bands.emplace(id, std::move(track));
        };

        // Band 0: 直射光 (午夜月光 / 黎明金粉 / 正午阳光 / 黄昏晚霞)
        addTrack(1, {
            glm::vec3(0.55f, 0.65f, 0.85f),
            glm::vec3(1.00f, 0.65f, 0.45f),
            glm::vec3(1.00f, 0.97f, 0.90f),
            glm::vec3(1.00f, 0.52f, 0.28f)
        });

        // Band 1: 环境光
        addTrack(2, {
            glm::vec3(0.10f, 0.13f, 0.22f),
            glm::vec3(0.38f, 0.32f, 0.40f),
            glm::vec3(0.45f, 0.52f, 0.62f),
            glm::vec3(0.38f, 0.28f, 0.30f)
        });

        // Band 2: 天顶色
        addTrack(3, {
            glm::vec3(0.02f, 0.03f, 0.08f),
            glm::vec3(0.18f, 0.28f, 0.55f),
            glm::vec3(0.18f, 0.42f, 0.82f),
            glm::vec3(0.12f, 0.18f, 0.45f)
        });

        // Band 3: 中空色
        addTrack(4, {
            glm::vec3(0.04f, 0.06f, 0.14f),
            glm::vec3(0.42f, 0.40f, 0.62f),
            glm::vec3(0.32f, 0.58f, 0.90f),
            glm::vec3(0.32f, 0.24f, 0.48f)
        });

        // Band 4: 中低空色
        addTrack(5, {
            glm::vec3(0.07f, 0.10f, 0.20f),
            glm::vec3(0.78f, 0.52f, 0.55f),
            glm::vec3(0.46f, 0.68f, 0.94f),
            glm::vec3(0.72f, 0.38f, 0.36f)
        });

        // Band 5: 地平线上方色
        addTrack(6, {
            glm::vec3(0.10f, 0.14f, 0.26f),
            glm::vec3(0.95f, 0.68f, 0.50f),
            glm::vec3(0.58f, 0.78f, 0.97f),
            glm::vec3(0.95f, 0.52f, 0.30f)
        });

        // Band 6: 地平线色
        addTrack(7, {
            glm::vec3(0.13f, 0.17f, 0.30f),
            glm::vec3(1.00f, 0.72f, 0.48f),
            glm::vec3(0.70f, 0.84f, 0.98f),
            glm::vec3(1.00f, 0.60f, 0.30f)
        });

        // Band 7: 远景雾色
        addTrack(8, {
            glm::vec3(0.08f, 0.10f, 0.18f),
            glm::vec3(0.78f, 0.58f, 0.50f),
            glm::vec3(0.62f, 0.78f, 0.95f),
            glm::vec3(0.72f, 0.44f, 0.32f)
        });

        // Band 16: 浅水色
        addTrack(17, {
            glm::vec3(0.03f, 0.09f, 0.14f),
            glm::vec3(0.10f, 0.28f, 0.30f),
            glm::vec3(0.18f, 0.55f, 0.52f),
            glm::vec3(0.12f, 0.26f, 0.28f)
        });

        // Band 17: 深水色
        addTrack(18, {
            glm::vec3(0.01f, 0.02f, 0.06f),
            glm::vec3(0.02f, 0.08f, 0.16f),
            glm::vec3(0.03f, 0.16f, 0.30f),
            glm::vec3(0.02f, 0.07f, 0.14f)
        });

        LightPoint globalLight;
        globalLight.isGlobal = true;
        globalLight.clearParamsId = 1;
        globalLight.underwaterParamsId = 1;
        m_lights.push_back(globalLight);
    }
};

} // namespace coa::assets
