// Raw sensor encoding. No color transfer function is applied to confidence.
#ifndef SKU_RGBD_PNG_H
#define SKU_RGBD_PNG_H

#include "lodepng.h"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace sku_rgbd {
inline bool encodeGray(const std::vector<unsigned char>& pixels, int width, int height,
                       unsigned bitDepth, std::vector<unsigned char>& png) {
    lodepng::State state;
    state.info_raw.colortype = LCT_GREY;
    state.info_raw.bitdepth = bitDepth;
    state.info_png.color.colortype = LCT_GREY;
    state.info_png.color.bitdepth = bitDepth;
    // Auto conversion can turn confidence into palette/RGB or zero depth into 1-bit.
    // The consumer requires a single channel at exactly 8 or 16 bits.
    state.encoder.auto_convert = 0;
    png.clear();
    return lodepng::encode(png, pixels, unsigned(width), unsigned(height), state) == 0;
}

inline bool encodeDepth(const float* input, int width, int height, size_t stride,
                        std::vector<unsigned char>& png) {
    if (!input || width <= 0 || height <= 0 || stride < size_t(width) * sizeof(float)
        || stride % alignof(float) != 0) return false;
    std::vector<unsigned char> pixels(size_t(width) * height * 2);
    for (int y = 0; y < height; ++y) {
        const float* row = reinterpret_cast<const float*>(
            reinterpret_cast<const unsigned char*>(input) + y * stride);
        for (int x = 0; x < width; ++x) {
            double mm = std::round(double(row[x]) * 1000.0);
            uint16_t value = std::isfinite(mm) && mm > 0 && mm <= 65535 ? uint16_t(mm) : 0;
            size_t index = (size_t(y) * width + x) * 2;
            pixels[index] = value >> 8; // PNG 16-bit samples are big endian.
            pixels[index + 1] = value & 0xff;
        }
    }
    return encodeGray(pixels, width, height, 16, png);
}

inline bool encodeConfidence(const uint8_t* input, int width, int height, size_t stride,
                             std::vector<unsigned char>& png) {
    if (!input || width <= 0 || height <= 0 || stride < size_t(width)) return false;
    std::vector<unsigned char> pixels(size_t(width) * height);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            uint8_t value = input[size_t(y) * stride + x];
            if (value > 2) return false;
            pixels[size_t(y) * width + x] = value;
        }
    }
    return encodeGray(pixels, width, height, 8, png);
}
} // namespace sku_rgbd
#endif
