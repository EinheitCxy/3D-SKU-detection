#include "RgbdPng.h"
#include <cassert>
#include <iostream>
#include <limits>
#include <string>

int main(int argc, char** argv) {
    // Six useful floats per row plus two padding values that must never be read.
    const float depth[] = {
        0, 1.25f, -1, std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(), 100.0f, 9, 9,
        2, 3.5f, 0.125f, 10, 60, 0.001f, 9, 9
    };
    const unsigned expected[] = {0, 1250, 0, 0, 0, 0, 2000, 3500, 125, 10000, 60000, 1};
    std::vector<unsigned char> depthPNG, decoded;
    assert(sku_rgbd::encodeDepth(depth, 6, 2, 8 * sizeof(float), depthPNG));
    assert(depthPNG[24] == 16 && depthPNG[25] == LCT_GREY);
    unsigned width = 0, height = 0;
    assert(lodepng::decode(decoded, width, height, depthPNG, LCT_GREY, 16) == 0);
    assert(width == 6 && height == 2 && decoded.size() == 24);
    for (size_t i = 0; i < 12; ++i)
        assert(((unsigned(decoded[2 * i]) << 8) | decoded[2 * i + 1]) == expected[i]);

    const uint8_t confidence[] = {0, 1, 2, 255, 2, 1, 0, 255};
    std::vector<unsigned char> confidencePNG;
    assert(sku_rgbd::encodeConfidence(confidence, 3, 2, 4, confidencePNG));
    assert(confidencePNG[24] == 8 && confidencePNG[25] == LCT_GREY);
    decoded.clear(); // lodepng's C++ decode appends to the destination vector.
    assert(lodepng::decode(decoded, width, height, confidencePNG, LCT_GREY, 8) == 0);
    assert(width == 3 && height == 2);
    assert((decoded == std::vector<unsigned char>{0, 1, 2, 2, 1, 0}));
    assert(!sku_rgbd::encodeConfidence(confidence, 4, 2, 4, decoded));
    assert(!sku_rgbd::encodeDepth(depth, 6, 2, 5 * sizeof(float), decoded));
    const float zeros[] = {0, 0, 0, 0};
    std::vector<unsigned char> zeroPNG;
    assert(sku_rgbd::encodeDepth(zeros, 2, 2, 2 * sizeof(float), zeroPNG));
    assert(zeroPNG[24] == 16 && zeroPNG[25] == LCT_GREY);
    if (argc == 2) {
        assert(lodepng::save_file(depthPNG, std::string(argv[1]) + "/depth.png") == 0);
        assert(lodepng::save_file(confidencePNG, std::string(argv[1]) + "/confidence.png") == 0);
    }
    std::cout << "PASS: PNG depth stride/mm/invalid/endianness; confidence stride/raw levels/rejection\n";
}
