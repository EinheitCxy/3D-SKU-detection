// Copyright © 2022 Stray Robots. MIT license; see LICENSE.md.
#import "PngEncoder.h"
#include "RgbdPng.h"

@implementation PngEncoder {
    std::vector<unsigned char> png;
}
- (instancetype)initWithDepth:(const float*)content width:(int)width height:(int)height bytesPerRow:(size_t)stride {
    self = [super init];
    if (self && !sku_rgbd::encodeDepth(content, width, height, stride, png)) return nil;
    return self;
}

- (instancetype)initWithConfidence:(const uint8_t*)content width:(int)width height:(int)height bytesPerRow:(size_t)stride {
    self = [super init];
    if (self && !sku_rgbd::encodeConfidence(content, width, height, stride, png)) return nil;
    return self;
}

- (NSData*)fileContents {
    if (png.empty()) return nil;
    return [NSData dataWithBytes:png.data() length:png.size()];
}
@end
