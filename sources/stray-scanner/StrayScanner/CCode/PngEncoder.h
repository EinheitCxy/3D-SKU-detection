//
//  PngWriter.h
//  StrayScanner
//
//  Created by Kenneth Blomqvist on 2/15/22.
//  Copyright © 2022 Stray Robots. All rights reserved.
//

#ifndef PngEncoder_h
#define PngEncoder_h
#include <Foundation/Foundation.h>

@interface PngEncoder : NSObject

- (nullable instancetype) initWithDepth:(const float *)content width:(int)width height:(int)height bytesPerRow:(size_t)stride;
- (nullable instancetype) initWithConfidence:(const uint8_t *)content width:(int)width height:(int)height bytesPerRow:(size_t)stride;
- (nullable NSData*) fileContents;

@end
#endif /* PngWriter_h */
