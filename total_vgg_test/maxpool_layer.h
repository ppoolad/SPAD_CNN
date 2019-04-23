#include <string>
#ifndef _MAXPOOL_LAYER_H
#define _MAXPOOL_LAYER_H

// Limits
// /#define MAX_BATCH 10
// #define MAX_INPUT_DIMS 512
// #define MAX_OUTPUT_DIMS 512
// #define MAX_INPUT_WIDTH 226
// #define MAX_INPUT_HEIGHT 226
// #define MAX_OUTPUT_WIDTH 224
// #define MAX_OUTPUT_HEIGHT 224
// #define MAX_CONV_INPUT MAX_INPUT_DIMS*MAX_INPUT_WIDTH*MAX_INPUT_HEIGHT 
// #define MAX_CONV_OUTPUT MAX_OUTPUT_DIMS*MAX_OUTPUT_WIDTH*MAX_OUTPUT_HEIGHT 
// #define MAXPOOL_WINDOW 2

void maxpool_layer(float * mem,            // global memory pointer
                int input_offset,       // offset of inputs
                int output_offset,      // offset of outputs
                const int b,            // batch size
                const int od,           // output dimensions
                const int ox,           // output width
                const int oy,           // output height
                const int id,           // input dimensions
                const int ix,           // input width
                const int iy           // input height
                );
void hw_maxpool_layer(int target,             // control register target
                   float * mem,            // global memory pointer
                   int input_offset,       // offset of inputs
                   int output_offset,      // offset of outputs
                   const int b,            // batch size
                   const int od,           // output dimensions
                   const int ox,           // output width
                   const int oy,           // output height
                   const int id,           // input dimensions
                   const int ix,           // input width
                   const int iy           // input height
                );       

int run_maxpool(std::string prevLayer, int numBatches);
#endif
