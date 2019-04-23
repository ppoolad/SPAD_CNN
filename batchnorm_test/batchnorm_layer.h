#ifndef _BATCHNORM_LAYER_H
#define _BATCHNORM_LAYER_H

// Limits
#define MAX_BATCH 10
#define MAX_KERNEL_SIZE 3
#define MAX_INPUT_DIMS 512
#define MAX_OUTPUT_DIMS 512
#define MAX_INPUT_WIDTH 226
#define MAX_INPUT_HEIGHT 226
#define MAX_OUTPUT_WIDTH 224
#define MAX_OUTPUT_HEIGHT 224
#define MAX_CONV_INPUT MAX_INPUT_DIMS*MAX_INPUT_WIDTH*MAX_INPUT_HEIGHT 
#define MAX_CONV_OUTPUT MAX_OUTPUT_DIMS*MAX_OUTPUT_WIDTH*MAX_OUTPUT_HEIGHT 
#define MAX_WEIGHT_SIZE MAX_OUTPUT_DIMS*MAX_INPUT_DIMS*MAX_KERNEL_SIZE*MAX_KERNEL_SIZE /* ALL TO BE DEFINED*/

void batchnorm_layer(float * mem,            // global memory pointer
                int input_offset,       // offset of inputs
                int output_offset,      // offset of outputs
                const int b,            // batch size
                const int id,           // input dimensions
                const int ix,           // input width
                const int iy,
                const float mean,
                const float var,
                const float epsilon,
                const float gamma,
                const float beta);            // input height
void hw_batchnorm_layer(int target,
                float * mem,            // global memory pointer
                int input_offset,       // offset of inputs
                int output_offset,      // offset of outputs
                const int b,            // batch size
                const int id,           // input dimensions
                const int ix,           // input width
                const int iy,
                const float mean,
                const float var,
                const float epsilon,
                const float gamma,
                const float beta)  ;          // input height
#endif
