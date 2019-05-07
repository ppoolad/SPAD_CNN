#ifndef _CONV_LAYER_H
#define _CONV_LAYER_H

// Limits
#define MAX_BATCH           1
#define MAX_KERNEL_SIZE     9
#define MAX_INPUT_DIMS      1024
#define MAX_OUTPUT_DIMS     1024
#define MAX_OUTPUT_CHANNELS 40
#define MAX_INPUT_CHANNELS  40
#define MAX_INPUT_WIDTH     64
#define MAX_INPUT_HEIGHT    64
#define MAX_OUTPUT_WIDTH    64
#define MAX_OUTPUT_HEIGHT   64
#define MAX_CONV_INPUT      MAX_INPUT_CHANNELS*MAX_INPUT_DIMS*MAX_INPUT_WIDTH*MAX_INPUT_HEIGHT
#define MAX_CONV_OUTPUT     MAX_OUTPUT_CHANNELS*MAX_OUTPUT_DIMS*MAX_OUTPUT_WIDTH*MAX_OUTPUT_HEIGHT
#define MAX_WEIGHT_SIZE     MAX_INPUT_CHANNELS*MAX_OUTPUT_CHANNELS*MAX_KERNEL_SIZE*MAX_KERNEL_SIZE*MAX_KERNEL_SIZE

#define NUM_BNORM_PARAMS    4

#define Tc      2   // <!!!>if changing this change Tn //tile for channels // keep this 4 other wise change conv_compute
#define Tod		3   //tile for input  dimension
#define Toy		3
#define Tox		3

#define Tn      8//Tc*4 // for batch normalization

#define ind_size	Tod+MAX_KERNEL_SIZE-1
#define iny_size	Toy+MAX_KERNEL_SIZE-1
#define inx_size	Tox+MAX_KERNEL_SIZE-1

#define ADD_PRAGMA_INNER(x) _Pragma (#x)
#define ADD_PRAGMA(x) ADD_PRAGMA_INNER(x)

#define EPSILON 0.00001
void conv3d_layer(float * mem,            // global memory pointer
                  int input_offset,       // offset of inputs
                  int parameters_offset,  // offset of parameters
                  int output_offset,      // offset of outputs
                  const int b,            // batch size
                  const int od,           // output dimensions
                  const int ox,           // output width
                  const int oy,           // output height
                  const int oc,           // output channel
                  const int ic,           // input channel
                  const int id,           // input dimensions
                  const int ix,           // input width
                  const int iy,           // input height
                  const int s,            // stride
                  const int k,            // kernel size
                  const int pad,          // padding
                  const int relu,         //relu enable
                  const int bnorm);       // batch norm enable
void hw_conv3d_layer(int target,             // control register target
                     float * mem,            // global memory pointer
                     int input_offset,       // offset of inputs
                     int parameters_offset,  // offset of parameters
                     int output_offset,      // offset of outputs
                     const int b,            // batch size
                     const int od,           // output dimensions
                     const int ox,           // output width
                     const int oy,           // output height
                     const int oc,           // output channel
                     const int ic,           // input channel
                     const int id,           // input dimensions
                     const int ix,           // input width
                     const int iy,           // input height
                     const int s,            // stride
                     const int k,            // kernel size
                     const int pad,          // padding
                     const int relu,         //relu enable
                     const int bnorm);       // batch norm enable
#endif