#include <cstddef>
#include <algorithm>
#include <float.h>
#include <math.h>
#include "conv_layer.h"
#include <ap_fixed.h>
#define NUM_ELEM 32
//#define NUM_ELEM2 16
typedef float myDataType;
//typedef ap_fixed<24,12,AP_RND,AP_SAT> myDataType;

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
                const float beta)            // input height
{

// Global memory interface
#pragma HLS INTERFACE m_axi port=mem depth=2147483648
// Bind all control ports to a single bundle
#pragma HLS INTERFACE s_axilite port=b bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=od bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=ox bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=oy bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=id bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=ix bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=iy bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=s bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=k bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=input_offset
#pragma HLS INTERFACE s_axilite port=output_offset
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL_BUS
 

int num_input = b*id*ix*iy;
myDataType denum = (myDataType) sqrt(var + epsilon);

//           // Batch
BATCH:  for (int b_=0; b_< b; b_++)
        {
            // Output Dimensions (Feature Maps)
OD:         for (int i_d = 0; i_d < id; i_d++)
            {
              // Output Y Dimension
OY:             for (int i_y = 0; i_y < iy; i_y++)
                {
                // Output X Dimension
OX:                 for (int i_x = 0; i_x < ix; i_x++)
                    {
                        mem[output_offset] = ((mem[input_offset/sizeof(float) +b_*ix*iy*id + i_d*ix*iy +i_y*ix +i_x] - mean)/denum) * gamma + beta;
                    }
                }
            }
        }
}