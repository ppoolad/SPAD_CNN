#include <algorithm>
#include <float.h>
#include "conv_trans3d_layer.h"
#include "math.h"
#include <iostream>
#define EPSILON 0.00001

void conv_trans3d_layer(float * mem,            // global memory pointer
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
                const int bnorm       // batch norm enable
                )

{

// Global memory interface
#pragma HLS INTERFACE m_axi port=mem depth=2147483648
// Bind all control ports to a single bundle
#pragma HLS INTERFACE s_axilite port=b bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=od bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=ox bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=oy bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=oc bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=id bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=ix bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=iy bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=ic bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=s bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=k bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=pad bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=relu bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=bnorm bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=input_offset
#pragma HLS INTERFACE s_axilite port=parameters_offset
#pragma HLS INTERFACE s_axilite port=output_offset
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL_BUS
 
  int num_weights = ic*oc*k*k*k;
  int num_biases = oc;
  int num_input = b*ic*id*ix*iy;
  int num_output = b*oc*od*ox*oy;
  int num_bnorm  = oc; //mean + var + beta + ghama
  // input weight + bias + input + 
  // Batch
  for (int b_=0; b_< b; b_++)
  {
    // Output Channels
    for(int o_c = 0; o_c < oc; o_c++ )
    {
      float mean  = mem[parameters_offset/sizeof(float) + num_weights + oc +                o_c];
      float var   = mem[parameters_offset/sizeof(float) + num_weights + oc +  num_bnorm*1 + o_c];
      float gamma = mem[parameters_offset/sizeof(float)+ num_weights + oc +  num_bnorm*2 + o_c];
      float beta  = mem[parameters_offset/sizeof(float)  + num_weights + oc +  num_bnorm*3 + o_c];
      float num   =  gamma/sqrt(var + EPSILON);
      // Output Dimensions (Feature Maps)
      for (int o_d = 0; o_d < od; o_d++)
      {
        // Output Y Dimension
        for (int o_y = 0; o_y < oy; o_y++)
        {
          // Output X Dimension
          for (int o_x = 0; o_x < ox; o_x++)
          {
            // Set bias 
            float output_element = mem[parameters_offset/sizeof(float) + num_weights + o_c];
            //std::cout<<"O[ " << o_d << ',' << o_y << ',' << o_x << ']' << std::endl;
            // Weighted Sum:
            for(int i_c = 0; i_c < ic; i_c++)
            {
            
            // Input Dimensions (Feature Maps)
              for (int i_d = o_d-pad, iid = 0; i_d < o_d-pad+k; i_d++, iid++)
              {
                // Input Y Dimension
                for (int i_y = o_y-pad, iiy = 0; i_y < o_y-pad+k; i_y++, iiy++)
                {
                  // Input X Dimension
                  for (int i_x = o_x-pad, iix = 0; i_x < o_x-pad+k; i_x++, iix++)
                  {
                    //float ifmap = 0.0;
                    if((i_x >= 0) && (i_y >= 0) && (i_d >= 0) && (i_x < s*ix) && (i_y < s*iy) && (i_d < s*id)){//check padding
                        if(((i_x%(s)) == 0) && ((i_y%(s)) == 0) && ((i_d%(s)) == 0)){
                        // calculate
                      //ifmap = mem[input_offset/sizeof(float) +b_*id*ix*iy + i_d*ix*iy + i_y*ix + i_x];
                            int ni_x = i_x/s; int ni_y = i_y/s; int ni_d = i_d/s; 
                            //std::cout << "in[" << ni_d << ',' << ni_y<<',' << ni_x << "] * w[" << k-1-iid << ',' << k-1-iiy << ',' << k-1-iix << ']' << std::endl; 
                            output_element += mem[input_offset/sizeof(float) +b_*ic*id*ix*iy+ i_c*id*ix*iy + ni_d*ix*iy + ni_y*ix + ni_x] * //+ num_weights+num_biases+ b_*id*ix*iy + i_d*ix*iy + i_y*ix + i_x]*
                                      mem[parameters_offset/sizeof(float) + i_c*oc*k*k*k + o_c*k*k*k + (k-1-iid)*k*k + (k-1-iiy)*k + k-1-iix];
                        }
                      }
                  }
                }
              }
            }
            // Write output
            if(bnorm){
              //TBC
              output_element = (output_element-mean)*num + beta;
            }
            if(relu) output_element = std::max(0.0f, output_element);
            mem[output_offset/sizeof(float) + b_*oc*od*ox*oy + o_c*od*ox*oy+ o_d*ox*oy + o_y*ox + o_x] = output_element;
          }
        }
      }
    }
  }
}
