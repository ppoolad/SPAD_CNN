#include <algorithm>
#include <float.h>
#include "conv3d_layer.h"
#include "math.h"
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

    int w_offset = parameters_offset/sizeof(float);
    int b_offset = parameters_offset/sizeof(float) + num_weights;
    int n_offset = parameters_offset/sizeof(float) + num_weights + num_biases;
    int i_offset = input_offset/sizeof(float);
    int o_offset = output_offset/sizeof(float);

    //on-chip BRAM buffer/////////////
    //PING-PONG RAM is appied, again for the detail please refer to the
    //FPGA 2015 paper : "Optimizing FPGA-based Accelerator Design for Deep Convolutional Neural Networks"
    float biasBRAM[MAX_OUTPUT_CHANNELS/To][To];
    #pragma HLS array_partition variable=biasBRAM complete dim=2
    float inputBRAM_ping[Ti][ind_size][iny_size][inx_size];
    float weightBRAM_ping[To][Ti][MAX_KERNEL_SIZE*MAX_KERNEL_SIZE*MAX_KERNEL_SIZE];
    #pragma HLS array_partition variable=inputBRAM_ping complete dim=1
    #pragma HLS array_partition variable=weightBRAM_ping complete dim=2
    #pragma HLS array_partition variable=weightBRAM_ping complete dim=1

    float inputBRAM_pong[Ti][ind_size][iny_size][inx_size];
    float weightBRAM_pong[To][Ti][MAX_KERNEL_SIZE*MAX_KERNEL_SIZE*MAX_KERNEL_SIZE];
    #pragma HLS array_partition variable=inputBRAM_pong complete dim=1
    #pragma HLS array_partition variable=weightBRAM_pong complete dim=2
    #pragma HLS array_partition variable=weightBRAM_pong complete dim=1

    float outputBRAM[To][Tod][Toy][Tox];
    #pragma HLS array_partition variable=outputBRAM complete dim=1
    /////////////////////////////////
    const int od_limit = (od >= Tod) ? Tod : od;
    const int oy_limit = (oy >= Toy) ? Toy : oy;
    const int ox_limit = (ox >= Tox) ? Tox : ox;


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

            // Weighted Sum:
            for(int i_c = 0; i_c < ic; i_c++)
            {
            
            // Input Dimensions (Feature Maps)
              for (int i_d = o_d*s-pad, iid = 0; i_d < o_d*s-pad+k; i_d++, iid++)
              {
                // Input Y Dimension
                for (int i_y = o_y*s-pad, iiy = 0; i_y < o_y*s-pad+k; i_y++, iiy++)
                {
                  // Input X Dimension
                  for (int i_x = o_x*s-pad, iix = 0; i_x < o_x*s-pad+k; i_x++, iix++)
                  {
                    //float ifmap = 0.0;
                    if((i_x >= 0) && (i_y >= 0) && (i_d >= 0) && (i_x < ix) && (i_y < iy) && (i_d < id)){
                      //ifmap = mem[input_offset/sizeof(float) +b_*id*ix*iy + i_d*ix*iy + i_y*ix + i_x];
                      output_element += mem[input_offset/sizeof(float) +b_*ic*id*ix*iy+ i_c*id*ix*iy + i_d*ix*iy + i_y*ix + i_x] * //+ num_weights+num_biases+ b_*id*ix*iy + i_d*ix*iy + i_y*ix + i_x]*
                                      mem[parameters_offset/sizeof(float) + o_c*ic*k*k*k + i_c*k*k*k + iid*k*k + iiy*k + iix];
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
