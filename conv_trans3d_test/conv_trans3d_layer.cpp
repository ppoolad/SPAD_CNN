#include <algorithm>
#include <float.h>
#include "conv_trans3d_layer.h"
#include "math.h"
#include <iostream>
#define EPSILON 0.00001

using namespace std;

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
                  const int bnorm         // batch norm enable
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
#pragma HLS INTERFACE s_axilite port=relu bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=bnorm bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=input_offset
#pragma HLS INTERFACE s_axilite port=parameters_offset
#pragma HLS INTERFACE s_axilite port=output_offset
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL_BUS

    int num_weights = ic*oc*k*k*k;
    int num_biases = oc;
    int num_input = b*id*ix*iy;
    int num_output = b*oc*od*ox*oy;
    int num_bnorm  = oc; //mean + var + beta + ghama

    int r = (k - 1) / 2; //radius;
    int sim_x = ix*s - s +1;
    int sim_y = iy*s - s +1;
    int sim_d = id*s - s +1;
    //int sim_pad = k - pad -1; // for now pad is zero di
    //int x_border = sim_pad == 0? r : 0;
    //int y_border = sim_pad == 0? r : 0;
    //int d_border = sim_pad == 0? r : 0;

    // input weight + bias + input +
    // Batch
    for (int b_=0; b_< b; b_++)
    {
        // Output Channels
        for(int o_c = 0; o_c < oc; o_c++ )
        {
            float mean  = mem[parameters_offset/sizeof(float) + num_weights + oc +                o_c];
            float var   = mem[parameters_offset/sizeof(float) + num_weights + oc +  num_bnorm*1 + o_c];
            float gamma = mem[parameters_offset/sizeof(float) + num_weights + oc +  num_bnorm*2 + o_c];
            float beta  = mem[parameters_offset/sizeof(float) + num_weights + oc +  num_bnorm*3 + o_c];
            float ratio =  gamma/sqrt(var + EPSILON);
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
                            for (int iid = -r; iid <= r; iid++)
                            {
                                // Input Y Dimension
                                for (int iiy = -r; iiy <= r; iiy++)
                                {
                                    // Input X Dimension
                                    for (int iix = -r; iix <= r; iix++)
                                    {
                                        //float ifmap = 0.0;
                                        if((o_x + iix) >= 0 && (o_y + iiy) >= 0 && (o_d+iid) >= 0 && (o_x + iix) < sim_x && (o_y+iiy) < sim_y && (o_d + iid) < sim_d){ // boundry check
                                            //ifmap = mem[input_offset/sizeof(float) +b_*id*ix*iy + i_d*ix*iy + i_y*ix + i_x];
                                            if ( (o_x + iix) % s != 0 || (o_y + iiy) % s != 0 || (o_d + iid) % s != 0) continue;
                                            #ifdef PRINT
                                            cout << "output[" << o_d << "][" << o_y << "][" << o_x <<  "] += input" << "[" << o_d +iid << "][" << o_y+iiy << "][" << o_x+iix
                                            << "] * filter[" << "[" << r+iid << "][" << r+iiy << "][" << r+iix << "]" << endl;
                                            #endif
                                            output_element += mem[input_offset/sizeof(float) +b_*id*ix*iy + (o_d +iid)/s*ix*iy + (o_y+iiy)/s*ix + (o_x+iix)/s] * //+ num_weights+num_biases+ b_*id*ix*iy + i_d*ix*iy + i_y*ix + i_x]*
                                                              mem[parameters_offset/sizeof(float) + o_c*ic*k*k*k + i_c*k*k*k + (r+iid)*k*k + (r+iiy)*k + r+iix];
                                        }
                                    }
                                }
                            }
                        }
                        // Write output
                        if(bnorm){
                            //TBC
                            output_element = (output_element-mean)*ratio + beta;
                        }
                        if(relu) output_element = std::max(0.0f, output_element);
                        mem[output_offset/sizeof(float) + b_*od*ox*oy + o_d*ox*oy + o_y*ox + o_x] = output_element;
                    }
                }
            }
        }
    }
}
