#include <algorithm>
#include <float.h>
#include "conv3d_layer.h"
#include "conv3d_functions.h"
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
    int num_bnorm  = oc*4; //mean + var + beta + ghama
    // input weight + bias + input +
    int w_offset = parameters_offset/sizeof(float);
    int b_offset = parameters_offset/sizeof(float) + num_weights;
    int n_offset = parameters_offset/sizeof(float) + num_weights + num_biases;
    int i_offset = input_offset/sizeof(float);
    int o_offset = output_offset/sizeof(float);

    //on-chip BRAM buffer/////////////
    //PING-PONG RAM is appied, again for the detail please refer to the
    //FPGA 2015 paper : "Optimizing FPGA-based Accelerator Design for Deep Convolutional Neural Networks"
    float biasBRAM[MAX_OUTPUT_CHANNELS/Tc][Tc];
    #pragma HLS array_partition variable=biasBRAM complete dim=2
    float normBRAM[MAX_OUTPUT_CHANNELS/Tc][Tn];
    #pragma HLS array_partition variable=normBRAM complete dim=2


    // should not usign ping pong for the weights since the ic and oc are small yet 16 but for bigger layers should change
    float weightBRAM_ping  [Tc][Tc][MAX_KERNEL_SIZE*MAX_KERNEL_SIZE*MAX_KERNEL_SIZE];
    #pragma HLS array_partition variable=weightBRAM_ping complete dim=2
    #pragma HLS array_partition variable=weightBRAM_ping complete dim=1

    float weightBRAM_pong   [Tc][Tc][MAX_KERNEL_SIZE*MAX_KERNEL_SIZE*MAX_KERNEL_SIZE];
    #pragma HLS array_partition variable=weightBRAM_pong complete dim=2
    #pragma HLS array_partition variable=weightBRAM_pong complete dim=1

    float inputBRAM_ping    [Tc][ind_size][iny_size][inx_size];
    #pragma HLS array_partition variable=inputBRAM_ping complete dim=1

    float inputBRAM_pong    [Tc][ind_size][iny_size][inx_size];
    #pragma HLS array_partition variable=inputBRAM_pong complete dim=1


    float outputBRAM[Tc][Tod][Toy][Tox];
    #pragma HLS array_partition variable=outputBRAM complete dim=1
    /////////////////////////////////
    const int od_limit = (od >= Tod) ? Tod : od;
    const int oy_limit = (oy >= Toy) ? Toy : oy;
    const int ox_limit = (ox >= Tox) ? Tox : ox;

    //load biases
    read_bias(biasBRAM, mem, b_offset,  oc);
    //load bnorm parmas
    read_bnorm(normBRAM, mem, n_offset, oc);

    // Batch
    batch_loop:
    for (int bb=0; bb< b; bb++)
    {
        ADD_PRAGMA(HLS loop_tripcount max = MAX_BATCH)
        // Output Y Dimension
        oy_loop:
        for (int o_y = 0; o_y < oy; o_y+=Toy)
        {
            ADD_PRAGMA(HLS loop_tripcount max = MAX_OUTPUT_HEIGHT/Toy)
            // Output X Dimension
            ox_loop:
            for (int o_x = 0; o_x < ox; o_x+=Tox)
            {
                ADD_PRAGMA(HLS loop_tripcount max = MAX_OUTPUT_WIDTH/Tox)
                // Output Dimensions (Feature Maps)
                od_loop:
                for (int o_d = 0; o_d < od; o_d+=Tod)
                {
                    ADD_PRAGMA(HLS loop_tripcount max = MAX_OUTPUT_DIMS/Tod)
                    // Output Channels
                    oc_loop:
                    for(int o_c = 0; o_c < oc; o_c+=Tc )
                    {
                        ADD_PRAGMA(HLS loop_tripcount max = MAX_OUTPUT_CHANNELS/Tc)
                        // Set bias
                        read_bias_to_output(outputBRAM,biasBRAM,o_c,bb,od_limit,oy_limit,ox_limit);

                        //PING-PONG RAM applied here
                        //the time spent on convolution computation is fully converd by the time spent on memory transaction
                        mem_read_weight(mem, w_offset, weightBRAM_ping, k,oc,ic, o_c, 0);
                        mem_read_input(mem,i_offset,inputBRAM_ping,ic,id,ix,iy,k,s,bb,o_y,o_x,o_d,o_c,0,oy_limit,ox_limit,od_limit);

                        for(int i_c = Tc; i_c < ic; i_c+=Tc )
                        {
                            ADD_PRAGMA(HLS loop_tripcount max = MAX_INPUT_CHANNELS/Tc )
                            if ((i_c/Tc)%2)
                            {
                                conv_compute(outputBRAM,inputBRAM_ping,weightBRAM_ping,k,s,od_limit,oy_limit,ox);
                                mem_read_weight(mem, w_offset, weightBRAM_pong, k,oc,ic, o_c, i_c);
                                mem_read_input(mem,i_offset,inputBRAM_pong,ic,id,ix,iy,k,s,bb,o_y,o_x,o_d,o_c,0,oy_limit,ox_limit,od_limit);
                            }
                            else
                            {
                                conv_compute(outputBRAM,inputBRAM_pong,weightBRAM_pong,k,s,od_limit,oy_limit,ox);
                                mem_read_weight(mem, w_offset, weightBRAM_ping, k,oc,ic, o_c, i_c);
                                mem_read_input(mem,i_offset,inputBRAM_ping,ic,id,ix,iy,k,s,bb,o_y,o_x,o_d,o_c,0,oy_limit,ox_limit,od_limit);
                            }
                        }
                        //for the last one to choose between ping and pong
                        if ((ic/Tc) % 2)
                            conv_compute(outputBRAM,inputBRAM_ping,weightBRAM_ping,k,s,od_limit,oy_limit,ox);
                        else
                            conv_compute(outputBRAM,inputBRAM_pong,weightBRAM_pong,k,s,od_limit,oy_limit,ox);
                        // Write output
                        mem_write(mem,o_offset,outputBRAM,normBRAM,oc,od,oy,ox,bb,o_c,o_d,o_y,o_x,od_limit,oy_limit,ox_limit,bnorm,relu);
                    }
                }
            }
        }
    }
}