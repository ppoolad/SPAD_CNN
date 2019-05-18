#include <iostream>
#include <algorithm>
#include <float.h>
#include "conv3d_layer.h"
#include "conv3d_functions.h"
#include "math.h"
#define EPSILON 0.00001

using namespace std;

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
                  const int relu,         // relu enable
                  const int bnorm        // batch norm enable
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
    //int num_bnorm  = oc*4; //mean + var + beta + ghama
    // input weight + bias + input +
    int w_offset = parameters_offset/sizeof(float);
    int b_offset = parameters_offset/sizeof(float) + num_weights;
    int n_offset = parameters_offset/sizeof(float) + num_weights + num_biases;
    int i_offset = input_offset/sizeof(float);
    int o_offset = output_offset/sizeof(float);
    int strd = abs(s);
    int sconv = strd;
    if(s<0) sconv=1;
    //on-chip BRAM buffer/////////////
    //PING-PONG RAM is appied, again for the detail please refer to the
    //FPGA 2015 paper : "Optimizing FPGA-based Accelerator Design for Deep Convolutional Neural Networks"
    // float biasBRAM[MAX_OUTPUT_CHANNELS/TCO][TCO];
    // #pragma HLS array_partition variable=biasBRAM complete dim=2
    float biasBRAM[MAX_OUTPUT_CHANNELS] = {0.0};
    #pragma HLS array_partition variable=biasBRAM complete

    // float normBRAM[MAX_OUTPUT_CHANNELS/TCO][TN];
    // #pragma HLS array_partition variable=normBRAM complete dim=2
    float normBRAM[MAX_OUTPUT_CHANNELS][4]={0.0f};
    #pragma HLS array_partition variable=normBRAM complete dim=0
    // should not usign ping pong for the weights since the ic and oc are small yet 16 but for bigger layers should change
    float inputBRAM_ping   [TCI][IND_SIZE][INY_SIZE][INX_SIZE] = {0.0};
    float weightBRAM_ping  [TCO][TCI][MAX_KERNEL_SIZE*MAX_KERNEL_SIZE*MAX_KERNEL_SIZE]= {0.0};
    #pragma HLS array_partition variable=inputBRAM_ping complete dim=1
    #pragma HLS array_partition variable=weightBRAM_ping complete dim=2
    #pragma HLS array_partition variable=weightBRAM_ping complete dim=1

    float inputBRAM_pong    [TCI][IND_SIZE][INY_SIZE][INX_SIZE]= {0.0};
    float weightBRAM_pong   [TCO][TCI][MAX_KERNEL_SIZE*MAX_KERNEL_SIZE*MAX_KERNEL_SIZE]= {0.0};
    #pragma HLS array_partition variable=inputBRAM_pong complete dim=1
    #pragma HLS array_partition variable=weightBRAM_pong complete dim=2
    #pragma HLS array_partition variable=weightBRAM_pong complete dim=1


    float outputBRAM[TCO][TOD][TOY][TOX]= {0.0};
    #pragma HLS array_partition variable=outputBRAM complete dim=1
    /////////////////////////////////
    const int od_limit = (od >= TOD) ? TOD : od;
    const int oy_limit = (oy >= TOY) ? TOY : oy;
    const int ox_limit = (ox >= TOX) ? TOX : ox;

    //load biases
    read_bias_full(biasBRAM, mem, b_offset,  num_biases);

    //load bnorm parmas
    read_bnorm_complete(normBRAM, mem, n_offset, oc);

    // Batch
    batch_loop:
    for (int bb=0; bb< b; bb++)
    {
        ADD_PRAGMA(HLS loop_tripcount max = MAX_BATCH)
        // Output Y Dimension
        oy_loop:
        for (int o_y = 0; o_y < oy; o_y+=TOY)
        {
            ADD_PRAGMA(HLS loop_tripcount max = MAX_OUTPUT_HEIGHT/TOY)
            // Output X Dimension
            ox_loop:
            for (int o_x = 0; o_x < ox; o_x+=TOX)
            {
                ADD_PRAGMA(HLS loop_tripcount max = MAX_OUTPUT_WIDTH/TOX)
                // Output Dimensions (Feature Maps)
                od_loop:
                for (int o_d = 0; o_d < od; o_d+=TOD)
                {
                    ADD_PRAGMA(HLS loop_tripcount max = MAX_OUTPUT_DIMS/TOD)
                    //std::cout << "bmorm[0][0] = " << normBRAM[0][0] << "\n";
                    //std::cout << "bias[0][0] = "  << biasBRAM[0][0] << "\n";
                    // Output Channels
                    oc_loop:
                    for(int o_c = 0; o_c < oc; o_c+=TCO )
                    {
                        ADD_PRAGMA(HLS loop_tripcount max = MAX_OUTPUT_CHANNELS/TCO)
                        // Set bias
                        read_fullbias_to_output(outputBRAM,biasBRAM,o_c,bb,od_limit,oy_limit,ox_limit);

                        //PING-PONG RAM applied here
                        //the time spent on convolution computation is fully converd by the time spent on memory transaction
                        if(s>0){
                            mem_read_weight(mem, w_offset, weightBRAM_ping, k, oc, ic, o_c, 0);
                            mem_read_input(mem, i_offset, inputBRAM_ping, ic, id, ix, iy, k, strd, bb, o_y, o_x, o_d, o_c, 0, oy_limit, ox_limit, od_limit);
                        }else{
                            mem_read_weight_transpose(mem, w_offset, weightBRAM_ping, k, oc, ic, o_c, 0);
                            mem_read_input_transpose(mem, i_offset, inputBRAM_ping, ic, id, ix, iy, k, strd, bb, o_y, o_x, o_d, o_c, 0, oy_limit, ox_limit, od_limit);
                        }
                        //std::cout << "read init"<<  weightBRAM_ping[0][0][0] << " and " << inputBRAM_ping[0][0][0][0] << "\n";
                        //std::cout << "ic = " << ic << "Tc = " << Tc << "residue" << (ic/Tc)%2 << "\n";
                        for(int i_c = TCI; i_c < ic; i_c+=TCI )
                        {
                            //std::cout << "i_c = " << i_c << "\n";
                            // unroll II
                            ADD_PRAGMA(HLS loop_tripcount max = MAX_INPUT_CHANNELS/TCI )
                            if ((i_c/TCI)%2)
                            {
                               // std :: cout << "read ping"<<  weightBRAM_ping[0][0][0] << " and " << inputBRAM_ping[0][0][0][0] << "\n";
                                conv_compute(outputBRAM,inputBRAM_ping,weightBRAM_ping,k,sconv,od_limit,oy_limit,ox_limit, o_c,i_c,o_x, o_y, o_d);
                                if(s>0){
                                    mem_read_weight(mem, w_offset, weightBRAM_pong, k,oc,ic, o_c, i_c);
                                    mem_read_input(mem,i_offset,inputBRAM_pong,ic,id,ix,iy,k,strd,bb,o_y,o_x,o_d,o_c,i_c,oy_limit,ox_limit,od_limit);
                                }else{
                                    mem_read_weight_transpose(mem, w_offset, weightBRAM_pong, k,oc,ic, o_c, i_c);
                                    mem_read_input_transpose(mem,i_offset,inputBRAM_pong,ic,id,ix,iy,k,strd,bb,o_y,o_x,o_d,o_c,i_c,oy_limit,ox_limit,od_limit); 
                                }
                            }
                            else
                            {
                                //std :: cout << "read pong "<<  weightBRAM_ping[0][0][0] << " and " << inputBRAM_ping[0][0][0][0] << "\n";
                                conv_compute(outputBRAM,inputBRAM_pong,weightBRAM_pong,k,sconv,od_limit,oy_limit,ox_limit,o_c,i_c,o_x, o_y, o_d);
                                if(s>0){
                                    mem_read_weight(mem, w_offset, weightBRAM_ping, k,oc,ic, o_c, i_c);
                                    mem_read_input(mem,i_offset,inputBRAM_ping,ic,id,ix,iy,k,strd,bb,o_y,o_x,o_d,o_c,i_c,oy_limit,ox_limit,od_limit);
                                }else{
                                    mem_read_weight_transpose(mem, w_offset, weightBRAM_ping, k,oc,ic, o_c, i_c);
                                    mem_read_input_transpose(mem,i_offset,inputBRAM_ping,ic,id,ix,iy,k,strd,bb,o_y,o_x,o_d,o_c,i_c,oy_limit,ox_limit,od_limit);
                                }
                            }
                        }
                        //for the last one to choose between ping and pong
                        if (((ic/TCI) % 2) || ic<TCI)
                            conv_compute(outputBRAM,inputBRAM_ping,weightBRAM_ping,k,sconv,od_limit,oy_limit,ox_limit,o_c,ic-1,o_x, o_y, o_d);
                        else
                            conv_compute(outputBRAM,inputBRAM_pong,weightBRAM_pong,k,sconv,od_limit,oy_limit,ox_limit,o_c,ic-1,o_x, o_y, o_d);
                        // Write output
                        mem_write_fullnorm(mem, o_offset, outputBRAM, normBRAM, oc, od, oy, ox, bb, o_c, o_d, o_y, o_x, od_limit, oy_limit, ox_limit, bnorm, relu);
                    }
                }
            }
        }
    }
}

//add padding
