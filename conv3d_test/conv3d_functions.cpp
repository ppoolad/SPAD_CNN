#include <algorithm>
#include <float.h>
#include "conv3d_layer.h"
#include "math.h"

void read_bias(
        float biasBRAM[MAX_OUTPUT_CHANNELS/Tc][Tc],
        float * mem,            // global memory pointer
        int bias_offset,
        const int oc)
{
    for (int i = 0; i < oc; i++) {
        ADD_PRAGMA(HLS loop_tripcount max = MAX_OUTPUT_CHANNELS)
        #pragma HLS pipeline II=1
        biasBRAM[i/Tc][i%Tc] = mem[bias_offset+ i];
    }
}

void read_bnorm(
        float normBRAM[MAX_OUTPUT_CHANNELS/Tc][Tn],
        float * mem,            // global memory pointer
        int norm_offset,
        const int oc)
{
    for (int i = 0; i < oc*NUM_BNORM_PARAMS; i++) {
        ADD_PRAGMA(HLS loop_tripcount max = MAX_OUTPUT_CHANNELS*NUM_BNORM_PARAMS)
        #pragma HLS pipeline II=1
        int ii = i%Tn;
        normBRAM[i/Tn][ii] = mem[norm_offset+ (ii%NUM_BNORM_PARAMS)*oc + ii/NUM_BNORM_PARAMS];
    }
}

void read_bias_to_output(
        float outputBRAM[Tc][Tod][Toy][Tox],
        float biasBRAM[MAX_OUTPUT_CHANNELS/Tc][Tc],
        int o_c,	//current output dimension index
        int bb,		//current batch index
        const int od_limit,
        const int oy_limit,
        const int ox_limit)
{
    for (int d = 0; d < od_limit; d++) {
        ADD_PRAGMA(HLS loop_tripcount max = Tod)
        for (int y = 0; y < oy_limit; y++) {
            ADD_PRAGMA(HLS loop_tripcount max = Toy)
            for (int x = 0; x < ox_limit; x++) {
                ADD_PRAGMA(HLS loop_tripcount max = Tox)
                #pragma HLS pipeline II = 1
                for (int o_cc = 0; o_cc < Tc; o_cc++) {
                    #pragma HLS unroll
                    outputBRAM[o_cc][d][y][x] = biasBRAM[o_c/Tc][o_cc];
                }
            }
        }
    }
}


void mem_read_weight(
        float * mem,            // global memory pointer
        int   weight_offset,    // offset of weights
        float weightBRAM[Tc][Tc][MAX_KERNEL_SIZE*MAX_KERNEL_SIZE*MAX_KERNEL_SIZE],
        const int k,
        const int oc,
        const int ic,
        int o_c,    //current output channel
        int i_c	//current input  channel index
        )
{
    //read weight
    for (int i = 0; i < k*k*k; i++)
    {
        ADD_PRAGMA(HLS loop_tripcount max = MAX_KERNEL_SIZE*MAX_KERNEL_SIZE*MAX_KERNEL_SIZE)
        for (int o_cc = 0; o_cc < Tc; o_cc++)
        {
            for (int i_cc=0; i_cc < Tc; i_cc++)
            {
                #pragma HLS pipeline II=1
                weightBRAM[o_cc][i_cc][i] = mem[weight_offset + (i_c+i_cc)*oc*k*k*k + (o_c+o_cc)*k*k + i];
            }
        }
    }
}
///////////////function to read tile of input//////////////////////////
void mem_read_input(
        float * mem,            // global memory pointer
        int   input_offset,     // offset of inputs
        float inputBRAM [Tc][ind_size][iny_size][inx_size],
        const int ic,
        const int id,
        const int ix,
        const int iy,
        const int k,
        const int s,
        int bb,		//current batch index
        int o_y,	//current output y index
        int o_x,	//current output x index
        int o_d,	//current output dimension index
        int o_c,    //current output channel
        int i_c,	//current input  channel index
        const int oy_limit,
        const int ox_limit,
        const int od_limit)
{
    //read input
    for (int l = 0; l < k; l++) {
        ADD_PRAGMA(HLS loop_tripcount max = MAX_KERNEL_SIZE)
        for (int i = 0; i < k; i++) {
            ADD_PRAGMA(HLS loop_tripcount max = MAX_KERNEL_SIZE)
            for (int j = 0; j < k; j++) {
                ADD_PRAGMA(HLS loop_tripcount max = MAX_KERNEL_SIZE)
                for (int d = 0; d < od_limit; d++) {
                    ADD_PRAGMA(HLS loop_tripcount max = Tod)
                    for (int y = 0; y < oy_limit; y++) {
                        ADD_PRAGMA(HLS loop_tripcount max = Toy)
                        for (int x = 0; x < ox_limit; x++) {
                            ADD_PRAGMA(HLS loop_tripcount max = Tox)
                            for (int i_cc=0; i_cc < Tc; i_cc++) {
                                #pragma HLS pipeline II=1
                                inputBRAM[i_cc][s*d+l][s*y+i][s*x+j] = mem[input_offset+ bb*ic*id*iy*ix + (i_c+i_cc)*id*iy*ix +(s*(o_d+d)+l)*iy*ix+ (s*(o_y+y)+i)*ix + (s*(o_x+x)+j)];
                            }
                        }
                    }
                }
            }
        }
    }
}

////////////////////////to do convolution//////////////////////
void conv_compute(
        float outputBRAM[Tc][Tod][Toy][Tox],
        float inputBRAM[Tc][ind_size][iny_size][inx_size],
        float weightBRAM[Tc][Tc][MAX_KERNEL_SIZE*MAX_KERNEL_SIZE*MAX_KERNEL_SIZE],
        const int k,
        const int s,
        const int od_limit,
        const int oy_limit,
        const int ox_limit)
{
    for (int l = 0; l < k; l++) {
        ADD_PRAGMA(HLS loop_tripcount max = MAX_KERNEL_SIZE)
        for (int i = 0; i < k; i++) {
            ADD_PRAGMA(HLS loop_tripcount max = MAX_KERNEL_SIZE)
            for (int j = 0; j < k; j++) {
                ADD_PRAGMA(HLS loop_tripcount max = MAX_KERNEL_SIZE)
                for (int d = 0; d < od_limit; d++) {
                    ADD_PRAGMA(HLS loop_tripcount max = Tod)
                    for (int y = 0; y < oy_limit; y++) {
                        ADD_PRAGMA(HLS loop_tripcount max = Toy)
                        for (int x = 0; x < ox_limit; x++) {
                            ADD_PRAGMA(HLS loop_tripcount max = Tox)
                            #pragma HLS pipeline II=1
                            for (int o_cc = 0; o_cc < Tc; o_cc++) {
                                #pragma HLS unroll
                                #pragma HLS dependence variable=inputBRAM inter false
                                #pragma HLS dependence variable=weightBRAM inter false
                                #pragma HLS dependence variable=outputBRAM inter false
                                float mul1_1;
                                float mul1_2;
                                float mul1_3;
                                float mul1_4;
                                float mul2_1;
                                float mul2_2;
                                float mul3_1;
                                #pragma HLS RESOURCE variable=mul1_1 core=FMul_meddsp
                                #pragma HLS RESOURCE variable=mul1_2 core=FMul_meddsp
                                #pragma HLS RESOURCE variable=mul1_3 core=FMul_meddsp
                                #pragma HLS RESOURCE variable=mul1_4 core=FMul_meddsp
                                #pragma HLS RESOURCE variable=mul2_1 core=FAddSub_nodsp
                                #pragma HLS RESOURCE variable=mul2_2 core=FAddSub_nodsp
                                mul1_1 = inputBRAM[0][s*d+l][s*y+i][s*x+j] * weightBRAM[o_cc][0][l*k*k+i*k+j];
                                mul1_2 = inputBRAM[1][s*d+l][s*y+i][s*x+j] * weightBRAM[o_cc][1][l*k*k+i*k+j];
                                mul1_3 = inputBRAM[2][s*d+l][s*y+i][s*x+j] * weightBRAM[o_cc][2][l*k*k+i*k+j];
                                mul1_4 = inputBRAM[3][s*d+l][s*y+i][s*x+j] * weightBRAM[o_cc][3][l*k*k+i*k+j];

                                mul2_1 = mul1_1 + mul1_2;
                                mul2_2 = mul1_3 + mul1_4;
                                mul3_1 = mul2_1 + mul2_2;
                                outputBRAM[o_cc][d][y][x] += mul3_1;
                            }
                        }
                    }
                }
            }
        }
    }
}

/////// writng memory //////////////////////
void mem_write(
        float * mem,            // global memory pointer
        int   output_offset,       // offset of inputs
        float outputBRAM[Tc][Tod][Toy][Tox],
        float normBRAM[MAX_OUTPUT_CHANNELS/Tc][Tn],
        const int oc,
        const int od,
        const int oy,
        const int ox,
        int bb,
        int o_c,
        int o_d,
        int o_y,
        int o_x,
        const int od_limit,
        const int oy_limit,
        const int ox_limit,
        int bnorm,
        int relu)
{
    for (int o_cc=0; o_cc < Tc; o_cc++) {
        ADD_PRAGMA(HLS loop_tripcount max = Tc)
        for (int d = 0; d < od_limit; d++) {
            ADD_PRAGMA(HLS loop_tripcount max = Tod)
            for (int y = 0; y < oy_limit; y++) {
                ADD_PRAGMA(HLS loop_tripcount max = Toy)
                for (int x = 0; x < ox_limit; x++) {
                    ADD_PRAGMA(HLS loop_tripcount max = Tox)
                    #pragma HLS pipeline II=1
                    float output_element = outputBRAM[o_cc][d][y][x];
                    if(bnorm)
                    {
                        int i0 = o_c/Tc;
                        int i1 = o_cc*4;
                        //(output-mean)*gamma/sqrt(var + EPSILON)-beta;
                        output_element = (output_element - normBRAM[i0][i1])/sqrt(normBRAM[i0][i1+1] + EPSILON)*normBRAM[i0][i1+2]-normBRAM[i0][i1+3];
                    }
                    if (relu) output_element = std::max(0.0f,output_element);
                    mem[output_offset+ bb*oc*od*oy*ox + (o_c+o_cc)*od*oy*ox + (o_d+d)*ox*oy + (o_y+y)*ox + o_x+x] = output_element;
                }
            }
        }
    }
}
