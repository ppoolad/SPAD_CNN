//
// Created by zbete on 5/2/2019.
//

#ifndef SPAD_CNN_CONV3D_FUNCTIONS_H
#define SPAD_CNN_CONV3D_FUNCTIONS_H

void read_bias(
        float biasBRAM[MAX_OUTPUT_CHANNELS/TCO][TCO],
        float * mem,            // global memory pointer
        int bias_offset,
        const int oc);

void read_bnorm(
        float normBRAM[MAX_OUTPUT_CHANNELS/TCO][TN],
        float * mem,            // global memory pointer
        int norm_offset,
        const int oc);

void read_bias_to_output(
        float outputBRAM[TCO][TOD][TOY][TOX],
        float biasBRAM[MAX_OUTPUT_CHANNELS/TCO][TCO],
        int o_c,	//current output dimension index
        int bb,		//current batch index
        const int od_limit,
        const int oy_limit,
        const int ox_limit);


void mem_read_weight(
        float * mem,            // global memory pointer
        int   weight_offset,    // offset of weights
        float weightBRAM[TCO][TCI][MAX_KERNEL_SIZE*MAX_KERNEL_SIZE*MAX_KERNEL_SIZE],
        const int k,
        const int oc,
        const int ic,
        int o_c,    //current output channel
        int i_c 	//current input  channel index
);

void mem_read_input(
        float * mem,            // global memory pointer
        int   input_offset,     // offset of inputs
        float inputBRAM [TCI][IND_SIZE][INY_SIZE][INX_SIZE],
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
        const int od_limit);

void conv_compute(
        float outputBRAM[TCO][TOD][TOY][TOX],
        float inputBRAM [TCI][IND_SIZE][INY_SIZE][INX_SIZE],
        float weightBRAM[TCO][TCI][MAX_KERNEL_SIZE*MAX_KERNEL_SIZE*MAX_KERNEL_SIZE],
        const int k,
        const int s,
        const int od_limit,
        const int oy_limit,
        const int ox_limit,
        int o_c, //for test
        int i_c,
        int o_x,
        int o_y,
        int o_d);

void mem_write(
        float * mem,            // global memory pointer
        int   output_offset,       // offset of inputs
        float outputBRAM[TCO][TOD][TOY][TOX],
        float normBRAM  [MAX_OUTPUT_CHANNELS/TCO][TN],
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
        int relu);

#endif //SPAD_CNN_CONV3D_FUNCTIONS_H
