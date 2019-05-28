#include <iostream>
#include <algorithm>
#include <float.h>
#include "conv3d_layer.h"
#include "math.h"

void read_bias(
        float biasBRAM[MAX_OUTPUT_CHANNELS/TCO][TCO],
        float * mem,            // global memory pointer
        int bias_offset,
        const int oc)
{
    for (int i = 0; i < oc; i++) {
        ADD_PRAGMA(HLS loop_tripcount max = MAX_OUTPUT_CHANNELS)
        #pragma HLS pipeline II=1
        biasBRAM[i/TCO][i%TCO] = mem[bias_offset+ i];
        //std::cout << "bias i = " << biasBRAM[i/Tc][i%Tc] << "\n";
    }

}


void read_bias_full(
        float biasBRAM[MAX_OUTPUT_CHANNELS],
        float * mem,            // global memory pointer
        int bias_offset,
        const int oc)
{
    for (int i = 0; i < oc; i++) {
        ADD_PRAGMA(HLS loop_tripcount max = MAX_OUTPUT_CHANNELS)
        #pragma HLS pipeline II=1
        biasBRAM[i] = mem[bias_offset+ i];
        //std::cout << "bias i = " << biasBRAM[i/Tc][i%Tc] << "\n";
    }

}

void read_bnorm(
        float normBRAM[MAX_OUTPUT_CHANNELS/TCO][TN],
        float * mem,            // global memory pointer
        int norm_offset,
        const int on)
{
    int oc = on/NUM_BNORM_PARAMS;
    for (int i = 0; i < on; i++) {
        ADD_PRAGMA(HLS loop_tripcount max = MAX_OUTPUT_CHANNELS*NUM_BNORM_PARAMS)
        #pragma HLS pipeline II=1
        int ii = i%TN;
        int iii = i/TN;
        //std::cout << i << " normBRAM[" << i/TN << "][" << ii << "]\n";
        normBRAM[i/TN][ii] = mem[norm_offset+ (ii%NUM_BNORM_PARAMS)*oc + ii/NUM_BNORM_PARAMS + iii*TN/NUM_BNORM_PARAMS];
        //std::cout << "Tn = " << TN << ">>"  << i << ","<< ii <<  " normBRAM[" << i/TN << "][" << ii << "] = mem[ " << ii%NUM_BNORM_PARAMS << "*oc + " << ii/NUM_BNORM_PARAMS << "] | [" << (ii%NUM_BNORM_PARAMS)*oc + ii/NUM_BNORM_PARAMS + iii*TN/NUM_BNORM_PARAMS <<  "] = " << normBRAM[i/TN][ii] << "\n";
    }
}

void read_bnorm_complete(
        float normBRAM[MAX_OUTPUT_CHANNELS][4],
        float * mem,            // global memory pointer
        int norm_offset,
        const int on)
{
    //int oc = on;///NUM_BNORM_PARAMS;
    for (int i = 0; i < on; i++) {
        ADD_PRAGMA(HLS loop_tripcount max = MAX_OUTPUT_CHANNELS)
        #pragma HLS pipeline II=1
        //int ii = i%TN;
        //int iii = i/TN;
        //std::cout << i << " normBRAM[" << i/TN << "][" << ii << "]\n";
        normBRAM[i][0] = mem[norm_offset + i];
    }
    for (int i = 0; i < on; i++) {
        ADD_PRAGMA(HLS loop_tripcount max = MAX_OUTPUT_CHANNELS)
        #pragma HLS pipeline II=1
        normBRAM[i][1] = mem[norm_offset + on + i];
    }
    for (int i = 0; i < on; i++) {
        ADD_PRAGMA(HLS loop_tripcount max = MAX_OUTPUT_CHANNELS)
        #pragma HLS pipeline II=1
        normBRAM[i][2] = mem[norm_offset + on*2 + i];
    }
    for (int i = 0; i < on; i++) {
        ADD_PRAGMA(HLS loop_tripcount max = MAX_OUTPUT_CHANNELS)
        #pragma HLS pipeline II=1
        normBRAM[i][3] = mem[norm_offset + on*3 + i];
        //normBRAM[i/TN][ii] = mem[norm_offset+ (ii%NUM_BNORM_PARAMS)*oc + ii/NUM_BNORM_PARAMS + iii*TN/NUM_BNORM_PARAMS];
        // std::cout << " normBRAM[" << i<< "][" << 0 << "] = mem[ " << i*4 + 0 << "] =" << mem[norm_offset + i*4 + 0] << "  \n";
        // std::cout << " normBRAM[" << i<< "][" << 1 << "] = mem[ " << i*4 + 1 << "] =" << mem[norm_offset + i*4 + 1]<< "   \n";
        // std::cout << " normBRAM[" << i<< "][" << 2 << "] = mem[ " << i*4 + 2 << "] =" << mem[norm_offset + i*4 + 2]<< "   \n";
        // std::cout << " normBRAM[" << i<< "][" << 3 << "] = mem[ " << i*4 + 3 << "] =" << mem[norm_offset + i*4 + 3]<< "   \n";
    }
}

void read_bias_to_output(
        float outputBRAM[TCO][TOD][TOY][TOX],
        float biasBRAM[MAX_OUTPUT_CHANNELS/TCO][TCO],
        int o_c,	//current output dimension index
        int bb,		//current batch index
        int od_limit,
        int oy_limit,
        int ox_limit)
{
    for (int d = 0; d < od_limit; d++) {
        ADD_PRAGMA(HLS loop_tripcount max = TOD)
        for (int y = 0; y < oy_limit; y++) {
            ADD_PRAGMA(HLS loop_tripcount max = TOY)
            for (int x = 0; x < ox_limit; x++) {
                ADD_PRAGMA(HLS loop_tripcount max = TOX)
                #pragma HLS pipeline II = 1
                for (int o_cc = 0; o_cc < TCO; o_cc++) {
                    #pragma HLS unroll
                    outputBRAM[o_cc][d][y][x] = biasBRAM[o_c/TCO][o_cc];
                    //std::cout << "output BRAM[" << o_c+o_cc << "] = " << outputBRAM[o_cc][d][y][x] << " comp " << biasBRAM[o_c/TCO][o_cc] << "\n";
                }
            }
        }
    }
}

void read_fullbias_to_output(
        float outputBRAM[TCO][TOD][TOY][TOX],
        float biasBRAM[MAX_OUTPUT_CHANNELS],
        int o_c,	//current output dimension index
        int bb,		//current batch index
        int od_limit,
        int oy_limit,
        int ox_limit)
{
    for (int d = 0; d < od_limit; d++) {
        ADD_PRAGMA(HLS loop_tripcount max = TOD)
        for (int y = 0; y < oy_limit; y++) {
            ADD_PRAGMA(HLS loop_tripcount max = TOY)
            for (int x = 0; x < ox_limit; x++) {
                ADD_PRAGMA(HLS loop_tripcount max = TOX)
                #pragma HLS pipeline II = 1
                for (int o_cc = 0; o_cc < TCO; o_cc++) {
                    #pragma HLS unroll
                    //if(o_c + o_cc <  )
                    outputBRAM[o_cc][d][y][x] = biasBRAM[o_c+o_cc];
                    //std::cout << "output BRAM[" << o_c+o_cc << "] = " << outputBRAM[o_cc][d][y][x] << " comp " << biasBRAM[o_c/TCO][o_cc] << "\n";
                }
            }
        }
    }
}

void mem_read_weight_transpose(
        float * mem,            // global memory pointer
        int   weight_offset,    // offset of weights
        float weightBRAM[TCO][TCI][MAX_KERNEL_SIZE*MAX_KERNEL_SIZE*MAX_KERNEL_SIZE],
        const int k,
        const int oc,
        const int ic,
        int o_c,    //current output channel
        int i_c	//current input  channel index
        )
{
    //read weight
    //std::cout << "reading weights[" << o_c << "][" << i_c << "]\n";
    for (int i = 0; i < k; i++)
    {
        for (int j = 0; j < k; j ++)
        {
            for(int q = 0; q < k; q++)
            {

                ADD_PRAGMA(HLS loop_tripcount max = MAX_KERNEL_SIZE*MAX_KERNEL_SIZE*MAX_KERNEL_SIZE)
                for (int o_cc = 0; o_cc < TCO; o_cc++)
                {
                    for (int i_cc=0; i_cc < TCI; i_cc++)
                    {
                        #pragma HLS pipeline II=1
                        weightBRAM[o_cc][i_cc][i*k*k + j*k + q] = mem[weight_offset + (o_c+o_cc)*k*k*k + (i_c+i_cc)*oc*k*k*k + (k-1-i)*k*k + (k-1-j)*k + (k-1-q)];
                        //std::cout<< "o_cc = " << o_cc << " i_cc = " << i_cc <<  " kernel = " << i << "\n";
                        //std::cout << "reading weight[" << o_c + o_cc << "][" << i_c + i_cc << "][" << i << "] = "
                        //<< mem[weight_offset + (o_c+o_cc)*ic*k*k*k + (i_c+i_cc)*k*k + i] << "=" << weightBRAM[o_cc][i_cc][i] << "\n";
                    }
                }
            }
        }
    }
}
void mem_read_weight(
        float * mem,            // global memory pointer
        int   weight_offset,    // offset of weights
        float weightBRAM[TCO][TCI][MAX_KERNEL_SIZE*MAX_KERNEL_SIZE*MAX_KERNEL_SIZE],
        const int k,
        const int oc,
        const int ic,
        int o_c,    //current output channel
        int i_c	//current input  channel index
        )
{
    //read weight
    //std::cout << "reading weights[" << o_c << "][" << i_c << "]\n";

    for (int o_cc = 0; o_cc < TCO; o_cc++)
    {
        for (int i_cc=0; i_cc < TCI; i_cc++)
        {
            for (int i = 0; i < k*k*k; i++)
            {
        ADD_PRAGMA(HLS loop_tripcount max = MAX_KERNEL_SIZE*MAX_KERNEL_SIZE*MAX_KERNEL_SIZE)
                #pragma HLS pipeline II=1
                weightBRAM[o_cc][i_cc][i] = mem[weight_offset + (o_c+o_cc)*ic*k*k*k + (i_c+i_cc)*k*k*k + i];
                //std::cout<< "o_cc = " << o_cc << " i_cc = " << i_cc <<  " kernel = " << i << "\n";
                //std::cout << "reading weight[" << o_c + o_cc << "][" << i_c + i_cc << "][" << i << "] = "
                //<< mem[weight_offset + (o_c+o_cc)*ic*k*k*k + (i_c+i_cc)*k*k + i] << "=" << weightBRAM[o_cc][i_cc][i] << "\n";
            }
        }
    }
}
///////////////function to read tile of input//////////////////////////
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
        int oy_limit,
        int ox_limit,
        int od_limit)
{
    //read input
    // padding also embedded
    int pad = (k)/2;
    int i_x = (o_x)*s - pad;
    int i_y = (o_y)*s - pad;
    int i_d = (o_d)*s - pad;
    for (int iid = 0; iid < k + s*od_limit; iid++) {
        ADD_PRAGMA(HLS loop_tripcount max = MAX_KERNEL_SIZE)
        for (int iiy = 0; iiy < k + s*oy_limit; iiy++) {
            ADD_PRAGMA(HLS loop_tripcount max = MAX_KERNEL_SIZE)
            for (int iix = 0; iix < k + s*ox_limit; iix++) {
                ADD_PRAGMA(HLS loop_tripcount max = MAX_KERNEL_SIZE)
                // for (int d = 0; d < od_limit; d++) {
                //     ADD_PRAGMA(HLS loop_tripcount max = TOD)
                //     for (int y = 0; y < oy_limit; y++) {
                //         ADD_PRAGMA(HLS loop_tripcount max = TOY)
                //         for (int x = 0; x < ox_limit; x++) {
                //             ADD_PRAGMA(HLS loop_tripcount max = TOX)
                            for (int i_cc=0; i_cc < TCI; i_cc++) {
                                #pragma HLS pipeline II=1
                                //reading some more than once
                                if(((i_x+iix) >= 0) && ((i_y+iiy) >= 0) && ((i_d+iid) >= 0) && ((i_x+iix) < ix) && ((i_y+iiy) < iy) && ((i_d+iid) < id) && (i_cc+i_c)<ic) //
                                    inputBRAM[i_cc][iid][iiy][iix] = mem[input_offset+ bb*ic*id*iy*ix + (i_c+i_cc)*id*ix*iy + (i_d+iid)*ix*iy + (i_y+iiy)*ix + i_x + iix];
                                else
                                    inputBRAM[i_cc][iid][iiy][iix] = 0;
                                //std::cout << "pad = " << pad << " o_c = " << o_c << "o_d = " << o_d + d << " o_y= " << o_y +y << "o_x = " << o_x +x  << "\n" ;
                                //std::cout << "pad = " << pad << " i_c = " << i_c + i_cc << "i_d = " << i_d << " i_y= " << i_y << "i_x = " << i_x  << "\n" ;
                                //std::cout << "reading input[" << (i_c+i_cc)*id*ix*iy + i_d*ix*iy + i_y*ix + i_x << "] as input [" << i_cc << "][" << s*d+iid << "][" << s*y+iiy << "][" << s*x+iix <<  "] =" << inputBRAM[i_cc][s*d+iid][s*y+iiy][s*x+iix] << "\n";
                            }
                //         }
                //     }
                // }
            }
        }
    }
}

void mem_read_input_transpose(
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
        int oy_limit,
        int ox_limit,
        int od_limit)
{
    //read input
    // padding also embedded
    int pad = (k)/2;
    for (int iid = 0; iid < k + s*od_limit; iid++) {
        ADD_PRAGMA(HLS loop_tripcount max = MAX_KERNEL_SIZE)
        for (int iiy = 0; iiy < (k + s*oy_limit); iiy++) {
            ADD_PRAGMA(HLS loop_tripcount max = MAX_KERNEL_SIZE)
            for (int iix = 0; iix < (k + s*ox_limit); iix++) {
                ADD_PRAGMA(HLS loop_tripcount max = MAX_KERNEL_SIZE)
                // for (int d = 0; d < od_limit; d++) {
                //     ADD_PRAGMA(HLS loop_tripcount max = TOD)
                //     for (int y = 0; y < oy_limit; y++) {
                //         ADD_PRAGMA(HLS loop_tripcount max = TOY)
                //         for (int x = 0; x < ox_limit; x++) {
                //             ADD_PRAGMA(HLS loop_tripcount max = TOX)
                            for (int i_cc=0; i_cc < TCI; i_cc++) {
                                #pragma HLS pipeline II=1
                                int i_x = ((o_x) - pad + iix);
                                int i_y = ((o_y) - pad + iiy);
                                int i_d = ((o_d) - pad + iid);
                                //reading some more than once
                                if((i_x >= 0) && (i_y >= 0) && (i_d >= 0) && (i_x < s*ix) && (i_y < s*iy) && (i_d < s*id) && (i_cc+i_c)<ic){ //padding
                                 if(((i_x%(s)) == 0) && ((i_y%(s)) == 0) && ((i_d%(s)) == 0)){ //transpose stride
                                    int ni_x = i_x/s; int ni_y = i_y/s; int ni_d = i_d/s; 
                                    inputBRAM[i_cc][iid][iiy][iix] = mem[input_offset+ bb*ic*id*iy*ix + (i_c+i_cc)*id*ix*iy + ni_d*ix*iy + ni_y*ix + ni_x];
                                 }else
                                 {
                                      inputBRAM[i_cc][iid][iiy][iix] = 0;
                                 }
                                 
                                }
                                else
                                    inputBRAM[i_cc][iid][iiy][iix] = 0;

                                
                                //std::cout << "pad = " << pad << " o_c = " << o_c << "o_d = " << o_d + d << " o_y= " << o_y +y << "o_x = " << o_x +x  << "\n" ;
                                //std::cout << "pad = " << pad << " i_c = " << i_c + i_cc << "i_d = " << i_d << " i_y= " << i_y << "i_x = " << i_x  << "\n" ;
                                //std::cout << "reading input[" << (i_c+i_cc)*id*ix*iy + i_d*ix*iy + i_y*ix + i_x << "] as input [" << i_cc << "][" << s*d+iid << "][" << s*y+iiy << "][" << s*x+iix <<  "] =" << inputBRAM[i_cc][s*d+iid][s*y+iiy][s*x+iix] << "\n";
                            }
                //         }
                //     }
                // }
                //std::cout <<  inputBRAM[0][iid][iiy][iix] << '\t';
                
            }
            //std::cout << std::endl;
        }
        //std::cout << std::endl;
    }
}

////////////////////////to do convolution//////////////////////
void conv_compute(
        float outputBRAM[TCO][TOD][TOY][TOX],
        float inputBRAM[TCI][IND_SIZE][INY_SIZE][INX_SIZE],
        float weightBRAM[TCO][TCI][MAX_KERNEL_SIZE*MAX_KERNEL_SIZE*MAX_KERNEL_SIZE],
        const int k,
        const int s,
        int od_limit,
        int oy_limit,
        int ox_limit,
        int o_c, //for test
        int i_c,
        int o_x,
        int o_y,
        int o_d)
{
    //std :: cout << "read "<<  weightBRAM[0][0][0] << " and " << inputBRAM[0][0][0][0] << "\n";
    for (int d = 0; d < od_limit; d++) {
        ADD_PRAGMA(HLS loop_tripcount max = TOD)
        for (int y = 0; y < oy_limit; y++) {
            ADD_PRAGMA(HLS loop_tripcount max = TOY)
            for (int x = 0; x < ox_limit; x++) {
                ADD_PRAGMA(HLS loop_tripcount max = TOX)
                for (int l = 0; l < k; l++) {
                    ADD_PRAGMA(HLS loop_tripcount max = MAX_KERNEL_SIZE)
                    for (int i = 0; i < k; i++) {
                        ADD_PRAGMA(HLS loop_tripcount max = MAX_KERNEL_SIZE)
                        for (int j = 0; j < k; j++) {
                            ADD_PRAGMA(HLS loop_tripcount max = MAX_KERNEL_SIZE)
                            float outarray[TCO];
                            #pragma HLS pipeline II=1
                            #pragma HLS array_partition variable outarray complete
                            for (int q = 0; q < TCO; q++)
                            {
                                #pragma HLS unroll
                                outarray[q] = outputBRAM[q][d][y][x];
                            }
                            
                            
                             for (int o_cc = 0; o_cc < TCO; o_cc++) {
                                #pragma HLS unroll
                                //#pragma HLS dependence variable=inputBRAM inter false
                                //#pragma HLS dependence variable=weightBRAM inter false
                               //#pragma HLS dependence variable=outputBRAM inter FALSE
                                //#pragma HLS pipeline
                                float mul1_1 = 0;
                                //#pragma HLS array_partition variable mul1_1 complete
                                float mul1_2 = 0;
                                float mul1_3 = 0;
                                float mul1_4 = 0;
                                float mul1_5 = 0;
                                float mul1_6 = 0;
                                float mul1_7 = 0;
                                float mul1_8 = 0;
                                float mul2_1 = 0;
                                float mul2_2 = 0;
                                float mul3_1 = 0;
                                //#pragma HLS RESOURCE variable=mul1_1 core=FMul_meddsp
                                //#pragma HLS RESOURCE variable=mul1_2 core=FMul_meddsp
                                //#pragma HLS RESOURCE variable=mul1_3 core=FMul_meddsp
                                //#pragma HLS RESOURCE variable=mul1_4 core=FMul_meddsp
                                //#pragma HLS RESOURCE variable=mul2_1 core=FAddSub_nodsp
                                //#pragma HLS RESOURCE variable=mul2_2 core=FAddSub_nodsp
                                mul1_1 =            inputBRAM[0][s*d+l][s*y+i][s*x+j] * weightBRAM[o_cc][0][l*k*k+i*k+j] ;
                                mul1_2 =            inputBRAM[1][s*d+l][s*y+i][s*x+j] * weightBRAM[o_cc][1][l*k*k+i*k+j] ;
                                mul1_3 =            inputBRAM[2][s*d+l][s*y+i][s*x+j] * weightBRAM[o_cc][2][l*k*k+i*k+j] ;
                                mul1_4 =            inputBRAM[3][s*d+l][s*y+i][s*x+j] * weightBRAM[o_cc][3][l*k*k+i*k+j] ;
                                mul1_5 =            inputBRAM[4][s*d+l][s*y+i][s*x+j] * weightBRAM[o_cc][4][l*k*k+i*k+j] ;
                                mul1_6 =            inputBRAM[5][s*d+l][s*y+i][s*x+j] * weightBRAM[o_cc][5][l*k*k+i*k+j] ;
                                mul1_7 =            inputBRAM[6][s*d+l][s*y+i][s*x+j] * weightBRAM[o_cc][6][l*k*k+i*k+j] ;
                                mul1_8 =            inputBRAM[7][s*d+l][s*y+i][s*x+j] * weightBRAM[o_cc][7][l*k*k+i*k+j] ;


                                mul2_1 = mul1_1 + mul1_2 + mul1_3 + mul1_4;
                                mul2_2 = mul1_5 + mul1_6 + mul1_7 + mul1_8;
                                mul3_1 = mul2_1 + mul2_2;


                                // mul1_1[1] =      inputBRAM[0][s*d+l][s*y+i][s*x+j] * weightBRAM[1][0][l*k*k+i*k+j] +
                                //                     inputBRAM[1][s*d+l][s*y+i][s*x+j] * weightBRAM[1][1][l*k*k+i*k+j] +
                                //                     inputBRAM[2][s*d+l][s*y+i][s*x+j] * weightBRAM[1][2][l*k*k+i*k+j] +
                                //                     inputBRAM[3][s*d+l][s*y+i][s*x+j] * weightBRAM[1][3][l*k*k+i*k+j] ;

                                // mul1_1[2] =      inputBRAM[0][s*d+l][s*y+i][s*x+j] * weightBRAM[2][0][l*k*k+i*k+j] +
                                //                     inputBRAM[1][s*d+l][s*y+i][s*x+j] * weightBRAM[2][1][l*k*k+i*k+j] +
                                //                     inputBRAM[2][s*d+l][s*y+i][s*x+j] * weightBRAM[2][2][l*k*k+i*k+j] +
                                //                     inputBRAM[3][s*d+l][s*y+i][s*x+j] * weightBRAM[2][3][l*k*k+i*k+j] ;                                                

                                // mul1_1[3] =      inputBRAM[0][s*d+l][s*y+i][s*x+j] * weightBRAM[3][0][l*k*k+i*k+j] +
                                //                     inputBRAM[1][s*d+l][s*y+i][s*x+j] * weightBRAM[3][1][l*k*k+i*k+j] +
                                //                     inputBRAM[2][s*d+l][s*y+i][s*x+j] * weightBRAM[3][2][l*k*k+i*k+j] +
                                //                     inputBRAM[3][s*d+l][s*y+i][s*x+j] * weightBRAM[3][3][l*k*k+i*k+j] ;

                                // mul1_1[4] =      inputBRAM[0][s*d+l][s*y+i][s*x+j] * weightBRAM[4][0][l*k*k+i*k+j] +
                                //                     inputBRAM[1][s*d+l][s*y+i][s*x+j] * weightBRAM[4][1][l*k*k+i*k+j] +
                                //                     inputBRAM[2][s*d+l][s*y+i][s*x+j] * weightBRAM[4][2][l*k*k+i*k+j] +
                                //                     inputBRAM[3][s*d+l][s*y+i][s*x+j] * weightBRAM[4][3][l*k*k+i*k+j] ;

                                // mul1_1[5] =      inputBRAM[0][s*d+l][s*y+i][s*x+j] * weightBRAM[5][0][l*k*k+i*k+j] +
                                //                     inputBRAM[1][s*d+l][s*y+i][s*x+j] * weightBRAM[5][1][l*k*k+i*k+j] +
                                //                     inputBRAM[2][s*d+l][s*y+i][s*x+j] * weightBRAM[5][2][l*k*k+i*k+j] +
                                //                     inputBRAM[3][s*d+l][s*y+i][s*x+j] * weightBRAM[5][3][l*k*k+i*k+j] ;

                                // mul1_1[6] =      inputBRAM[0][s*d+l][s*y+i][s*x+j] * weightBRAM[6][0][l*k*k+i*k+j] +
                                //                     inputBRAM[1][s*d+l][s*y+i][s*x+j] * weightBRAM[6][1][l*k*k+i*k+j] +
                                //                     inputBRAM[2][s*d+l][s*y+i][s*x+j] * weightBRAM[6][2][l*k*k+i*k+j] +
                                //                     inputBRAM[3][s*d+l][s*y+i][s*x+j] * weightBRAM[6][3][l*k*k+i*k+j] ;

                                //  mul1_1[7] =      inputBRAM[0][s*d+l][s*y+i][s*x+j] * weightBRAM[7][0][l*k*k+i*k+j] +
                                //                     inputBRAM[1][s*d+l][s*y+i][s*x+j] * weightBRAM[7][1][l*k*k+i*k+j] +
                                //                     inputBRAM[2][s*d+l][s*y+i][s*x+j] * weightBRAM[7][2][l*k*k+i*k+j] +
                                //                     inputBRAM[3][s*d+l][s*y+i][s*x+j] * weightBRAM[7][3][l*k*k+i*k+j] ;                                                                                                                                                                                                               
                                // mul1_2 =    inputBRAM[4][s*d+l][s*y+i][s*x+j] * weightBRAM[o_cc][4][l*k*k+i*k+j] +
                                //             inputBRAM[5][s*d+l][s*y+i][s*x+j] * weightBRAM[o_cc][5][l*k*k+i*k+j] +
                                //             inputBRAM[6][s*d+l][s*y+i][s*x+j] * weightBRAM[o_cc][6][l*k*k+i*k+j] +
                                //             inputBRAM[7][s*d+l][s*y+i][s*x+j] * weightBRAM[o_cc][7][l*k*k+i*k+j] ;
                                            //inputBRAM[4][s*d+l][s*y+i][s*x+j] * weightBRAM[o_cc][4][l*k*k+i*k+j];
                                //mul1_3 = inputBRAM[2][s*d+l][s*y+i][s*x+j] * weightBRAM[o_cc][2][l*k*k+i*k+j];
                                //mul1_4 = inputBRAM[3][s*d+l][s*y+i][s*x+j] * weightBRAM[o_cc][3][l*k*k+i*k+j];
                                // std::cout << inputBRAM[0][s*d+l][s*y+i][s*x+j] << " * " << weightBRAM[o_cc][0][l*k*k+i*k+j] << " + "
                                //             << inputBRAM[1][s*d+l][s*y+i][s*x+j] << " * " << weightBRAM[o_cc][1][l*k*k+i*k+j] << "\t";
                                // mul2_1 = mul1_1 + mul1_2;
                                //mul2_2 = mul1_3 + mul1_4;
                                //mul3_1 = mul2_1 + mul2_2;
                                //mul3_1 = mul2_1;
                                //float prev = outputBRAM[o_cc][d][y][x];
                                //std::cout << "writing to [" << o_cc << "][" << d << "][" << y << "][" << x << "]\n";
                                //outputBRAM[o_cc][d][y][x] += mul3_1;
                                outputBRAM[o_cc][d][y][x] = outarray[o_cc] + mul3_1;
                                // outputBRAM[1][d][y][x] += mul1_1[1];
                                // outputBRAM[2][d][y][x] += mul1_1[2];
                                // outputBRAM[3][d][y][x] += mul1_1[3];
                                // outputBRAM[4][d][y][x] += mul1_1[4];
                                // outputBRAM[5][d][y][x] += mul1_1[5];
                                // outputBRAM[6][d][y][x] += mul1_1[6];
                                // outputBRAM[7][d][y][x] += mul1_1[7];
                                //std::cout << "O: " << outputBRAM[o_cc][d][y][x] << "\t";
                                //std::cout << inputBRAM[0][s*d+l][s*y+i][s*x+j] << " x " << weightBRAM[o_cc][0][l*k*k+i*k+j] << " + " << inputBRAM[1][s*d+l][s*y+i][s*x+j] << " x " << weightBRAM[o_cc][1][l*k*k+i*k+j] << " = " << mul1_1 << "   ";
//                                if (o_cc == 0 && d == 1 && y == 0 && x == 0)
//                                    std::cout << " in[" << l << "][" << i << "][" << j << "] .. =" << inputBRAM[0][s*d+l][s*y+i][s*x+j] << "x" << weightBRAM[o_cc][0][l*k*k+i*k+j]<< "\n";
//                                    std::cout << "in 1 =" << inputBRAM[1][s*d+l][s*y+i][s*x+j] << "x" << weightBRAM[o_cc][1][l*k*k+i*k+j] << "\n";
//                                    std::cout << "in 2 =" <<  inputBRAM[2][s*d+l][s*y+i][s*x+j] << "x" << weightBRAM[o_cc][2][l*k*k+i*k+j] << "\n";
//                                    std::cout << "in 3 =" <<  inputBRAM[3][s*d+l][s*y+i][s*x+j] << "x" << weightBRAM[o_cc][3][l*k*k+i*k+j] << "\n";
                                //std::cout << "out[" << o_cc << "][" << d << "][" << y << "][" << x << "] = " << mul1_1 << " + " << mul1_2 << " + " << mul1_3 << " + " << mul1_4 << "\n";
                                //int r = (k-1)/2;
                                //if ((o_cc+o_c) == 0 && (o_d+d )== 1 && (o_y+y) == 0 && (o_x + x) == 2){
                                //std::cout <<"out[" << o_cc + o_c << "][" << o_d + d << "][" << o_y + y << "][" << o_x + x << "] += "
                                //<< " in[" << i_c+0 << "][" << s*(d+o_d) + l -r << "][" << s*(y+o_y) + i - r  << "][" << s*(x+o_x) + j - r << "] x w[" << o_cc+o_c << "][" << 0 << "][" << l << "][" << i << "][" << j << "] = "
                                //<< inputBRAM[0][s*d+l][s*y+i][s*x+j] << "x" << weightBRAM[o_cc][0][l*k*k+i*k+j] << "=" << mul1_1 << "+" << mul1_2 << "+" << mul1_3 << "+" << mul1_4 << " = " << outputBRAM[o_cc][d][y][x] - prev <<" = +" << outputBRAM[o_cc][d][y][x] << "\n";}
                            }
                        }
                    }
                }
            }
        }
    }
}

// typedef float myDataType;
// static myDataType adder_tree(myDataType window0[TCO])
// {
// #pragma HLS PIPELINE
// 	myDataType window1[TCO/2] = {0.0};
// #pragma HLS ARRAY_PARTITION variable=window1 complete
// //#pragma HLS RESOURCE variable=window1 core=FAddSub_nodsp
// 	myDataType window2[TCO/4] = {0.0};
// #pragma HLS ARRAY_PARTITION variable=window2 complete


// 	myDataType result = 0.0;
// 	L1: for(int x=0; x<TCO/2; x++)
//     {
//     	 window1[x] = window0[2*x] +  window0[1+2*x];
// 	}
// 	L2: for(int x=0; x<TCO/4; x++)
//     {
//     	 window2[x] = window1[x] +  window1[TCO/4+x];

// 	}


// 	result = window2[0] + window2[1];
// 	return result;
// }

/////// writng memory //////////////////////
void mem_write(
        float * mem,            // global memory pointer
        int   output_offset,       // offset of inputs
        float outputBRAM[TCO][TOD][TOY][TOX],
        float normBRAM[MAX_OUTPUT_CHANNELS/TCO][TN],
        const int oc,
        const int od,
        const int oy,
        const int ox,
        int bb,
        int o_c,
        int o_d,
        int o_y,
        int o_x,
        int od_limit,
        int oy_limit,
        int ox_limit,
        int bnorm,
        int relu)
{
    for (int o_cc=0; o_cc < TCO; o_cc++) {
        ADD_PRAGMA(HLS loop_tripcount max = TCO)
        for (int d = 0; d < od_limit; d++) {
            ADD_PRAGMA(HLS loop_tripcount max = TOD)
            for (int y = 0; y < oy_limit; y++) {
                ADD_PRAGMA(HLS loop_tripcount max = TOY)
                for (int x = 0; x < ox_limit; x++) {
                    ADD_PRAGMA(HLS loop_tripcount max = TOX)
                    #pragma HLS pipeline II=1
                    float output_element = outputBRAM[o_cc][d][y][x];
                    // std::cout << "writing " << output_element << "\n";
                    if(bnorm)
                    {
                        int i0 = o_c/TCO;
                        int i1 = o_cc*NUM_BNORM_PARAMS;
                        //(output-mean)*gamma/sqrt(var + EPSILON)-beta;
                        //if ((o_cc+o_c)==0)
                        float ratio =  normBRAM[i0][i1+2]/sqrt(normBRAM[i0][i1+1] + EPSILON);
                        //std::cout << "for outBRAM[ " << o_cc+o_c << "][" << o_d+d << "][" << o_y+y << "][" << o_x+x << "] =" << ratio << "x" << output_element - normBRAM[i0][i1] << " + " << normBRAM[i0][i1+3] << " , mean = " << normBRAM[i0][i1] << " var = " << normBRAM[i0][i1+1] << "gamma = " << normBRAM[i0][i1+2] << " beta = " << normBRAM[i0][i1+3] << "\n";
                        output_element = (output_element - normBRAM[i0][i1])*ratio+normBRAM[i0][i1+3];
                        //std::cout << "writing " << output_element << "\n";
                    }
                    //std::cout << "writing after bnorm " << output_element << "\n";
                    if (relu) output_element = std::max(0.0f,output_element);
                    //std::cout << " = " << output_element << "\n";
                    if ((o_c+o_cc) < oc && (o_d+d) < od && (o_y+y) < oy && (o_x+x) < ox)
                        //printf("O: %f", output_element);
                        mem[output_offset + bb*oc*od*ox*oy + (o_c+o_cc)*od*ox*oy+ (o_d+d)*ox*oy + (o_y+y)*ox + o_x+x]  = output_element;
                    //std::cout << " = " << mem[output_offset + 0*oc*od*ox*oy + 0*od*ox*oy+ 1*ox*oy + 0*ox + 2] << "\n";
                }
            }
        }
    }
    //if (o_c == 0)
    //    std::cout << "done writing\n";
}

void mem_write_fullnorm(
        float * mem,            // global memory pointer
        int   output_offset,       // offset of inputs
        float outputBRAM[TCO][TOD][TOY][TOX],
        float normBRAM[MAX_OUTPUT_CHANNELS][4],
        const int oc,
        const int od,
        const int oy,
        const int ox,
        int bb,
        int o_c,
        int o_d,
        int o_y,
        int o_x,
        int od_limit,
        int oy_limit,
        int ox_limit,
        int bnorm,
        int relu)
{
    for (int o_cc=0; o_cc < TCO; o_cc++) {
        ADD_PRAGMA(HLS loop_tripcount max = TCO)
        for (int d = 0; d < od_limit; d++) {
            ADD_PRAGMA(HLS loop_tripcount max = TOD)
            for (int y = 0; y < oy_limit; y++) {
                ADD_PRAGMA(HLS loop_tripcount max = TOY)
                for (int x = 0; x < ox_limit; x++) {
                    ADD_PRAGMA(HLS loop_tripcount max = TOX)
                    #pragma HLS pipeline II=1
                    float output_element = outputBRAM[o_cc][d][y][x];
                     //std::cout << "writing[ " << o_cc+o_c << ", " << o_d+d << ", "<< o_y + y << ", "<< o_x + x << "]: " << output_element << "\n";
                    if(bnorm==1)
                    {

                        float ratio =  normBRAM[o_c+o_cc][2]/sqrt(normBRAM[o_c+o_cc][1] + EPSILON);
                        //std::cout << "for outBRAM[ " << o_cc+o_c << "][" << o_d+d << "][" << o_y+y << "][" << o_x+x << "] =" << ratio << "x" << output_element - normBRAM[o_c+o_cc][0] << " + " << normBRAM[o_c+o_cc][3] << " , mean = " << normBRAM[o_c+o_cc][0] << " var = " << normBRAM[o_c+o_cc][1] << "gamma = " << normBRAM[o_c+o_cc][2] << " beta = " << normBRAM[o_c+o_cc][3] << "\n";
                        output_element = (output_element - normBRAM[o_c+o_cc][0])*ratio+normBRAM[o_c+o_cc][3];
                        //std::cout << "writing " << output_element << "\n";
                    }
                    //std::cout << "writing after bnorm " << output_element << "\n";
                    if (relu==1){ output_element = std::max(0.0f,output_element);}
                    //std::cout << " = " << output_element << "\n";
                    if (((o_c+o_cc) < oc) && ((o_d+d) < od) && ((o_y+y) < oy) && ((o_x+x) < ox)){
                        //printf("O: %f", output_element);
                        mem[output_offset + bb*oc*od*ox*oy + (o_c+o_cc)*od*ox*oy+ (o_d+d)*ox*oy + (o_y+y)*ox + o_x+x]  = output_element;
                    }
                    //std::cout << " = " << mem[output_offset + 0*oc*od*ox*oy + 0*od*ox*oy+ 1*ox*oy + 0*ox + 2] << "\n";
                }
            }
        }
    }
    //if (o_c == 0)
    //    std::cout << "done writing\n";
}
