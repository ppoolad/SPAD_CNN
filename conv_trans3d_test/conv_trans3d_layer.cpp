#include <cstddef>
#include <algorithm>
#include <float.h>
#include "conv_layer.h"
#include <ap_fixed.h>
#define NUM_ELEM 32
//#define NUM_ELEM2 16
typedef float myDataType;
//typedef ap_fixed<24,12,AP_RND,AP_SAT> myDataType;

static myDataType hls_fp_accumulator(myDataType window0[NUM_ELEM]);
void conv_layer(float * mem,            // global memory pointer
                int input_offset,       // offset of inputs
                int output_offset,      // offset of outputs
                const int b,            // batch size
                const int od,           // output dimensions
                const int ox,           // output width
                const int oy,           // output height
                const int id,           // input dimensions
                const int ix,           // input width
                const int iy,           // input height
                const int s,            // stride
                const int k)            // kernel size
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
 

  myDataType sumArrz[MAX_KERNEL_SIZE][NUM_ELEM]={0.0};
#pragma HLS ARRAY_PARTITION variable=sumArrz cyclic factor=32 dim=2

static float obf[MAX_BATCH][MAX_OUTPUT_WIDTH];
static float kbuf[MAX_INPUT_DIMS][MAX_KERNEL_SIZE*MAX_KERNEL_SIZE]={0.0};
#pragma HLS ARRAY_PARTITION variable=kbuf cyclic factor=32 dim=1
static myDataType ibuf[MAX_INPUT_DIMS][MAX_KERNEL_SIZE]={0.0};
#pragma HLS ARRAY_PARTITION variable=ibuf cyclic factor=32 dim=1

//#pragma HLS ARRAY_PARTITION variable=batch_out complete dim=1

  int num_weights = id*od*k*k;
  int num_biases = od;
  int num_input = b*id*ix*iy;
  int num_output = b*od*ox*oy;

//           // Batch
// BATCH:    for (int b_=0; b_< b; b_++)
//           {
            // Output Dimensions (Feature Maps)
OD:         for (int o_d = 0; o_d < od; o_d++)
            {
#pragma HLS LOOP_TRIPCOUNT max=512
              myDataType tmpbias = (myDataType) mem[input_offset/sizeof(float) + num_weights + o_d];

// buffering kernels
KBUF2:        for(int j = 0; j < id; j++)
              {
#pragma HLS LOOP_TRIPCOUNT min=1 max=512
KBUF:           for(int i = 0; i < k*k; i++)
                {
#pragma HLS LOOP_TRIPCOUNT min=1 max=9
#pragma HLS PIPELINE
                  kbuf[j][i] = mem[input_offset/sizeof(float) + o_d*id*k*k + j*k*k +i];//weights[o_d*id*k*k + j*k*k + i];
                }
              }    


              // Output Y Dimension
OY:           for (int o_y = 0; o_y < oy; o_y++)
              {
#pragma HLS LOOP_TRIPCOUNT max=224                
                // Output X Dimension
OX:             for (int o_x = 0; o_x < ox; o_x++)
                {
#pragma HLS LOOP_TRIPCOUNT max=224
                  // Set bias 
                  //float output_element = mem[input_offset/sizeof(float) + num_weights + o_d];

                  // Weighted Sum:
                  // Batch
BATCH:            for (int b_=0; b_< b; b_++)
                  {
#pragma HLS LOOP_TRIPCOUNT max=10
                    myDataType output_element = tmpbias;
//                   // Input Dimensions (Feature Maps)
// ID:               for (int i_d = 0; i_d < id; i_d++)
//                   {

                    // Input Y Dimension
IY:                 for (int i_y = o_y*s, iiy = 0; i_y < o_y*s+k; i_y++, iiy++)
                    {
#pragma HLS LOOP_TRIPCOUNT max=3
ibuf:                 for(int j = 0; j < id; j++)
                      {
#pragma HLS LOOP_TRIPCOUNT min=1 max=3
                        for(int i = 0; i < k; i++)
                        {
#pragma HLS LOOP_TRIPCOUNT min=1 max=512
#pragma HLS PIPELINE
                          ibuf[j][i] = (myDataType) mem[input_offset/sizeof(float) + num_weights+num_biases+ b_*id*ix*iy + j*ix*iy + i_y*ix + s*o_x + i];
                        }
                
                      }

                      
                      // Input X Dimension
IX:                   for (int i_x = o_x*s, iix = 0; i_x < o_x*s+k; i_x++, iix++)
                      {
#pragma HLS LOOP_TRIPCOUNT max=3
                        //myDataType sumd2[MAX_OUTPUT_DIMS/32] = {0.0};
//#pragma HLS ARRAY_PARTITION variable=sumd2 complete
                        // Input Dimensions (Feature Maps)
			int iteration = id/32;
			if(iteration == 0)iteration = 1;
ID:                     for (int i_d = 0; i_d < iteration; i_d++)
                        {
#pragma HLS LOOP_TRIPCOUNT max=16
#pragma HLS PIPELINE
                          for(int q = 0; q < 32; q++)
                          {
                            sumArrz[iix][q] = ibuf[32*i_d+ q][iix] * (myDataType)kbuf[32*i_d+q][iiy*k  + iix];
                          }
                          output_element += hls_fp_accumulator(sumArrz[iix]);//adder tree adds 32 elements

                        }

                        //output_element += hls_fp_accumulator(sumd2);
                    }
                  }
                  // Write output
                  if(output_element<0.0)output_element=0;//reLU
                    obf[b_][o_x] = (float) output_element;//write to buffer
                }
              }
SAVE:         for(size_t j = 0; j < b; j++)
              {
#pragma HLS LOOP_TRIPCOUNT max=10
                for(size_t i = 0; i < ix; i++)
                {
#pragma HLS LOOP_TRIPCOUNT max=224
#pragma HLS PIPELINE
                 mem[output_offset/sizeof(float) + j*od*ox*oy + o_d*ox*oy + o_y*ox + i]=obf[j][i];
                }
              }

            }
          }
}


static myDataType hls_fp_accumulator(myDataType window0[NUM_ELEM])
{
#pragma HLS PIPELINE
	myDataType window1[NUM_ELEM/2] = {0.0};
#pragma HLS ARRAY_PARTITION variable=window1 complete
//#pragma HLS RESOURCE variable=window1 core=FAddSub_nodsp
	myDataType window2[NUM_ELEM/4] = {0.0};
#pragma HLS ARRAY_PARTITION variable=window2 complete
	myDataType window3[NUM_ELEM/8] = {0.0};
#pragma HLS ARRAY_PARTITION variable=window3 complete
	myDataType window4[NUM_ELEM/16]= {0.0};
#pragma HLS ARRAY_PARTITION variable=window4 complete
// 	myDataType window5[NUM_ELEM/32];//= {0.0};
// #pragma HLS ARRAY_PARTITION variable=window5 complete

//	myDataType window6[NUM_ELEM/64];//= {0.0};
//#pragma HLS ARRAY_PARTITION variable=window6 complete

//   myDataType window7[NUM_ELEM/128];//= {0.0};
// #pragma HLS ARRAY_PARTITION variable=window7 complete


	myDataType result = 0.0;
	L1: for(int x=0; x<NUM_ELEM/2; x++)
    {
    	 window1[x] = window0[2*x] +  window0[1+2*x];
	}
	L2: for(int x=0; x<NUM_ELEM/4; x++)
    {
    	 window2[x] = window1[x] +  window1[NUM_ELEM/4+x];

	}
	L3: for(int x=0; x<NUM_ELEM/8; x++)
    {
    	 window3[x] = window2[x] +  window2[NUM_ELEM/8+x];

	}
	L4: for(int x=0; x<NUM_ELEM/16; x++)
    {
    	 window4[x] = window3[x] +  window3[NUM_ELEM/16+x];
	}


	result = window4[0] + window4[1];
	return result;
}

