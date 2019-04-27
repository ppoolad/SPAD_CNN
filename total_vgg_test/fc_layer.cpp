#include <cstddef>
#include <algorithm>
#include "fc_layer.h"
#include "constants.h"
#include <ap_fixed.h>

typedef ap_fixed<24,10,AP_RND,AP_SAT> myDataType;
//typedef float myDataType;
#define NUM_ELEM 32
myDataType hls_fp_accumulator(myDataType window0[NUM_ELEM]);

void fc_layer(float * mem,
              int input_offset,
              int output_offset,
              const int batch_size,
              const int num_inputs,
              const int num_outputs,
              const int enable_relu)
{
#pragma HLS INTERFACE m_axi port=mem depth=2147483648
#pragma HLS INTERFACE s_axilite port=input_offset bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=output_offset bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=batch_size bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=num_inputs bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=num_outputs bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=enable_relu bundle=CTRL_BUS
#pragma HLS INTERFACE s_axilite port=return bundle=CTRL_BUS

  const int num_weights = num_inputs*num_outputs;
  const int num_biases =  num_outputs;
float output_b[MAX_BATCH][FC_MAX_OUTPUT_SIZE] = {0.0};
#pragma HLS ARRAY_PARTITION variable=output_b cyclic factor=32 dim=2
myDataType input_b[FC_MAX_INPUT_SIZE];//= {0.0};
#pragma HLS ARRAY_PARTITION variable=input_b cyclic factor=32
myDataType tree[32];//={0.0};
#pragma HLS ARRAY_PARTITION variable=tree cyclic factor=32
static myDataType weights_b[FC_MAX_INPUT_SIZE];//= {0.0};
#pragma HLS ARRAY_PARTITION variable=weights_b cyclic factor=32
static myDataType biases[FC_MAX_OUTPUT_SIZE];//= {0.0};
#pragma HLS ARRAY_PARTITION variable=weights_b cyclic factor=32

for(int i = 0; i < num_outputs; i++)// CAHCE BIASSES
{
#pragma HLS LOOP_TRIPCOUNT max=4096
#pragma HLS PIPELINE
  biases[i] = (myDataType) mem[input_offset/sizeof(float) + num_weights + i];
}


//   for (int b = 0; b < batch_size; b++) {
// #pragma HLS LOOP_TRIPCOUNT max=10
    // Output Node Iterator
  for (int o = 0; o < num_outputs; o++) {
#pragma HLS LOOP_TRIPCOUNT max=4096
      // Set bias
    
    for(int i = 0; i < num_inputs; i++) //CAHCE WEIGHTS
    {
#pragma HLS LOOP_TRIPCOUNT max=25088
#pragma HLS PIPELINE
      weights_b[i] = (myDataType) mem[input_offset/sizeof(float) + o*num_inputs+i];
    }
    
    for (int b = 0; b < batch_size; b++) {
#pragma HLS LOOP_TRIPCOUNT max=10
	myDataType output_element = biases[o];
      for(int i = 0; i < num_inputs; i++) //CACHE INPUTS
      {
#pragma HLS LOOP_TRIPCOUNT max=25088
#pragma HLS PIPELINE
        input_b[i] = (myDataType) mem[input_offset/sizeof(float) + num_weights + num_biases + b*num_inputs+i];
      }
            
      // Accumulate weighted sum
      for(int j = 0; j < num_inputs/32; j++)
      {
#pragma HLS LOOP_TRIPCOUNT max=49
#pragma HLS PIPELINE
        for(int i = 0; i < 32; i++)
        {
//         float input_element = mem[input_offset/sizeof(float) + num_weights + num_biases + b*num_inputs+i];
//         float weight_element = mem[input_offset/sizeof(float) + o*num_inputs+i];
//         output_element += input_element * weight_element;
          // float input_element = mem[input_offset/sizeof(float) + num_weights + num_biases + b*num_inputs+i];
          // float weight_element = mem[input_offset/sizeof(float) + o*num_inputs+i];
          tree[i] = input_b[j*32 + i] * weights_b[j*32 + i];     
		//output_element += input_b[j*512 + i] * weights_b[j*512 + i];       
        }
        output_element += hls_fp_accumulator(tree);
      }
 
/*     
       for (int i = 0; i < num_inputs; i++) {
 #pragma HLS LOOP_TRIPCOUNT max=25088
 #pragma PIPELINE
         float input_element = mem[input_offset/sizeof(float) + num_weights + num_biases + b*num_inputs+i];
         float weight_element = mem[input_offset/sizeof(float) + o*num_inputs+i];
         output_element += input_element * weight_element;
       }
*/
      // Compute activation
      if (enable_relu){
         //mem[output_offset/sizeof(float) + b*num_outputs+o] = std::max(0.0f, output_element);
	if(output_element<0.0)output_element=0.0;
	output_b[b][o] = (float) output_element;
      }else{
	output_b[b][o] = (float) output_element;}
         //mem[output_offset/sizeof(float) + b*num_outputs+o] = output_element;
    }
  }


   for(int b__=0;b__<batch_size;b__++)
	{
#pragma HLS LOOP_TRIPCOUNT max=10
	for(int out_=0;out_<num_outputs;out_++){
#pragma HLS LOOP_TRIPCOUNT max=4096
#pragma HLS PIPELINE
		mem[output_offset/sizeof(float) + b__*num_outputs+out_] = output_b[b__][out_];}
	}


}



myDataType hls_fp_accumulator(myDataType window0[NUM_ELEM])
{
#pragma HLS PIPELINE
	myDataType window1[NUM_ELEM/2];// = {0.0};
#pragma HLS ARRAY_PARTITION variable=window1 complete
//#pragma HLS RESOURCE variable=window1 core=FAddSub_nodsp
	myDataType window2[NUM_ELEM/4];// = {0.0};
#pragma HLS ARRAY_PARTITION variable=window2 complete
	myDataType window3[NUM_ELEM/8];// = {0.0};
#pragma HLS ARRAY_PARTITION variable=window3 complete
	myDataType window4[NUM_ELEM/16];//= {0.0};
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
