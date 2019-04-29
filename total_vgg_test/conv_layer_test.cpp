//
// Created by zbete on 4/28/2019.
//

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <map>
#include <string>
#include "conv_test/conv_layer.h"
#include "constants.h"
#include "util/shared.h"
#include <sstream>
#include <chrono>

#define HW_CTRL_ADDR 0x00000000

using namespace std;


//#define PRINT


int run_single_test_conv(string imageDir, map<string, int> layer_params, float * &dma_input, float * gold_outputs){

    int num_outputs = layer_params["output_dim"]*layer_params["output_width"]*
                      layer_params["output_height"];
    int num_inputs = layer_params["input_dim"]*layer_params["input_width"]*
                     layer_params["input_height"];
    int num_weights = layer_params["input_dim"]*layer_params["output_dim"]*
                      layer_params["kernel_size"]*layer_params["kernel_size"];
    int num_biases = layer_params["output_dim"];

    // very basic input checking
    if (layer_params["input_dim"] > CONV_MAX_INPUT_DIMS ||
        layer_params["input_width"] > CONV_MAX_INPUT_WIDTH ||
        layer_params["input_height"] > CONV_MAX_INPUT_WIDTH ||
        layer_params["output_dim"] > CONV_MAX_OUTPUT_DIMS ||
        layer_params["output_width"] > CONV_MAX_OUTPUT_WIDTH ||
        layer_params["output_height"] > CONV_MAX_OUTPUT_WIDTH ||
        layer_params["batch_size"] > CONV_MAX_BATCH)
    {
        cerr << "Problem with layer params\n";
        return 1;
    } else {
        int b = layer_params["batch_size"];
        int od = layer_params["output_dim"];
        int ox = layer_params["output_width"];
        int oy = layer_params["output_height"];
        int id = layer_params["input_dim"];
        int ix = layer_params["input_width"];
        int iy = layer_params["input_height"];
        int k = layer_params["kernel_size"];
        int s = layer_params["stride"];

#ifdef PRINT
        cout << "Begin Test\n"
       << "Batch Size: " << b << endl
       << "Num Inputs: " << num_inputs << endl
       << "Num Outputs: " << num_outputs << endl
       << "Num Weights: " << num_weights << endl
       << "Num Biases: " << num_biases << endl
       << "Input Dimensions " << b << " x " << id << " x " << ix << " x " << iy << endl
       << "Output Dimensions " << b << " x " << od << " x " << ox << " x " << oy << endl
       << "Kernel Dimensions " << od << " x " << id << " x " << k << " x " << k << endl
       << "Stride Size: " << s << endl;
#endif

        // Run Accelerator
#ifdef HW_TEST
        hw_conv_layer(HW_CTRL_ADDR, dma_input, 0,
                  sizeof(float)*(b*num_inputs+num_biases + num_weights),
                  b, od, ox, oy, id, ix, iy, s, k);
#else
        conv_layer(dma_input, 0, sizeof(float)*(b*num_inputs+num_biases + num_weights),
                   b, od, ox, oy, id, ix, iy, s, k);
#endif

    }

    return 0;

}




