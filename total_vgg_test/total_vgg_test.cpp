#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <map>
#include <string>
#include "constants.h"
#include "total_vgg_test.h"
#include "util/shared.h"
#include <sstream>
#include <chrono>
//#include "constants.h"
#define HW_CTRL_ADDR 0x00000000

using namespace std;

#define NUM_LAYERS 5
//#define PRINT


int run_single_test_bytype(string imageDir, map<string, int> layer_params, float * &dma_input, float * gold_outputs, int layerType){

    if(layerType == CONV3DT){
        if(run_single_test_conv_trans3d(imageDir, layer_params, dma_input, gold_outputs)!=0)
            return 1;
    }
    else if (layerType == CONV3D)
    {
        if(run_single_test_conv3d(imageDir, layer_params, dma_input, gold_outputs)!=0)
            return 1;
    }
    else{
        cout << "urecognized type of layer " << layerType << endl;
        return 1;
    }
    return 0;
}


int main(int argc, char** argv)
{
    string imageRootDir = "data/SPAD/batch_";
    int numBatches = 1;
    string layers[NUM_LAYERS]           = {"conv0"   ,"conv1"   , "conv2"   , "conv3"};
    string input_file_names[NUM_LAYERS] = {"spadfile","ds1out"  , "ds2out"  , "ds3out"};
    int layers_type[NUM_LAYERS]         = {CONV3D,    CONV3D,     CONV3D,     CONV3D};
    vector<float *> dma_input_vec;
    vector<float *> gold_outputs_vec;
    //static float dma_in[
    //        MAX_BATCH * (MAX_WEIGHT_SIZE + 5 * MAX_OUTPUT_CHANNELS + MAX_CONV_INPUT + MAX_CONV_OUTPUT)];
    int max_allocated;
    //MAX_BATCH * (MAX_WEIGHT_SIZE + 5 * MAX_OUTPUT_CHANNELS + MAX_CONV_INPUT + MAX_CONV_OUTPUT);

    std::cout << "Starting Test with " << numBatches << " batches" <<  endl;
    for (int l = 0; l < NUM_LAYERS; l++) {
        int middle_layer_count = 0;
        dma_input_vec.clear();
        float total_error = 0.0;
        vector <map<string, int>> batch_layer_params;
        // handeling middle layers
        while ((layers[l] == "conv0" || layers[l] == "conv1" || layers[l] == "conv2") && middle_layer_count <= 6) {
            ostringstream ss,mlc;
            mlc << middle_layer_count;
            string layer = layers[l]+ "." + mlc.str();
            string layer_prv = (l == 0) ? layers[0] : layers[l-1];
            // read all the batches params
            batch_layer_params = readBatchParams(imageRootDir, numBatches, layer);
            // if this fist of layer allocate memory, pointers will be in dma_input_vec
            if (middle_layer_count == 0 )
                max_allocated = allocate_memory(dma_input_vec,batch_layer_params, numBatches, layers_type[l]);
            // read inputs into allocated memory
            if (readInputBatchesWithNorm(imageRootDir, dma_input_vec, batch_layer_params, numBatches, layers[l], middle_layer_count, layer_prv,
                                         max_allocated, layers_type[l], input_file_names[l], false)) return 1;
            // start calculation
            auto start = chrono::system_clock::now();
            for (int i = 0; i < numBatches; i++) {
                ostringstream ss;
                ss << i;
#ifdef PRINT
                cout << "Running batch" << i << endl;
#endif
                string imageDir = imageRootDir + ss.str() + "/" + layer;
                std::cout << "ImageDir is " << imageDir << endl;
                if (run_single_test_bytype(imageDir, batch_layer_params[i], dma_input_vec[i], gold_outputs_vec[i], layers_type[l] ) != 0)
                    return 1;
            }
            middle_layer_count+=3;
            auto end = chrono::system_clock::now();
            auto elapsed = end - start;
            std::cout << "Computation for layer"<< layer << " took  " << chrono::duration_cast<chrono::milliseconds>(elapsed).count() << " ms"
                      << endl;
        }

        if (readOutputBatches(imageRootDir, batch_layer_params, numBatches, layers[l] , MAX_BATCH*CONV3D_MAX_CONV_OUTPUT,
                              gold_outputs_vec, layers_type[l]))
            return 1;
        float avg_error = get_mean_squared_error_and_write_file(dma_input_vec, gold_outputs_vec, numBatches,
                                                                batch_layer_params, imageRootDir, layers[l], layers_type[l]);

        std::cout << "Mean Square Error " << avg_error << endl;
        std::cout << "DONE" << std::endl;
        for (int d = 0; d < dma_input_vec.size(); d++) // clearing memroy allocated for next layer
            delete [] dma_input_vec[d];
    }
    return 0;
}

