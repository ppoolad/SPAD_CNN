#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <map>
#include <string>
#include "conv_trans3d_layer.h"
#include "util/shared.h"
#include <sstream>
#include <chrono>
//#include "constants.h"
#define HW_CTRL_ADDR 0x00010000

using namespace std;


//#define PRINT


static int myreadFile(const string fname,
                              float * fptr,
                              const int read_alloc,
                              const int max_alloc){

  int retval = 0;
  std::cout << "Reading: " << fname << " size: " << read_alloc << std::endl;
  ifstream in_file(fname.c_str(), ios::in | ios::binary);
  if (in_file.is_open())
  {
    //fptr = new float[max_alloc];
    if (read_alloc <= max_alloc) {
      if (!in_file.read(reinterpret_cast<char*>(fptr), sizeof(float)*read_alloc))
      {
    	  std::cout << "Read Error in myRead" << endl;
          retval = 1;
      }
    } else {
      cerr << "Desired dimensions too large: " << read_alloc << " > " << max_alloc << "\n";
      retval = 1;
    }
    in_file.close();
  }
  else{
    cerr << "Couldn't open file: " << fname << endl;
    retval = 1;
  }
  //if (retval) delete [] fptr;
  return retval;
}



static int run_single_test(string imageDir, map<string, int> layer_params, float * &dma_input, float * gold_outputs){
  
  


  int num_outputs = layer_params["output_channel"]*layer_params["output_dim"]*layer_params["output_width"]*
                    layer_params["output_height"];
  int num_inputs = layer_params["input_channel"]*layer_params["input_dim"]*layer_params["input_width"]*
                   layer_params["input_height"];
  int num_weights = layer_params["input_channel"]*layer_params["output_channel"]*
                    layer_params["kernel_size"]*layer_params["kernel_size"]*layer_params["kernel_size"];
  int num_biases = layer_params["output_channel"];
  int num_bnormparams = layer_params["output_channel"]*4;

  // very basic input checking
  if (layer_params["input_dim"] > MAX_INPUT_DIMS ||
      layer_params["input_width"] > MAX_INPUT_WIDTH ||
      layer_params["input_height"] > MAX_INPUT_WIDTH ||
      layer_params["output_dim"] > MAX_OUTPUT_DIMS ||
      layer_params["output_width"] > MAX_OUTPUT_WIDTH ||
      layer_params["output_height"] > MAX_OUTPUT_WIDTH ||
      layer_params["batch_size"] > MAX_BATCH)
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
    int oc = layer_params["output_channel"];
    int ic = layer_params["input_channel"];
    int pad = layer_params["pad"];
    
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
    hw_conv_trans3d_layer(HW_CTRL_ADDR, dma_input, sizeof(float)*(num_biases + num_weights + num_bnormparams), 0 ,sizeof(float)*(b*num_inputs+num_biases + num_weights + num_bnormparams),
                  b, od, ox, oy, oc, ic, id, ix, iy, s, k,1,1,1);
    #else
    std::cout << "od: " << od << " oy: " << oy << " ox: " << ox <<" id: "<<id << " iy: " << iy << " ix: " << ix << " k: " << k << " s: " << s << " pad: " << pad << std::endl;  
    conv_trans3d_layer(dma_input, sizeof(float)*(num_biases + num_weights + num_bnormparams), 0 ,sizeof(float)*(b*num_inputs+num_biases + num_weights + num_bnormparams),
               b, od, ox, oy, oc, ic, id, ix, iy, s, k,pad,1,1);
    #endif

  }

  return 0;

}


int main(int argc, char** argv)
{

  string imageRootDir = "data/SPAD/batch_";
  int numBatches = 1;
  string layer = "up3";
  // string prev_layer = prevLayer;
  string feed = "/dma_in";
  // if(prevLayer.empty()){
  //   prev_layer = layer;
  //   feed = "dma_in";
  // }

  string imageDir, imageDir_current;
  ostringstream ss;
  std::cout << "Starting Test with " << numBatches << " batches" <<  endl;

  vector<map<string, int> > batch_layer_params = readBatchParams(imageRootDir, numBatches, layer);
  vector<float *> dma_input_vec;
  vector<float *> gold_outputs_vec;
  // if(prevLayer.empty()){
  //   if(readInputBatches(imageRootDir, batch_layer_params, numBatches, layer, FC_MAX_WEIGHT_SIZE+FC_MAX_OUTPUT_SIZE+MAX_BATCH*FC_MAX_INPUT_SIZE+MAX_BATCH*FC_MAX_OUTPUT_SIZE, dma_input_vec, FC))
  //     return 1;
  // }else{
    for(int i = 0; i < numBatches; i++)
    {
      //static float dma_in[MAX_WEIGHT_SIZE+5*MAX_OUTPUT_CHANNELS+MAX_CONV_INPUT+MAX_CONV_OUTPUT];
      float* dma_in = new float[MAX_WEIGHT_SIZE+5*MAX_OUTPUT_CHANNELS+MAX_CONV_INPUT+MAX_CONV_OUTPUT];
      float * ptr = dma_in;
      ss.str("");
        ss << i;
      imageDir = imageRootDir + ss.str() + "/" + layer;
      imageDir_current = imageRootDir + ss.str() + "/" + layer;
  		int size = batch_layer_params[i]["kernel_size"]*batch_layer_params[i]["kernel_size"]*batch_layer_params[i]["kernel_size"]+
	               batch_layer_params[i]["output_channel"] + 4*batch_layer_params[i]["output_channel"]+
	               batch_layer_params[i]["input_channel"]*batch_layer_params[i]["batch_size"]*batch_layer_params[i]["input_dim"]*batch_layer_params[i]["input_height"]*batch_layer_params[i]["input_width"];

      int wsize = batch_layer_params[i]["output_channel"]*batch_layer_params[i]["input_channel"]*batch_layer_params[i]["kernel_size"]*batch_layer_params[i]["kernel_size"]*batch_layer_params[i]["kernel_size"];
      int bsize = batch_layer_params[i]["output_channel"];
      int isize = batch_layer_params[i]["input_channel"]*batch_layer_params[i]["batch_size"]*batch_layer_params[i]["input_dim"]*batch_layer_params[i]["input_height"]*batch_layer_params[i]["input_width"];
      string fname;
      /*Reading weights*/
      if (myreadFile(imageDir_current + "/up3.0.weight", ptr, wsize, MAX_WEIGHT_SIZE )) {
        std::cout << "Read Error";
        return 1;
      }

      ptr += wsize;
      /*Reading Biases*/
      if (myreadFile(imageDir_current + "/up3.0.bias", ptr, bsize, MAX_CONV_OUTPUT )) {
        std::cout << "Read Error";
        return 1;
      }
      
      ptr += bsize;

      /*reading bnorm params*/
      if (myreadFile(imageDir_current + "/up3.1.running_mean", ptr, bsize, MAX_OUTPUT_CHANNELS )) {
        std::cout << "Read Error";
        return 1;
      }
      ptr += bsize;
      if (myreadFile(imageDir_current + "/up3.1.running_var", ptr, bsize, MAX_OUTPUT_CHANNELS )) {
        std::cout << "Read Error";
        return 1;
      }
      ptr += bsize;
      if (myreadFile(imageDir_current + "/up3.1.weight", ptr, bsize, MAX_OUTPUT_CHANNELS )) {
        std::cout << "Read Error";
        return 1;
      }
      ptr += bsize;
      if (myreadFile(imageDir_current + "/up3.1.bias", ptr, bsize, MAX_OUTPUT_CHANNELS )) {
        std::cout << "Read Error";
        return 1;
      }      

      ptr += bsize;
      /*Reading Inputs*/
      if (myreadFile(imageDir_current + "/conv3out", ptr, isize, 1*MAX_CONV_INPUT )) {
        std::cout << "Read Error";
        return 1;
      }
      dma_input_vec.push_back(dma_in);
      string outdir = imageRootDir + ss.str() + "/" + layer + "/" +"created_dma_in";
      ofstream myFile(outdir.c_str(), ios::out | ios::binary);
      myFile.write((char *)dma_in, size*sizeof(float));    
      myFile.close();

    }
  //}


  if(readOutputBatches("/up3out",imageRootDir, batch_layer_params, numBatches, layer, 1*MAX_CONV_OUTPUT, gold_outputs_vec, CONV3D)) return 1;

  auto start = chrono::system_clock::now(); 
  for(int i=0; i<numBatches; i++){
    
    ostringstream ss;
    ss << i;
   
#ifdef PRINT
    cout << "Running batch" << i << endl;
#endif
    imageDir = imageRootDir + ss.str() + "/" + layer;
    std::cout << "ImageDir is " << imageDir << endl;  
    if(run_single_test(imageDir, batch_layer_params[i], dma_input_vec[i], gold_outputs_vec[i])!=0)
	return 1;
  }
  auto end = chrono::system_clock::now(); 
  auto elapsed = end - start;

  float avg_error = get_mean_squared_error_and_write_file(dma_input_vec, gold_outputs_vec, numBatches, batch_layer_params, imageRootDir, layer, CONV3D);
  
  std::cout << "Mean Square Error " << avg_error << endl;
  std::cout << "Computation took  " << chrono::duration_cast<chrono::milliseconds> (elapsed).count() << " ms" << endl;
  std::cout << "DONE" << std::endl;
  return 0;
}

