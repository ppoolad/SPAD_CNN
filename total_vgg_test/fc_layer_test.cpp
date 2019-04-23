#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <map>
//#include <string>
#include <sstream>
#include "fc_layer.h"
#include "constants.h"
#include <assert.h>
#include <chrono>
#include "util/shared.h"

#define HW_CTRL_ADDR 0x00010000

using namespace std;

#define PRINT

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
    	  cout << "Read Error in myRead" << endl;
          retval = 1;
      }
    } else {
      cerr << "Desired dimensions too large: " << read_alloc << " > " << max_alloc << "\n";
      retval = 1;
    }
    in_file.close();
  }
  else
    cerr << "Couldn't open file: " << fname << endl;

  //if (retval) delete [] fptr;
  return retval;
}

static int run_single_test(string imageDir, map<string, int> layer_params, float *dma_input, float * gold_outputs)
{


  int num_outputs = layer_params["output_dim"];
  int num_inputs = layer_params["input_dim"];
  int num_weights = layer_params["input_dim"]*layer_params["output_dim"];
  int num_biases = layer_params["output_dim"];

  // very basic input checking
  if (layer_params["input_dim"] > FC_MAX_INPUT_SIZE ||
      layer_params["output_dim"] > FC_MAX_OUTPUT_SIZE ||
      layer_params["batch_size"] > MAX_BATCH)
  {
    cerr << "Problem with layer params\n";
    return 1;
  } else {
    int b = layer_params["batch_size"];
    int er = layer_params["enable_relu"];

#ifdef PRINT
    cout << "Begin Test\n"
       << "Batch Size: " << b << endl
       << "Num Inputs: " << num_inputs << endl
       << "Num Outputs: " << num_outputs << endl
       << "Enable ReLU: " << er << endl;
#endif

    // Run Accelerator
    #ifdef HW_TEST
    hw_fc_layer(HW_CTRL_ADDR, dma_input, 0,
                sizeof(float)*(b*num_inputs+num_biases + num_weights),
                b, num_inputs, num_outputs, er);
    #else
    fc_layer(dma_input, 0, sizeof(float)*(b*num_inputs+num_biases + num_weights),
             b, num_inputs, num_outputs, er);
    #endif
   

  }

  return 0;
}


int run_fc(string prevLayer, string _layer, int numBatches)
{

  string imageRootDir = "data/vgg_batches/batch_";
  //int numBatches = 10;
  string layer = _layer;
  string prev_layer = prevLayer;
  string feed = "/dma_out";
  if(prevLayer.empty()){
    prev_layer = layer;
    feed = "dma_in";
  }

  string imageDir, imageDir_current;
  ostringstream ss;
  cout << "Starting Test with " << numBatches << " batches" <<  endl;

  vector<map<string, int> > batch_layer_params = readBatchParams(imageRootDir, numBatches, layer);
  vector<float *> dma_input_vec;
  vector<float *> gold_outputs_vec;
  if(prevLayer.empty()){
    if(readInputBatches(imageRootDir, batch_layer_params, numBatches, layer, FC_MAX_WEIGHT_SIZE+FC_MAX_OUTPUT_SIZE+MAX_BATCH*FC_MAX_INPUT_SIZE+MAX_BATCH*FC_MAX_OUTPUT_SIZE, dma_input_vec, FC))
      return 1;
  }else{
    for(int i = 0; i < numBatches; i++)
    {
      static float dma_in[FC_MAX_WEIGHT_SIZE+FC_MAX_OUTPUT_SIZE+MAX_BATCH*FC_MAX_INPUT_SIZE+MAX_BATCH*FC_MAX_OUTPUT_SIZE];
      
      float * ptr = dma_in;
      ss.str("");
        ss << i;
      imageDir = imageRootDir + ss.str() + "/" + prevLayer;
      imageDir_current = imageRootDir + ss.str() + "/" + layer;
  		int size = batch_layer_params[i]["input_dim"]*batch_layer_params[i]["output_dim"]+
	               batch_layer_params[i]["output_dim"] +
	               batch_layer_params[i]["batch_size"]*batch_layer_params[i]["input_dim"];

      int wsize = batch_layer_params[i]["input_dim"]*batch_layer_params[i]["output_dim"];
      int bsize = batch_layer_params[i]["output_dim"];
      int isize = batch_layer_params[i]["batch_size"]*batch_layer_params[i]["input_dim"];
      string fname;
      /*Reading weights*/
      if (myreadFile(imageDir_current + "/weights", ptr, wsize, FC_MAX_WEIGHT_SIZE )) {
        cout << "Read Error";
        return 1;
      }

      ptr += wsize;
      /*Reading Biases*/
      if (myreadFile(imageDir_current + "/biases", ptr, bsize, FC_MAX_OUTPUT_SIZE )) {
        cout << "Read Error";
        return 1;
      }
      /*Reading Inputs*/
      ptr += bsize;
      if (myreadFile(imageDir + feed, ptr, isize, MAX_BATCH*FC_MAX_INPUT_SIZE )) {
        cout << "Read Error";
        return 1;
      }
      dma_input_vec.push_back(dma_in);
      string outdir = imageRootDir + ss.str() + "/" + layer + "/" +"created_dma_in";
      ofstream myFile(outdir.c_str(), ios::out | ios::binary);
      myFile.write((char *)dma_in, size*sizeof(float));    
      myFile.close();

    }
  }


  if(readOutputBatches(imageRootDir, batch_layer_params, numBatches, _layer, MAX_BATCH*FC_MAX_OUTPUT_SIZE, gold_outputs_vec, FC)) return 1;

  auto start = chrono::system_clock::now(); 
  for(int i=0; i<numBatches; i++){
    
    ostringstream ss;
    ss << i;
   
#ifdef PRINT
    cout << "Running batch" << i << endl;
#endif
    imageDir = imageRootDir + ss.str() + "/" + layer;
    cout << "ImageDir is " << imageDir << endl;  
    if(run_single_test(imageDir, batch_layer_params[i], dma_input_vec[i], gold_outputs_vec[i])!=0)
	return 1;
  }
  auto end = chrono::system_clock::now(); 
  auto elapsed = end - start;

  float avg_error = get_mean_squared_error_and_write_file(dma_input_vec, gold_outputs_vec, numBatches, batch_layer_params, imageRootDir, layer, FC);
  
  cout << "Mean Square Error " << avg_error << endl;
  cout << "Computation took  " << chrono::duration_cast<chrono::milliseconds> (elapsed).count() << " ms" << endl;
  std::cout << "DONE" << std::endl;
  return 0;
}

