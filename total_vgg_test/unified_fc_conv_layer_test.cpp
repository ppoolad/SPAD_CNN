#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <map>
//#include <string>
#include "conv_layer.h"
#include "constants.h"
#include "util/shared.h"
#include <sstream>
#include <chrono>

#define HW_CTRL_ADDR 0x00000000

using namespace std;


#define PRINT

#define CONV_LAYER 0
#define FC_LAYER 1

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

int run_single_test(string imageDir, map<string, int> layer_params, float * &dma_input, float * gold_outputs, int layer_type){
  
  if (layer_type == CONV) {

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
                  b, od, ox, oy, id, ix, iy, s, k,1);
    #else
    unified_fc_conv_layer(dma_input, 0, sizeof(float)*(b*num_inputs+num_biases + num_weights),
               b, od, ox, oy, id, ix, iy, s, k,1);
    #endif

  }
  }else if(layer_type == FC){

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
      int od = layer_params["output_dim"];
      int ox = 1;
      int oy = 1;
      int id = layer_params["input_dim"];
      int ix = 1;
      int iy = 1;
      int k = 1;
      int s = 1;
      int relu = layer_params["enable_relu"];

  #ifdef PRINT
      cout << "Begin Test\n"
        << "Layer Type: " << "FC" << endl
        << "Batch Size: " << b << endl
        << "Num Inputs: " << num_inputs << endl
        << "Num Outputs: " << num_outputs << endl
        << "Num Weights: " << num_weights << endl 
        << "Num Biases: " << num_biases << endl 
        << "Input Dimensions " << b << " x " << id << " x " << ix << " x " << iy << endl
        << "Output Dimensions " << b << " x " << od << " x " << ox << " x " << oy << endl;
  #endif

      // Run Accelerator
      #ifdef HW_TEST
      hw_conv_layer(HW_CTRL_ADDR, dma_input, 0,
                    sizeof(float)*(b*num_inputs+num_biases + num_weights),
                    b, od, ox, oy, id, ix, iy, s, k, relu);
      #else
      unified_fc_conv_layer(dma_input, 0, sizeof(float)*(b*num_inputs+num_biases + num_weights),
                b, od, ox, oy, id, ix, iy, s, k, relu);
      #endif

  }
  }else{
    cout << "arg error,should be fc or conv" << endl;
    return 1;
  }

  return 0;

}


int run_unified_fc_conv(string prevLayer, string _layer, int numBatches, int layer_type)
{

  //int layer_type = CONV_LAYER;
  // if (argc > 1) {
  //   if(std::string(argv[1]) == "fc"){
  //     layer_type = FC_LAYER;
  //   }else if(std::string(argv[1]) == "conv"){
  //     layer_type = CONV_LAYER;
  //   }else{
  //     cout<< "Unsupported layer type, use fc or conv"<< endl;
  //     return 1;
  //   }
  // }
  
  string imageRootDir = "data/vgg_batches/batch_";
  //int numBatches = 1;
  string layer = _layer;
  //if(layer_type == FC)layer = "fc6";
  string prev_layer = prevLayer;
  string feed = "/dma_out";
  if(prevLayer.empty()){
    prev_layer = layer;
    feed = "dma_in";
  }

  string imageDir, imageDir_current;
  ostringstream ss;
  //float total_error = 0.0;
  cout << "Reading Input for " << numBatches << " batches" <<  endl;

  vector<map<string, int> > batch_layer_params = readBatchParams(imageRootDir, numBatches, layer);
  vector<float *> dma_input_vec;
  vector<float *> gold_outputs_vec;
  if(prevLayer.empty()){  
    if(readInputBatches(imageRootDir, batch_layer_params, numBatches, layer, CONV_MAX_WEIGHT_SIZE+CONV_MAX_OUTPUT_DIMS+MAX_BATCH*CONV_MAX_CONV_INPUT+MAX_BATCH*CONV_MAX_CONV_OUTPUT, dma_input_vec, layer_type))
	    return 1;
  }else{
    if (layer_type == CONV) {
      for(int i = 0; i < numBatches; i++)
      {
        //static float dma_in[CONV_MAX_WEIGHT_SIZE+CONV_MAX_CONV_OUTPUT+MAX_BATCH*CONV_MAX_CONV_INPUT+MAX_BATCH*CONV_MAX_CONV_OUTPUT];
        
        float *dma_in = new float[CONV_MAX_WEIGHT_SIZE+CONV_MAX_CONV_OUTPUT+MAX_BATCH*CONV_MAX_CONV_INPUT+MAX_BATCH*CONV_MAX_CONV_OUTPUT];
        float * ptr = dma_in;
        ss.str("");
          ss << i;
        imageDir = imageRootDir + ss.str() + "/" + prevLayer;
        imageDir_current = imageRootDir + ss.str() + "/" + layer;
        int size =  batch_layer_params[i]["input_dim"]*batch_layer_params[i]["output_dim"]*
                    batch_layer_params[i]["kernel_size"]*batch_layer_params[i]["kernel_size"]+
                    batch_layer_params[i]["output_dim"]+
                    batch_layer_params[i]["input_dim"]*batch_layer_params[i]["input_width"]*
                    batch_layer_params[i]["input_height"]*batch_layer_params[i]["batch_size"];
        int wsize = batch_layer_params[i]["input_dim"]*batch_layer_params[i]["output_dim"]*
                    batch_layer_params[i]["kernel_size"]*batch_layer_params[i]["kernel_size"];
        int bsize = batch_layer_params[i]["output_dim"];
        int isize = batch_layer_params[i]["input_dim"]*batch_layer_params[i]["input_width"]*
                    batch_layer_params[i]["input_height"]*batch_layer_params[i]["batch_size"];
        //string fname;
        /*Reading weights*/
        if (myreadFile(imageDir_current + "/weights", ptr, wsize, CONV_MAX_WEIGHT_SIZE )) {
          cout << "Read Error";
          delete [] dma_in;
          return 1;
        }

        ptr += wsize;
        /*Reading Biases*/
        if (myreadFile(imageDir_current + "/biases", ptr, bsize, CONV_MAX_OUTPUT_DIMS )) {
          cout << "Read Error";
          delete [] dma_in;
          return 1;
        }
        /*Reading Inputs*/
        ptr += bsize;
        if (myreadFile(imageDir + feed, ptr, isize, MAX_BATCH*CONV_MAX_CONV_INPUT )) {
          cout << "Read Error";
          delete [] dma_in;
          return 1;
          
        }
        dma_input_vec.push_back(dma_in);
        string outdir = imageRootDir + ss.str() + "/" + layer + "/" +"created_dma_in";
        ofstream myFile(outdir.c_str(), ios::out | ios::binary);
        myFile.write((char *)dma_in, size*sizeof(float));    
        myFile.close();
        

      }
    }
    else if(layer_type == FC) {
      
      for(int i = 0; i < numBatches; i++)
      {
        //static float dma_in[FC_MAX_WEIGHT_SIZE+FC_MAX_OUTPUT_SIZE+MAX_BATCH*FC_MAX_INPUT_SIZE+MAX_BATCH*FC_MAX_OUTPUT_SIZE];
        float * dma_in = new  float[FC_MAX_WEIGHT_SIZE+FC_MAX_OUTPUT_SIZE+MAX_BATCH*FC_MAX_INPUT_SIZE+MAX_BATCH*FC_MAX_OUTPUT_SIZE];
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
          delete [] dma_in;
          return 1;
        }

        ptr += wsize;
        /*Reading Biases*/
        if (myreadFile(imageDir_current + "/biases", ptr, bsize, FC_MAX_OUTPUT_SIZE )) {
          cout << "Read Error";
          delete [] dma_in;
          return 1;
        }
        /*Reading Inputs*/
        ptr += bsize;
        if (myreadFile(imageDir + feed, ptr, isize, MAX_BATCH*FC_MAX_INPUT_SIZE )) {
          cout << "Read Error";
          delete [] dma_in;
          return 1;
        }
        dma_input_vec.push_back(dma_in);
        string outdir = imageRootDir + ss.str() + "/" + layer + "/" +"created_dma_in";
        ofstream myFile(outdir.c_str(), ios::out | ios::binary);
        myFile.write((char *)dma_in, size*sizeof(float));    
        myFile.close();
        //delete [] dma_in;
      }

    }
    else {
      cout << "Not supported layer type"<< endl;
      return 1;
    }
    



  }
  if(readOutputBatches(imageRootDir, batch_layer_params, numBatches, layer, MAX_BATCH*CONV_MAX_CONV_OUTPUT, gold_outputs_vec, layer_type)) return 1;


  cout << "Starting Test with " << numBatches << " batches" <<  endl;


  auto start = chrono::system_clock::now(); 
  for(int i=0; i<numBatches; i++){
    ss << i;
#ifdef PRINT
    cout << "Running batch" << i << endl;
#endif
    imageDir = imageRootDir + ss.str() + "/" + layer;
    
    if(run_single_test(imageDir, batch_layer_params[i], dma_input_vec[i], gold_outputs_vec[i], layer_type)!=0)
	return 1;
  }
  auto end = chrono::system_clock::now(); 
  auto elapsed = end - start;

  float avg_error = get_mean_squared_error_and_write_file(dma_input_vec, gold_outputs_vec, numBatches, batch_layer_params, imageRootDir, layer, layer_type);

  cout << "Mean Square Error " << avg_error << endl;
  cout << "Computation took  " << chrono::duration_cast<chrono::seconds> (elapsed).count() << " seconds" << endl;
  std::cout << "DONE" << std::endl;

  int milis = (int) chrono::duration_cast<chrono::milliseconds> (elapsed).count();
  return milis;
}

