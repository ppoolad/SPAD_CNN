#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <sstream>
#include <cmath>
#include <stdint.h>
#include <assert.h>

#include "shared.h"

/* ltoh: little to host */
/* htol: little to host */
#if __BYTE_ORDER == __LITTLE_ENDIAN
#  define ltohl(x)       (x)
#  define ltohs(x)       (x)
#  define htoll(x)       (x)
#  define htols(x)       (x)
#elif __BYTE_ORDER == __BIG_ENDIAN
#  define ltohl(x)     __bswap_32(x)
#  define ltohs(x)     __bswap_16(x)
#  define htoll(x)     __bswap_32(x)
#  define htols(x)     __bswap_16(x)
#endif

    #define PRINT

using namespace std;


std::map<std::string, int> readParams(const std::string fname)
{
  std::map<std::string, int> params;
  ifstream in_file(fname.c_str(), ios::in);
  std::string key, svalue;
  int ivalue;

  // Warning: don't do much error checking
  while (in_file >> key)
  {
    if (!key.compare("type")) {
      in_file >> svalue;
      if (!svalue.compare("Convolution"))
        params[key] = Convolution;
      else if (!svalue.compare("InnerProduct"))
        params[key] = InnerProduct;
      else if (!svalue.compare("Pooling"))
        params[key] = Pooling;
      else if (!svalue.compare("TransConvolution"))
        params[key] = CONV3DT;
      else {
        cerr << "Invalid Layer Type " << svalue << " !\n";
        return std::map<std::string, int>();
      }
    } else if (!key.compare("name")) { 
      in_file >> svalue;
//      std::cout << "Parsing " << svalue << std::endl;
    }
    else {
       in_file >> ivalue;
       params[key] = ivalue;
    }
  }
  in_file.close();
 // std::cout << "DONE\n";
  return params;
}

int readRawFile(const string fname,
                              float *& fptr,
                              const int read_alloc,
                              const int max_alloc)
{
  int retval = 0;

  std::cout << "Reading: " << fname << " size: " << read_alloc << std::endl;
  ifstream in_file(fname.c_str(), ios::in | ios::binary);
  if (in_file.is_open())
  {
    fptr = new float[max_alloc];
    if (read_alloc <= max_alloc) {
      if (!in_file.read(reinterpret_cast<char*>(&fptr[0]), sizeof(float)*read_alloc))
      {
    	  cout << "Read Error" << endl;
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

  if (retval) delete [] fptr;
  return retval;
}

std::vector <int> readFile(const string fname,
                     float *& fptr,
                     const int max_alloc)
{
  int* read_sizes = NULL;

  std::vector<int> read_sizes_vec;
  ifstream in_file(fname.c_str(), ios::in | ios::binary);
  if (in_file.is_open())
  {
    // Read header
    int read_dims;
    in_file.read(reinterpret_cast<char*>(&read_dims), sizeof(int));
    fptr = new float[max_alloc];
    read_sizes = new int[read_dims];

    // Read dimension data
    in_file.read(reinterpret_cast<char*>(read_sizes),
                 sizeof(int)*read_dims);

    // Read the array
    // Use a hack to read dma data:
    int read_alloc = 1;
    if (read_dims == 9) {
    	cout << "Reading DMA Input" << endl;
    	read_alloc = read_sizes[0]*read_sizes[1] *   // weights
    			     read_sizes[2]*read_sizes[3] +
    			     read_sizes[4] +                 // biases
					 read_sizes[5]*read_sizes[6] *   // inputs
					 read_sizes[7]*read_sizes[8];
    	for (int i = 0; i < read_dims; i++)
    	  read_sizes_vec.push_back(read_sizes[i]);
    }
    else
    {
    	cout << "READING NORMAL INPUT" << endl;
      read_alloc = 1;
      for (int i = 0; i < read_dims; i++){
    	cout << i << " = " << read_sizes[i] << endl;
        read_alloc *= read_sizes[i];
        read_sizes_vec.push_back(read_sizes[i]);
      }
    }
    cout << "READ ALLOC " << read_alloc << " for " << read_dims << endl;
    if (read_alloc <= max_alloc) {
      if (!in_file.read(reinterpret_cast<char*>(&fptr[0]), sizeof(float)*read_alloc))
      {
    	  cout << "Read Error" << endl;
    	  delete [] read_sizes;
    	  delete [] fptr;
    	  read_sizes = NULL;
      }
    }
    else
    {
      cerr << "Desired dimensions too large: " << read_alloc << " > " << max_alloc << "\n";
      delete [] read_sizes;
      delete [] fptr;
      read_sizes = NULL;
    }

    for (int i = 0; i < 10; i++)
    	cout << fptr[i] << endl;
    in_file.close();
  }
  else
    cerr << "Couldn't open file: " << fname << endl;

  delete [] read_sizes;
  return read_sizes_vec;
}

vector<map<string, int> > readBatchParams(string imageRootDir, int numBatches, string layer){

   vector<map<string,int> > retVec;
   ostringstream ss;
   for(int i=0; i<numBatches; i++){
        ss.str("");
    	ss << i;
  	string imageDir = imageRootDir + ss.str() + "/" + layer;
  	map<string, int> layer_params = readParams(imageDir + "/params");
	retVec.push_back(layer_params);
   }

   return retVec;
}

int readInputBatches(string imageRootDir, vector<map<string, int> > batch_layer_params, int numBatches, string layer, const int max_alloc, vector<float *> &ptr, int layerType){
  float * dma_input;
  ostringstream ss;
  

  // Read inputs
  // Inputs are packed together as weights, biases and input values
  // Allocate enough space for outputs

  for(int i=0; i<numBatches; i++){
        ss.str("");
    	ss << i;
	int size;
        if(layerType == CONV){
  		size = batch_layer_params[i]["input_dim"]*batch_layer_params[i]["output_dim"]*
                  batch_layer_params[i]["kernel_size"]*batch_layer_params[i]["kernel_size"]+
                  batch_layer_params[i]["output_dim"]+
                  batch_layer_params[i]["input_dim"]*batch_layer_params[i]["input_width"]*
                  batch_layer_params[i]["input_height"]*batch_layer_params[i]["batch_size"];
        }
	else if(layerType == FC){
  		size = batch_layer_params[i]["input_dim"]*batch_layer_params[i]["output_dim"]+
	               batch_layer_params[i]["output_dim"] +
	               batch_layer_params[i]["batch_size"]*batch_layer_params[i]["input_dim"];

        }
    else{ 
        size = batch_layer_params[i]["input_dim"]*batch_layer_params[i]["input_width"]*
                  batch_layer_params[i]["input_height"]*batch_layer_params[i]["batch_size"];
    }
  	string imageDir = imageRootDir + ss.str() + "/" + layer;
  	if (readRawFile(imageDir + "/dma_in",
                  dma_input,
                  size,
		  max_alloc
                 ))
    	return 1;
	ptr.push_back(dma_input);
  }
  return 0;

}

int readOutputBatches(string imageRootDir, vector<map<string, int> > batch_layer_params, int numBatches, string layer, const int max_alloc, vector <float *> &ptr, int layerType){
  float * gold_outputs;
  ostringstream ss;
  

  // Read inputs
  // Inputs are packed together as weights, biases and input values
  // Allocate enough space for outputs

  int size;
  for(int i=0; i<numBatches; i++){
        ss.str("");
    	ss << i;
  	string imageDir = imageRootDir + ss.str() + "/" + layer;
  	if(layerType == CONV || layerType == POOL ){
  	  size =      batch_layer_params[i]["output_dim"]*batch_layer_params[i]["output_width"]*
                  batch_layer_params[i]["output_height"]*batch_layer_params[i]["batch_size"];
  	}
     else if(layerType == CONV3D || layerType == CONV3DT ){
          size =  batch_layer_params[i]["output_channel"]*
                  batch_layer_params[i]["output_dim"]*batch_layer_params[i]["output_width"]*
                  batch_layer_params[i]["output_height"]*batch_layer_params[i]["batch_size"];
      }
	else{ //POOL
  	  size =     batch_layer_params[i]["output_dim"]*
  	             batch_layer_params[i]["batch_size"];
        }
  	// Read gold outputs
  	if (readRawFile(imageDir + "/testoutput",// "/up3out",// this is for test change it to "out",
                  gold_outputs,
                  size,
                  max_alloc))
    	return 1;
	ptr.push_back(gold_outputs);
    	gold_outputs += size;
  }
  return 0;

}

float get_mean_squared_error_and_write_file(vector<float *> mem, vector <float *> golden_output, int numBatches, vector<map<string,int> >batch_layer_params, string imageRootDir, string layer, int layerType){
  

  float total = 0.0f;

  ostringstream ss;

  int totalNumOutputs = 0;

  for(int i=0; i<numBatches; i++){
    int b = batch_layer_params[i]["batch_size"];

    int num_inputs;
    int num_biases;
    int num_weights;
    int num_outputs;
    int num_bnormpars;
    float * outputs;
    if(layerType == CONV){
    	num_inputs = batch_layer_params[i]["input_dim"]*batch_layer_params[i]["input_width"]*
        batch_layer_params[i]["input_height"];
    	num_biases = batch_layer_params[i]["output_dim"];
    	num_weights = batch_layer_params[i]["input_dim"]*batch_layer_params[i]["output_dim"]*batch_layer_params[i]["kernel_size"]*batch_layer_params[i]["kernel_size"];
    	num_outputs = batch_layer_params[i]["output_dim"]*batch_layer_params[i]["output_width"]*batch_layer_params[i]["output_height"];
    	totalNumOutputs += b*num_outputs;
    	outputs = mem[i] + b*num_inputs+num_biases+num_weights;
    }
    else if(layerType == CONV3D || layerType == CONV3DT){
    	num_inputs = batch_layer_params[i]["input_dim"]*batch_layer_params[i]["input_width"]*
        batch_layer_params[i]["input_height"]*batch_layer_params[i]["input_channel"];
    	num_biases = batch_layer_params[i]["output_channel"];
        num_bnormpars = batch_layer_params[i]["output_channel"]*4;
    	num_weights = batch_layer_params[i]["input_channel"]*batch_layer_params[i]["output_channel"]*batch_layer_params[i]["kernel_size"]*batch_layer_params[i]["kernel_size"]*batch_layer_params[i]["kernel_size"];
    	num_outputs = batch_layer_params[i]["output_dim"]*batch_layer_params[i]["output_width"]*batch_layer_params[i]["output_height"]*batch_layer_params[i]["output_channel"];
    	totalNumOutputs += b*num_outputs;
    	outputs = mem[i] + b*num_inputs+num_biases+num_weights+num_bnormpars;
    }
    else if(layerType == FC){
    	num_inputs = batch_layer_params[i]["input_dim"];
    	num_biases = batch_layer_params[i]["output_dim"];
    	num_weights = batch_layer_params[i]["input_dim"]*batch_layer_params[i]["output_dim"];
    	num_outputs = batch_layer_params[i]["output_dim"];
    	totalNumOutputs += b*num_outputs;
    	outputs = mem[i] + b*num_inputs+num_biases+num_weights;
      //std::cout << "id: " << num_inputs << " b: "<< num_biases << " nw: " << num_weights << " od: " << num_outputs;

    }
    else { //POOL
    	num_inputs = batch_layer_params[i]["input_dim"]*batch_layer_params[i]["input_width"]*
        batch_layer_params[i]["input_height"];
    	num_outputs = batch_layer_params[i]["output_dim"]*batch_layer_params[i]["output_width"]*batch_layer_params[i]["output_height"];
    	totalNumOutputs += b*num_outputs;
    	outputs = mem[i] + b*num_inputs;

    }
    for (int j = 0; j < b*num_outputs; j++)
    {
      float err = fabs(outputs[j] - golden_output[i][j]);
#ifdef PRINT
      int b1 = batch_layer_params[i]["output_dim"];
      int w4 = batch_layer_params[i]["output_width"];
      int h3 = batch_layer_params[i]["output_height"];
      int c2 = batch_layer_params[i]["output_channel"];
      if (err > 0.1) {
          int w4_2 = j%(w4);
          int h3_2 = (j/w4) % (h3);
          int c2_2 = (j/(w4*h3)) % c2;
          std::cout << "output[" << c2_2 << "][" << h3_2 << "][" << w4_2 << "] = "<< outputs[j] << "VS" << golden_output[i][j]<< '\n';
      }
#endif
      
      total += err*err;
      //printf("HW: %.6f GOLD:%.6f\n",fabs(outputs[j]),golden_output[i][j] );
    }
	//std::cout << "hw: "<< fabs(outputs[1] << "gold "<<golden_output[i][1]<<"\n";
	
    ss.str("");
    ss << i;
    string imageDir = imageRootDir + ss.str() + "/" + layer + "/dma_out"; // not for test "dma_out"
    char * buffer = (char *)outputs;
    ofstream myFile(imageDir.c_str(), ios::out | ios::binary);
    myFile.write(buffer, b*num_outputs*sizeof(float));    
    myFile.close();
  }
    return total/(totalNumOutputs);

}

void write_int(volatile void* map_base, int offset, int value)
{
  volatile void* virt_addr = (volatile void*)((char*)map_base + offset); 
  *((uint32_t *) virt_addr) = htoll(value);
}
int read_int(volatile void* map_base, int offset)
{
  volatile void* virt_addr = (volatile void*)((char*)map_base + offset); 
  return ltohl(*((uint32_t *) virt_addr));
}

void timespec_sub(struct timespec *t1, const struct timespec *t2)
{
  assert(t1->tv_nsec >= 0);
  assert(t1->tv_nsec < 1000000000);
  assert(t2->tv_nsec >= 0);
  assert(t2->tv_nsec < 1000000000);
  t1->tv_sec -= t2->tv_sec;
  t1->tv_nsec -= t2->tv_nsec;
  if (t1->tv_nsec >= 1000000000)
  {
    t1->tv_sec++;
    t1->tv_nsec -= 1000000000;
  }
  else if (t1->tv_nsec < 0)
  {
    t1->tv_sec--;
    t1->tv_nsec += 1000000000;
  }
}

////////// added functions ////////
int readRawFileNoAlloc(const string fname,
                              float * fptr,
                              const int read_alloc,
                              const int max_alloc)
{
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
  else
    cerr << "Couldn't open file: " << fname << endl;

  //if (retval) delete [] fptr;
  return retval;
}

int readInputBatchesWithNorm(string imageRootDir,vector<float *>dma_input_vec, vector<map<string, int> > batch_layer_params, int numBatches, string layer, int layer_index, string layer_prv, const int max_alloc, int layerType, bool ReadinputFlag){
  ostringstream ss, sindex, sindexplus;
  // Read inputs
  // Inputs are packed together as weights, biases and input values from different places
  // Allocate enough space for outputs
  sindex.str("");
  sindexplus.str("");
  sindex << layer_index;
  sindexplus << (layer_index+1);

  for(int i=0; i<numBatches; i++){
        ss.str("");
        ss << i;
        int input_size;
        int weight_size;
        int bias_size;
        float * dma_input = dma_input_vec[i];
        string imageDir     = imageRootDir + ss.str()+ "/" + layer+ "." + sindex.str();
        string imageDir_plus= imageRootDir + ss.str()+ "/" + layer+ "." + sindexplus.str();
        string imageDir_prv = imageRootDir + ss.str() + "/" + layer_prv;
            if(layerType == CONV){
                  weight_size = batch_layer_params[i]["input_dim"]*batch_layer_params[i]["output_dim"]*
                                batch_layer_params[i]["kernel_size"]*batch_layer_params[i]["kernel_size"];
                  bias_size   = batch_layer_params[i]["output_dim"];
                  input_size  = batch_layer_params[i]["input_dim"]*batch_layer_params[i]["input_width"]*
                                batch_layer_params[i]["input_height"]*batch_layer_params[i]["batch_size"];

                  if (readRawFile(imageDir + "/weights", dma_input, weight_size, max_alloc))
                    return 1;
                  if (readRawFileNoAlloc(imageDir + "/biases", dma_input, bias_size, weight_size))
                    return 1;
                  if (readRawFileNoAlloc(imageDir_prv + "/dma_out", dma_input, input_size, weight_size+bias_size))
                    return 1;
                  printf("ptr adr %p \n", dma_input);
            }
            else if(layerType == CONV3D || layerType == CONV3DT){
                weight_size = batch_layer_params[i]["input_channel"]*batch_layer_params[i]["output_channel"]*
                              batch_layer_params[i]["kernel_size"]*batch_layer_params[i]["kernel_size"]*batch_layer_params[i]["kernel_size"];
                bias_size   = batch_layer_params[i]["output_channel"];
                input_size  = batch_layer_params[i]["input_dim"]*batch_layer_params[i]["input_width"]*batch_layer_params[i]["input_height"]*
                              batch_layer_params[i]["input_channel"]*
                              batch_layer_params[i]["batch_size"];

                if (readRawFileNoAlloc(imageDir + ".weight", dma_input, weight_size, max_alloc))
                    return 1;
                dma_input += weight_size;
                if (readRawFileNoAlloc(imageDir + ".bias", dma_input, bias_size, max_alloc))
                    return 1;
                dma_input += bias_size;
                if (readRawFileNoAlloc(imageDir_plus + ".running_mean", dma_input, input_size, weight_size+bias_size))
                    return 1;
                dma_input += bias_size;
                if (readRawFileNoAlloc(imageDir_plus + ".running_var", dma_input, input_size, weight_size+bias_size))
                    return 1;
                dma_input += bias_size;
                if (readRawFileNoAlloc(imageDir_plus + ".weight", dma_input, input_size, weight_size+bias_size))
                    return 1;
                dma_input += bias_size;
                if (readRawFileNoAlloc(imageDir_plus + ".bias", dma_input, input_size, weight_size+bias_size))
                    return 1;
                dma_input += bias_size;
                if (ReadinputFlag){
                    if (readRawFileNoAlloc(imageDir_prv + ".dma_out", dma_input, input_size, weight_size+bias_size))
                        return 1;
                }
                printf("ptr adr %p \n", dma_input);
            }
      }
      return 0;

}

int allocate_memory( vector<float *> &dma_input_vec, vector<map<string, int> > batch_layer_params, int numBatches, int layerType){
    int output_size, input_size ,weight_size, bias_size, size;
    ostringstream ss;

    for(int i=0; i<numBatches; i++) {
        ss.str("");
        ss << i;
        if (layerType == CONV)
        {
            weight_size = batch_layer_params[i]["input_dim"] * batch_layer_params[i]["output_dim"] *
                          batch_layer_params[i]["kernel_size"] * batch_layer_params[i]["kernel_size"];
            bias_size = batch_layer_params[i]["output_dim"];
            input_size = batch_layer_params[i]["input_dim"] * batch_layer_params[i]["input_width"] *
                         batch_layer_params[i]["input_height"] * batch_layer_params[i]["batch_size"];
            output_size = batch_layer_params[i]["output_dim"] * batch_layer_params[i]["output_width"] *
                          batch_layer_params[i]["output_height"] * batch_layer_params[i]["batch_size"];

            size = weight_size + bias_size + input_size + output_size;

        } else if (layerType == CONV3D || layerType == CONV3DT) {
            weight_size = batch_layer_params[i]["input_channel"] * batch_layer_params[i]["output_channel"] *
                          batch_layer_params[i]["kernel_size"] * batch_layer_params[i]["kernel_size"] *
                          batch_layer_params[i]["kernel_size"];
            bias_size = batch_layer_params[i]["output_channel"];
            input_size = batch_layer_params[i]["input_dim"] * batch_layer_params[i]["input_width"] *
                         batch_layer_params[i]["input_height"] *
                         batch_layer_params[i]["input_channel"] *
                         batch_layer_params[i]["batch_size"];
            output_size= batch_layer_params[i]["output_dim"] * batch_layer_params[i]["output_width"] *
                         batch_layer_params[i]["output_height"] *
                         batch_layer_params[i]["output_channel"] *
                         batch_layer_params[i]["batch_size"];

            size = weight_size + 5*bias_size + input_size + output_size;
        }

        float* ptr = new float[size];
        dma_input_vec.push_back(ptr);
    }

    return size;
}