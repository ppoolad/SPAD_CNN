#ifndef _SHARED_H
#define _SHARED_H

#include <string>
#include <vector>
#include <map>

//#define MAX_BATCH 1
#define NUM_BIAS_DIMENSIONS 1
#define CONV 0
#define FC 1
#define POOL 2
#define CONV3D 3
#define CONV3DT 4
float get_mean_squared_error_and_write_file(std::vector<float *> mem, std::vector <float *> golden_output, int numBatches, std::vector<std::map<std::string, int> >, std::string imageRootDir, std::string layer, int layerType  );
std::vector<std::map<std::string, int> > readBatchParams(std::string imageRootDir, int numBatches, std::string layer);
float get_mean_squared_error_and_write_file_of(std::vector<float *> mem, int output_offset, std::vector <float *> golden_output, int numBatches, std::vector<std::map<std::string, int> >, std::string imageRootDir, std::string layer, int layerType  );
std::vector<std::map<std::string, int> > readBatchParams(std::string imageRootDir, int numBatches, std::string layer);
int readInputBatches(std::string imageRootDir, std::vector<std::map<std::string, int> > batch_layer_params, int numBatches, std::string layer, const int max_alloc, std::vector<float *> &ptr, int layerType);
int readOutputBatches(std::string filename, std::string imageRootDir, std::vector<std::map<std::string, int> > batch_layer_params, int numBatches, std::string layer, const int max_alloc, std::vector<float *> &ptr, int layerType);

int readInputBatchesWithNorm(std::string imageRootDir,std::vector<float *>dma_input_vec, std::vector<std::map<std::string, int> > batch_layer_params, int numBatches,std::string layer, int layer_index, std::string layer_prv, const int max_alloc, int layerType, std::string input_name, bool ReadinputFlag);

int allocate_memory(std::vector<float *> &dma_input_vec, std::vector<std::map<std::string, int> > batch_layer_params, int numBatches, int layerType);

int readInputBatchesWithNorm(std::string imageRootDir,std::vector<float *>dma_input_vec, std::vector<std::map<std::string, int> > batch_layer_params, int numBatches,std::string layer, int layer_index, std::string layer_prv, const int max_alloc, int layerType, bool ReadinputFlag);

int allocate_memory(std::vector<float *> &dma_input_vec, std::vector<std::map<std::string, int> > batch_layer_params, int numBatches, int layerType);

std::vector<int> readFile(const std::string fname,
                          float *& fptr,
                          const int max_alloc);
int readRawFile(const std::string fname,
                float *& fptr,
                const int read_alloc,
                const int max_alloc);

int readRawFileNoAlloc(const std::string fname,
               float * fptr,
               const int read_alloc,
               const int max_alloc);

std::map<std::string, int> readParams(const std::string fname);
enum LayerTypes {Convolution, Pooling, InnerProduct, TransConvolution};
void timespec_sub(struct timespec *t1, const struct timespec *t2);
int read_int(volatile void* map_base, int offset);
void write_int(volatile void* map_base, int offset, int value);
#endif
