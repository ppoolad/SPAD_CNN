#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <map>
#include <string>
#include "conv_layer.h"
#include "fc_layer.h"
#include "maxpool_layer.h"
#include "util/shared.h"
#include <sstream>
#include <chrono>

std::string vgg16layers[] = {"", "conv1_1", "conv1_2",
                        "pool1",
                        "conv2_1", "conv2_2",
                        "pool2",
                        "conv3_1", "conv3_2", "conv3_3",
                        "pool3",
                        "conv4_1", "conv4_2","conv4_3",
                        "pool4",
                        "conv5_1", "conv5_2", "conv5_3",
                        "pool5",
                        "fc6","fc7","fc8"};


int main(int argc, char const *argv[])
{
  int numBatches = 10;
  int start_layer = 17;
  int total_vgglayers = 22;
  if (argc > 1) {
    numBatches = atoi(argv[1]);
    if(argc>2)start_layer = atoi(argv[2]);
  }
  int millis = 0;
  // run_conv(numBatches);
  // run_maxpool("conv5_3", numBatches);
  // run_fc("pool5","fc6",numBatches);
  // run_fc("fc6","fc7", numBatches);
  //run_fc("fc7","fc8", numBatches);
  //millis += run_unified_fc_conv("", "conv5_2", numBatches, CONV);
  millis += run_unified_fc_conv("", "conv5_3", numBatches, CONV);
  millis += run_maxpool("conv5_3", numBatches);
  millis += run_unified_fc_conv("pool5","fc6", numBatches,FC);
  millis += run_unified_fc_conv("fc6","fc7", numBatches,FC);
  millis += run_unified_fc_conv("fc7","fc8", numBatches,FC);

  std::cout << "VGG's Computation DONE, TOTAL Time Elapsed for calculation: " << millis << "ms" << std::endl;

  return 0;
}


