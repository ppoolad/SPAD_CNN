//
// Created by zbete on 4/28/2019.
//

#ifndef _TOTAL_VGG_TEST_H
#define _TOTAL_VGG_TEST_H

#include <string>
#include <vector>
#include <map>

int run_single_test_conv_trans3d (std::string imageDir, std::map<std::string, int> layer_params, float * &dma_input, float * gold_outputs);

int run_single_test_conv3d  (std::string imageDir, std::map<std::string, int> layer_params, float * &dma_input, float * gold_outputs);

#endif //SPAD_CNN_TOTAL_SPAD_TEST_H
