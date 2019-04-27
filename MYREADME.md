# my project is in folder ECE1373A2

# ECE1373_assignment2

Welcome to the second assignment in ECE1373

This will describe how to run the provided sample code in this directory. 
This description assumes you have read the assignment handout in doc/assignment2.pdf

## Code Organization

The source code is organized as follows:
- fc directory has files for fully connected layer
- conv directory has files for the convolution layer
- conv_fc_unified_test directory has the final engine which is used for both conv and FC
- nn_params stores binaries for the weights, biases, inputs and reference output. This also contains a script extractParams to create new binaries for other layers. 
- util directory has the shared function to read input.
- hls_proj contains tcl scripts to create a vivado_hls project for convolution and fc.
- pci_tests includes tests to read and write to pcie.
- 8v3_shell contains files and projects for the hypervisor and user appplications

- total_vgg_test has necessary files to run whole network
    - total_vgg_test.cpp just runs layers sequentially
    - each (hw_)<layer>_test.cpp has necessary funciton to read inputs and write outputs for that layer



To create a project with the sample unoptimized code and run csim and synth design do the following:

## RUN PROJECT:
in addition to the explanations in README.MD

- make pr builds the latest version of project. make my_pr does same but does not compile unnecessary blocks
- "make unified_fc_conv_layer" makes software simulation for mac engine which is used by both fc and conv layers.
    it generates "conv_fc_unified_layer" which takes as arguments : fc or conv to simulate for one of them :
    example : ./conv_fc_unified_layer fc
- "make hw_unified_fc_conv_layer" makes hardware testbench for mac engine which is used by both fc and conv layers.
    it generates "hw_conv_fc_unified_layer" which takes as arguments : fc or conv to simulate for one of them :
    example : ./hw_conv_fc_unified_layer fc

- "make maxpool_layer" and "make hw_maxpool_layer" makes software and hw testbenches for maxpool layer

- "make vgg16" and "make hw_vgg16" makes software and hardware testbenches for running last 5 layers of vgg16. it takes as argument number of batches to run. 
    usage: ./hw_vgg16 1 makes runs 1 batch on hardware

- partial bitstream and clear bitstream are saved in default location. 

#NOTE: To software simulation fc or conv, makesure myDataType is defined as floats not ap_fixed. otherwise it will result in a huge runtime.