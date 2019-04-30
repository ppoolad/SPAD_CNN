#this bash script runs the following 4 programs:
#	1: write_file:	writes contents of a file to a memory address through PCIe
#	2: read_file: 	reads contents of a certain address of DMA and copies them into a local file
#	3: writeint: 	sends integer values (defined in a given file) through AXILite bus
#	4: runip:		sends ap_start signal to an IP core

#this script can be used as follows:


write_file ./input 0 #writes the contents of this file (could be inputs/wheights/biases) to offset 0 of memory
#if need to write multiple set of data to memory, simply add write_file commands with different file paths and memory offsets
# e.g.
#	write_file ./weights n (n is the size of weights) 
#	write_file ./biases m (m is the size of biases)

writeint file ./conv5_3/conv_params_n.txt #writes contents of this file to AXILite (see contents of this file for details)
runip 0x0 	#run this IP; address 0x0 is the base address if this conv IP (defined in vivado).
			#make sure to use correct address for different IP cores

read_file conv_output 17000000 14688000 	#reads 40000 bytes starting from address 0 into a file called conv_output
								#17M and 14.688M are put as example. make sure you read the offset that output has been written to
								#offset is specified in params file. and size (in this case 14688000 is the size of outputs from
								#the first conv layer)


		# CONNECTING SEVERAL LAYERS:
#if you need to connect numerous layers, simply replicate the above 4 steps and make sure the correct memory addresses
#are being read. Meaning, input offset of layer 2 should math output offset of layer 1 (because layer 1 feeds into layer 2)
#
