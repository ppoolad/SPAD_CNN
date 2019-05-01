./write_file ./spadfile 0
./write_file ./conv0.0.weight 16777216
./write_file ./conv0.0.bias 16788880
./write_file ./conv0.1.running_mean 16788896
./write_file ./conv0.1.running_var 16788912
./write_file ./conv0.1.weight 16788928
./write_file ./conv0.1.bias 16788944
./writeint file ./conv_params.txt
./runip 0x0
./read_file conv_output 17000000 67108864