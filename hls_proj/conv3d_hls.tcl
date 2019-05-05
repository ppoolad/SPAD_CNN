cd hls_proj
open_project conv3d_proj
set_top conv3d_layer 
add_files ../conv3d_test/conv3d_layer.cpp
add_files ../conv3d_test/conv3d_functions.cpp
add_files -tb ../conv3d_test/conv3d_layer_test.cpp -cflags "-I .  -std=c++0x"
add_files -tb ../util/shared.cpp
add_files -tb ../data
open_solution "solution1"
set_part {xcvu095-ffvc1517-2-e} -tool vivado
create_clock -period 250MHz -name default
config_interface -m_axi_addr64 -m_axi_offset off -register_io off
#csim_design -compiler gcc
csynth_design
#cosim_design
export_design -format ip_catalog
