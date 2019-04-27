cd hls_proj
open_project maxpool_proj
set_top maxpool_layer 
add_files ../maxpool_test/maxpool_layer.cpp
add_files -tb ../maxpool_test/maxpool_layer_test.cpp -cflags "-I .  -std=c++0x"
add_files -tb ../util/shared.cpp
add_files -tb ../util/shared.h
add_files -tb ../data
open_solution "solution1"
set_part {xcvu095-ffvc1517-2-e} -tool vivado
create_clock -period 250MHz -name default
config_interface -m_axi_addr64 -m_axi_offset off -register_io off
csynth_design
#csim_design -compiler gcc
#csynth_design
#cosim_design
export_design -format ip_catalog
