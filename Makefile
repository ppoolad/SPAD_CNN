DCP = static_routed_v1.dcp
PR_SRCS = conv_test/conv_layer.cpp conv_test/conv_layer.h conv3d_test/conv3d_layer.cpp conv3d_test/conv3d_layer.h 8v3_shell/create_pr2_nn.tcl 8v3_shell/create_pr2_0.tcl 8v3_shell/create_pr2_1.tcl 8v3_shell/pr_region_2_bd.tcl
PROJNAME = pr_region_test_proj 



all: conv_layer fc_layer hw_conv_layer hw_fc_layer

maxpool_layer: maxpool_test/* util/*
	g++ maxpool_test/*.cpp maxpool_test/*.c util/*.cpp -I maxpool_test -I./ -o maxpool_layer -std=c++11

hw_maxpool_layer: maxpool_test/* util/*
	g++ -DHW_TEST maxpool_test/*.cpp maxpool_test/*.c util/*.cpp -I maxpool_test -I./ -o hw_maxpool_layer -std=c++11

conv_layer: conv_test/* util/*
	g++ conv_test/*.cpp conv_test/*.c util/*.cpp -I conv_test -I./ -I/opt/Xilinx/Vivado_HLS/2017.2/include -o conv_layer -std=c++11

conv3d_layer: conv3d_test/* util/*
	g++ conv3d_test/*.cpp conv3d_test/*.c util/*.cpp -I conv3d_test -I./ -I/opt/Xilinx/Vivado_HLS/2017.2/include -o conv3d_layer -std=c++11

hw_conv3d_layer: conv3d_test/* util/*
	g++ -DHW_TEST conv3d_test/*.cpp conv3d_test/*.c util/*.cpp -I conv3d_test -I./ -I/opt/Xilinx/Vivado_HLS/2017.2/include  -o hw_conv3d_layer -std=c++11


conv_trans3d_layer: conv_trans3d_test/* util/*
	g++ -g conv_trans3d_test/*.cpp conv_trans3d_test/*.c util/*.cpp -I conv_trans3d_test -I./ -I/opt/Xilinx/Vivado_HLS/2017.2/include -o conv_trans3d_layer -std=c++11

hw_conv_trans3d_layer: conv3d_test/* util/*
	g++ -DHW_TEST conv_trans3d_test/*.cpp conv_trans3d_test/*.c util/*.cpp -I conv_trans3d_test -I./ -I/opt/Xilinx/Vivado_HLS/2017.2/include  -o hw_conv_trans3d_layer -std=c++11


unified_fc_conv_layer: conv_test/* util/*
	g++ conv_fc_unified_test/*.cpp conv_fc_unified_test/*.c util/*.cpp -I conv_fc_unified_test -I./ -I/opt/Xilinx/Vivado_HLS/2017.2/include -o conv_fc_unified_layer -std=c++11

hw_unified_fc_conv_layer: conv_test/* util/*
	g++ -g -DHW_TEST conv_fc_unified_test/*.cpp conv_fc_unified_test/*.c util/*.cpp -I conv_fc_unified_test -I./ -I/opt/Xilinx/Vivado_HLS/2017.2/include -o conv_fc_unified_layer -std=c++11

fc_layer: fc_test/* util/*
	g++ fc_test/*.cpp fc_test/*.c util/*.cpp -I fc_test -I./ -I/opt/Xilinx/Vivado_HLS/2017.2/include -o fc_layer -std=c++11

hw_fc_layer: fc_test/* util/*
	g++ -g -DHW_TEST fc_test/*.cpp fc_test/*.c util/*.cpp -I fc_test -I./ -I/opt/Xilinx/Vivado_HLS/2017.2/include -o hw_fc_layer -std=c++11

vgg16: total_vgg_test/* util/*
	g++ total_vgg_test/*.cpp total_vgg_test/*.c util/*.cpp -I total_vgg_test -I./ -I/opt/Xilinx/Vivado_HLS/2017.2/include -o vgg16 -std=c++11

hw_vgg16: total_vgg_test/* util/*
	g++ -g -DHW_TEST total_vgg_test/*.cpp total_vgg_test/*.c util/*.cpp -I total_vgg_test -I./ -I/opt/Xilinx/Vivado_HLS/2017.2/include -o hw_vgg16 -std=c++11

conv_hls: conv_test/* util/* 
	vivado_hls hls_proj/conv_hls.tcl

conv3d_hls: conv3d_test/* util/* 
	vivado_hls hls_proj/conv3d_hls.tcl

fc_hls: fc_test/*  util/*
	vivado_hls hls_proj/fc_hls.tcl

maxpool_hls: maxpool_test/* util/* 
	vivado_hls hls_proj/maxpool_hls.tcl

unified_fc_conv_hls: conv_fc_unified_test/* util/* 
	vivado_hls hls_proj/unified_fc_conv_hls.tcl


pr:     $(PR_SRCS) dcp conv_hls conv3d_hls ##fc_hls maxpool_hls unified_fc_conv_hls
	vivado -mode batch -source 8v3_shell/create_pr2_nn.tcl -tclargs $(DCP) $(PROJNAME)  0 

pr_modify: $(PR_SRCS) dcp conv_hls conv3d_hls## fc_hls maxpool_hls unified_fc_conv_hls
	vivado -mode gui -source 8v3_shell/create_pr2_nn.tcl -tclargs $(DCP) $(PROJNAME)  1




static:
	vivado -mode tcl -source 8v3_shell/create_mig_shell.tcl 


clean_sw: 
	@rm -rf fc_layer hw_fc_layer conv_layer hw_conv_layer

clean_pr:
	@rm -rf 8v3_shell/pr_region_test_proj 

clean_static:
	@rm -rf 8v3_shell/mig_shell_ila_proj

clean_dcp:
	@rm -rf 8v3_shell/$(DCP)


dcp: 
	@ls 8v3_shell/$(DCP) 2> /dev/null &&  echo "DCP file exists" || echo "Error: No DCP file found!"       

