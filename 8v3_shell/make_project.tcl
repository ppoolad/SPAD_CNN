
set projName [lindex $argv 0]
set proj 8v3_shell/$projName/mig_shell.xpr




if {![file exists $proj]} { 
    source 8v3_shell/test.tcl
} else {
    open_project $proj
    open_bd_design 8v3_shell/$projName/mig_shell.srcs/sources_1/bd/static_region/static_region.bd
    upgrade_ip [get_ips]
    make_wrapper -files [get_files 8v3_shell/$projName/mig_shell.srcs/sources_1/bd/static_region/static_region.bd] -top
    add_files -norecurse 8v3_shell/$projName/mig_shell.srcs/sources_1/bd/static_region/hdl/static_region.v
    update_compile_order -fileset sources_1

    generate_target {synthesis implementation} [get_files 8v3_shell/$projName/mig_shell.srcs/sources_1/bd/static_region/static_region.bd]
    export_ip_user_files -of_objects [get_files 8v3_shell/$projName/mig_shell.srcs/sources_1/bd/static_region/static_region.bd] -no_script -sync -force -quiet

    set ooc_runs [get_runs -filter {IS_SYNTHESIS && name != "synth_1"} ]
    foreach run $ooc_runs { reset_run $run }

    create_ip_run [get_files -of_objects [get_fileset sources_1] 8v3_shell/$projName/mig_shell.srcs/sources_1/bd/static_region/static_region.bd]
    launch_runs [get_runs *_synth*] -jobs 12
    foreach run_name [get_runs *_synth*] {
    wait_on_run ${run_name}
    }
    synth_design -top static_region -mode out_of_context
    #write_checkpoint -force 8v3_shell/$projName.dcp

    opt_design -directive Explore
    place_design -directive Explore
    phys_opt_design -directive Explore
    route_design -directive Explore
    #write_checkpoint -force 8v3_shell/${projName}_routed.dcp
    #write_checkpoint -cell static_region_i/pr_region rp2_route_design.dcp
    write_bitstream -force 8v3_shell/$projName.bit 
    report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose -max_paths 10 -input_pins -name timing_1 -file 8v3_shell/$projName.timing
    report_utilization -hierarchical -name utilization_1 -file 8v3_shell/$projName.util


}