// Dual-instance wrapper for the tpu_nxn N=2 full-chip equivalence test
// (roadmap item 5d-2). Instantiates the LEGACY tpu and the new tpu_nxn
// side by side sharing every input; test_tpu_nxn_equiv.py drives the
// shared inputs with the full test_tpu.py instruction script (forward +
// backward + gradient descent) and compares the two UB memory images
// word for word at the end (the legacy chip is the spec). Written by the
// harness author, not the agent.
`timescale 1ns/1ps
`default_nettype none

module dump();

    logic clk;
    logic rst;

    logic [15:0] ub_wr_host_data_in [0:1];
    logic ub_wr_host_valid_in [0:1];

    logic ub_rd_start_in;
    logic ub_rd_transpose;
    logic [8:0] ub_ptr_select;
    logic [15:0] ub_rd_addr_in;
    logic [15:0] ub_rd_row_size;
    logic [15:0] ub_rd_col_size;

    logic [15:0] learning_rate_in;
    logic [6:0] vpu_data_pathway;
    logic sys_switch_in;
    logic [15:0] vpu_leak_factor_in;
    logic [15:0] inv_batch_size_times_two_in;

    tpu #(
        .SYSTOLIC_ARRAY_WIDTH(2)
    ) tpu_legacy (
        .clk(clk),
        .rst(rst),
        .ub_wr_host_data_in(ub_wr_host_data_in),
        .ub_wr_host_valid_in(ub_wr_host_valid_in),
        .ub_rd_start_in(ub_rd_start_in),
        .ub_rd_transpose(ub_rd_transpose),
        .ub_ptr_select(ub_ptr_select),
        .ub_rd_addr_in(ub_rd_addr_in),
        .ub_rd_row_size(ub_rd_row_size),
        .ub_rd_col_size(ub_rd_col_size),
        .learning_rate_in(learning_rate_in),
        .vpu_data_pathway(vpu_data_pathway),
        .sys_switch_in(sys_switch_in),
        .vpu_leak_factor_in(vpu_leak_factor_in),
        .inv_batch_size_times_two_in(inv_batch_size_times_two_in)
    );

    tpu_nxn #(
        .SYSTOLIC_ARRAY_WIDTH(2)
    ) tpu_nxn_inst (
        .clk(clk),
        .rst(rst),
        .ub_wr_host_data_in(ub_wr_host_data_in),
        .ub_wr_host_valid_in(ub_wr_host_valid_in),
        .ub_rd_start_in(ub_rd_start_in),
        .ub_rd_transpose(ub_rd_transpose),
        .ub_ptr_select(ub_ptr_select),
        .ub_rd_addr_in(ub_rd_addr_in),
        .ub_rd_row_size(ub_rd_row_size),
        .ub_rd_col_size(ub_rd_col_size),
        .learning_rate_in(learning_rate_in),
        .vpu_data_pathway(vpu_data_pathway),
        .sys_switch_in(sys_switch_in),
        .vpu_leak_factor_in(vpu_leak_factor_in),
        .inv_batch_size_times_two_in(inv_batch_size_times_two_in)
    );

    initial begin
        $dumpfile("tpu_nxn_equiv.vcd");
        $dumpvars(1, dump);
    end

endmodule
