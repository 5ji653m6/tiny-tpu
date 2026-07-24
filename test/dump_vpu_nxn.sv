// Parameterized wrapper for the vpu_nxn exact-value test (roadmap item
// 5c). The Makefile's N=4 target overrides the width with
// iverilog -Pdump.N=4 and exports VPU_NXN_N to the cocotb test.
// Written by the harness author, not the agent.
`timescale 1ns/1ps
`default_nettype none

module dump #(parameter int N = 2)();

    logic clk;
    logic rst;
    logic [6:0] vpu_data_pathway;

    logic [15:0] vpu_data_in [N];
    logic vpu_valid_in [N];
    logic [15:0] bias_scalar_in [N];
    logic [15:0] lr_leak_factor_in;
    logic [15:0] Y_in [N];
    logic [15:0] inv_batch_size_times_two_in;
    logic [15:0] H_in [N];

    logic [15:0] vpu_data_out [N];
    logic vpu_valid_out [N];

    vpu_nxn #(.SYSTOLIC_ARRAY_WIDTH(N)) vpu_nxn (
        .clk(clk),
        .rst(rst),
        .vpu_data_pathway(vpu_data_pathway),
        .vpu_data_in(vpu_data_in),
        .vpu_valid_in(vpu_valid_in),
        .bias_scalar_in(bias_scalar_in),
        .lr_leak_factor_in(lr_leak_factor_in),
        .Y_in(Y_in),
        .inv_batch_size_times_two_in(inv_batch_size_times_two_in),
        .H_in(H_in),
        .vpu_data_out(vpu_data_out),
        .vpu_valid_out(vpu_valid_out)
    );

    initial begin
        $dumpfile("vpu_nxn.vcd");
        $dumpvars(1, dump);
    end

endmodule
