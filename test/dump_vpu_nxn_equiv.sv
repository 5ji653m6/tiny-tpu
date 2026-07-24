// Dual-instance wrapper for the vpu_nxn N=2 equivalence test (roadmap
// item 5c). Instantiates the LEGACY vpu and the new vpu_nxn side by side
// sharing every input; test_vpu_nxn_equiv.py drives the shared inputs once
// and compares the two instances' outputs cycle by cycle (the legacy
// module is the spec). Written by the harness author, not the agent.
`timescale 1ns/1ps
`default_nettype none

module dump();

    logic clk;
    logic rst;
    logic [6:0] vpu_data_pathway;

    logic [15:0] vpu_data_in [2];
    logic vpu_valid_in [2];
    logic [15:0] bias_scalar_in [2];
    logic [15:0] lr_leak_factor_in;
    logic [15:0] Y_in [2];
    logic [15:0] inv_batch_size_times_two_in;
    logic [15:0] H_in [2];

    vpu vpu_legacy (
        .clk(clk),
        .rst(rst),
        .vpu_data_pathway(vpu_data_pathway),
        .vpu_data_in_1(vpu_data_in[0]),
        .vpu_data_in_2(vpu_data_in[1]),
        .vpu_valid_in_1(vpu_valid_in[0]),
        .vpu_valid_in_2(vpu_valid_in[1]),
        .bias_scalar_in_1(bias_scalar_in[0]),
        .bias_scalar_in_2(bias_scalar_in[1]),
        .lr_leak_factor_in(lr_leak_factor_in),
        .Y_in_1(Y_in[0]),
        .Y_in_2(Y_in[1]),
        .inv_batch_size_times_two_in(inv_batch_size_times_two_in),
        .H_in_1(H_in[0]),
        .H_in_2(H_in[1]),
        .vpu_data_out_1(),
        .vpu_data_out_2(),
        .vpu_valid_out_1(),
        .vpu_valid_out_2()
    );

    vpu_nxn vpu_nxn (
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
        .vpu_data_out(),
        .vpu_valid_out()
    );

    initial begin
        $dumpfile("vpu_nxn_equiv.vcd");
        $dumpvars(1, dump);
    end

endmodule
