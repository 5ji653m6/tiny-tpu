`timescale 1ns/1ps
`default_nettype none

// Two-lane SiLU parent: a thin shell of two silu_child instances,
// mirroring gelu_parent exactly (roadmap item 13).
module silu_parent (
    input logic clk,
    input logic rst,

    input logic silu_valid_1_in,
    input logic silu_valid_2_in,

    input logic signed [15:0] silu_data_1_in,
    input logic signed [15:0] silu_data_2_in,

    output logic signed [15:0] silu_data_1_out,
    output logic signed [15:0] silu_data_2_out,

    output logic silu_valid_1_out,
    output logic silu_valid_2_out,

    output logic silu_overflow_out_1,
    output logic silu_overflow_out_2
);

    silu_child silu_col_1 (
        .clk(clk),
        .rst(rst),
        .silu_valid_in(silu_valid_1_in),
        .silu_data_in(silu_data_1_in),
        .silu_data_out(silu_data_1_out),
        .silu_valid_out(silu_valid_1_out),
        .silu_overflow_out(silu_overflow_out_1)
    );

    silu_child silu_col_2 (
        .clk(clk),
        .rst(rst),
        .silu_valid_in(silu_valid_2_in),
        .silu_data_in(silu_data_2_in),
        .silu_data_out(silu_data_2_out),
        .silu_valid_out(silu_valid_2_out),
        .silu_overflow_out(silu_overflow_out_2)
    );

endmodule
