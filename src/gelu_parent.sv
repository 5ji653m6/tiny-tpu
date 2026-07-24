`timescale 1ns/1ps
`default_nettype none

module gelu_parent (
    input logic clk,
    input logic rst,

    input logic gelu_valid_1_in,
    input logic gelu_valid_2_in,

    input logic signed [15:0] gelu_data_1_in,
    input logic signed [15:0] gelu_data_2_in,

    output logic signed [15:0] gelu_data_1_out,
    output logic signed [15:0] gelu_data_2_out,

    output logic gelu_valid_1_out,
    output logic gelu_valid_2_out,

    output logic gelu_overflow_out_1,
    output logic gelu_overflow_out_2
);

    gelu_child gelu_col_1 (
        .clk(clk),
        .rst(rst),
        .gelu_valid_in(gelu_valid_1_in),
        .gelu_data_in(gelu_data_1_in),
        .gelu_data_out(gelu_data_1_out),
        .gelu_valid_out(gelu_valid_1_out),
        .gelu_overflow_out(gelu_overflow_out_1)
    );

    gelu_child gelu_col_2 (
        .clk(clk),
        .rst(rst),
        .gelu_valid_in(gelu_valid_2_in),
        .gelu_data_in(gelu_data_2_in),
        .gelu_data_out(gelu_data_2_out),
        .gelu_valid_out(gelu_valid_2_out),
        .gelu_overflow_out(gelu_overflow_out_2)
    );

endmodule
