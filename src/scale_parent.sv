`timescale 1ns/1ps
`default_nettype none

// Item 18a SCALE stage (two-lane parent), mirroring bias_parent: each
// child handles one feature column of the systolic output stream. The
// per-lane operand (scale_scalar_in) rides the existing bias operand
// channel; the per-lane scale_valid_in window gates the multiply
// beat-by-beat.

module scale_parent(
    input logic clk,
    input logic rst,

    input logic signed [15:0] scale_scalar_in_1,
    input logic signed [15:0] scale_scalar_in_2, // scale operands fetched from the unified buffer via the bias channel

    input wire scale_valid_in_1,
    input wire scale_valid_in_2,

    output logic scale_Z_valid_out_1,
    output logic scale_Z_valid_out_2,

    input wire signed [15:0] scale_sys_data_in_1,
    input wire signed [15:0] scale_sys_data_in_2,

    input wire scale_sys_valid_in_1,
    input wire scale_sys_valid_in_2,

    output logic signed [15:0] scale_z_data_out_1,
    output logic signed [15:0] scale_z_data_out_2,

    output logic scale_overflow_out_1,
    output logic scale_overflow_out_2

);
    // Each scale module handles a feature column for a pre-activation matrix.

    scale_child column_1 (
        .clk(clk),
        .rst(rst),
        .scale_scalar_in(scale_scalar_in_1),
        .scale_valid_in(scale_valid_in_1),
        .scale_Z_valid_out(scale_Z_valid_out_1),
        .scale_sys_data_in(scale_sys_data_in_1),
        .scale_sys_valid_in(scale_sys_valid_in_1),
        .scale_z_data_out(scale_z_data_out_1),
        .scale_overflow_out(scale_overflow_out_1)
    );

    scale_child column_2 (
        .clk(clk),
        .rst(rst),
        .scale_scalar_in(scale_scalar_in_2),
        .scale_valid_in(scale_valid_in_2),
        .scale_Z_valid_out(scale_Z_valid_out_2),
        .scale_sys_data_in(scale_sys_data_in_2),
        .scale_sys_valid_in(scale_sys_valid_in_2),
        .scale_z_data_out(scale_z_data_out_2),
        .scale_overflow_out(scale_overflow_out_2)
    );


endmodule
