`timescale 1ns/1ps
`default_nettype none

// Item 18a SCALE stage (leaf): the per-element multiply half of adaLN
// timestep conditioning (the shift half is the item-17a residual add
// through the bias stage). Sits at the HEAD of the VPU chain so the
// phase computes C = (A @ W) . S before any add -- mul-then-add.
//
// Structure mirrors bias_child: a combinational fxp paired with each
// systolic-valid beat and a registered output (the usual 1-cycle stage
// latency). Unlike bias, the operand applies ONLY on scale-valid beats
// (the per-lane operand window the UB drives while rd_bias_scale is
// set); every other beat passes the systolic data through unchanged,
// so a STALE-armed phase (flag set by an earlier phase, no operand
// read in this one) is an exact passthrough.

module scale_child (
    input logic clk,
    input logic rst,

    input logic signed [15:0] scale_scalar_in, // operand beat (rides the bias operand channel)
    input wire scale_valid_in, // operand-window valid for this beat
    output logic scale_Z_valid_out,
    input wire signed [15:0] scale_sys_data_in, // data from systolic array
    input wire scale_sys_valid_in, // valid signal from the systolic array

    output logic signed [15:0] scale_z_data_out,
    output logic scale_overflow_out
);
    logic signed [15:0] z_scaled;
    logic mul_overflow;

    // Q8.8 x Q8.8 -> Q8.8, fxp_zoom ROUND=1: the same rounding as the
    // PE MAC and the item-13 SiLU multiply (clamp16 of (a*b + 0x80) >> 8).
    fxp_mul mul_inst(
        .ina(scale_sys_data_in),
        .inb(scale_scalar_in),
        .out(z_scaled),
        .overflow(mul_overflow)
    );
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            scale_Z_valid_out <= 1'b0;
            scale_z_data_out  <= 16'b0;
            scale_overflow_out <= 1'b0;
        end else begin
            if (scale_sys_valid_in) begin
                scale_Z_valid_out <= 1'b1;
                // Multiply only on operand-valid beats; passthrough otherwise.
                scale_z_data_out  <= scale_valid_in ? z_scaled : scale_sys_data_in;
                scale_overflow_out <= scale_overflow_out | (scale_valid_in & mul_overflow); // sticky
            end else begin
                scale_Z_valid_out <= 1'b0;
                scale_z_data_out  <= 16'b0;
            end
        end
    end

endmodule
