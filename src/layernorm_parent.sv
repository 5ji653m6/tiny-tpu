`timescale 1ns/1ps
`default_nettype none

// LayerNorm over a 2-lane beat (Q8.8 fixed point).
// The two column values arriving together on one beat form the group:
//   mean = (x1 + x2) / 2
//   var  = ((x1 - mean)^2 + (x2 - mean)^2) / 2
//   y_i  = (x_i - mean) / sqrt(var + eps)
//
// For a 2-element group, (x1 - mean) = (x1 - x2)/2 = half_diff and
// (x2 - mean) = -half_diff, so var = half_diff^2 and y2 = -y1.
// A single squarer, square root, and divider therefore suffice.
//
// eps is a small fixed-point constant (LN_EPS) added to var before the
// square root to keep the divisor away from zero.
module layernorm_parent (
    input logic clk,
    input logic rst,

    input logic ln_valid_1_in,
    input logic ln_valid_2_in,

    input logic signed [15:0] ln_data_1_in,
    input logic signed [15:0] ln_data_2_in,

    output logic signed [15:0] ln_data_1_out,
    output logic signed [15:0] ln_data_2_out,

    output logic ln_valid_1_out,
    output logic ln_valid_2_out,

    output logic ln_overflow_out_1,
    output logic ln_overflow_out_2
);

    // eps = 1/16 in Q16.8 (the variance domain below)
    localparam logic [23:0] LN_EPS = 24'h000010;

    // half_diff = (x1 - x2) / 2 in Q8.8.
    // x1 - x2 fits in 17-bit Q9.8; the arithmetic shift right drops back to Q8.8
    // without loss of range since |(x1-x2)/2| < 128.
    logic signed [16:0] diff_ext;
    logic signed [15:0] half_diff;
    assign diff_ext  = {ln_data_1_in[15], ln_data_1_in} - {ln_data_2_in[15], ln_data_2_in};
    assign half_diff = diff_ext[16:1];

    // var = half_diff^2 in Q16.8 (max ~16384.0, fits in 24-bit Q16.8)
    logic [23:0] var_sq;
    logic        sq_overflow;
    fxp_mul #(
        .WIIA (8), .WIFA (8),
        .WIIB (8), .WIFB (8),
        .WOI  (16), .WOF (8),
        .ROUND(1)
    ) sq_inst (
        .ina      (half_diff),
        .inb      (half_diff),
        .out      (var_sq),
        .overflow (sq_overflow)
    );

    // var + eps (Q16.8, non-negative)
    logic [23:0] var_eps;
    assign var_eps = var_sq + LN_EPS;

    // std = sqrt(var + eps) in Q8.8
    logic [15:0] std;
    logic        sqrt_overflow;
    fxp_sqrt #(
        .WII  (16), .WIF (8),
        .WOI  (8),  .WOF (8),
        .ROUND(1)
    ) sqrt_inst (
        .in       (var_eps),
        .out      (std),
        .overflow (sqrt_overflow)
    );

    // y1 = half_diff / std in Q8.8; y2 = -y1
    logic [15:0] y1;
    logic        div_overflow;
    fxp_div #(
        .WIIA (8), .WIFA (8),
        .WIIB (8), .WIFB (8),
        .WOI  (8), .WOF (8),
        .ROUND(1)
    ) div_inst (
        .dividend (half_diff),
        .divisor  (std),
        .out      (y1),
        .overflow (div_overflow)
    );

    logic signed [15:0] y2;
    assign y2 = -$signed(y1);

    logic any_overflow;
    assign any_overflow = sq_overflow | sqrt_overflow | div_overflow;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            ln_data_1_out     <= 16'b0;
            ln_data_2_out     <= 16'b0;
            ln_valid_1_out    <= 1'b0;
            ln_valid_2_out    <= 1'b0;
            ln_overflow_out_1 <= 1'b0;
            ln_overflow_out_2 <= 1'b0;
        end else begin
            // the two lanes form one group, so both must be valid to normalize
            if (ln_valid_1_in && ln_valid_2_in) begin
                ln_data_1_out     <= y1;
                ln_data_2_out     <= y2;
                ln_valid_1_out    <= 1'b1;
                ln_valid_2_out    <= 1'b1;
                ln_overflow_out_1 <= ln_overflow_out_1 | any_overflow; // sticky
                ln_overflow_out_2 <= ln_overflow_out_2 | any_overflow; // sticky
            end else begin
                ln_valid_1_out <= 1'b0;
                ln_valid_2_out <= 1'b0;
                ln_data_1_out  <= 16'b0;
                ln_data_2_out  <= 16'b0;
            end
        end
    end

endmodule
