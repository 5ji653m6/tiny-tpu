`timescale 1ns/1ps
`default_nettype none

// GELU activation: piecewise fixed-point approximation (Q8.8)
//   x >= +2.0 : gelu(x) = x
//   x <= -2.0 : gelu(x) = 0
//   else      : gelu(x) = x/2 + x^2/4   (i.e. x * (0.5 + x/4))
// Continuous at both thresholds: at x=+2, x/2 + x^2/4 = 1 + 1 = 2;
// at x=-2, x/2 + x^2/4 = -1 + 1 = 0.
module gelu_child (
    input logic clk,
    input logic rst,
    input logic gelu_valid_in,
    input logic signed [15:0] gelu_data_in,
    output logic signed [15:0] gelu_data_out,
    output logic gelu_valid_out,
    output logic gelu_overflow_out
);

    // +/- 2.0 in Q8.8 fixed point
    localparam logic signed [15:0] GELU_POS_THRESH = 16'sh0200;
    localparam logic signed [15:0] GELU_NEG_THRESH = -16'sh0200;

    logic signed [15:0] sq_out;
    logic sq_overflow;
    fxp_mul mul_inst(
        .ina(gelu_data_in),
        .inb(gelu_data_in),
        .out(sq_out),
        .overflow(sq_overflow)
    );

    // Middle-region approximation: x/2 + x^2/4 (arithmetic shifts, Q8.8)
    logic signed [15:0] mid_out;
    assign mid_out = (gelu_data_in >>> 1) + (sq_out >>> 2);

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            gelu_data_out     <= 16'b0;
            gelu_valid_out    <= 0;
            gelu_overflow_out <= 1'b0;
        end else begin
            if (gelu_valid_in) begin
                if (gelu_data_in >= GELU_POS_THRESH) begin
                    gelu_data_out <= gelu_data_in;
                end else if (gelu_data_in <= GELU_NEG_THRESH) begin
                    gelu_data_out <= 16'b0;
                end else begin
                    // |x| < 2.0 here, so x^2 < 4.0 always fits in Q8.8
                    gelu_data_out     <= mid_out;
                    gelu_overflow_out <= gelu_overflow_out | sq_overflow; // sticky
                end
                gelu_valid_out <= 1;
            end else begin
                gelu_valid_out <= 0;
                gelu_data_out  <= 16'b0;
            end
        end
    end

endmodule
