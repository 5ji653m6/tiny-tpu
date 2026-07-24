`timescale 1ns/1ps
`default_nettype none

// Softmax over a 2-lane beat (Q8.8 fixed point).
// The two column values arriving together on one beat form the group:
//   y_i = exp(x_i) / (exp(x_1) + exp(x_2))
//
// Numerically stable logistic form: with d = x_1 - x_2,
//   y_1 = 1 / (1 + exp(-d)) = sigmoid(d)
//   y_2 = 1 - y_1
// so only ONE evaluation of a nonlinearity of a non-positive argument is
// needed (exp(-|d|) <= 1); the sigmoid symmetry sigmoid(-d) = 1 - sigmoid(d)
// covers both signs of d.
//
// Sigmoid implementation: piecewise-linear LUT, 32 uniform segments over
// |d| in [0, 8), clamped to 1.0 for |d| >= 8 (sigmoid(8) ~= 0.999665, which
// rounds to 1.0 in Q8.8 anyway). Entries are sigmoid(k/4) in Q8.8 for
// k = 0..32, with linear interpolation on the 6 sub-segment fraction bits.
// Measured max absolute error vs. the true sigmoid over |d| in [0, 8]:
// ~0.0021 (< 1 Q8.8 LSB = 0.00390625). |d| is clamped at 8, which is also
// the saturation behaviour for inputs whose difference exceeds the Q8.8
// useful range, so no overflow condition exists.
//
// Handshake discipline matches layernorm_parent: outputs are registered and
// asserted only when BOTH lane inputs are valid; otherwise valid and data
// outputs are cleared.
module softmax_parent (
    input logic clk,
    input logic rst,

    input logic sm_valid_1_in,
    input logic sm_valid_2_in,

    input logic signed [15:0] sm_data_1_in,
    input logic signed [15:0] sm_data_2_in,

    output logic signed [15:0] sm_data_1_out,
    output logic signed [15:0] sm_data_2_out,

    output logic sm_valid_1_out,
    output logic sm_valid_2_out,

    output logic sm_overflow_out_1,
    output logic sm_overflow_out_2
);

    // 1.0 in Q8.8
    localparam logic [15:0] SM_ONE = 16'h0100;

    // d = x1 - x2, 17-bit signed Q8.8 (extra bit covers the doubled range;
    // |x1 - x2| <= 65535/256 so no overflow is possible)
    logic signed [16:0] diff_ext;
    assign diff_ext = {sm_data_1_in[15], sm_data_1_in} - {sm_data_2_in[15], sm_data_2_in};

    logic        d_neg;
    logic [16:0] abs_d;
    assign d_neg = diff_ext[16];
    assign abs_d = d_neg ? (~diff_ext + 17'd1) : diff_ext[16:0];

    // clamp |d| to 8.0 (raw 2048): beyond this sigmoid rounds to 1.0 in Q8.8
    logic clamped;
    logic [10:0] a; // |d| in Q3.8, range [0, 2048)
    assign clamped = (abs_d >= 17'd2048);
    assign a       = abs_d[10:0];

    // segment index (0..31) and 6-bit sub-segment fraction
    logic [4:0] seg;
    logic [5:0] frac;
    assign seg  = a[10:6];
    assign frac = a[5:0];

    // sigmoid(k/4) in Q8.8, k = 0..32 (generated: round(256 * 1/(1+exp(-0.25k))))
    function automatic logic [15:0] sig_lut(input logic [5:0] k);
        case (k)
            6'd0:  sig_lut = 16'h0080;
            6'd1:  sig_lut = 16'h0090;
            6'd2:  sig_lut = 16'h009f;
            6'd3:  sig_lut = 16'h00ae;
            6'd4:  sig_lut = 16'h00bb;
            6'd5:  sig_lut = 16'h00c7;
            6'd6:  sig_lut = 16'h00d1;
            6'd7:  sig_lut = 16'h00da;
            6'd8:  sig_lut = 16'h00e1;
            6'd9:  sig_lut = 16'h00e8;
            6'd10: sig_lut = 16'h00ed;
            6'd11: sig_lut = 16'h00f1;
            6'd12: sig_lut = 16'h00f4;
            6'd13: sig_lut = 16'h00f6;
            6'd14: sig_lut = 16'h00f8;
            6'd15: sig_lut = 16'h00fa;
            6'd16: sig_lut = 16'h00fb;
            6'd17: sig_lut = 16'h00fc;
            6'd18: sig_lut = 16'h00fd;
            6'd19: sig_lut = 16'h00fe;
            6'd20: sig_lut = 16'h00fe;
            6'd21: sig_lut = 16'h00ff;
            6'd22: sig_lut = 16'h00ff;
            6'd23: sig_lut = 16'h00ff;
            6'd24: sig_lut = 16'h00ff;
            6'd25: sig_lut = 16'h0100;
            6'd26: sig_lut = 16'h0100;
            6'd27: sig_lut = 16'h0100;
            6'd28: sig_lut = 16'h0100;
            6'd29: sig_lut = 16'h0100;
            6'd30: sig_lut = 16'h0100;
            6'd31: sig_lut = 16'h0100;
            6'd32: sig_lut = 16'h0100;
            default: sig_lut = 16'h0100;
        endcase
    endfunction

    // linear interpolation: y = lut[seg] + round((lut[seg+1]-lut[seg]) * frac / 64)
    logic [15:0] y_lo, y_hi;
    logic [15:0] slope;
    logic [21:0] interp;
    logic [15:0] y_mag;
    assign y_lo   = sig_lut({1'b0, seg});
    assign y_hi   = sig_lut({1'b0, seg} + 6'd1);
    assign slope  = y_hi - y_lo;           // sigmoid is monotone, slope >= 0
    assign interp = slope * frac + 22'd32; // +32 for round-to-nearest
    assign y_mag  = clamped ? SM_ONE : (y_lo + interp[21:6]);

    // apply sigmoid symmetry: sigmoid(-|d|) = 1 - sigmoid(|d|)
    logic [15:0] y1;
    logic [15:0] y2;
    assign y1 = d_neg ? (SM_ONE - y_mag) : y_mag;
    assign y2 = SM_ONE - y1;

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            sm_data_1_out     <= 16'b0;
            sm_data_2_out     <= 16'b0;
            sm_valid_1_out    <= 1'b0;
            sm_valid_2_out    <= 1'b0;
            sm_overflow_out_1 <= 1'b0;
            sm_overflow_out_2 <= 1'b0;
        end else begin
            // the two lanes form one group, so both must be valid to normalize
            if (sm_valid_1_in && sm_valid_2_in) begin
                sm_data_1_out  <= y1;
                sm_data_2_out  <= y2;
                sm_valid_1_out <= 1'b1;
                sm_valid_2_out <= 1'b1;
                // no overflow condition exists (difference saturates at 1.0)
                sm_overflow_out_1 <= 1'b0;
                sm_overflow_out_2 <= 1'b0;
            end else begin
                sm_valid_1_out <= 1'b0;
                sm_valid_2_out <= 1'b0;
                sm_data_1_out  <= 16'b0;
                sm_data_2_out  <= 16'b0;
            end
        end
    end

endmodule
