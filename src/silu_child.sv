`timescale 1ns/1ps
`default_nettype none

// SiLU activation (x * sigmoid(x)) in Q8.8 fixed point, one lane
// (roadmap item 13). Built entirely from already-gated arithmetic:
// the softmax_group_nxn exp LUT (verbatim), fxp_div, and fxp_mul.
//
//   a     = |x|                       (17-bit two's-complement magnitude)
//   e     = exp_lut(a)                (piecewise-linear exp LUT, clamps to
//                                      0 for a >= 8.0)
//   den   = 256 + e                   (1.0 + e in Q8.8 raw, in [1.0, 2.0])
//   num   = 256 if x >= 0 else e      (sigmoid = 1/(1+e^-x) / e^x/(1+e^x))
//   sigma = fxp_div(num, den)         (WIIB=9 covers the [1.0,2.0] divisor)
//   silu  = fxp_mul(x, sigma)
//
// The divider always sees the SMALL exponential, so the divisor stays in
// [1.0, 2.0] and no widened geometry beyond WIIB=9 is needed. Saturation
// falls out of the LUT clamp for free: |x| >= 8.0 -> e = 0 -> sigma =
// 1.0 (x >= 0) or 0.0 (x < 0), so silu(x) = x exactly / 0 exactly.
// There is NO overflow condition (|silu(x)| <= |x|), so silu_overflow_out
// is tied to 0.
//
// Handshake mirrors gelu_child: all arithmetic is combinational into ONE
// registered output stage; valid_out asserts one cycle after valid_in;
// when valid_in is low, valid_out deasserts and data_out clears to 0.
module silu_child (
    input logic clk,
    input logic rst,
    input logic silu_valid_in,
    input logic signed [15:0] silu_data_in,
    output logic signed [15:0] silu_data_out,
    output logic silu_valid_out,
    output logic silu_overflow_out
);

    // exp(-0.25k) in Q8.8, k = 0..32
    // (generated: round(256 * exp(-0.25*k)))
    // VERBATIM copy of the exp LUT in softmax_group_nxn.sv.
    function automatic logic [15:0] exp_lut(input logic [5:0] k);
        case (k)
            6'd0:  exp_lut = 16'h0100;
            6'd1:  exp_lut = 16'h00c7;
            6'd2:  exp_lut = 16'h009b;
            6'd3:  exp_lut = 16'h0079;
            6'd4:  exp_lut = 16'h005e;
            6'd5:  exp_lut = 16'h0049;
            6'd6:  exp_lut = 16'h0039;
            6'd7:  exp_lut = 16'h002c;
            6'd8:  exp_lut = 16'h0023;
            6'd9:  exp_lut = 16'h001b;
            6'd10: exp_lut = 16'h0015;
            6'd11: exp_lut = 16'h0010;
            6'd12: exp_lut = 16'h000d;
            6'd13: exp_lut = 16'h000a;
            6'd14: exp_lut = 16'h0008;
            6'd15: exp_lut = 16'h0006;
            6'd16: exp_lut = 16'h0005;
            6'd17: exp_lut = 16'h0004;
            6'd18: exp_lut = 16'h0003;
            6'd19: exp_lut = 16'h0002;
            6'd20: exp_lut = 16'h0002;
            6'd21: exp_lut = 16'h0001;
            6'd22: exp_lut = 16'h0001;
            6'd23: exp_lut = 16'h0001;
            6'd24: exp_lut = 16'h0001;
            6'd25: exp_lut = 16'h0000;
            6'd26: exp_lut = 16'h0000;
            6'd27: exp_lut = 16'h0000;
            6'd28: exp_lut = 16'h0000;
            6'd29: exp_lut = 16'h0000;
            6'd30: exp_lut = 16'h0000;
            6'd31: exp_lut = 16'h0000;
            6'd32: exp_lut = 16'h0000;
            default: exp_lut = 16'h0000;
        endcase
    endfunction

    // ---- a = |x| (17-bit two's-complement magnitude, same construction
    // as softmax_group_nxn's abs_e) ----
    logic signed [16:0] x_ext;
    logic [16:0]        a;
    assign x_ext = {silu_data_in[15], silu_data_in};
    assign a     = x_ext[16] ? (~x_ext + 17'd1) : x_ext[16:0];

    // clamp |x| to 8.0 (raw 2048): beyond this exp rounds to 0 in Q8.8
    logic        clamped;
    assign clamped = (a >= 17'd2048);

    // segment index (0..31) and 6-bit sub-segment fraction
    logic [4:0] seg;
    logic [5:0] frac;
    assign seg  = a[10:6];
    assign frac = a[5:0];

    // linear interpolation (same geometry as softmax_group_nxn):
    //   exp = lut[seg] + round((lut[seg+1]-lut[seg]) * frac / 64)
    // slope is SIGNED (exp is monotone decreasing, slope <= 0) and
    // the product shift is arithmetic.
    logic [15:0]        lut_lo, lut_hi;
    logic signed [16:0] slope;
    logic signed [23:0] interp;
    logic signed [23:0] interp_shift;
    logic [15:0]        e;
    assign lut_lo       = exp_lut({1'b0, seg});
    assign lut_hi       = exp_lut({1'b0, seg} + 6'd1);
    assign slope        = $signed({1'b0, lut_hi}) - $signed({1'b0, lut_lo});
    assign interp       = slope * $signed({2'b0, frac}) + 24'sd32; // +32 for round-to-nearest
    assign interp_shift = interp >>> 6;
    assign e            = clamped ? 16'h0000
                                  : (lut_lo + interp_shift[15:0]);

    // ---- sigmoid = num / den ----
    // den = 1.0 + e, always in [1.0, 2.0] (17-bit Q9.8, WIIB=9)
    // num = 1.0 (x >= 0) else e  -> sigmoid = 1/(1+e^-x) / e^x/(1+e^x)
    logic [16:0] den;
    logic [15:0] num;
    assign den = 17'h00100 + {1'b0, e};
    assign num = silu_data_in[15] ? e : 16'h0100;

    logic [15:0] sigma;
    logic        div_overflow_unused;
    fxp_div #(
        .WIIA (8), .WIFA (8),
        .WIIB (9), .WIFB (8),
        .WOI  (8), .WOF  (8),
        .ROUND(1)
    ) div_inst (
        .dividend (num),
        .divisor  (den),
        .out      (sigma),
        .overflow (div_overflow_unused)
    );

    // ---- silu = x * sigma (Q8.8; sigma in [0, 1.0] is a valid signed
    // Q8.8 input) ----
    logic signed [15:0] silu_mult;
    logic               mul_overflow_unused;
    fxp_mul mul_inst (
        .ina(silu_data_in),
        .inb(sigma),
        .out(silu_mult),
        .overflow(mul_overflow_unused)
    );

    // No overflow condition exists (|silu(x)| <= |x|): tied low.
    assign silu_overflow_out = 1'b0;

    // One registered output stage: single-cycle latency, fully pipelined
    // (mirrors gelu_child's handshake).
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            silu_data_out  <= 16'b0;
            silu_valid_out <= 0;
        end else begin
            if (silu_valid_in) begin
                silu_data_out  <= silu_mult;
                silu_valid_out <= 1;
            end else begin
                silu_valid_out <= 0;
                silu_data_out  <= 16'b0;
            end
        end
    end

endmodule
