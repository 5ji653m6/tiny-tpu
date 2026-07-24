`timescale 1ns/1ps
`default_nettype none

// N-lane group LayerNorm (Q8.8 fixed point), generalizing the hardcoded
// 2-lane layernorm_parent.sv to SYSTOLIC_ARRAY_WIDTH = N lanes (N a power
// of two, >= 2). The N lane values arriving together on one beat form the
// group:
//   mean = (sum_i x_i) >>> log2(N)              (Q8.8, truncating shift)
//   dev_i = x_i - mean                          (17-bit Q9.8)
//   sq_i  = dev_i^2                             (Q18.8 via fxp_mul)
//   var   = (sum_i sq_i) >>> log2(N)            (Q18.8)
//   std   = sqrt(var + eps)                     (Q8.8 via fxp_sqrt)
//   y_i   = dev_i / std                         (Q8.8 via fxp_div)
//
// eps is a small fixed-point constant (LN_EPS = 16 raw = 1/16 in the Q*.8
// variance domain) added to var before the square root to keep the divisor
// away from zero, same as legacy layernorm_parent.
//
// dev_i is carried in 17-bit Q9.8 because a deviation can reach +/-256 for
// extreme inputs and does NOT fit Q8.8. fxp_sqrt natively supports the
// WII=18 variance input, so no saturation of var_eps is required.
//
// Handshake discipline matches layernorm_parent: outputs are registered and
// asserted only when ALL N lane inputs are valid in the same cycle;
// otherwise all valid and data outputs are cleared. The group stage assumes
// its N inputs are ALIGNED to the same cycle (skew compensation is the
// integrator's job, not the leaf's).
//
// BUG-TOOLS-1 (iverilog 11): unpacked-array output ports are wires assigned
// from internal _r mirror registers; `output logic` arrays are not safe for
// module-to-module chaining (pattern copied from src/vpu_nxn.sv).
module layernorm_group_nxn #(
    parameter int SYSTOLIC_ARRAY_WIDTH = 4
)(
    input logic clk,
    input logic rst,

    input logic ln_valid_in [SYSTOLIC_ARRAY_WIDTH],
    input logic signed [15:0] ln_data_in [SYSTOLIC_ARRAY_WIDTH],

    output wire signed [15:0] ln_data_out [SYSTOLIC_ARRAY_WIDTH],
    output wire ln_valid_out [SYSTOLIC_ARRAY_WIDTH],
    output wire ln_overflow_out [SYSTOLIC_ARRAY_WIDTH]
);

    // BUG-TOOLS-1 mirrors
    logic signed [15:0] ln_data_out_r [SYSTOLIC_ARRAY_WIDTH];
    logic ln_valid_out_r [SYSTOLIC_ARRAY_WIDTH];
    logic ln_overflow_out_r [SYSTOLIC_ARRAY_WIDTH];

    assign ln_data_out     = ln_data_out_r;
    assign ln_valid_out    = ln_valid_out_r;
    assign ln_overflow_out = ln_overflow_out_r;

    localparam int N     = SYSTOLIC_ARRAY_WIDTH;
    localparam int LOG2N = $clog2(N);

    // The shifts that divide by N are only exact for powers of two.
    generate
        if (N < 2 || (N & (N - 1)) != 0) begin : g_bad_width
            initial begin
                $error("layernorm_group_nxn: SYSTOLIC_ARRAY_WIDTH=%0d is not a power of two >= 2", N);
            end
        end
    endgenerate

    // eps = 1/16 in Q18.8 (the variance domain below), same raw value as
    // legacy layernorm_parent's LN_EPS.
    localparam logic [25:0] LN_EPS = 26'h0000010;

    // ---- mean = (sum_i x_i) >>> LOG2N (Q8.8, truncating shift) ----
    // Sum of N signed Q8.8 values needs 16 + LOG2N + 1 bits (sign included).
    logic signed [16+LOG2N:0] sum_ext;
    always_comb begin
        sum_ext = '0;
        for (int i = 0; i < N; i++) begin
            // both operands are signed, so the narrower lane value is
            // sign-extended to the sum width automatically
            sum_ext = sum_ext + ln_data_in[i];
        end
    end

    // Arithmetic-shift-right by LOG2N with truncation, exactly like the
    // legacy half_diff = diff_ext[16:1] slice. The mean of Q8.8 values fits
    // back in Q8.8, so the dropped upper bits are sign copies.
    logic signed [15:0] mean;
    assign mean = sum_ext[15+LOG2N:LOG2N];

    // ---- dev_i = x_i - mean in 17-bit Q9.8 ----
    logic signed [16:0] dev [N];
    logic [25:0]        sq [N];
    logic               sq_overflow [N];

    // ---- shared variance/std pipeline ----
    logic [25+LOG2N:0] sum_sq;
    logic [25:0]       var_eps;
    logic [15:0]       std;
    logic              sqrt_overflow;

    logic [15:0] y [N];
    logic        div_overflow [N];

    genvar gi;
    generate
        for (gi = 0; gi < N; gi++) begin : g_lane
            assign dev[gi] = {ln_data_in[gi][15], ln_data_in[gi]}
                           - {mean[15], mean};

            // sq_i = dev_i^2 in Q18.8 (max ~65536.0, fits Q18.8)
            fxp_mul #(
                .WIIA (9), .WIFA (8),
                .WIIB (9), .WIFB (8),
                .WOI  (18), .WOF (8),
                .ROUND(1)
            ) sq_inst (
                .ina      (dev[gi]),
                .inb      (dev[gi]),
                .out      (sq[gi]),
                .overflow (sq_overflow[gi])
            );

            // y_i = dev_i / std in Q8.8
            fxp_div #(
                .WIIA (9), .WIFA (8),
                .WIIB (8), .WIFB (8),
                .WOI  (8), .WOF (8),
                .ROUND(1)
            ) div_inst (
                .dividend (dev[gi]),
                .divisor  (std),
                .out      (y[gi]),
                .overflow (div_overflow[gi])
            );
        end
    endgenerate

    // var = (sum_i sq_i) >>> LOG2N (Q18.8); sq_i are non-negative so the
    // plain slice implements the shift exactly.
    always_comb begin
        sum_sq = '0;
        for (int i = 0; i < N; i++) begin
            sum_sq = sum_sq + sq[i];
        end
    end

    assign var_eps = sum_sq[25+LOG2N:LOG2N] + LN_EPS;

    // std = sqrt(var + eps) in Q8.8 (fxp_sqrt supports WII=18 directly)
    fxp_sqrt #(
        .WII  (18), .WIF (8),
        .WOI  (8),  .WOF (8),
        .ROUND(1)
    ) sqrt_inst (
        .in       (var_eps),
        .out      (std),
        .overflow (sqrt_overflow)
    );

    // sticky OR of the per-beat squarer/sqrt/divider overflows, like legacy
    logic any_overflow;
    always_comb begin
        any_overflow = sqrt_overflow;
        for (int i = 0; i < N; i++) begin
            any_overflow = any_overflow | sq_overflow[i] | div_overflow[i];
        end
    end

    logic all_valid;
    always_comb begin
        all_valid = 1'b1;
        for (int i = 0; i < N; i++) begin
            all_valid = all_valid & ln_valid_in[i];
        end
    end

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            for (int i = 0; i < N; i++) begin
                ln_data_out_r[i]     <= 16'b0;
                ln_valid_out_r[i]    <= 1'b0;
                ln_overflow_out_r[i] <= 1'b0;
            end
        end else begin
            // the N lanes form one group, so all must be valid to normalize
            if (all_valid) begin
                for (int i = 0; i < N; i++) begin
                    ln_data_out_r[i]     <= y[i];
                    ln_valid_out_r[i]    <= 1'b1;
                    ln_overflow_out_r[i] <= ln_overflow_out_r[i] | any_overflow; // sticky
                end
            end else begin
                for (int i = 0; i < N; i++) begin
                    ln_valid_out_r[i] <= 1'b0;
                    ln_data_out_r[i]  <= 16'b0;
                end
            end
        end
    end

endmodule
