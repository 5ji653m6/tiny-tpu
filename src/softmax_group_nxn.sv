`timescale 1ns/1ps
`default_nettype none

// N-lane group Softmax (Q8.8 fixed point), generalizing the hardcoded
// 2-lane softmax_parent.sv to SYSTOLIC_ARRAY_WIDTH = N lanes (N a power of
// two, >= 2). The N lane values arriving together on one beat form the
// group:
//   m    = max_i x_i                            (signed, combinational)
//   e_i  = x_i - m                              (17-bit Q9.8, <= 0)
//   y_i  = exp(e_i) / sum_j exp(e_j)            (Q8.8 via fxp_div)
//
// exp approximation: piecewise-linear LUT over |e| in [0, 8), mirroring the
// legacy sigmoid LUT construction exactly (same segment geometry, same
// rounding). lut[k] = round(256 * exp(-0.25*k)) for k = 0..32 in Q8.8, with
// linear interpolation on the 6 sub-segment fraction bits:
//   seg  = |e|[10:6], frac = |e|[5:0]
//   slope = lut[seg+1] - lut[seg]   (SIGNED -- negative here)
//   exp_i = lut[seg] + ((slope*frac + 32) >>> 6)   (arithmetic shift)
// |e| >= 8 clamps to 0 (exp(-8) rounds to 0 in Q8.8 anyway), which is also
// the saturation behaviour for inputs whose deviation exceeds the Q8.8
// useful range, so no overflow condition exists.
//
// sum = total of the N exp_i in Q(8+log2(N)).8 unsigned (max N.0). The
// divider's divisor width is widened (WIIB = 8+log2(N)) so sum > 1.0 is
// represented exactly; fxp_div accepts the widened divisor natively, so no
// rescaling is required. sum >= 1.0 always (the max lane contributes
// lut[0] = 1.0), so the divisor is never zero and y_i in [0, 1].
//
// Handshake discipline matches softmax_parent: outputs are registered and
// asserted only when ALL N lane inputs are valid in the same cycle;
// otherwise all valid and data outputs are cleared. The group stage assumes
// its N inputs are ALIGNED to the same cycle (skew compensation is the
// integrator's job, not the leaf's).
//
// BUG-TOOLS-1 (iverilog 11): unpacked-array output ports are wires assigned
// from internal _r mirror registers; `output logic` arrays are not safe for
// module-to-module chaining (pattern copied from src/vpu_nxn.sv).
module softmax_group_nxn #(
    parameter int SYSTOLIC_ARRAY_WIDTH = 4
)(
    input logic clk,
    input logic rst,

    input logic sm_valid_in [SYSTOLIC_ARRAY_WIDTH],
    input logic signed [15:0] sm_data_in [SYSTOLIC_ARRAY_WIDTH],

    output wire signed [15:0] sm_data_out [SYSTOLIC_ARRAY_WIDTH],
    output wire sm_valid_out [SYSTOLIC_ARRAY_WIDTH],
    output wire sm_overflow_out [SYSTOLIC_ARRAY_WIDTH]
);

    // BUG-TOOLS-1 mirrors
    logic signed [15:0] sm_data_out_r [SYSTOLIC_ARRAY_WIDTH];
    logic sm_valid_out_r [SYSTOLIC_ARRAY_WIDTH];
    logic sm_overflow_out_r [SYSTOLIC_ARRAY_WIDTH];

    assign sm_data_out     = sm_data_out_r;
    assign sm_valid_out    = sm_valid_out_r;
    assign sm_overflow_out = sm_overflow_out_r;

    localparam int N     = SYSTOLIC_ARRAY_WIDTH;
    localparam int LOG2N = $clog2(N);

    // The group-sum width geometry is only defined for powers of two.
    generate
        if (N < 2 || (N & (N - 1)) != 0) begin : g_bad_width
            initial begin
                $error("softmax_group_nxn: SYSTOLIC_ARRAY_WIDTH=%0d is not a power of two >= 2", N);
            end
        end
    endgenerate

    // exp(-0.25k) in Q8.8, k = 0..32
    // (generated: round(256 * exp(-0.25*k)))
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

    // ---- m = max of the N inputs (signed comparator reduction) ----
    logic signed [15:0] max_x;
    always_comb begin
        max_x = sm_data_in[0];
        for (int i = 1; i < N; i++) begin
            if ($signed(sm_data_in[i]) > max_x) begin
                max_x = sm_data_in[i];
            end
        end
    end

    // ---- per-lane exp(x_i - m) via the piecewise-linear LUT ----
    logic [15:0] exp_i [N];

    genvar gi;
    generate
        for (gi = 0; gi < N; gi++) begin : g_lane
            // e_i = x_i - m, 17-bit signed Q9.8 (<= 0; extra bit covers the
            // doubled range so no overflow is possible)
            logic signed [16:0] e_ext;
            logic [16:0]        abs_e;
            assign e_ext = {sm_data_in[gi][15], sm_data_in[gi]}
                         - {max_x[15], max_x};
            assign abs_e = e_ext[16] ? (~e_ext + 17'd1) : e_ext[16:0];

            // clamp |e| to 8.0 (raw 2048): beyond this exp rounds to 0 in Q8.8
            logic        clamped;
            logic [10:0] a; // |e| in Q3.8, range [0, 2048)
            assign clamped = (abs_e >= 17'd2048);
            assign a       = abs_e[10:0];

            // segment index (0..31) and 6-bit sub-segment fraction
            logic [4:0] seg;
            logic [5:0] frac;
            assign seg  = a[10:6];
            assign frac = a[5:0];

            // linear interpolation:
            //   exp = lut[seg] + round((lut[seg+1]-lut[seg]) * frac / 64)
            // slope is SIGNED (exp is monotone decreasing, slope <= 0) and
            // the product shift is arithmetic.
            logic [15:0]        lut_lo, lut_hi;
            logic signed [16:0] slope;
            logic signed [23:0] interp;
            logic signed [23:0] interp_shift;
            assign lut_lo        = exp_lut({1'b0, seg});
            assign lut_hi        = exp_lut({1'b0, seg} + 6'd1);
            assign slope         = $signed({1'b0, lut_hi}) - $signed({1'b0, lut_lo});
            assign interp        = slope * $signed({2'b0, frac}) + 24'sd32; // +32 for round-to-nearest
            assign interp_shift  = interp >>> 6;
            assign exp_i[gi]     = clamped ? 16'h0000
                                           : (lut_lo + interp_shift[15:0]);
        end
    endgenerate

    // ---- sum = total of the N exp_i in Q(8+LOG2N).8 unsigned (max N.0) ----
    logic [15+LOG2N:0] exp_sum;
    always_comb begin
        exp_sum = '0;
        for (int i = 0; i < N; i++) begin
            exp_sum = exp_sum + exp_i[i];
        end
    end

    // ---- y_i = exp_i / sum (divisor widened to Q(8+LOG2N).8) ----
    logic [15:0] y [N];
    generate
        for (gi = 0; gi < N; gi++) begin : g_div
            fxp_div #(
                .WIIA (8),        .WIFA (8),
                .WIIB (8+LOG2N),  .WIFB (8),
                .WOI  (8),        .WOF  (8),
                .ROUND(1)
            ) div_inst (
                .dividend (exp_i[gi]),
                .divisor  (exp_sum),
                .out      (y[gi]),
                .overflow ()
            );
        end
    endgenerate

    logic all_valid;
    always_comb begin
        all_valid = 1'b1;
        for (int i = 0; i < N; i++) begin
            all_valid = all_valid & sm_valid_in[i];
        end
    end

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            for (int i = 0; i < N; i++) begin
                sm_data_out_r[i]     <= 16'b0;
                sm_valid_out_r[i]    <= 1'b0;
                sm_overflow_out_r[i] <= 1'b0;
            end
        end else begin
            // the N lanes form one group, so all must be valid to normalize
            if (all_valid) begin
                for (int i = 0; i < N; i++) begin
                    sm_data_out_r[i]  <= y[i];
                    sm_valid_out_r[i] <= 1'b1;
                    // no overflow condition exists (y_i in [0, 1])
                    sm_overflow_out_r[i] <= 1'b0;
                end
            end else begin
                for (int i = 0; i < N; i++) begin
                    sm_valid_out_r[i] <= 1'b0;
                    sm_data_out_r[i]  <= 16'b0;
                end
            end
        end
    end

endmodule
