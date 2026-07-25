`timescale 1ns/1ps
`default_nettype none

// N-lane vector processing unit, generalizing the hardcoded 2-lane vpu.sv
// to SYSTOLIC_ARRAY_WIDTH = N lanes (N even, >= 2).
//
// Every stage parent is a two-lane module; this unit instantiates N/2
// instances of each parent in a generate loop over pairs p, wiring parent
// lane _1 to array lane 2p and parent lane _2 to array lane 2p+1. The
// combinational stage-chain routing
//   bias -> leaky_relu -> gelu -> silu -> layernorm -> softmax -> loss
//     -> leaky_relu_derivative
// is replicated per lane with the same bypass semantics as vpu.sv (a
// disabled stage gets zeroed inputs and its upstream wires pass through),
// including the BUG-VPU-1..4 fixes per lane (registered outputs, last_H
// cache per lane cleared when loss is inactive, lr_d H-source mux selecting
// last_H when pathway[1] else H_in, zeroed lr_d inputs when disabled).
//
// GROUP STAGES (layernorm, softmax) are N-dependent (roadmap item 7b);
// lane k leads lane k+1 by one cycle in the systolic dataflow
// (BUG-SKEW-1 in vpu.sv). A generate switch selects the structure:
//   N == 2: the legacy pair structure, textually identical to the
//     pre-parameterization code -- per pair p a 1-cycle shift register
//     delays the EVEN lane's (2p) data+valid at the group stage input
//     (align), and a 1-cycle shift register delays the ODD lane's (2p+1)
//     data+valid at the group stage output (re-skew). This branch is the
//     bit-identity anchor for legacy vpu.sv (test_vpu_nxn_equiv compares
//     cycle-by-cycle).
//   N > 2: ONE layernorm_group_nxn and ONE softmax_group_nxn instance of
//     width N. The group is ALL N lanes of one beat: lane k's data+valid
//     is de-skewed at the group input by a shift-register chain of depth
//     (N-1-k), and the group outputs are re-skewed by a chain of depth k
//     per lane. The shift registers are free-running with reset clear
//     (same discipline as the legacy BUG-SKEW-1 registers) and are used
//     only when the stage is enabled. At N==2 these chains have depths
//     1/0 and 0/1 -- i.e. the legacy align/re-skew registers are the
//     N==2 special case of this structure.

/*
vpu_data_pathway is 8 bits: |silu(7)| |sm(6)| |ln(5)| |gelu(4)| |bias(3)| |lr(2)| |loss(1)| |lr_d(0)|

00000000: activate no modules
00001100: forward pass pathway (sys --> bias --> leaky relu --> output)
00001111: transition pathway (sys --> bias --> leaky relu --> loss --> leaky relu derivative --> output)
00000001: backward pass pathway (sys --> leaky relu derivative --> output)
0001xxxx: gelu stage enabled (inserted between leaky relu and silu)
001xxxxx: layernorm stage enabled (inserted between silu and softmax)
01xxxxxx: softmax stage enabled (inserted between layernorm and loss)
1xxxxxxx: silu stage enabled (inserted between gelu and layernorm)
*/

module vpu_nxn #(
    parameter int SYSTOLIC_ARRAY_WIDTH = 2
)(
    input logic clk,
    input logic rst,

    input logic [7:0] vpu_data_pathway, // 1 bit per stage; bit 4 = gelu, bit 5 = layernorm, bit 6 = softmax, bit 7 = silu

    // Inputs from systolic array (one per lane; index 0 = column 1)
    input logic [15:0] vpu_data_in [SYSTOLIC_ARRAY_WIDTH],
    input logic vpu_valid_in [SYSTOLIC_ARRAY_WIDTH],

    // Inputs from UB
    input logic [15:0] bias_scalar_in [SYSTOLIC_ARRAY_WIDTH],   // For bias modules
    input logic [15:0] lr_leak_factor_in,                       // For leaky relu modules (shared)
    input logic [15:0] Y_in [SYSTOLIC_ARRAY_WIDTH],             // For loss modules
    input logic [15:0] inv_batch_size_times_two_in,             // For loss modules (shared)
    input logic [15:0] H_in [SYSTOLIC_ARRAY_WIDTH],             // For leaky relu derivative modules

    // Outputs to UB
    output wire [15:0] vpu_data_out [SYSTOLIC_ARRAY_WIDTH],
    output wire vpu_valid_out [SYSTOLIC_ARRAY_WIDTH]
);

    // BUG-TOOLS-1 (iverilog 11): variable unpacked-array
    // output ports propagate X to connected parent nets; the
    // public ports are wires assigned from these mirrors.
    logic [15:0] vpu_data_out_r [SYSTOLIC_ARRAY_WIDTH];  // BUG-TOOLS-1 mirror
    logic vpu_valid_out_r [SYSTOLIC_ARRAY_WIDTH];  // BUG-TOOLS-1 mirror

    assign vpu_data_out = vpu_data_out_r;
    assign vpu_valid_out = vpu_valid_out_r;

    localparam int N = SYSTOLIC_ARRAY_WIDTH;
    localparam int PAIRS = SYSTOLIC_ARRAY_WIDTH / 2;

    // bias
    logic signed [15:0] bias_data_in_s [N];
    logic bias_valid_in_s [N];
    logic signed [15:0] bias_data_out_s [N];
    logic bias_valid_out_s [N];

    // bias to lr intermediate values
    logic signed [15:0] b_to_lr_data [N];
    logic b_to_lr_valid [N];

    // lr
    logic signed [15:0] lr_data_in_s [N];
    logic lr_valid_in_s [N];
    logic signed [15:0] lr_data_out_s [N];
    logic lr_valid_out_s [N];

    // lr to loss intermediate values
    logic signed [15:0] lr_to_loss_data [N];
    logic lr_to_loss_valid [N];

    // gelu
    logic signed [15:0] gelu_data_in_s [N];
    logic gelu_valid_in_s [N];
    logic signed [15:0] gelu_data_out_s [N];
    logic gelu_valid_out_s [N];

    // gelu to loss intermediate values
    logic signed [15:0] gelu_to_loss_data [N];
    logic gelu_to_loss_valid [N];

    // silu (elementwise stage, like gelu: no de-skew/re-skew)
    logic signed [15:0] silu_data_in_s [N];
    logic silu_valid_in_s [N];
    logic signed [15:0] silu_data_out_s [N];
    logic silu_valid_out_s [N];

    // silu to loss intermediate values
    logic signed [15:0] silu_to_loss_data [N];
    logic silu_to_loss_valid [N];

    // layernorm
    // BUG-TOOLS-1 (iverilog 11): the group-stage output arrays are driven
    // by whole unpacked-array output ports at N>2 (layernorm_group_nxn /
    // softmax_group_nxn), which only propagate to a parent net declared
    // 'wire' -- a parent 'logic' array silently reads X. Scalar element
    // connections (the N==2 pair parents) work on wires too, so these are
    // wires in both generate branches.
    logic signed [15:0] ln_data_in_s [N];
    logic ln_valid_in_s [N];
    wire signed [15:0] ln_data_out_s [N];
    wire ln_valid_out_s [N];

    // layernorm to loss intermediate values
    logic signed [15:0] ln_to_loss_data [N];
    logic ln_to_loss_valid [N];

    // softmax (outputs are wires: BUG-TOOLS-1, see layernorm above)
    logic signed [15:0] sm_data_in_s [N];
    logic sm_valid_in_s [N];
    wire signed [15:0] sm_data_out_s [N];
    wire sm_valid_out_s [N];

    // softmax to loss intermediate values
    logic signed [15:0] sm_to_loss_data [N];
    logic sm_to_loss_valid [N];

    // Group-stage outputs as consumed by the stage-chain routing: one
    // uniform per-lane view in both generate branches. The generate
    // branches below drive these from their own scoped signals --
    // N == 2: even lanes take the group output directly, odd lanes the
    //   1-cycle re-skewed output (legacy BUG-SKEW-1 wiring);
    // N > 2: lane k takes the group output delayed by k cycles (re-skew
    //   chain restoring the native column skew).
    logic signed [15:0] ln_data_out_routed [N];
    logic               ln_valid_out_routed [N];
    logic signed [15:0] sm_data_out_routed [N];
    logic               sm_valid_out_routed [N];

    // loss
    logic signed [15:0] loss_data_in_s [N];
    logic loss_valid_in_s [N];
    logic signed [15:0] loss_data_out_s [N];
    logic loss_valid_out_s [N];

    // loss to lrd intermediate values
    logic signed [15:0] loss_to_lrd_data [N];
    logic loss_to_lrd_valid [N];

    // lr_d
    logic signed [15:0] lr_d_data_in_s [N];
    logic lr_d_valid_in_s [N];
    logic signed [15:0] lr_d_data_out_s [N];
    logic lr_d_valid_out_s [N];
    logic signed [15:0] lr_d_H_in_s [N];

    // temp 'last H matrix' cache, per lane
    logic signed [15:0] last_H_data_in_s [N];  // combinational input to H-cache register
    logic signed [15:0] last_H_data_out_s [N];

    // BUG-VPU-1 fix: intermediate mux signals; vpu_data_out_r is registered in always_ff
    logic signed [15:0] vpu_data_mux [N];
    logic               vpu_valid_mux [N];

    // N/2 instances of each two-lane stage parent; pair p serves lanes 2p (_1) and 2p+1 (_2)
    genvar p;
    generate
        for (p = 0; p < PAIRS; p++) begin : gen_pair

            bias_parent bias_parent_inst (
                .clk(clk),
                .rst(rst),
                .bias_sys_data_in_1(bias_data_in_s[2*p]),
                .bias_sys_data_in_2(bias_data_in_s[2*p+1]),
                .bias_sys_valid_in_1(bias_valid_in_s[2*p]),
                .bias_sys_valid_in_2(bias_valid_in_s[2*p+1]),

                .bias_scalar_in_1(bias_scalar_in[2*p]),
                .bias_scalar_in_2(bias_scalar_in[2*p+1]),

                .bias_Z_valid_out_1(bias_valid_out_s[2*p]),
                .bias_Z_valid_out_2(bias_valid_out_s[2*p+1]),
                .bias_z_data_out_1(bias_data_out_s[2*p]),
                .bias_z_data_out_2(bias_data_out_s[2*p+1]),
                .bias_overflow_out_1(),   // BUG-OVF-1: observable via hierarchical reference
                .bias_overflow_out_2()
            );

            leaky_relu_parent leaky_relu_parent_inst (
                .clk(clk),
                .rst(rst),

                .lr_data_1_in(lr_data_in_s[2*p]),
                .lr_data_2_in(lr_data_in_s[2*p+1]),
                .lr_valid_1_in(lr_valid_in_s[2*p]),
                .lr_valid_2_in(lr_valid_in_s[2*p+1]),

                .lr_leak_factor_in(lr_leak_factor_in),

                .lr_data_1_out(lr_data_out_s[2*p]),
                .lr_data_2_out(lr_data_out_s[2*p+1]),
                .lr_valid_1_out(lr_valid_out_s[2*p]),
                .lr_valid_2_out(lr_valid_out_s[2*p+1]),
                .lr_overflow_out_1(),   // BUG-OVF-1: observable via hierarchical reference
                .lr_overflow_out_2()
            );

            gelu_parent gelu_parent_inst (
                .clk(clk),
                .rst(rst),

                .gelu_data_1_in(gelu_data_in_s[2*p]),
                .gelu_data_2_in(gelu_data_in_s[2*p+1]),
                .gelu_valid_1_in(gelu_valid_in_s[2*p]),
                .gelu_valid_2_in(gelu_valid_in_s[2*p+1]),

                .gelu_data_1_out(gelu_data_out_s[2*p]),
                .gelu_data_2_out(gelu_data_out_s[2*p+1]),
                .gelu_valid_1_out(gelu_valid_out_s[2*p]),
                .gelu_valid_2_out(gelu_valid_out_s[2*p+1]),
                .gelu_overflow_out_1(),   // BUG-OVF-1: observable via hierarchical reference
                .gelu_overflow_out_2()
            );

            silu_parent silu_parent_inst (
                .clk(clk),
                .rst(rst),

                .silu_data_1_in(silu_data_in_s[2*p]),
                .silu_data_2_in(silu_data_in_s[2*p+1]),
                .silu_valid_1_in(silu_valid_in_s[2*p]),
                .silu_valid_2_in(silu_valid_in_s[2*p+1]),

                .silu_data_1_out(silu_data_out_s[2*p]),
                .silu_data_2_out(silu_data_out_s[2*p+1]),
                .silu_valid_1_out(silu_valid_out_s[2*p]),
                .silu_valid_2_out(silu_valid_out_s[2*p+1]),
                .silu_overflow_out_1(),   // BUG-OVF-1: observable via hierarchical reference
                .silu_overflow_out_2()
            );

            loss_parent loss_parent_inst (
                .clk(clk),
                .rst(rst),
                .H_1_in(loss_data_in_s[2*p]),
                .H_2_in(loss_data_in_s[2*p+1]),
                .valid_1_in(loss_valid_in_s[2*p]),
                .valid_2_in(loss_valid_in_s[2*p+1]),

                .Y_1_in(Y_in[2*p]),
                .Y_2_in(Y_in[2*p+1]),
                .inv_batch_size_times_two_in(inv_batch_size_times_two_in),

                .gradient_1_out(loss_data_out_s[2*p]),
                .gradient_2_out(loss_data_out_s[2*p+1]),
                .valid_1_out(loss_valid_out_s[2*p]),
                .valid_2_out(loss_valid_out_s[2*p+1]),
                .loss_overflow_out_1(),   // BUG-OVF-1: observable via hierarchical reference
                .loss_overflow_out_2()
            );

            leaky_relu_derivative_parent leaky_relu_derivative_parent_inst (
                .clk(clk),
                .rst(rst),
                .lr_d_data_1_in(lr_d_data_in_s[2*p]),
                .lr_d_data_2_in(lr_d_data_in_s[2*p+1]),
                .lr_d_valid_1_in(lr_d_valid_in_s[2*p]),
                .lr_d_valid_2_in(lr_d_valid_in_s[2*p+1]),

                .lr_d_H_1_in(lr_d_H_in_s[2*p]),
                .lr_d_H_2_in(lr_d_H_in_s[2*p+1]),
                .lr_leak_factor_in(lr_leak_factor_in),

                .lr_d_data_1_out(lr_d_data_out_s[2*p]),
                .lr_d_data_2_out(lr_d_data_out_s[2*p+1]),
                .lr_d_valid_1_out(lr_d_valid_out_s[2*p]),
                .lr_d_valid_2_out(lr_d_valid_out_s[2*p+1]),
                .lr_d_overflow_out_1(),   // BUG-OVF-1: observable via hierarchical reference
                .lr_d_overflow_out_2()
            );

        end
    endgenerate

    // ------------------------------------------------------------------
    // Group stages (layernorm, softmax): N-dependent structure.
    //   N == 2: legacy per-pair 2-lane parents + BUG-SKEW-1 align/re-skew
    //     registers, textually identical to the pre-7b code (bit-identity
    //     anchor for vpu.sv).
    //   N > 2: one N-wide layernorm_group_nxn / softmax_group_nxn with
    //     full de-skew (lane k delayed N-1-k cycles at the input) and
    //     re-skew (lane k delayed k cycles at the output). All shift
    //     registers are free-running with reset clear, the same
    //     discipline as the legacy BUG-SKEW-1 registers.
    // Both branches drive the module-scope *_routed view consumed by the
    // stage-chain routing below.
    genvar gp, gk;
    generate
        if (N == 2) begin : gen_group_pair
            // BUG-SKEW-1 fix, per pair: align the even lane to the odd lane at each
            // group stage's input (1-cycle delay), then re-apply the native skew at
            // its output (delay the odd lane by 1 cycle) so downstream per-lane
            // stages and VPU output timing are unchanged. Indexed by pair p.
            logic signed [15:0] ln_data_in_al [PAIRS];
            logic               ln_valid_in_al [PAIRS];
            logic signed [15:0] ln_data_out_rs [PAIRS];
            logic               ln_valid_out_rs [PAIRS];
            logic signed [15:0] sm_data_in_al [PAIRS];
            logic               sm_valid_in_al [PAIRS];
            logic signed [15:0] sm_data_out_rs [PAIRS];
            logic               sm_valid_out_rs [PAIRS];

            // Legacy pair structure, textually identical to the pre-7b code.
            for (gp = 0; gp < PAIRS; gp++) begin : gen_pair
                layernorm_parent layernorm_parent_inst (
                    .clk(clk),
                    .rst(rst),

                    .ln_data_1_in(ln_data_in_al[gp]),     // BUG-SKEW-1: even lane aligned to odd lane
                    .ln_data_2_in(ln_data_in_s[2*gp+1]),
                    .ln_valid_1_in(ln_valid_in_al[gp]),   // BUG-SKEW-1: even lane aligned to odd lane
                    .ln_valid_2_in(ln_valid_in_s[2*gp+1]),

                    .ln_data_1_out(ln_data_out_s[2*gp]),
                    .ln_data_2_out(ln_data_out_s[2*gp+1]),
                    .ln_valid_1_out(ln_valid_out_s[2*gp]),
                    .ln_valid_2_out(ln_valid_out_s[2*gp+1]),
                    .ln_overflow_out_1(),   // BUG-OVF-1: observable via hierarchical reference
                    .ln_overflow_out_2()
                );

                softmax_parent softmax_parent_inst (
                    .clk(clk),
                    .rst(rst),

                    .sm_data_1_in(sm_data_in_al[gp]),     // BUG-SKEW-1: even lane aligned to odd lane
                    .sm_data_2_in(sm_data_in_s[2*gp+1]),
                    .sm_valid_1_in(sm_valid_in_al[gp]),   // BUG-SKEW-1: even lane aligned to odd lane
                    .sm_valid_2_in(sm_valid_in_s[2*gp+1]),

                    .sm_data_1_out(sm_data_out_s[2*gp]),
                    .sm_data_2_out(sm_data_out_s[2*gp+1]),
                    .sm_valid_1_out(sm_valid_out_s[2*gp]),
                    .sm_valid_2_out(sm_valid_out_s[2*gp+1]),
                    .sm_overflow_out_1(),   // BUG-OVF-1: observable via hierarchical reference
                    .sm_overflow_out_2()
                );
            end

            // BUG-SKEW-1 fix, per pair: 1-cycle shift registers implementing the
            // lane alignment described at the declarations above. Free-running (no
            // enable) so valid pulse trains are delayed exactly one cycle; when a
            // group stage is bypassed these shift zeros and are unused.
            always_ff @(posedge clk or posedge rst) begin
                if (rst) begin
                    for (int p = 0; p < PAIRS; p++) begin
                        ln_data_in_al[p]   <= '0;
                        ln_valid_in_al[p]  <= '0;
                        ln_data_out_rs[p]  <= '0;
                        ln_valid_out_rs[p] <= '0;
                        sm_data_in_al[p]   <= '0;
                        sm_valid_in_al[p]  <= '0;
                        sm_data_out_rs[p]  <= '0;
                        sm_valid_out_rs[p] <= '0;
                    end
                end else begin
                    for (int p = 0; p < PAIRS; p++) begin
                        // align: the even lane arrives one cycle early, hold it for the odd lane
                        ln_data_in_al[p]   <= ln_data_in_s[2*p];
                        ln_valid_in_al[p]  <= ln_valid_in_s[2*p];
                        sm_data_in_al[p]   <= sm_data_in_s[2*p];
                        sm_valid_in_al[p]  <= sm_valid_in_s[2*p];
                        // re-skew: group-stage outputs are paired; delay the odd lane
                        // to restore the native one-cycle column skew downstream
                        ln_data_out_rs[p]  <= ln_data_out_s[2*p+1];
                        ln_valid_out_rs[p] <= ln_valid_out_s[2*p+1];
                        sm_data_out_rs[p]  <= sm_data_out_s[2*p+1];
                        sm_valid_out_rs[p] <= sm_valid_out_s[2*p+1];
                    end
                end
            end

            // Uniform per-lane routed view: even lanes direct, odd lanes
            // re-skewed (the legacy _1/_2 wiring).
            for (gp = 0; gp < PAIRS; gp++) begin : gen_route
                assign ln_data_out_routed[2*gp]    = ln_data_out_s[2*gp];
                assign ln_valid_out_routed[2*gp]   = ln_valid_out_s[2*gp];
                assign ln_data_out_routed[2*gp+1]  = ln_data_out_rs[gp];
                assign ln_valid_out_routed[2*gp+1] = ln_valid_out_rs[gp];
                assign sm_data_out_routed[2*gp]    = sm_data_out_s[2*gp];
                assign sm_valid_out_routed[2*gp]   = sm_valid_out_s[2*gp];
                assign sm_data_out_routed[2*gp+1]  = sm_data_out_rs[gp];
                assign sm_valid_out_routed[2*gp+1] = sm_valid_out_rs[gp];
            end
        end else begin : gen_group_nxn
            // De-skewed (beat-aligned) group inputs: lane k delayed N-1-k cycles.
            logic signed [15:0] ln_data_in_al [N];
            logic               ln_valid_in_al [N];
            logic signed [15:0] sm_data_in_al [N];
            logic               sm_valid_in_al [N];

            // Group-stage overflow outputs are unused here (BUG-OVF-1 style
            // observability is via the leaf's own mirrors).
            logic ln_overflow_unused [N];
            logic sm_overflow_unused [N];

            // One N-wide group stage each; the N lanes of one beat form the group.
            layernorm_group_nxn #(
                .SYSTOLIC_ARRAY_WIDTH(N)
            ) layernorm_group_nxn_inst (
                .clk(clk),
                .rst(rst),

                .ln_valid_in(ln_valid_in_al),
                .ln_data_in(ln_data_in_al),

                .ln_data_out(ln_data_out_s),
                .ln_valid_out(ln_valid_out_s),
                .ln_overflow_out(ln_overflow_unused)
            );

            softmax_group_nxn #(
                .SYSTOLIC_ARRAY_WIDTH(N)
            ) softmax_group_nxn_inst (
                .clk(clk),
                .rst(rst),

                .sm_valid_in(sm_valid_in_al),
                .sm_data_in(sm_data_in_al),

                .sm_data_out(sm_data_out_s),
                .sm_valid_out(sm_valid_out_s),
                .sm_overflow_out(sm_overflow_unused)
            );

            // Per-lane de-skew / re-skew shift-register chains. Free-running
            // with reset clear (same discipline as the BUG-SKEW-1 registers);
            // when a group stage is bypassed these shift zeros and are unused.
            for (gk = 0; gk < N; gk++) begin : gen_skew
                localparam int D_IN  = N - 1 - gk;  // de-skew depth: lane k waits for lane N-1
                localparam int D_OUT = gk;          // re-skew depth: restore the column skew

                if (D_IN == 0) begin : gen_din_direct
                    assign ln_data_in_al[gk]  = ln_data_in_s[gk];
                    assign ln_valid_in_al[gk] = ln_valid_in_s[gk];
                    assign sm_data_in_al[gk]  = sm_data_in_s[gk];
                    assign sm_valid_in_al[gk] = sm_valid_in_s[gk];
                end else begin : gen_din_chain
                    logic signed [15:0] ln_d [D_IN];
                    logic               ln_v [D_IN];
                    logic signed [15:0] sm_d [D_IN];
                    logic               sm_v [D_IN];
                    always_ff @(posedge clk or posedge rst) begin
                        if (rst) begin
                            for (int i = 0; i < D_IN; i++) begin
                                ln_d[i] <= '0;
                                ln_v[i] <= 1'b0;
                                sm_d[i] <= '0;
                                sm_v[i] <= 1'b0;
                            end
                        end else begin
                            ln_d[0] <= ln_data_in_s[gk];
                            ln_v[0] <= ln_valid_in_s[gk];
                            sm_d[0] <= sm_data_in_s[gk];
                            sm_v[0] <= sm_valid_in_s[gk];
                            for (int i = 1; i < D_IN; i++) begin
                                ln_d[i] <= ln_d[i-1];
                                ln_v[i] <= ln_v[i-1];
                                sm_d[i] <= sm_d[i-1];
                                sm_v[i] <= sm_v[i-1];
                            end
                        end
                    end
                    assign ln_data_in_al[gk]  = ln_d[D_IN-1];
                    assign ln_valid_in_al[gk] = ln_v[D_IN-1];
                    assign sm_data_in_al[gk]  = sm_d[D_IN-1];
                    assign sm_valid_in_al[gk] = sm_v[D_IN-1];
                end

                if (D_OUT == 0) begin : gen_dout_direct
                    assign ln_data_out_routed[gk]  = ln_data_out_s[gk];
                    assign ln_valid_out_routed[gk] = ln_valid_out_s[gk];
                    assign sm_data_out_routed[gk]  = sm_data_out_s[gk];
                    assign sm_valid_out_routed[gk] = sm_valid_out_s[gk];
                end else begin : gen_dout_chain
                    logic signed [15:0] ln_d [D_OUT];
                    logic               ln_v [D_OUT];
                    logic signed [15:0] sm_d [D_OUT];
                    logic               sm_v [D_OUT];
                    always_ff @(posedge clk or posedge rst) begin
                        if (rst) begin
                            for (int i = 0; i < D_OUT; i++) begin
                                ln_d[i] <= '0;
                                ln_v[i] <= 1'b0;
                                sm_d[i] <= '0;
                                sm_v[i] <= 1'b0;
                            end
                        end else begin
                            ln_d[0] <= ln_data_out_s[gk];
                            ln_v[0] <= ln_valid_out_s[gk];
                            sm_d[0] <= sm_data_out_s[gk];
                            sm_v[0] <= sm_valid_out_s[gk];
                            for (int i = 1; i < D_OUT; i++) begin
                                ln_d[i] <= ln_d[i-1];
                                ln_v[i] <= ln_v[i-1];
                                sm_d[i] <= sm_d[i-1];
                                sm_v[i] <= sm_v[i-1];
                            end
                        end
                    end
                    assign ln_data_out_routed[gk]  = ln_d[D_OUT-1];
                    assign ln_valid_out_routed[gk] = ln_v[D_OUT-1];
                    assign sm_data_out_routed[gk]  = sm_d[D_OUT-1];
                    assign sm_valid_out_routed[gk] = sm_v[D_OUT-1];
                end
            end
        end
    endgenerate

    // Combinational stage-chain routing, replicated per lane. Group-stage
    // outputs are consumed through the uniform per-lane *_routed view,
    // which the generate branches drive with the N-appropriate re-skew
    // (legacy odd-lane 1-cycle delay at N=2, per-lane delay-k chain at
    // N>2), so this block is identical for both structures.
    always @(*) begin
        for (int k = 0; k < N; k++) begin
            // Default assignments for all intermediate signals to prevent latch inference.
            // These are overridden by the routing logic below when rst is not asserted.
            b_to_lr_data[k]      = 16'b0;
            b_to_lr_valid[k]     = 1'b0;
            lr_to_loss_data[k]   = 16'b0;
            lr_to_loss_valid[k]  = 1'b0;
            gelu_to_loss_data[k]  = 16'b0;
            gelu_to_loss_valid[k] = 1'b0;
            silu_to_loss_data[k]  = 16'b0;
            silu_to_loss_valid[k] = 1'b0;
            ln_to_loss_data[k]   = 16'b0;
            ln_to_loss_valid[k]  = 1'b0;
            sm_to_loss_data[k]   = 16'b0;
            sm_to_loss_valid[k]  = 1'b0;
            loss_to_lrd_data[k]  = 16'b0;
            loss_to_lrd_valid[k] = 1'b0;
            last_H_data_in_s[k]  = 16'b0;
            lr_d_H_in_s[k]       = 16'b0;

            if (rst) begin
                vpu_data_mux[k]  = 16'b0;
                vpu_valid_mux[k] = 1'b0;

                // default internal wire assignments during reset
                bias_data_in_s[k]  = 16'b0;
                bias_valid_in_s[k] = 1'b0;
                lr_data_in_s[k]    = 16'b0;
                lr_valid_in_s[k]   = 1'b0;
                gelu_data_in_s[k]  = 16'b0;
                gelu_valid_in_s[k] = 1'b0;
                silu_data_in_s[k]  = 16'b0;
                silu_valid_in_s[k] = 1'b0;
                ln_data_in_s[k]    = 16'b0;
                ln_valid_in_s[k]   = 1'b0;
                sm_data_in_s[k]    = 16'b0;
                sm_valid_in_s[k]   = 1'b0;
                loss_data_in_s[k]  = 16'b0;
                loss_valid_in_s[k] = 1'b0;
                lr_d_data_in_s[k]  = 16'b0;
                lr_d_valid_in_s[k] = 1'b0;
            end else begin
                // bias module
                if (vpu_data_pathway[3]) begin
                    // connect vpu inputs to bias module
                    bias_data_in_s[k]  = vpu_data_in[k];
                    bias_valid_in_s[k] = vpu_valid_in[k];

                    // connect bias output to intermediate values
                    b_to_lr_data[k]  = bias_data_out_s[k];
                    b_to_lr_valid[k] = bias_valid_out_s[k];
                end else begin
                    // disable inputs
                    bias_data_in_s[k]  = 16'b0;
                    bias_valid_in_s[k] = 1'b0;

                    // connect vpu input to intermediate values
                    b_to_lr_data[k]  = vpu_data_in[k];
                    b_to_lr_valid[k] = vpu_valid_in[k];
                end

                // leaky relu module
                if (vpu_data_pathway[2]) begin
                    // connect lr inputs to intermediate values
                    lr_data_in_s[k]  = b_to_lr_data[k];
                    lr_valid_in_s[k] = b_to_lr_valid[k];

                    // connect lr outputs to intermediate values
                    lr_to_loss_data[k]  = lr_data_out_s[k];
                    lr_to_loss_valid[k] = lr_valid_out_s[k];
                end else begin
                    // disable inputs
                    lr_data_in_s[k]  = 16'b0;
                    lr_valid_in_s[k] = 1'b0;

                    // connect intermediate values to each other
                    lr_to_loss_data[k]  = b_to_lr_data[k];
                    lr_to_loss_valid[k] = b_to_lr_valid[k];
                end

                // gelu module
                if (vpu_data_pathway[4]) begin
                    // connect gelu inputs to intermediate values
                    gelu_data_in_s[k]  = lr_to_loss_data[k];
                    gelu_valid_in_s[k] = lr_to_loss_valid[k];

                    // connect gelu outputs to intermediate values
                    gelu_to_loss_data[k]  = gelu_data_out_s[k];
                    gelu_to_loss_valid[k] = gelu_valid_out_s[k];
                end else begin
                    // disable inputs
                    gelu_data_in_s[k]  = 16'b0;
                    gelu_valid_in_s[k] = 1'b0;

                    // bypass: connect intermediate values to each other
                    gelu_to_loss_data[k]  = lr_to_loss_data[k];
                    gelu_to_loss_valid[k] = lr_to_loss_valid[k];
                end

                // silu module (elementwise stage, like gelu)
                if (vpu_data_pathway[7]) begin
                    // connect silu inputs to intermediate values
                    silu_data_in_s[k]  = gelu_to_loss_data[k];
                    silu_valid_in_s[k] = gelu_to_loss_valid[k];

                    // connect silu outputs to intermediate values
                    silu_to_loss_data[k]  = silu_data_out_s[k];
                    silu_to_loss_valid[k] = silu_valid_out_s[k];
                end else begin
                    // disable inputs
                    silu_data_in_s[k]  = 16'b0;
                    silu_valid_in_s[k] = 1'b0;

                    // bypass: connect intermediate values to each other
                    silu_to_loss_data[k]  = gelu_to_loss_data[k];
                    silu_to_loss_valid[k] = gelu_to_loss_valid[k];
                end

                // layernorm module (group stage: operates on the pair)
                if (vpu_data_pathway[5]) begin
                    // connect layernorm inputs to intermediate values
                    ln_data_in_s[k]  = silu_to_loss_data[k];
                    ln_valid_in_s[k] = silu_to_loss_valid[k];

                    // connect layernorm outputs to intermediate values
                    // (uniform per-lane view; re-skew lives in the generate branches)
                    ln_to_loss_data[k]  = ln_data_out_routed[k];
                    ln_to_loss_valid[k] = ln_valid_out_routed[k];
                end else begin
                    // disable inputs
                    ln_data_in_s[k]  = 16'b0;
                    ln_valid_in_s[k] = 1'b0;

                    // bypass: connect intermediate values to each other
                    ln_to_loss_data[k]  = silu_to_loss_data[k];
                    ln_to_loss_valid[k] = silu_to_loss_valid[k];
                end

                // softmax module (group stage: operates on the pair)
                if (vpu_data_pathway[6]) begin
                    // connect softmax inputs to intermediate values
                    sm_data_in_s[k]  = ln_to_loss_data[k];
                    sm_valid_in_s[k] = ln_to_loss_valid[k];

                    // connect softmax outputs to intermediate values
                    // (uniform per-lane view; re-skew lives in the generate branches)
                    sm_to_loss_data[k]  = sm_data_out_routed[k];
                    sm_to_loss_valid[k] = sm_valid_out_routed[k];
                end else begin
                    // disable inputs
                    sm_data_in_s[k]  = 16'b0;
                    sm_valid_in_s[k] = 1'b0;

                    // bypass: connect intermediate values to each other
                    sm_to_loss_data[k]  = ln_to_loss_data[k];
                    sm_to_loss_valid[k] = ln_to_loss_valid[k];
                end

                // loss module
                if (vpu_data_pathway[1]) begin
                    // connect loss inputs to intermediate values
                    loss_data_in_s[k]  = sm_to_loss_data[k];
                    loss_valid_in_s[k] = sm_to_loss_valid[k];

                    // connect loss outputs to intermediate values
                    loss_to_lrd_data[k]  = loss_data_out_s[k];
                    loss_to_lrd_valid[k] = loss_valid_out_s[k];

                    // Cache and use 'last H matrix'
                    last_H_data_in_s[k] = lr_data_out_s[k];
                    lr_d_H_in_s[k]      = last_H_data_out_s[k];
                end else begin
                    // disable inputs
                    loss_data_in_s[k]  = 16'b0;
                    loss_valid_in_s[k] = 1'b0;

                    // connect intermediate values to each other
                    loss_to_lrd_data[k]  = sm_to_loss_data[k];
                    loss_to_lrd_valid[k] = sm_to_loss_valid[k];

                    // BUG-VPU-2 fix: clear last_H cache inputs when loss is not active
                    last_H_data_in_s[k] = 16'b0;
                    // BUG-VPU-3 fix: use UB H-input only during backward pass (pathway[0]=1)
                    lr_d_H_in_s[k]      = H_in[k];
                end

                // leaky relu derivative module
                if (vpu_data_pathway[0]) begin
                    lr_d_data_in_s[k]  = loss_to_lrd_data[k];
                    lr_d_valid_in_s[k] = loss_to_lrd_valid[k];

                    // connect lr_d outputs to vpu mux output
                    vpu_data_mux[k]  = lr_d_data_out_s[k];
                    vpu_valid_mux[k] = lr_d_valid_out_s[k];
                end else begin
                    // BUG-VPU-4 fix: zero lrd module inputs when disabled to prevent spurious state
                    lr_d_data_in_s[k]  = 16'b0;
                    lr_d_valid_in_s[k] = 1'b0;

                    // bypass: connect intermediate values directly to vpu mux output
                    vpu_data_mux[k]  = loss_to_lrd_data[k];
                    vpu_valid_mux[k] = loss_to_lrd_valid[k];
                end
            end
        end
    end

    // BUG-VPU-1 fix: register VPU outputs to prevent combinational glitches
    // BUG-VPU-2 fix: last_H cache cleared per lane when loss is inactive
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            for (int k = 0; k < N; k++) begin
                last_H_data_out_s[k] <= '0;
                vpu_data_out_r[k]      <= '0;
                vpu_valid_out_r[k]     <= '0;
            end
        end else begin
            for (int k = 0; k < N; k++) begin
                vpu_data_out_r[k]  <= vpu_data_mux[k];
                vpu_valid_out_r[k] <= vpu_valid_mux[k];
                if (vpu_data_pathway[1]) begin
                    last_H_data_out_s[k] <= last_H_data_in_s[k];
                end else begin
                    last_H_data_out_s[k] <= '0;
                end
            end
        end
    end

endmodule
