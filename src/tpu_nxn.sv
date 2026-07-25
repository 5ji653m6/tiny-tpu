`timescale 1ns/1ps
`default_nettype none

// Full-chip top, generalizing the hardcoded-2x2 integration in tpu.sv to
// SYSTOLIC_ARRAY_WIDTH = N lanes (N even, >= 2).
//
// Wires unified_buffer_nxn + systolic_nxn + vpu_nxn together. Same port
// list as tpu.sv; no outputs (results land in UB memory). The control_unit
// is intentionally NOT instantiated here: its instruction word carries only
// two host-write lanes, and widening it is a separate ISA item. The
// testbench drives the instruction fields directly, exactly as test_tpu.py
// does for tpu.sv.
//
// Behavioral notes:
//  - systolic_nxn has a SINGLE start port (sys_start_1): the valid stream is
//    chained through the whole array (BUG-SYS-1 design), so only lane 0 of
//    the UB's per-lane input valids drives it; lanes 1..N-1 are staggered by
//    the UB and ride the chained valid.
//  - systolic_nxn registers its left-edge inputs once on entry, so
//    intermediate cycle timing differs from tpu.sv by one cycle; final UB
//    memory contents are identical at N=2.

module tpu_nxn #(
    parameter int SYSTOLIC_ARRAY_WIDTH = 2
)(
    input logic clk,
    input logic rst,

    // UB wires (writing from host to UB)
    input logic [15:0] ub_wr_host_data_in [0:SYSTOLIC_ARRAY_WIDTH-1],
    input logic ub_wr_host_valid_in [0:SYSTOLIC_ARRAY_WIDTH-1],

    // UB wires (inputting reading instructions from host)
    input logic ub_rd_start_in,
    input logic ub_rd_transpose,
    input logic [8:0] ub_ptr_select,
    input logic [15:0] ub_rd_addr_in,
    input logic [15:0] ub_rd_row_size,
    input logic [15:0] ub_rd_col_size,

    // Learning rate
    input logic [15:0] learning_rate_in,

    // VPU data pathway (see tpu.sv / vpu_nxn.sv for bit definitions)
    input logic [7:0] vpu_data_pathway,

    input logic sys_switch_in,
    input logic [15:0] vpu_leak_factor_in,
    input logic [15:0] inv_batch_size_times_two_in
);

    localparam int N = SYSTOLIC_ARRAY_WIDTH;

    // BUG-TOOLS-1 (iverilog 11): an unpacked-array output port only
    // propagates to a parent net declared 'wire' — a parent 'logic'
    // array silently reads X (proven with a minimal repro). All array
    // interconnects in this top are therefore wires. Scalars are
    // unaffected and stay logic.
    // UB write-back wires (from VPU)
    wire [15:0] ub_wr_data_in [N];
    wire ub_wr_valid_in [N];

    // UB outputs to systolic array (left side)
    wire [15:0] ub_rd_input_data_out [N];
    wire ub_rd_input_valid_out [N];

    // UB outputs to systolic array (top)
    wire [15:0] ub_rd_weight_data_out [N];
    wire ub_rd_weight_valid_out [N];

    // UB outputs to VPU
    wire [15:0] ub_rd_bias_data_out [N];
    wire [15:0] ub_rd_Y_data_out [N];
    wire [15:0] ub_rd_H_data_out [N];

    // Column-count handshake to systolic array
    logic [15:0] ub_rd_col_size_out;
    logic ub_rd_col_size_valid_out;

    // Systolic array outputs to VPU
    wire [15:0] sys_data_out [N];
    wire sys_valid_out [N];

    // VPU outputs (write back into UB)
    wire [15:0] vpu_data_out [N];
    wire vpu_valid_out [N];

    // SKEW COMPENSATION: systolic_nxn registers its left-edge inputs once
    // on entry, so every systolic output beat reaches the VPU one cycle
    // later than in legacy tpu.sv given identical UB read timing. The UB's
    // bias/Y/H read streams are choreographed (by the instruction script)
    // to legacy systolic latency, and the VPU's leaf children pair these
    // operands COMBINATIONALLY with each systolic-valid beat (see
    // bias_child.sv). Without compensation the last beat of each stream
    // pairs with a zero/stale operand (observed: H row N computed with
    // bias 0). Register the three UB->VPU operand streams once here so
    // beat k of each operand stream coincides with systolic beat k.
    logic [15:0] bias_d [N];
    logic [15:0] Y_d [N];
    logic [15:0] H_d [N];
    always_ff @(posedge clk) begin
        for (int i = 0; i < N; i++) begin
            bias_d[i] <= ub_rd_bias_data_out[i];
            Y_d[i]    <= ub_rd_Y_data_out[i];
            H_d[i]    <= ub_rd_H_data_out[i];
        end
    end

    // SKEW COMPENSATION (gradient descent): the UB's gradient-capture
    // window is opened by ptr-5/6 read commands and sized to legacy
    // systolic latency, so the +1-shifted VPU gradient stream slides
    // one beat past the window close (observed: legacy captures lane 0
    // and clips lane 1; uncompensated nxn clips BOTH lanes). Delaying
    // exactly the grad read commands by one cycle re-aligns the window
    // and the value_old_in walk with the delayed stream, reproducing
    // legacy clip semantics beat for beat. All other read commands feed
    // the systolic left edge (which owns its own entry register) or the
    // compensated operand streams above, and must NOT be delayed.
    logic        grad_cmd_pending;
    logic [8:0]  grad_ptr_q;
    logic [15:0] grad_addr_q;
    logic [15:0] grad_row_q;
    logic [15:0] grad_col_q;

    wire grad_cmd = ub_rd_start_in &&
                    (ub_ptr_select == 9'd5 || ub_ptr_select == 9'd6);

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            grad_cmd_pending <= 1'b0;
            grad_ptr_q       <= '0;
            grad_addr_q      <= '0;
            grad_row_q       <= '0;
            grad_col_q       <= '0;
        end else begin
            grad_cmd_pending <= grad_cmd;
            if (grad_cmd) begin
                grad_ptr_q  <= ub_ptr_select;
                grad_addr_q <= ub_rd_addr_in;
                grad_row_q  <= ub_rd_row_size;
                grad_col_q  <= ub_rd_col_size;
            end
        end
    end

    wire        ub_rd_start_d1 = grad_cmd_pending ||
                                 (ub_rd_start_in && !grad_cmd);
    wire [8:0]  ub_ptr_d1  = grad_cmd_pending ? grad_ptr_q  : ub_ptr_select;
    wire [15:0] ub_addr_d1 = grad_cmd_pending ? grad_addr_q : ub_rd_addr_in;
    wire [15:0] ub_row_d1  = grad_cmd_pending ? grad_row_q  : ub_rd_row_size;
    wire [15:0] ub_col_d1  = grad_cmd_pending ? grad_col_q  : ub_rd_col_size;

    genvar i;
    generate
        for (i = 0; i < N; i++) begin : g_wb
            assign ub_wr_data_in[i]  = vpu_data_out[i];
            assign ub_wr_valid_in[i] = vpu_valid_out[i];
        end
    endgenerate

    unified_buffer_nxn #(
        .SYSTOLIC_ARRAY_WIDTH(SYSTOLIC_ARRAY_WIDTH)
    ) ub_inst (
        .clk(clk),
        .rst(rst),

        .ub_wr_data_in(ub_wr_data_in),
        .ub_wr_valid_in(ub_wr_valid_in),

        .ub_wr_host_data_in(ub_wr_host_data_in),
        .ub_wr_host_valid_in(ub_wr_host_valid_in),

        .ub_rd_start_in(ub_rd_start_d1),
        .ub_rd_transpose(ub_rd_transpose),
        .ub_ptr_select(ub_ptr_d1),
        .ub_rd_addr_in(ub_addr_d1),
        .ub_rd_row_size(ub_row_d1),
        .ub_rd_col_size(ub_col_d1),

        .learning_rate_in(learning_rate_in),

        .ub_rd_input_data_out(ub_rd_input_data_out),
        .ub_rd_input_valid_out(ub_rd_input_valid_out),

        .ub_rd_weight_data_out(ub_rd_weight_data_out),
        .ub_rd_weight_valid_out(ub_rd_weight_valid_out),

        .ub_rd_bias_data_out(ub_rd_bias_data_out),
        .ub_rd_Y_data_out(ub_rd_Y_data_out),
        .ub_rd_H_data_out(ub_rd_H_data_out),

        .ub_rd_col_size_out(ub_rd_col_size_out),
        .ub_rd_col_size_valid_out(ub_rd_col_size_valid_out)
    );

    systolic_nxn #(
        .SYSTOLIC_ARRAY_WIDTH(SYSTOLIC_ARRAY_WIDTH)
    ) sys_inst (
        .clk(clk),
        .rst(rst),

        .sys_data_in(ub_rd_input_data_out),
        // Single start port: the valid stream is chained through the array
        // (BUG-SYS-1 design); only lane 0 drives it.
        .sys_start_1(ub_rd_input_valid_out[0]),

        .sys_data_out(sys_data_out),
        .sys_valid_out(sys_valid_out),

        .sys_weight_in(ub_rd_weight_data_out),
        .sys_accept_w(ub_rd_weight_valid_out),

        .sys_switch_in(sys_switch_in),

        .ub_rd_col_size_in(ub_rd_col_size_out),
        .ub_rd_col_size_valid_in(ub_rd_col_size_valid_out)
    );

    vpu_nxn #(
        .SYSTOLIC_ARRAY_WIDTH(SYSTOLIC_ARRAY_WIDTH)
    ) vpu_inst (
        .clk(clk),
        .rst(rst),

        .vpu_data_pathway(vpu_data_pathway),

        .vpu_data_in(sys_data_out),
        .vpu_valid_in(sys_valid_out),

        .bias_scalar_in(bias_d),
        .lr_leak_factor_in(vpu_leak_factor_in),
        .Y_in(Y_d),
        .inv_batch_size_times_two_in(inv_batch_size_times_two_in),
        .H_in(H_d),

        .vpu_data_out(vpu_data_out),
        .vpu_valid_out(vpu_valid_out)
    );

endmodule

`default_nettype wire
