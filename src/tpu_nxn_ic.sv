`timescale 1ns/1ps
`default_nettype none

// First instruction-consuming full-chip top (roadmap item 8b): wires
// control_unit_nxn in front of tpu_nxn so the host drives a single
// instruction word instead of the individual control fields.
//
// This is a THIN SHELL: decode + instantiate + wire. No new logic, no
// registers, no re-implementation of tpu_nxn internals. The decode is
// purely combinational (control_unit_nxn has no clock), so the host
// must hold each instruction word stable for exactly the cycles it
// means, exactly as the legacy testbenches hold the raw fields.
//
// The learning rate is NOT part of the instruction word; it stays a
// separate port (learning_rate_in) and is passed straight through to
// tpu_nxn.
//
// Instruction word: 134 + 17*(N-2) bits (see control_unit_nxn.sv for
// the field layout; legacy 133-bit images zero-extend and keep working).
// No outputs: results land in UB memory, as in tpu_nxn.
//
// BUG-TOOLS-1 (iverilog 11): the N-wide host-lane arrays between the
// two instances are declared 'wire' — unpacked-array outputs only
// propagate to parent nets declared wire (see tpu_nxn.sv).

module tpu_nxn_ic #(
    parameter int SYSTOLIC_ARRAY_WIDTH = 2,
    // UB depth in words (item 14: parameterized for larger composites)
    parameter int UNIFIED_BUFFER_WIDTH = 128
)(
    input logic clk,
    input logic rst,

    // 134 + 17*(N-2) bits; the SiLU pathway bit is the MSB
    input logic [133+17*(SYSTOLIC_ARRAY_WIDTH-2):0] instruction,

    // Learning rate (separate port, not in the instruction word)
    input logic [15:0] learning_rate_in
);

    localparam int N = SYSTOLIC_ARRAY_WIDTH;

    // Decoded scalar fields (combinational, straight off the instruction)
    logic        sys_switch_in;
    logic        ub_rd_start_in;
    logic        ub_rd_transpose;
    logic [15:0] ub_rd_col_size;
    logic [15:0] ub_rd_row_size;
    logic [15:0] ub_rd_addr_in;
    logic [8:0]  ub_ptr_select;
    logic [7:0]  vpu_data_pathway;
    logic [15:0] vpu_leak_factor_in;
    logic [15:0] inv_batch_size_times_two_in;

    // BUG-TOOLS-1 (iverilog 11): unpacked-array outputs only propagate
    // to parent nets declared 'wire' (same interconnect style as
    // tpu_nxn.sv). The signed/unsigned width-matched connection between
    // the control unit's signed data lanes and tpu_nxn's unsigned host
    // lanes is bit-for-bit identical.
    wire [15:0] ub_wr_host_data_in [N];
    wire        ub_wr_host_valid_in [N];

    control_unit_nxn #(
        .SYSTOLIC_ARRAY_WIDTH(SYSTOLIC_ARRAY_WIDTH)
    ) control_unit_inst (
        .instruction(instruction),

        .sys_switch_in(sys_switch_in),
        .ub_rd_start_in(ub_rd_start_in),
        .ub_rd_transpose(ub_rd_transpose),
        // legacy scalar host-lane aliases (lanes 0/1) left unconnected;
        // the N-wide arrays below carry the same decode
        .ub_wr_host_valid_in_1(),
        .ub_wr_host_valid_in_2(),

        .ub_rd_col_size(ub_rd_col_size),
        .ub_rd_row_size(ub_rd_row_size),
        .ub_rd_addr_in(ub_rd_addr_in),
        .ub_ptr_select(ub_ptr_select),

        .ub_wr_host_data_in_1(),
        .ub_wr_host_data_in_2(),

        .vpu_data_pathway(vpu_data_pathway),
        .inv_batch_size_times_two_in(inv_batch_size_times_two_in),
        .vpu_leak_factor_in(vpu_leak_factor_in),

        .ub_wr_host_valid_in(ub_wr_host_valid_in),
        .ub_wr_host_data_in(ub_wr_host_data_in)
    );

    tpu_nxn #(
        .SYSTOLIC_ARRAY_WIDTH(SYSTOLIC_ARRAY_WIDTH),
        .UNIFIED_BUFFER_WIDTH(UNIFIED_BUFFER_WIDTH)
    ) tpu_nxn_inst (
        .clk(clk),
        .rst(rst),

        .ub_wr_host_data_in(ub_wr_host_data_in),
        .ub_wr_host_valid_in(ub_wr_host_valid_in),

        .ub_rd_start_in(ub_rd_start_in),
        .ub_rd_transpose(ub_rd_transpose),
        .ub_ptr_select(ub_ptr_select),
        .ub_rd_addr_in(ub_rd_addr_in),
        .ub_rd_row_size(ub_rd_row_size),
        .ub_rd_col_size(ub_rd_col_size),

        .learning_rate_in(learning_rate_in),

        .vpu_data_pathway(vpu_data_pathway),

        .sys_switch_in(sys_switch_in),
        .vpu_leak_factor_in(vpu_leak_factor_in),
        .inv_batch_size_times_two_in(inv_batch_size_times_two_in)
    );

endmodule

`default_nettype wire
