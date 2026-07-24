`timescale 1ns/1ps
`default_nettype none

// Program-driven full-chip top (roadmap item 9b): wires the loadable
// instruction sequencer (instr_seq_nxn) in front of the
// instruction-consuming chip top (tpu_nxn_ic), so the host loads a
// program of instruction words once and a single `run` pulse replays
// the whole choreography (e.g. the item-8b training step) with no
// further per-cycle host driving.
//
// This is a THIN SHELL: sequencer + chip + wires. No new logic, no
// registers, no re-implementation of tpu_nxn_ic internals.
//   - seq_inst.instr_out drives tpu_nxn_ic_inst.instruction
//   - busy reflects the sequencer's replay-in-flight flag
//   - learning_rate_in passes straight through (it is a chip port,
//     not part of the instruction word)
// No outputs other than busy: results land in UB memory, as in tpu_nxn.
//
// Program load protocol (see instr_seq_nxn.sv): while busy = 0, drive
// prog_wr_en + prog_wr_data one word per cycle (loads work even while
// rst is held); pulse run for one cycle to replay. Loading a different
// program requires rst first.

module tpu_nxn_prog #(
    parameter int SYSTOLIC_ARRAY_WIDTH = 2,
    parameter int PROG_DEPTH = 256
) (
    input logic clk,
    input logic rst,

    // Program load + run (the instruction sequencer interface)
    input logic prog_wr_en,
    input logic [132+17*(SYSTOLIC_ARRAY_WIDTH-2):0] prog_wr_data,
    input logic run,
    output logic busy,

    // Learning rate (separate port, not in the instruction word)
    input logic [15:0] learning_rate_in
);

    localparam int N = SYSTOLIC_ARRAY_WIDTH;
    localparam int W = 133 + 17*(N-2);

    // Sequencer -> chip instruction stream
    logic [W-1:0] instr;

    instr_seq_nxn #(
        .SYSTOLIC_ARRAY_WIDTH(SYSTOLIC_ARRAY_WIDTH),
        .PROG_DEPTH(PROG_DEPTH)
    ) seq_inst (
        .clk(clk),
        .rst(rst),
        .prog_wr_en(prog_wr_en),
        .prog_wr_data(prog_wr_data),
        .run(run),
        .busy(busy),
        .instr_out(instr)
    );

    tpu_nxn_ic #(
        .SYSTOLIC_ARRAY_WIDTH(SYSTOLIC_ARRAY_WIDTH)
    ) tpu_nxn_ic_inst (
        .clk(clk),
        .rst(rst),
        .instruction(instr),
        .learning_rate_in(learning_rate_in)
    );

endmodule

`default_nettype wire
