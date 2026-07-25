// Parameterized dump wrapper for the tpu_nxn_prog program-driven
// full-chip test (roadmap item 9b). Compiled with -Pdump.N=4 (and
// -Pdump.PROG_DEPTH=512 for the item-10 two-step program, which needs
// 286 words). Outputs are left unconnected (results live in UB memory,
// read hierarchically by the cocotb test). Written by the harness
// author, not the agent.
//
// Item 12: prog_wr_data is W+1 bits — the program word is {ctrl,
// legacy_word}, ctrl the MSB (LOOP control-word construct). Until the
// item-12 RTL widens the DUT port to match, iverilog prunes the MSB
// (0 for all pre-item-12 programs) with a warning; the item-12 tests
// fail red by design in the meantime.
`timescale 1ns/1ps
`default_nettype none

module dump #(
    parameter int N = 2,
    parameter int PROG_DEPTH = 256,
    // UB depth passthrough (item 14: multi-head attention needs 256)
    parameter int UB_WIDTH = 128
)();

    localparam int W = 134 + 17*(N-2);   // item 13: SiLU bit

    logic clk;
    logic rst;

    // Program load + run (the instruction sequencer interface)
    logic prog_wr_en;
    logic [W:0] prog_wr_data;
    logic run;
    logic busy;

    logic [15:0] learning_rate_in;

    tpu_nxn_prog #(
        .SYSTOLIC_ARRAY_WIDTH(N),
        .PROG_DEPTH(PROG_DEPTH),
        .UNIFIED_BUFFER_WIDTH(UB_WIDTH)
    ) tpu_nxn_prog_inst (
        .clk(clk),
        .rst(rst),
        .prog_wr_en(prog_wr_en),
        .prog_wr_data(prog_wr_data),
        .run(run),
        .busy(busy),
        .learning_rate_in(learning_rate_in)
    );

    initial begin
        $dumpfile("tpu_nxn_prog.vcd");
        $dumpvars(1, dump);
    end

endmodule
