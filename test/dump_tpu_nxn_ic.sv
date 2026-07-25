// Parameterized dump wrapper for the tpu_nxn_ic instruction-driven
// full-chip test (roadmap item 8b). Compiled with -Pdump.N=4. Outputs
// are left unconnected (results live in UB memory, read hierarchically
// by the cocotb test). Written by the harness author, not the agent.
`timescale 1ns/1ps
`default_nettype none

module dump #(
    parameter int N = 2
)();

    logic clk;
    logic rst;

    // 134 + 17*(N-2)-bit instruction word (item-8a layout + the item-13
    // SiLU pathway bit appended at the top; pre-item-13-RTL iverilog
    // prunes the MSB — 0 for all old programs — into the 133-based DUT
    // port with a warning)
    logic [133+17*(N-2):0] instruction;   // item 13: SiLU bit (MSB)
    logic [15:0] learning_rate_in;

    tpu_nxn_ic #(
        .SYSTOLIC_ARRAY_WIDTH(N)
    ) tpu_nxn_ic_inst (
        .clk(clk),
        .rst(rst),
        .instruction(instruction),
        .learning_rate_in(learning_rate_in)
    );

    initial begin
        $dumpfile("tpu_nxn_ic.vcd");
        $dumpvars(1, dump);
    end

endmodule
