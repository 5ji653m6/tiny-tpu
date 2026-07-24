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

    // 133 + 17*(N-2)-bit instruction word (item-8a layout)
    logic [132+17*(N-2):0] instruction;
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
