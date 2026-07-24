// Parameterized dump wrapper for the tpu_nxn_prog program-driven
// full-chip test (roadmap item 9b). Compiled with -Pdump.N=4. Outputs
// are left unconnected (results live in UB memory, read hierarchically
// by the cocotb test). Written by the harness author, not the agent.
`timescale 1ns/1ps
`default_nettype none

module dump #(
    parameter int N = 2
)();

    localparam int W = 133 + 17*(N-2);

    logic clk;
    logic rst;

    // Program load + run (the instruction sequencer interface)
    logic prog_wr_en;
    logic [W-1:0] prog_wr_data;
    logic run;
    logic busy;

    logic [15:0] learning_rate_in;

    tpu_nxn_prog #(
        .SYSTOLIC_ARRAY_WIDTH(N)
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
