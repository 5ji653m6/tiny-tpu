`timescale 1ns/1ps
`default_nettype none

// Loadable instruction sequencer (roadmap item 9b): stores a program
// of 133+17*(N-2)-bit instruction words and replays it, one word per
// cycle, into an instruction consumer (tpu_nxn_ic).
//
// Usage model:
//   1. While NOT busy, the host loads the program one word per cycle
//      (prog_wr_en + prog_wr_data); the write pointer auto-increments
//      from 0. Writes while busy are IGNORED (a replay in flight is
//      never corrupted by host writes).
//   2. A 1-cycle `run` pulse while not busy starts the replay IF the
//      write pointer is nonzero; a run with an empty program is
//      ignored (busy stays 0). Starting the cycle AFTER the run pulse
//      is sampled, instr_out presents prog[0], prog[1], ..., prog[M-1]
//      (M = write-pointer count at the run pulse), one word per cycle,
//      with busy = 1. The cycle after prog[M-1], instr_out returns to
//      0 and busy drops.
//   3. A later run pulse replays the SAME program: running does NOT
//      clear the write pointer. Loading a different program requires
//      rst first.
//
// Reset-note: rst clears busy/instr_out every cycle it is held, but
// clears the write pointer only on ENTRY into reset (0->1) -- and a
// write takes priority even over that entry clear. A program loaded
// while the chip is held in reset therefore survives until rst
// releases (tpu_nxn_prog's testbench loads exactly this way), and a
// fresh rst still resets the pointer for loading a different program.
// All state is registered, synchronous to clk.

module instr_seq_nxn #(
    parameter int SYSTOLIC_ARRAY_WIDTH = 2,
    parameter int PROG_DEPTH = 256
) (
    input logic clk,
    input logic rst,

    // Program load interface (ignored while busy)
    input logic prog_wr_en,
    input logic [132+17*(SYSTOLIC_ARRAY_WIDTH-2):0] prog_wr_data,

    // 1-cycle pulse starts the replay (ignored while busy or empty)
    input logic run,

    output logic busy,
    output logic [132+17*(SYSTOLIC_ARRAY_WIDTH-2):0] instr_out
);

    localparam int N = SYSTOLIC_ARRAY_WIDTH;
    localparam int W = 133 + 17*(N-2);
    // Wide enough to represent PROG_DEPTH itself (full memory).
    localparam int PTR_W = $clog2(PROG_DEPTH + 1);

    // Program memory and load pointer
    logic [W-1:0] prog_mem [PROG_DEPTH];
    logic [PTR_W-1:0] wr_ptr;

    // Replay read pointer
    logic [PTR_W-1:0] rd_ptr;

    // Previous-cycle rst, for entry-into-reset detection. Initialized
    // deasserted so a rst already high at time 0 still clears wr_ptr.
    logic rst_d = 1'b0;

    always @(posedge clk) begin
        rst_d <= rst;
        if (rst) begin
            busy      <= 1'b0;
            instr_out <= '0;
            rd_ptr    <= '0;
            if (prog_wr_en && (wr_ptr < PROG_DEPTH[PTR_W-1:0])) begin
                // Write takes priority over the pointer clear so a
                // program can be loaded while the chip is held in rst.
                prog_mem[wr_ptr] <= prog_wr_data;
                wr_ptr           <= wr_ptr + 1'b1;
            end else if (!rst_d) begin
                // Entry into reset: clear the write pointer exactly once.
                wr_ptr <= '0;
            end
            // else: hold wr_ptr -- a program loaded during this reset
            // survives until rst releases.
        end else if (busy) begin
            // Replay in flight: writes are ignored (no prog_wr_en branch).
            if (rd_ptr < wr_ptr) begin
                instr_out <= prog_mem[rd_ptr];
                rd_ptr    <= rd_ptr + 1'b1;
            end else begin
                // prog[M-1] was presented last cycle: return to idle.
                instr_out <= '0;
                busy      <= 1'b0;
                rd_ptr    <= '0;
            end
        end else begin
            // Idle: host may load the program and/or pulse run.
            if (prog_wr_en && (wr_ptr < PROG_DEPTH[PTR_W-1:0])) begin
                prog_mem[wr_ptr] <= prog_wr_data;
                wr_ptr           <= wr_ptr + 1'b1;
            end
            if (run && (wr_ptr != '0)) begin
                // Run pulse sampled with a non-empty program: present
                // prog[0] the very next cycle (this NBA update), with
                // busy = 1. The write pointer is NOT cleared, so a
                // later run replays the same program.
                busy      <= 1'b1;
                instr_out <= prog_mem[0];
                rd_ptr    <= {{(PTR_W-1){1'b0}}, 1'b1};
            end
        end
    end

endmodule

`default_nettype wire
